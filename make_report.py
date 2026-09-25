"""Collect all run JSONs into results/summary.csv and plot WER by model and language.

Metrics are recomputed from the raw texts in each run CSV, so they always match common.normalize().
Usage: python make_report.py
"""
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import bootstrap_ci, normalize, score

LANG_COLORS = {"kk": "#2a78d6", "ru": "#eb6834", "en": "#1baf7a"}
LANG_NAMES = {"kk": "Kazakh", "ru": "Russian", "en": "English"}
WER_CAP = 1.0  # y-axis cap; bars above it are drawn clipped and labelled with the true value


def rescore(json_path):
    """Recompute WER/CER from the raw ref/hyp in the run's CSV with the current normalize(),
    update the JSON and CSV in place, and add the runaway share (normalized hypothesis
    >1.5x the reference length: loops/hallucinations)."""
    csv_path = json_path[:-5] + ".csv"
    d = pd.read_csv(csv_path, keep_default_na=False)
    d["ref_norm"] = d.ref.map(normalize)
    d["hyp_norm"] = d.hyp.map(normalize)
    # write to a temp file, then atomic rename, so an interrupted run never truncates raw results
    d.to_csv(csv_path + ".tmp", index=False)
    os.replace(csv_path + ".tmp", csv_path)
    r = json.load(open(json_path))
    r.update(score(list(d.ref), list(d.hyp)))
    r.update(bootstrap_ci(list(d.ref), list(d.hyp)))
    json.dump(r, open(json_path + ".tmp", "w"), indent=2, ensure_ascii=False)
    os.replace(json_path + ".tmp", json_path)
    r["runaway_share"] = float((d.hyp_norm.str.len() > 1.5 * d.ref_norm.str.len()).mean())
    return r


rows = []
for f in sorted(glob.glob("results/phase*/*.json")):
    r = rescore(f)
    r["phase"] = os.path.basename(os.path.dirname(f))
    # run name from the file, not from r["model"]: it also carries the compute type (phase 2)
    r["model_short"] = os.path.basename(f)[:-5].split("__")[0]
    rows.append(r)
df = pd.DataFrame(rows)
df.to_csv("results/summary.csv", index=False)

for phase, d in df.groupby("phase"):
    print(f"\n## {phase}")
    for metric in ["wer", "cer", "rtf", "runaway_share"]:
        print(f"\n{metric}:")
        print(d.pivot_table(index="model_short", columns="lang", values=metric).round(4).to_string())

# Plot: zero-shot GPU runs (+ fine-tuned models once phase 3 exists)
gpu = df[df.phase.isin(["phase1", "phase3", "phase6_lora"])]
# Whisper models by size, non-Whisper reference (MMS) last
models = list(gpu.sort_values(["model_short"]).assign(mms=gpu.model_short.str.contains("mms"))
              .sort_values(["mms", "params_m"]).model_short.drop_duplicates())
langs = [l for l in LANG_COLORS if l in set(gpu.lang)]
x = np.arange(len(models))
w = 0.8 / len(langs)

fig, ax = plt.subplots(figsize=(1.6 * len(models) + 2, 4.5))
for i, lang in enumerate(langs):
    vals = [gpu[(gpu.model_short == m) & (gpu.lang == lang)].wer.mean() for m in models]
    pos = x + (i - (len(langs) - 1) / 2) * w
    ax.bar(pos, np.minimum(vals, WER_CAP), w - 0.03, color=LANG_COLORS[lang], label=LANG_NAMES[lang])
    for p, v in zip(pos, vals):
        if v > WER_CAP:
            ax.text(p, WER_CAP + 0.02, f"{v:.2f}↑", ha="center", va="bottom", fontsize=8, color="#333")

ax.set_xticks(x, models, rotation=15, ha="right")
ax.set_ylim(0, WER_CAP + 0.15)
ax.set_ylabel("WER (FLEURS test, lower is better)")
ax.set_title("WER by model and language on FLEURS test (bars above 1.0 clipped, true value shown)\n"
             "All zero-shot except the two models fine-tuned on Kazakh", fontsize=11)
ax.yaxis.grid(True, color="#e5e5e5")
ax.set_axisbelow(True)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
ax.legend(frameon=False, ncol=len(langs), loc="upper right")
fig.tight_layout()
fig.savefig("results/wer_by_model_lang.png", dpi=150)
# Second plot: Kazakh in-domain (FLEURS test) vs out-of-domain (KSC test)
ksc = df[df.phase.isin(["phase4_ksc", "phase6_lora_ksc"])]
if len(ksc):
    fl = df[(df.lang == "kk") & (df.phase.isin(["phase1", "phase3", "phase6_lora"]))].set_index("model_short").wer
    names = [m for m in ["whisper-small", "whisper-small-kk-final", "whisper-small-kk-lora-final",
                         "whisper-large-v3-turbo", "mms-1b-all"]
             if m in set(ksc.model_short)]
    pairs = [("FLEURS test (read news sentences)", [fl[m] for m in names], "#2a78d6"),
             ("KSC test (out of domain)", [ksc.set_index("model_short").wer[m] for m in names], "#eb6834")]
    xx = np.arange(len(names))
    fig2, ax2 = plt.subplots(figsize=(1.9 * len(names) + 2, 4.2))
    for i, (label, vals, color) in enumerate(pairs):
        pos = xx + (i - 0.5) * 0.4
        ax2.bar(pos, np.minimum(vals, WER_CAP), 0.37, color=color, label=label)
        for px, v in zip(pos, vals):
            ax2.text(px, min(v, WER_CAP) + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=9, color="#333")
    ax2.set_xticks(xx, names, rotation=12, ha="right")
    ax2.set_ylim(0, WER_CAP)
    ax2.set_ylabel("Kazakh WER (lower is better)")
    ax2.set_title("Kazakh WER in domain vs out of domain\n"
                  "-kk-final (full fine-tuning) and -kk-lora-final were trained on FLEURS kk train", fontsize=11)
    ax2.yaxis.grid(True, color="#e5e5e5")
    ax2.set_axisbelow(True)
    for sp in ["top", "right"]:
        ax2.spines[sp].set_visible(False)
    ax2.legend(frameon=False, loc="upper right")
    fig2.tight_layout()
    fig2.savefig("results/kk_wer_fleurs_vs_ksc.png", dpi=150)

# Third plot: the trade-off between the target language and the others (LoRA learning-rate sweep)
TRADEOFF = [  # label, phase, model_short, label offset, sweep this point belongs to
    ("no fine-tuning", "phase1", "whisper-small", (-140, 14), None),
    ("full FT, lr 1e-5", "phase3", "whisper-small-kk-final", (-52, 16), None),
    ("LoRA r=32, lr 1e-4", "phase7_lora_lr1e-4", "lora-lr1e-4-final", (10, -6), "lr"),
    ("LoRA r=32, lr 3e-4", "phase7_lora_lr3e-4", "lora-lr3e-4-final", (8, 12), "lr"),
    ("LoRA r=32, lr 1e-3", "phase6_lora", "whisper-small-kk-lora-final", (12, -16), "lr rank"),
    ("LoRA r=8", "phase13_lora_r8", "lora-r8-final", (10, 8), "rank"),
    ("LoRA r=64", "phase13_lora_r64", "lora-r64-final", (12, -4), "rank"),
    ("full FT + 20% ru/en data", "phase11_mix", "whisper-small-kk-mix-final", (18, 10), None),
    ("LoRA r=32 + 20% ru/en data", "phase14_lora_mix", "lora-mix-final", (-34, -44), None),
]
pts = []
for label, phase, model, off, sweep in TRADEOFF:
    sel = df[(df.phase == phase) & (df.model_short == model)].set_index("lang").wer
    if {"kk", "ru"} <= set(sel.index):
        pts.append((label, sel["kk"], sel["ru"], off, sweep or ""))
if len(pts) >= 3:
    fig3, ax3 = plt.subplots(figsize=(7.6, 5.4))
    for sweep, color, name in [("lr", "#c3c2b7", "LoRA learning-rate sweep (r=32)"),
                               ("rank", "#9ec5f4", "LoRA rank sweep (lr 1e-3)")]:
        line = sorted([p for p in pts if sweep in p[4]], key=lambda p: p[1])
        if len(line) > 1:
            ax3.plot([p[1] for p in line], [p[2] for p in line], "-", color=color, zorder=1, label=name)
    for label, kk, ru, off, sweep in pts:
        color = "#1baf7a" if "ru/en" in label else ("#2a78d6" if "LoRA" in label else "#eb6834")
        ax3.scatter(kk, ru, s=90, color=color, zorder=2, edgecolor="white", linewidth=1.5)
        ax3.annotate(f"{label}\nkk {kk:.3f} / ru {ru:.3f}", (kk, ru), textcoords="offset points",
                     xytext=off, fontsize=8.5, color="#333")
    ax3.set_xlabel("Kazakh WER, FLEURS test (target language, lower is better)")
    ax3.set_ylabel("Russian WER, FLEURS test (forgetting, lower is better)")
    ax3.set_title("Adapting to Kazakh trades off against Russian,\nunless other languages are rehearsed during training",
                  fontsize=11)
    ax3.grid(True, color="#eeeeee")
    ax3.set_axisbelow(True)
    for sp in ["top", "right"]:
        ax3.spines[sp].set_visible(False)
    ax3.margins(0.2)
    ax3.legend(frameon=False, loc="upper left", fontsize=9)
    fig3.tight_layout()
    fig3.savefig("results/kk_vs_ru_tradeoff.png", dpi=150)

print("\nwrote results/summary.csv, results/wer_by_model_lang.png")
