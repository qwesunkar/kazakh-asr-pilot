"""Phase 1: zero-shot WER/CER, RTF and peak VRAM on FLEURS test (GPU).

Usage:
  python 01_zero_shot.py --limit 50 --out results/phase1_smoke
  python 01_zero_shot.py --out results/phase1
  python 01_zero_shot.py --models checkpoints/whisper-small-kk/best --out results/phase3
"""
import argparse
import csv
import json
import os
import time

import torch
from transformers import (AutoProcessor, Wav2Vec2ForCTC,
                          WhisperForConditionalGeneration)

from common import LANGS, audio_seconds, length_sorted, load_fleurs, load_ksc, normalize, score

DEFAULT_MODELS = [
    "openai/whisper-tiny",
    "openai/whisper-base",
    "openai/whisper-small",
    "openai/whisper-large-v3-turbo",
    "facebook/mms-1b-all",
]


def load_model(name):
    processor = AutoProcessor.from_pretrained(name)
    if "mms" in name:
        model = Wav2Vec2ForCTC.from_pretrained(name, dtype=torch.float16)
    else:
        model = WhisperForConditionalGeneration.from_pretrained(name, dtype=torch.float16)
    return processor, model.cuda().eval()


@torch.inference_mode()
def transcribe_batch(name, processor, model, audios, lang):
    if "mms" in name:
        inputs = processor(audios, sampling_rate=16000, return_tensors="pt", padding=True)
        logits = model(inputs.input_values.cuda().half(),
                       attention_mask=inputs.attention_mask.cuda()).logits
        return processor.batch_decode(logits.argmax(-1))
    if max(len(a) for a in audios) > 30 * 16000:
        # >30 s: Whisper's sequential long-form decoding instead of silently truncating to 30 s
        inputs = processor(audios, sampling_rate=16000, return_tensors="pt", truncation=False,
                           padding="longest", return_attention_mask=True)
        ids = model.generate(inputs.input_features.cuda().half(),
                             attention_mask=inputs.attention_mask.cuda(),
                             language=LANGS[lang]["whisper"], task="transcribe",
                             num_beams=1, do_sample=False, return_timestamps=True)
        return processor.batch_decode(ids, skip_special_tokens=True)
    feats = processor(audios, sampling_rate=16000, return_tensors="pt").input_features
    ids = model.generate(feats.cuda().half(), language=LANGS[lang]["whisper"], task="transcribe",
                         num_beams=1, do_sample=False)
    return processor.batch_decode(ids, skip_special_tokens=True)


def run(name, processor, model, lang, items, bs):
    if "mms" in name:
        processor.tokenizer.set_target_lang(LANGS[lang]["mms"])
        model.load_adapter(LANGS[lang]["mms"])

    order = length_sorted(items)
    # Whisper: utterances >30 s go one at a time (long-form); the rest in length-sorted batches
    n_long = 0 if "mms" in name else sum(len(items[j]["audio"]) > 30 * 16000 for j in order)
    batches = [[j] for j in order[:n_long]] + [order[i:i + bs] for i in range(n_long, len(order), bs)]
    transcribe_batch(name, processor, model, [items[order[-1]]["audio"]], lang)  # warm-up, not timed
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    hyps = [None] * len(items)
    t0 = time.perf_counter()
    for idx in batches:
        out = transcribe_batch(name, processor, model, [items[j]["audio"] for j in idx], lang)
        for j, h in zip(idx, out):
            hyps[j] = h.strip()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return hyps, elapsed, n_long


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--langs", nargs="+", default=["kk", "ru", "en"])
    ap.add_argument("--dataset", default="fleurs", choices=["fleurs", "ksc"])
    ap.add_argument("--split", default="test", help="FLEURS split to evaluate on")
    ap.add_argument("--limit", type=int, default=None, help="first N utterances per language")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", default="results/phase1")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.dataset == "ksc":
        assert args.langs == ["kk"], "KSC is Kazakh only"
        data = {"kk": load_ksc("test", args.limit)}
    else:
        data = {lang: load_fleurs(lang, args.split, args.limit) for lang in args.langs}

    for name in args.models:
        short = os.path.basename(name.rstrip("/"))
        if short in ("best", "final"):
            short = os.path.basename(os.path.dirname(name.rstrip("/"))) + "-" + short
        processor, model = load_model(name)
        n_params = sum(p.numel() for p in model.parameters())
        for lang in args.langs:
            items = data[lang]
            hyps, elapsed, n_long = run(name, processor, model, lang, items, args.batch_size)
            refs = [x["ref"] for x in items]
            dur = audio_seconds(items)
            res = {
                "model": name, "dataset": args.dataset, "lang": lang, "split": args.split, "n_utts": len(items),
                    "audio_hours": round(dur / 3600, 3), **score(refs, hyps),
                "rtf": elapsed / dur, "wall_s": elapsed,
                "peak_vram_alloc_gb": torch.cuda.max_memory_allocated() / 1e9,
                "peak_vram_reserved_gb": torch.cuda.max_memory_reserved() / 1e9,
                "params_m": n_params / 1e6, "device": torch.cuda.get_device_name(0),
                "dtype": "float16", "decoding": "ctc-greedy" if "mms" in name else "greedy (num_beams=1)",
                "batch_size": args.batch_size, "n_longform_gt30s": n_long, "forced_language": True,
            }
            base = f"{args.out}/{short}__{lang}"
            json.dump(res, open(base + ".json", "w"), indent=2, ensure_ascii=False)
            with open(base + ".csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["id", "ref", "hyp", "ref_norm", "hyp_norm"])
                for x, h in zip(items, hyps):
                    w.writerow([x["id"], x["ref"], h, normalize(x["ref"]), normalize(h)])

            print(f"\n=== {short} | {lang} | n={len(items)} | WER {res['wer']:.3f} CER {res['cer']:.3f} "
                  f"| RTF {res['rtf']:.4f} | VRAM {res['peak_vram_alloc_gb']:.2f} GB")
            for x, h in list(zip(items, hyps))[:3]:
                print("  REF:", normalize(x["ref"]))
                print("  HYP:", normalize(h))
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
