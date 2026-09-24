"""Phase 2: CPU inference with faster-whisper (CTranslate2), float32 vs int8.

Each HF model is converted twice (float32 and int8 weights) so that the reported
disk size matches the precision actually used. Decoding: greedy, forced language,
no temperature fallback, no VAD, no conditioning on previous text, batch size 1.

Usage:
  python 02_cpu_inference.py --limit 50 --out results/phase2_smoke
  python 02_cpu_inference.py --out results/phase2
"""
import argparse
import csv
import json
import os
import subprocess
import time

from faster_whisper import WhisperModel

from common import audio_seconds, load_fleurs, normalize, score


def convert(hf_name, quant):
    out = f"models/ct2/{os.path.basename(hf_name.rstrip('/'))}-{quant}"
    if not os.path.exists(os.path.join(out, "model.bin")):
        subprocess.run(["ct2-transformers-converter", "--model", hf_name, "--output_dir", out,
                        "--quantization", quant, "--copy_files", "tokenizer.json",
                        "preprocessor_config.json", "--force"], check=True)
    return out


def dir_size_mb(path):
    return sum(os.path.getsize(os.path.join(path, f)) for f in os.listdir(path)) / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["openai/whisper-small", "openai/whisper-large-v3-turbo"])
    ap.add_argument("--quants", nargs="+", default=["float32", "int8"])
    ap.add_argument("--langs", nargs="+", default=["kk", "ru", "en"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--threads", type=int, default=8, help="CPU threads (8 = physical cores here)")
    ap.add_argument("--out", default="results/phase2")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    data = {lang: load_fleurs(lang, "test", args.limit) for lang in args.langs}

    for name in args.models:
        for quant in args.quants:
            path = convert(name, quant)
            model = WhisperModel(path, device="cpu", compute_type=quant, cpu_threads=args.threads)
            short = os.path.basename(path)
            for lang in args.langs:
                items = data[lang]
                opts = dict(language=lang, task="transcribe", beam_size=1, temperature=0.0,
                            condition_on_previous_text=False, without_timestamps=True, vad_filter=False)
                list(model.transcribe(items[0]["audio"], **opts)[0])  # warm-up, not timed

                hyps = []
                t0 = time.perf_counter()
                for x in items:
                    segments, _ = model.transcribe(x["audio"], **opts)
                    hyps.append(" ".join(s.text.strip() for s in segments))  # generator: decoding happens here
                elapsed = time.perf_counter() - t0

                refs = [x["ref"] for x in items]
                dur = audio_seconds(items)
                res = {
                    "model": name, "engine": "faster-whisper/ctranslate2", "compute_type": quant,
                    "lang": lang, "split": "test", "n_utts": len(items),
                    "audio_hours": round(dur / 3600, 3), **score(refs, hyps),
                    "rtf": elapsed / dur, "wall_s": elapsed, "disk_mb": dir_size_mb(path),
                    "device": "cpu", "cpu_threads": args.threads, "batch_size": 1,
                    "decoding": "greedy (beam_size=1, temperature=0, no fallback)", "forced_language": True,
                }
                base = f"{args.out}/{short}__{lang}"
                json.dump(res, open(base + ".json", "w"), indent=2, ensure_ascii=False)
                with open(base + ".csv", "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["id", "ref", "hyp", "ref_norm", "hyp_norm"])
                    for x, h in zip(items, hyps):
                        w.writerow([x["id"], x["ref"], h, normalize(x["ref"]), normalize(h)])
                print(f"=== {short} | {lang} | n={len(items)} | WER {res['wer']:.3f} CER {res['cer']:.3f} "
                      f"| RTF {res['rtf']:.3f} | {res['disk_mb']:.0f} MB", flush=True)
            del model


if __name__ == "__main__":
    main()
