# Kazakh / Russian / English ASR pilot

Zero-shot evaluation, local CPU inference and Kazakh fine-tuning of Whisper and MMS on FLEURS.
Results and discussion: [RESULTS.md](RESULTS.md).

## Files

| File | Purpose |
|---|---|
| `common.py` | FLEURS loading, text normalization, WER/CER (shared by all phases) |
| `00_check_env.py` | Records versions and runs a GPU sanity check → `results/env.json` |
| `01_zero_shot.py` | Phase 1: zero-shot WER/CER, RTF, peak VRAM on GPU |
| `02_cpu_inference.py` | Phase 2: CPU inference with faster-whisper, float32 vs int8 |
| `03_finetune_kk.py` | Phase 3: fine-tune whisper-small on FLEURS Kazakh |
| `make_report.py` | Recomputes metrics from the per-run CSVs, writes `results/summary.csv` and the WER plot |

Each run writes `results/<phase>/<model>__<lang>.json` (metrics and settings) and `.csv` (per-utterance ref/hyp).

## Setup

Tested on Ubuntu 26.04 with an RTX 5060 Laptop GPU (Blackwell, needs CUDA 12.8+ wheels).

```bash
uv python install 3.12
uv venv --python 3.12 .venv
uv pip install --python .venv -r requirements.txt   # torch comes from the cu128 index listed in the file
.venv/bin/python 00_check_env.py
```

FLEURS is downloaded from the Hugging Face Hub on first use (parquet, no `trust_remote_code`).
Loading a language config downloads all of its splits: about 3.9 GB for kk_kz.

## Phase 1: zero-shot baseline

```bash
# smoke test: first 50 test utterances per language
.venv/bin/python 01_zero_shot.py --limit 50 --models openai/whisper-tiny openai/whisper-base openai/whisper-small openai/whisper-large-v3-turbo --out results/phase1_smoke
.venv/bin/python 01_zero_shot.py --limit 50 --models facebook/mms-1b-all --batch-size 4 --out results/phase1_smoke

# full test sets
.venv/bin/python 01_zero_shot.py --models openai/whisper-tiny openai/whisper-base openai/whisper-small openai/whisper-large-v3-turbo --out results/phase1
.venv/bin/python 01_zero_shot.py --models facebook/mms-1b-all --batch-size 4 --out results/phase1

.venv/bin/python make_report.py
```

MMS uses batch size 4 because with 16 its peak VRAM approached the 8 GB limit.

## Phase 2: CPU inference (faster-whisper, float32 vs int8)

```bash
# smoke test (first 50 utterances per language)
.venv/bin/python 02_cpu_inference.py --limit 50 --models openai/whisper-small --out results/phase2_smoke

# full test sets, about 4 h on the reference laptop
PATH="$PWD/.venv/bin:$PATH" .venv/bin/python 02_cpu_inference.py --models openai/whisper-small --out results/phase2

.venv/bin/python make_report.py
```

`ct2-transformers-converter` must be on PATH (it ships with faster-whisper, hence the `PATH=` prefix).
Converted models are written to `models/ct2/` (git-ignored): 971 MB for float32, 252 MB for int8.

## Phase 3: Kazakh fine-tuning

```bash
# smoke test: 20 steps, checks speed and VRAM
.venv/bin/python 03_finetune_kk.py --max-steps 20 --out checkpoints/smoke

# full run: 8 epochs on FLEURS kk train, about 70 min on the reference laptop
setsid nohup .venv/bin/python 03_finetune_kk.py --epochs 8 > phase3.log 2>&1 &
# if the run is interrupted, continue from the last checkpoint:
.venv/bin/python 03_finetune_kk.py --epochs 8 --resume

# evaluate the best checkpoint on the test splits and on kk validation
.venv/bin/python 01_zero_shot.py --models checkpoints/whisper-small-kk/final --out results/phase3
.venv/bin/python 01_zero_shot.py --split validation --langs kk --models openai/whisper-small checkpoints/whisper-small-kk/final --out results/phase3_val

.venv/bin/python make_report.py
```

Training writes `results/phase3_training.json` (settings, per-epoch validation WER/CER, peak VRAM) and the model to
`checkpoints/whisper-small-kk/final` (git-ignored, 926 MB).

## Phase 4: out-of-domain evaluation on KSC (Kazakh)

```bash
# 410 MB test parquet is downloaded on first use; no training data is fetched
.venv/bin/python 01_zero_shot.py --dataset ksc --langs kk --limit 50 --models openai/whisper-small checkpoints/whisper-small-kk/final --out results/phase4_ksc_smoke

.venv/bin/python 01_zero_shot.py --dataset ksc --langs kk --models openai/whisper-small checkpoints/whisper-small-kk/final openai/whisper-large-v3-turbo --out results/phase4_ksc
.venv/bin/python 01_zero_shot.py --dataset ksc --langs kk --batch-size 4 --models facebook/mms-1b-all --out results/phase4_ksc

.venv/bin/python make_report.py   # also writes results/kk_wer_fleurs_vs_ksc.png
```
