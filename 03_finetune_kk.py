"""Phase 3: fine-tune whisper-small on Kazakh (FLEURS kk train), evaluate on FLEURS kk validation.

With --lora, only low-rank adapters on the attention projections are trained; the adapter is merged
into the base weights before saving, so the result is an ordinary Whisper checkpoint.

Test data is never used here. Evaluate the result on the test splits with 01_zero_shot.py.

Usage:
  python 03_finetune_kk.py --max-steps 20 --out checkpoints/smoke   # smoke test
  python 03_finetune_kk.py
"""
import argparse
import json
import os

import torch
from torch.utils.data import Dataset
from transformers import (Seq2SeqTrainer, Seq2SeqTrainingArguments,
                          WhisperForConditionalGeneration, WhisperProcessor)

from common import LANGS, load_fleurs, score

MAX_AUDIO = 30 * 16000  # Whisper's context; longer utterances would be truncated against a full transcript


class FleursKK(Dataset):
    def __init__(self, items, processor, timestamps=False):
        self.items = items
        self.processor = processor
        self.timestamps = timestamps  # wrap the transcript in <|0.00|> ... <|duration|>

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        x = self.items[i]
        feats = self.processor.feature_extractor(x["audio"], sampling_rate=16000).input_features[0]
        # each example carries its own language token, so mixed-language batches stay labelled correctly
        self.processor.tokenizer.set_prefix_tokens(language=LANGS[x.get("lang", "kk")]["whisper"])
        text = x["ref"]
        if self.timestamps:  # Whisper's timestamp grid is 0.02 s, capped at 30 s
            end = min(round(len(x["audio"]) / 16000 / 0.02) * 0.02, 30.0)
            text = f"<|0.00|>{text}<|{end:.2f}|>"
        labels = self.processor.tokenizer(text).input_ids
        return {"input_features": feats, "labels": labels}


def collate(batch, processor):
    out = processor.feature_extractor.pad([{"input_features": b["input_features"]} for b in batch],
                                          return_tensors="pt")
    labels = processor.tokenizer.pad([{"input_ids": b["labels"]} for b in batch], return_tensors="pt")
    ids = labels.input_ids.masked_fill(labels.attention_mask == 0, -100)
    if (ids[:, 0] == processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")).all():
        ids = ids[:, 1:]  # the model prepends it via decoder_start_token_id
    out["labels"] = ids
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/whisper-small")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--max-steps", type=int, default=-1, help="overrides --epochs (smoke test)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=None, help="default: 1e-5 full, 1e-3 with --lora")
    ap.add_argument("--lora", action="store_true", help="train LoRA adapters instead of all weights")
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--timestamps", action="store_true",
                    help="train with timestamp tokens, which Whisper's long-form decoding needs")
    ap.add_argument("--mix", nargs="*", default=[], metavar="LANG",
                    help="also train on this many utterances of other languages, e.g. --mix ru en")
    ap.add_argument("--mix-utts", type=int, default=800, help="utterances per extra language")
    ap.add_argument("--out", default="checkpoints/whisper-small-kk")
    ap.add_argument("--resume", action="store_true", help="continue from the last checkpoint in --out")
    args = ap.parse_args()
    if args.lr is None:
        args.lr = 1e-3 if args.lora else 1e-5
    if args.lora and args.out == "checkpoints/whisper-small-kk":
        args.out = "checkpoints/whisper-small-kk-lora"
    if args.timestamps and args.out == "checkpoints/whisper-small-kk":
        args.out = "checkpoints/whisper-small-kk-ts"
    if args.mix and args.out == "checkpoints/whisper-small-kk":
        args.out = "checkpoints/whisper-small-kk-mix"

    processor = WhisperProcessor.from_pretrained(args.model, language="kazakh", task="transcribe",
                                                 predict_timestamps=args.timestamps)
    model = WhisperForConditionalGeneration.from_pretrained(args.model)
    model.generation_config.language = "kazakh"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    model.config.use_cache = False  # required with gradient checkpointing

    if args.lora:
        from peft import LoraConfig, get_peft_model
        model.enable_input_require_grads()  # needed for gradient checkpointing through frozen weights
        model = get_peft_model(model, LoraConfig(
            r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05, bias="none",
            target_modules=["q_proj", "v_proj"]))
        model.print_trainable_parameters()

    train = [dict(x, lang="kk") for x in load_fleurs("kk", "train") if len(x["audio"]) <= MAX_AUDIO]
    for lang in args.mix:  # keep the other languages alive by rehearsing a slice of their training data
        extra = [dict(x, lang=lang) for x in load_fleurs(lang, "train") if len(x["audio"]) <= MAX_AUDIO]
        train += extra[:args.mix_utts]
        print(f"+ {min(args.mix_utts, len(extra))} {lang} utts", flush=True)
    val = [dict(x, lang="kk") for x in load_fleurs("kk", "validation") if len(x["audio"]) <= MAX_AUDIO]
    print(f"train {len(train)} utts, validation {len(val)} utts (kk only)", flush=True)

    def compute_metrics(pred):
        label_ids = pred.label_ids.copy()
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        refs = processor.batch_decode(label_ids, skip_special_tokens=True)
        hyps = processor.batch_decode(pred.predictions, skip_special_tokens=True)
        return score(refs, hyps)

    trainer = Seq2SeqTrainer(
        model=model,
        args=Seq2SeqTrainingArguments(
            output_dir=args.out,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            per_device_eval_batch_size=4,
            learning_rate=args.lr,
            warmup_steps=100,
            num_train_epochs=args.epochs,
            max_steps=args.max_steps,
            bf16=True,
            gradient_checkpointing=True,
            predict_with_generate=True,
            generation_max_length=225,
            eval_strategy="epoch" if args.max_steps < 0 else "no",
            save_strategy="epoch" if args.max_steps < 0 else "no",
            load_best_model_at_end=args.max_steps < 0,
            metric_for_best_model="wer",
            greater_is_better=False,
            save_total_limit=1,
            logging_steps=25,
            dataloader_num_workers=2,
            report_to=[],
        ),
        train_dataset=FleursKK(train, processor, args.timestamps),
        eval_dataset=FleursKK(val, processor, args.timestamps),
        data_collator=lambda b: collate(b, processor),
        compute_metrics=compute_metrics,
        processing_class=processor,
    )

    result = trainer.train(resume_from_checkpoint=args.resume or None)
    final = os.path.join(args.out, "final")
    if args.lora:
        model = model.merge_and_unload()  # fold the adapter in: a plain Whisper checkpoint
        trainer.model = model
    model.generation_config.forced_decoder_ids = None
    if isinstance(model.generation_config.eos_token_id, list):  # a list breaks Whisper's long-form decoding
        model.generation_config.eos_token_id = model.generation_config.eos_token_id[0]
    trainer.save_model(final)
    processor.save_pretrained(final)

    summary = {
        "base_model": args.model, "lora": args.lora, "lora_r": args.lora_r if args.lora else None,
        "timestamps": args.timestamps, "mix": args.mix, "mix_utts": args.mix_utts,
        "train_utts": len(train), "val_utts": len(val),
        "epochs": args.epochs, "max_steps": args.max_steps, "lr": args.lr, "out": args.out,
        "batch_size": args.batch_size, "grad_accum": args.grad_accum,
        "train_runtime_s": result.metrics["train_runtime"],
        "peak_vram_alloc_gb": torch.cuda.max_memory_allocated() / 1e9,
        "log_history": trainer.state.log_history,
    }
    os.makedirs("results", exist_ok=True)
    # one history file per run, named after the output directory
    name = {"whisper-small-kk": "phase3_training", "whisper-small-kk-lora": "phase6_lora_training"}.get(
        os.path.basename(args.out.rstrip("/")), os.path.basename(args.out.rstrip("/")) + "_training")
    json.dump(summary, open(f"results/{name}.json", "w"), indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in summary.items() if k != "log_history"}, indent=2))


if __name__ == "__main__":
    main()
