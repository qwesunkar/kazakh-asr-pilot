"""Phase 3: fine-tune whisper-small on Kazakh (FLEURS kk train), evaluate on FLEURS kk validation.

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

from common import load_fleurs, score

MAX_AUDIO = 30 * 16000  # Whisper's context; longer utterances would be truncated against a full transcript


class FleursKK(Dataset):
    def __init__(self, items, processor):
        self.items = items
        self.processor = processor

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        x = self.items[i]
        feats = self.processor.feature_extractor(x["audio"], sampling_rate=16000).input_features[0]
        labels = self.processor.tokenizer(x["ref"]).input_ids
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
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--out", default="checkpoints/whisper-small-kk")
    ap.add_argument("--resume", action="store_true", help="continue from the last checkpoint in --out")
    args = ap.parse_args()

    processor = WhisperProcessor.from_pretrained(args.model, language="kazakh", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(args.model)
    model.generation_config.language = "kazakh"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    model.config.use_cache = False  # required with gradient checkpointing

    train = [x for x in load_fleurs("kk", "train") if len(x["audio"]) <= MAX_AUDIO]
    val = [x for x in load_fleurs("kk", "validation") if len(x["audio"]) <= MAX_AUDIO]
    print(f"train {len(train)} utts, validation {len(val)} utts", flush=True)

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
        train_dataset=FleursKK(train, processor),
        eval_dataset=FleursKK(val, processor),
        data_collator=lambda b: collate(b, processor),
        compute_metrics=compute_metrics,
        processing_class=processor,
    )

    result = trainer.train(resume_from_checkpoint=args.resume or None)
    final = os.path.join(args.out, "final")
    model.generation_config.forced_decoder_ids = None
    if isinstance(model.generation_config.eos_token_id, list):  # a list breaks Whisper's long-form decoding
        model.generation_config.eos_token_id = model.generation_config.eos_token_id[0]
    trainer.save_model(final)
    processor.save_pretrained(final)

    summary = {
        "base_model": args.model, "train_utts": len(train), "val_utts": len(val),
        "epochs": args.epochs, "max_steps": args.max_steps, "lr": args.lr,
        "batch_size": args.batch_size, "grad_accum": args.grad_accum,
        "train_runtime_s": result.metrics["train_runtime"],
        "peak_vram_alloc_gb": torch.cuda.max_memory_allocated() / 1e9,
        "log_history": trainer.state.log_history,
    }
    os.makedirs("results", exist_ok=True)
    json.dump(summary, open("results/phase3_training.json", "w"), indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in summary.items() if k != "log_history"}, indent=2))


if __name__ == "__main__":
    main()
