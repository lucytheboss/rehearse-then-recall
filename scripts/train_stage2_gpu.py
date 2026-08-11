"""Standalone stage-2 (B, elaborative rehearsal, rolling) training script for
a rented CUDA GPU — same logic as notebooks/07b_rehearsal_elaborative_stage2_train.ipynb's
§1-4 (load + train + save + held-out test eval), extracted so it doesn't need
Jupyter or this project's full src/ package on the remote machine.

Does NOT include 07b's §5/§6 (insert/revise branch check, collapse-in-the-
real-loop check) -- those need NIM API access (embeddings + rehearse_elaborative)
and are cheap enough to run locally afterward once the checkpoint is back.

Needs, copied over from the local repo:
    data/processed/rehearsal_elaborative_stage2_ratio0/   (from 06b's ratio=0.0 ablation)
    experiments/rehearsal_elaborative_flant5base/         (from train_stage1_gpu.py)

Usage on the remote instance:
    pip install torch transformers datasets evaluate accelerate rouge_score nltk pandas
    python train_stage2_gpu.py

Then copy experiments/rehearsal_elaborative_stage2_flant5base_ratio0/ back to
this repo's same path -- nb10/07b pick it up from there unchanged.
"""

from __future__ import annotations

from pathlib import Path

import datasets
import evaluate
import numpy as np
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

STAGE1_DIR = Path("experiments/rehearsal_elaborative_flant5base")
DATA_DIR = Path("data/processed/rehearsal_elaborative_stage2_ratio0")
OUTPUT_DIR = "experiments/rehearsal_elaborative_stage2_flant5base_ratio0"


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    if device == "cpu":
        print("WARNING: no CUDA GPU detected -- this script will be slow. Check your vast.ai instance.")

    data_ready = DATA_DIR.exists() and (DATA_DIR / "train").exists()
    stage1_ready = STAGE1_DIR.exists()
    if not (data_ready and stage1_ready):
        raise SystemExit(f"data ready: {data_ready} ({DATA_DIR}), stage1 ready: {stage1_ready} ({STAGE1_DIR}) "
                          f"-- copy both over first.")

    dataset = datasets.load_from_disk(str(DATA_DIR))
    print(dataset)

    tokenizer = AutoTokenizer.from_pretrained(STAGE1_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(STAGE1_DIR).to(device)
    print(f"loaded from {STAGE1_DIR}, parameters: {sum(p.numel() for p in model.parameters()):,}")

    rouge = evaluate.load("rouge")

    def compute_metrics(eval_preds):
        predictions, labels = eval_preds
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=False)
        return {"rougeL": result["rougeL"]}

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # Same two fixes as stage 1 (2026-08-09): bf16 not fp16 (T5 numerical
    # instability under fp16 -- produced loss=0/grad_norm=nan on this same
    # GPU class for stage 1), and predict_with_generate=False during
    # training eval (generation is slow and unnecessary per-epoch; a real
    # generation-quality check still runs once at the end below).
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=1e-4,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        predict_with_generate=False,
        logging_steps=50,
        report_to=[],
        bf16=(device == "cuda"),
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"],
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"saved to {OUTPUT_DIR}")

    # Final generation-quality check on the held-out test set (once, not per-epoch)
    test_trainer = Seq2SeqTrainer(
        model=model,
        args=Seq2SeqTrainingArguments(
            output_dir=OUTPUT_DIR,
            per_device_eval_batch_size=4,
            predict_with_generate=True,
            generation_max_length=256,
            report_to=[],
        ),
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    test_metrics = test_trainer.evaluate(eval_dataset=dataset["test"], metric_key_prefix="test")
    print("Final test-set metrics:", test_metrics)


if __name__ == "__main__":
    main()
