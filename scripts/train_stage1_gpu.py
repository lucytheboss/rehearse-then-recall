"""Standalone stage-1 (B, elaborative rehearsal) training script for a rented
CUDA GPU (e.g. vast.ai) — same logic as notebooks/07_rehearsal_elaborative_train.ipynb,
extracted so it doesn't need Jupyter or this project's full src/ package on the
remote machine. Only needs data/processed/rehearsal_elaborative/ (from 06,
built locally) copied over.

Usage on the remote instance:
    pip install torch transformers datasets evaluate accelerate rouge_score nltk pandas
    python train_stage1_gpu.py

Then copy experiments/rehearsal_elaborative_flant5base/ back to this repo's
same path — 07b picks it up from there via STAGE1_DIR unchanged.
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

MODEL_NAME = "google/flan-t5-base"
DATA_DIR = Path("data/processed/rehearsal_elaborative")
OUTPUT_DIR = "experiments/rehearsal_elaborative_flant5base"
TRAIN_SIZE = 3000  # full size -- a real GPU should make this fast; lower if you want a quicker sanity run first
VAL_SIZE = 300
TEST_SIZE = 300


def novel_ngram_ratio(generated_text: str, source_text: str, n: int = 3) -> float:
    """Fraction of generated text's n-grams not present in the source --
    copied from src/pipeline/rehearsal.py so this script has no project
    dependency beyond the tokenized data."""
    gen_words = generated_text.split()
    src_words = source_text.split()
    if len(gen_words) < n:
        return 0.0
    gen_ngrams = {tuple(gen_words[i : i + n]) for i in range(len(gen_words) - n + 1)}
    if not gen_ngrams:
        return 0.0
    src_ngrams = {tuple(src_words[i : i + n]) for i in range(len(src_words) - n + 1)}
    novel = gen_ngrams - src_ngrams
    return len(novel) / len(gen_ngrams)


def average_novel_ngram_ratio(model, tokenizer, eval_dataset, n: int = 3,
                               sample_size: int = 10, max_new_tokens: int = 64) -> float:
    examples = eval_dataset.select(range(min(sample_size, len(eval_dataset))))
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device

    ratios = []
    for example in examples:
        input_ids = torch.tensor([example["input_ids"]]).to(device)
        with torch.no_grad():
            output_ids = model.generate(input_ids=input_ids, max_new_tokens=max_new_tokens)
        generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        source = tokenizer.decode(example["input_ids"], skip_special_tokens=True)
        ratios.append(novel_ngram_ratio(generated, source, n=n))

    if was_training:
        model.train()
    return sum(ratios) / len(ratios) if ratios else 0.0


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    if device == "cpu":
        print("WARNING: no CUDA GPU detected -- this script will be slow. Check your vast.ai instance.")

    data_ready = all((DATA_DIR / split).exists() for split in ("train", "val", "test"))
    if not data_ready:
        raise SystemExit(f"No tokenized data at {DATA_DIR} -- copy it over from the local repo first "
                          f"(built by notebooks/06_rehearsal_elaborative_prep.ipynb).")

    full_train = datasets.Dataset.load_from_disk(str(DATA_DIR / "train"))
    full_val = datasets.Dataset.load_from_disk(str(DATA_DIR / "val"))
    full_test = datasets.Dataset.load_from_disk(str(DATA_DIR / "test"))

    train_dataset = full_train.select(range(min(TRAIN_SIZE, len(full_train))))
    val_dataset = full_val.select(range(min(VAL_SIZE, len(full_val))))
    test_dataset = full_test.select(range(min(TEST_SIZE, len(full_test))))
    print(f"Using {len(train_dataset)}/{len(full_train)} train, "
          f"{len(val_dataset)}/{len(full_val)} val, {len(test_dataset)}/{len(full_test)} test rows")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
    print(f"{MODEL_NAME} loaded, parameters: {sum(p.numel() for p in model.parameters()):,}")

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

    # predict_with_generate off during training (see 07's 2026-08-09 fix) --
    # a real CUDA GPU shouldn't need this, but keeping it off costs nothing
    # and avoids re-introducing the slow path if this ever runs on CPU/MPS.
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        predict_with_generate=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        # bf16, not fp16 (2026-08-09): fp16 produced loss=0/grad_norm=nan
        # within the first ~75 steps on an RTX 3090 -- a well-documented T5
        # instability (T5 was pretrained in bfloat16; fp16's narrower
        # dynamic range overflows in its attention/layernorm variants).
        # bf16 has fp32's exponent range so it doesn't hit that, and Ampere
        # (RTX 3090 and newer) supports it natively.
        bf16=(device == "cuda"),
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
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
            per_device_eval_batch_size=8,
            predict_with_generate=True,
            generation_max_length=64,
            report_to="none",
        ),
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    test_metrics = test_trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
    print("Final test-set metrics:", test_metrics)

    test_novel_ratio = average_novel_ngram_ratio(model, tokenizer, test_dataset, sample_size=30)
    print(f"Final test-set novel 3-gram ratio: {test_novel_ratio:.4f} (higher is better for B)")


if __name__ == "__main__":
    main()
