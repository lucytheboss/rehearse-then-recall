"""Extractive maintenance rehearsal (A) — RoBERTa sentence scorer.

Maintenance rehearsal keeps the source's own sentences (surface repetition),
so the natural model is a sentence *selector*, not a generator. The earlier
seq2seq implementation in `rehearsal.py` generated text and then forced it
back onto source sentences with `_snap_to_verbatim`; selecting directly makes
verbatim retention structural rather than a post-hoc repair.

Note that this does *not* make the novel n-gram ratio zero. Joining
non-adjacent sentences creates n-grams that span the gap between them, and
that grows with the compression rate. The invariant is per-sentence, not
per-output — see `seam_report`.

Prior work:
  - Sie, Beek, Bots, Brinkkemper & Gatt, NLLP 2024 (arXiv:2408.09777) §3.2 —
    RoBERTa with a single dependent-ratio extraction pass was the best
    extractive configuration in their 5x6x3 grid.
  - Liu & Lapata, EMNLP 2019 (BERTSUM/PreSumm) — encode the whole document
    with a marker token per sentence, classify each sentence, and supervise
    with greedy-ROUGE oracle labels. `04_rehearsal_maintenance_prep.ipynb`
    already builds those labels.

RoBERTa uses learned position embeddings, so 512 tokens is an architectural
ceiling: sentences past it get no representation and could never be selected.
`configs/chunking.yaml`'s max_words is set so a full chunk stays under it.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from transformers import AutoModel

MAX_TOKENS = 512


def encode_sentences(
    sentences: list[str],
    tokenizer,
    max_tokens: int = MAX_TOKENS,
) -> dict:
    """BERTSUM-style encoding: `<s> sent </s>` per sentence, concatenated.

    Each sentence's opening `<s>` is the token whose final representation
    classifies that sentence, so its position is recorded in `cls_positions`.

    Sentences that would push the sequence past `max_tokens` are dropped
    rather than truncated mid-sentence — a half-sentence has no meaningful
    label. `n_kept` reports how many survived so callers can trim labels to
    match; if this is regularly below len(sentences), max_words is too high
    for the backbone.
    """
    bos, eos = tokenizer.bos_token_id, tokenizer.eos_token_id
    input_ids: list[int] = []
    cls_positions: list[int] = []

    for sentence in sentences:
        piece = [bos] + tokenizer(sentence, add_special_tokens=False).input_ids + [eos]
        if len(input_ids) + len(piece) > max_tokens:
            break
        cls_positions.append(len(input_ids))
        input_ids.extend(piece)

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "cls_positions": cls_positions,
        "n_kept": len(cls_positions),
    }


def collate(batch: list[dict], pad_token_id: int) -> dict:
    """Pads a batch of `encode_sentences` outputs.

    Two independent ragged dimensions here: sequence length and sentence
    count. `cls_mask` marks which sentence slots are real, so padded slots are
    excluded from the loss rather than being learned as negatives.
    """
    max_len = max(len(item["input_ids"]) for item in batch)
    max_sents = max(len(item["cls_positions"]) for item in batch)

    out = {k: [] for k in ("input_ids", "attention_mask", "cls_positions", "cls_mask", "labels")}
    for item in batch:
        pad = max_len - len(item["input_ids"])
        out["input_ids"].append(item["input_ids"] + [pad_token_id] * pad)
        out["attention_mask"].append(item["attention_mask"] + [0] * pad)

        n = len(item["cls_positions"])
        sent_pad = max_sents - n
        # Padded slots point at position 0; cls_mask keeps them out of the loss.
        out["cls_positions"].append(item["cls_positions"] + [0] * sent_pad)
        out["cls_mask"].append([1] * n + [0] * sent_pad)
        if "labels" in item:
            out["labels"].append(item["labels"] + [0] * sent_pad)

    tensors = {k: torch.tensor(v) for k, v in out.items() if v}
    if "labels" in tensors:
        tensors["labels"] = tensors["labels"].float()
    return tensors


class SentenceScorer(nn.Module):
    """RoBERTa encoder + a linear head over each sentence's `<s>` token.

    Returns one logit per sentence: how salient it is *within its own chunk*.
    Query-agnostic by construction — nothing about a downstream question ever
    enters this model.
    """

    def __init__(self, model_name: str = "roberta-base", dropout: float = 0.1):
        super().__init__()
        # No pooler: classification reads each sentence's own <s> state, never a
        # whole-sequence pooled vector, so the pooler would be dead weight and
        # would emit a spurious "newly initialized weights" warning on load.
        self.encoder = AutoModel.from_pretrained(model_name, add_pooling_layer=False)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, 1)

    def forward(self, input_ids, attention_mask, cls_positions, cls_mask, labels=None):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        index = cls_positions.unsqueeze(-1).expand(-1, -1, hidden.size(-1))
        sentence_states = hidden.gather(1, index)
        logits = self.classifier(self.dropout(sentence_states)).squeeze(-1)

        loss = None
        if labels is not None:
            per_sentence = nn.functional.binary_cross_entropy_with_logits(
                logits, labels, reduction="none"
            )
            loss = (per_sentence * cls_mask).sum() / cls_mask.sum().clamp(min=1)
        return {"loss": loss, "logits": logits}


@torch.no_grad()
def score_sentences(sentences: list[str], model: SentenceScorer, tokenizer) -> list[float]:
    """Salience logit per sentence. Sentences dropped for length score -inf,
    so they are never selected — they were never seen by the encoder."""
    encoded = encode_sentences(sentences, tokenizer)
    if not encoded["cls_positions"]:
        return [float("-inf")] * len(sentences)

    device = next(model.parameters()).device
    batch = collate([encoded], tokenizer.pad_token_id)
    batch = {k: v.to(device) for k, v in batch.items()}
    logits = model(**batch)["logits"][0].tolist()
    return logits + [float("-inf")] * (len(sentences) - len(logits))


def seam_report(kept_sentences: list[str], source_text: str, n: int = 3) -> dict:
    """Checks that an extractive compression really is verbatim.

    A tempting but wrong check is "novel n-gram ratio must be 0". Selecting
    sentences is verbatim, but *joining non-adjacent* ones creates n-grams that
    span the gap: the last words of one kept sentence followed by the first
    words of the next produce a sequence that never occurred in the source.
    That is unavoidable for any extractive compressor that skips anything, and
    it grows with the compression rate.

    The invariant that does hold is per-sentence: every kept sentence must
    appear verbatim in the source, so every novel n-gram must **span a seam**.
    A novel n-gram sitting inside a single kept sentence means the sentence
    itself was altered — that is a real defect.
    """
    def norm(text: str) -> str:
        return " ".join(text.split())

    def grams(text: str) -> list[str]:
        words = norm(text).split()
        return [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]

    normalized_source = norm(source_text)
    not_verbatim = [s for s in kept_sentences if norm(s) not in normalized_source]

    source_grams = set(grams(source_text))
    within_sentences = {g for s in kept_sentences for g in grams(s)}
    compressed_grams = grams(" ".join(kept_sentences))
    novel = [g for g in compressed_grams if g not in source_grams]

    return {
        "ok": not not_verbatim and not [g for g in novel if g in within_sentences],
        "not_verbatim": not_verbatim,
        "novel_total": len(novel),
        "novel_inside_sentence": [g for g in novel if g in within_sentences],
        "novel_ratio": len(novel) / len(compressed_grams) if compressed_grams else 0.0,
        "seams": max(0, len(kept_sentences) - 1),
    }


def select_sentences(sentences: list[str], scores: list[float], ratio: float) -> list[str]:
    """Top `ceil(ratio * n)` sentences by score, returned in **source order**.

    Source order matters: this is maintenance rehearsal, and a compressed
    chunk that reshuffles the original is no longer a faithful retention of
    it. At least one sentence is always kept — an empty chunk would break the
    rolling context that later stages read.
    """
    if not sentences:
        return []
    budget = max(1, math.ceil(ratio * len(sentences)))
    ranked = sorted(range(len(sentences)), key=lambda i: scores[i], reverse=True)[:budget]
    return [sentences[i] for i in sorted(ranked)]
