"""Maintenance rehearsal (A) — intra-chunk salience compression.

Implements `11_되뇌기 구현 가이드.md` §3 (Obsidian). Each Chunk is fed
independently (no cross-chunk context, no rolling state — this is the key
difference from elaborative rehearsal B) into a seq2seq model to produce a
compressed version, then each generated sentence is checked against the
source for verbatim presence; if it isn't found, it's snapped to the
source sentence with the highest cosine similarity — enforcing verbatim
output purely via decoding-time post-processing, without changing the
architecture (guide §3.2).

Hard constraints (guide §2, shared by A/B):
- query-agnostic — no function in this module takes a question/query-like
  argument.
- Preserves chunk position metadata (index/paragraph_indices/char_start/
  char_end) — only `text` is replaced with the compressed version;
  `original_text` stores the pre-compression source.
- Text-level intervention — plain text in/out, not the model's internal
  hidden states.
"""

from __future__ import annotations

from dataclasses import replace

import torch

from src.pipeline.embeddings import cosine_similarity, embed_texts, embed_with_retry
from src.pipeline.gisting import split_into_sentences
from src.pipeline.types import Chunk


def _snap_to_verbatim(
    generated_sentences: list[str],
    source_sentences: list[str],
    embed_cfg: dict,
    embed_fn=embed_texts,
) -> list[str]:
    """Keeps each generated sentence as-is if it exists verbatim in
    source_sentences; otherwise replaces it with the source sentence with the
    highest cosine similarity.

    If the embedding API still fails after retries (embed_with_retry returns
    None), gives up on the verbatim guarantee and returns the generated
    sentences unchanged — the pipeline doesn't die (same philosophy as
    paginate_semantic's on_error="pass_through").
    """
    if not generated_sentences or not source_sentences:
        return generated_sentences

    source_set = set(source_sentences)
    needs_snap = [s for s in generated_sentences if s not in source_set]
    if not needs_snap:
        return generated_sentences

    source_embeddings = embed_with_retry(source_sentences, "passage", embed_cfg, embed_fn)
    if source_embeddings is None:
        return generated_sentences

    needed_embeddings = embed_with_retry(needs_snap, "passage", embed_cfg, embed_fn)
    if needed_embeddings is None:
        return generated_sentences

    snap_map: dict[str, str] = {}
    for gen_sentence, gen_embedding in zip(needs_snap, needed_embeddings):
        scores = [cosine_similarity(gen_embedding, src_embedding) for src_embedding in source_embeddings]
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        snap_map[gen_sentence] = source_sentences[best_idx]

    return [snap_map.get(s, s) for s in generated_sentences]


def rehearse_maintenance(
    chunks: list[Chunk],
    model,
    tokenizer,
    embed_cfg: dict,
    embed_fn=embed_texts,
    max_input_length: int = 512,
    max_new_tokens: int = 400,
) -> list[Chunk]:
    """Processes each Chunk independently (no cross-chunk context) and replaces
    it with a compressed version.

    Returned Chunks: text=the compressed version with verbatim post-processing
    already applied, original_text=the pre-compression source;
    index/paragraph_indices/char_start/char_end stay identical to the input.
    Takes no question-related argument (query-agnostic).

    max_input_length=512 is a common convention for t5-base-family
    summarization fine-tuning (the old pko-t5 backbone's 1300 was tuned to
    that model's own training setup and doesn't carry over). Re-measure
    against `configs/chunking.yaml`'s max_words and the actual English
    tokenizer ratio and adjust as needed — English generally runs fewer
    tokens per word than Korean (roughly 1.2-1.5 tokens/word with
    SentencePiece), so there's room to raise the chunk max_words a bit.
    """
    model.eval()
    device = next(model.parameters()).device

    result: list[Chunk] = []
    for chunk in chunks:
        inputs = tokenizer(
            chunk.text,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_length,
        ).to(device)

        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        generated_sentences = [s for s in split_into_sentences(generated_text) if s.strip()]
        source_sentences = split_into_sentences(chunk.text)
        verbatim_sentences = _snap_to_verbatim(generated_sentences, source_sentences, embed_cfg, embed_fn)

        compressed_text = " ".join(verbatim_sentences)
        result.append(replace(chunk, text=compressed_text, original_text=chunk.text))

    return result


def novel_ngram_ratio(generated_text: str, source_text: str, n: int = 3) -> float:
    """Fraction of the generated text's n-grams that are "novel" (not present
    in the source) — the manipulation check from guide §6.

    For A (maintenance), this should be close to 0 — the output should be
    almost entirely lifted verbatim from the source. A value meaningfully
    above 0 signals that verbatim post-processing (_snap_to_verbatim) isn't
    actually filtering out new phrasing. Uses word-level (whitespace-split)
    n-grams — kept simple rather than pulling in a morphological analyzer.
    """
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
