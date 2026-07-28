"""Material for chunk-summary (gisting) compression logic — compress_chunk
doesn't exist yet; that function will eventually consume the scores these
functions produce. Pure embedding API calls + vector math only, no
generative prompts.
"""

from __future__ import annotations

from nltk.tokenize import sent_tokenize

from src.pipeline.embeddings import _batched, cosine_similarity, embed_texts, embed_with_retry
from src.pipeline.types import Chunk


def split_into_sentences(text: str) -> list[str]:
    sentences = sent_tokenize(text)
    return sentences if sentences else [text]


def embed_chunks(chunks: list[Chunk], config: dict, embed_fn=embed_texts) -> list[list[float]] | None:
    """Batch-embeds chunks as passages. Returns None if it still fails after retries."""
    embeddings: list[list[float]] = []
    for batch in _batched(chunks, config["batch_size"]):
        batch_embeddings = embed_with_retry([c.text for c in batch], "passage", config, embed_fn)
        if batch_embeddings is None:
            return None
        embeddings.extend(batch_embeddings)
    return embeddings


def score_chunk_sentences(
    chunk: Chunk,
    chunk_embedding: list[float],
    config: dict,
    embed_fn=embed_texts,
) -> list[tuple[str, float]] | None:
    """Per-sentence importance score *within* a chunk. Treats the chunk's
    whole-chunk embedding as the centroid meaning and scores each sentence by
    cosine similarity to it. Drops nothing — returns (sentence, score) pairs
    only; the caller decides whether/what to compress.
    Returns None on API failure (retries exhausted).
    """
    sentences = split_into_sentences(chunk.text)
    sentence_embeddings = embed_with_retry(sentences, "passage", config, embed_fn)
    if sentence_embeddings is None:
        return None
    return [(s, cosine_similarity(emb, chunk_embedding)) for s, emb in zip(sentences, sentence_embeddings)]
