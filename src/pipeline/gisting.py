"""청크 요약(gisting) 압축 로직의 재료 — compress_chunk는 아직 없고, 이 함수들이
매기는 점수를 그 함수가 나중에 활용할 예정. 순수 임베딩 API 호출 + 벡터
연산만 하며, 생성형 프롬프트는 쓰지 않는다.
"""

from __future__ import annotations

from typing import cast

import kss

from src.pipeline.embeddings import _batched, cosine_similarity, embed_texts, embed_with_retry
from src.pipeline.types import Chunk


def split_into_sentences(text: str) -> list[str]:
    sentences = kss.split_sentences(text, backend="fast")
    if sentences and isinstance(sentences[0], list):
        return [s for group in sentences for s in group]
    if sentences:
        return cast(list, sentences)
    return [text]


def embed_chunks(chunks: list[Chunk], config: dict, embed_fn=embed_texts) -> list[list[float]] | None:
    """청크를 배치로 묶어 passage 임베딩. 재시도 후에도 실패하면 None."""
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
    """청크 "내부" 문장별 중요도 지표. 청크 전체 임베딩을 중심 의미(centroid)로
    보고 문장별 임베딩과의 코사인 유사도를 점수로 삼는다. 아무것도 버리지
    않고 (문장, 점수) 목록만 반환 — 압축 여부는 호출자가 결정.
    API 실패(재시도 소진) 시 None 반환.
    """
    sentences = split_into_sentences(chunk.text)
    sentence_embeddings = embed_with_retry(sentences, "passage", config, embed_fn)
    if sentence_embeddings is None:
        return None
    return [(s, cosine_similarity(emb, chunk_embedding)) for s, emb in zip(sentences, sentence_embeddings)]
