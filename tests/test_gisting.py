"""gisting(청크 요약 재료) 단위 테스트 — 전부 mock embed_fn으로 동작, API 키/네트워크 불필요."""

import os

import pytest

from src.pipeline.chuncking import Chunk
from src.pipeline.embeddings import EmbeddingAPIError, load_config
from src.pipeline.gisting import score_chunk_sentences


@pytest.fixture
def config():
    cfg = load_config("configs/importance_filter.yaml")
    os.environ[cfg["api_key_env"]] = "dummy-key-for-test"
    return cfg


@pytest.fixture
def chunks():
    return [
        Chunk(text="고양이가 창밖을 바라본다.", index=0),
        Chunk(text="고양이가 창밖을 바라본다. 주가 지수가 급락했다. 고양이는 낮잠을 잔다. 부동산 시장이 얼어붙었다.", index=1),
        Chunk(text="고양이는 낮잠을 잔다.", index=2),
    ]


def _fake_embed_by_keyword(texts, input_type, model, truncate, api_key, timeout=30.0, dimensions=None):
    """'고양이' 포함 여부로 벡터를 갈라 유사도 차이를 만드는 가짜 임베딩."""
    return [[1.0 if "고양이" in t else 0.0, 1.0 if "고양이" not in t else 0.0, 0.1] for t in texts]


def test_score_chunk_sentences_returns_one_score_per_sentence(chunks, config):
    chunk = chunks[1]  # 4문장
    chunk_embedding = [1.0, 0.0, 0.1]  # "고양이" 쪽 벡터로 가정
    scored = score_chunk_sentences(chunk, chunk_embedding, config=config, embed_fn=_fake_embed_by_keyword)

    assert scored is not None
    assert len(scored) == 4
    sentences_only = [s for s, _ in scored]
    assert sentences_only == ["고양이가 창밖을 바라본다.", "주가 지수가 급락했다.", "고양이는 낮잠을 잔다.", "부동산 시장이 얼어붙었다."]


def test_score_chunk_sentences_higher_score_for_topically_aligned_sentences(chunks, config):
    chunk = chunks[1]
    chunk_embedding = [1.0, 0.0, 0.1]  # "고양이" 쪽 벡터
    scored = score_chunk_sentences(chunk, chunk_embedding, config=config, embed_fn=_fake_embed_by_keyword)
    scores_by_sentence = dict(scored)

    assert scores_by_sentence["고양이가 창밖을 바라본다."] > scores_by_sentence["주가 지수가 급락했다."]


def test_score_chunk_sentences_returns_none_on_api_failure(chunks, config):
    def always_fails(*args, **kwargs):
        raise EmbeddingAPIError("simulated failure")

    config["max_retries"] = 0
    result = score_chunk_sentences(chunks[1], [1.0, 0.0, 0.1], config=config, embed_fn=always_fails)
    assert result is None


def test_missing_api_key_env_raises(chunks, config):
    del os.environ[config["api_key_env"]]
    with pytest.raises(EmbeddingAPIError):
        score_chunk_sentences(chunks[1], [1.0, 0.0, 0.1], config=config, embed_fn=_fake_embed_by_keyword)
