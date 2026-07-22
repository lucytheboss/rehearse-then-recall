"""embeddings(저수준 임베딩 API 공용 유틸) 단위 테스트 — API 키/네트워크 불필요."""

import os

import pytest

from src.pipeline.embeddings import (
    EmbeddingAPIError,
    cosine_similarity,
    embed_with_retry,
    last_embedding_error,
    load_config,
)


def test_cosine_similarity_identical_vectors():
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors():
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_similarity_zero_vector_returns_zero():
    assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


@pytest.fixture
def config():
    cfg = load_config("configs/importance_filter.yaml")
    os.environ[cfg["api_key_env"]] = "dummy-key-for-test"
    cfg["max_retries"] = 0
    return cfg


def test_embed_with_retry_logs_failure_reason_to_stderr(config, capsys):
    """pass_through로 조용히 넘어가는 경로에서도 실패 사실/원인이 stderr에 남아야 한다."""

    def always_fails(*args, **kwargs):
        raise EmbeddingAPIError("simulated failure detail")

    result = embed_with_retry(["텍스트"], "passage", config, always_fails)

    assert result is None
    captured = capsys.readouterr()
    assert "simulated failure detail" in captured.err
    assert "EmbeddingAPIError" in captured.err


def test_last_embedding_error_reflects_most_recent_call(config):
    def always_fails(*args, **kwargs):
        raise EmbeddingAPIError("simulated failure")

    def always_succeeds(texts, input_type, model, truncate, api_key, timeout=30.0, dimensions=None):
        return [[0.0] for _ in texts]

    embed_with_retry(["텍스트"], "passage", config, always_fails)
    assert isinstance(last_embedding_error(), EmbeddingAPIError)

    # 이후 호출이 성공하면 이전 실패 기록은 초기화된다.
    embed_with_retry(["텍스트"], "passage", config, always_succeeds)
    assert last_embedding_error() is None
