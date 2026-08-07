"""Unit tests for embeddings (low-level embedding API shared utilities) — no API key/network required."""

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
    """Even on the pass_through path (silently continuing), the failure and its cause must be logged to stderr."""

    def always_fails(*args, **kwargs):
        raise EmbeddingAPIError("simulated failure detail")

    result = embed_with_retry(["text"], "passage", config, always_fails)

    assert result is None
    captured = capsys.readouterr()
    assert "simulated failure detail" in captured.err
    assert "EmbeddingAPIError" in captured.err


class _CountingRateLimiter:
    def __init__(self):
        self.calls = 0

    def acquire(self):
        self.calls += 1


def test_embed_with_retry_acquires_rate_limiter_before_every_attempt(config):
    config = dict(config)
    config["max_retries"] = 2
    limiter = _CountingRateLimiter()

    def always_fails(*args, **kwargs):
        raise EmbeddingAPIError("simulated failure")

    embed_with_retry(["text"], "passage", config, always_fails, rate_limiter=limiter)
    assert limiter.calls == 3  # initial attempt + 2 retries


def test_embed_with_retry_works_without_rate_limiter(config):
    """rate_limiter is optional — existing callers must be unaffected."""
    def always_succeeds(texts, input_type, model, truncate, api_key, timeout=30.0, dimensions=None):
        return [[0.0] for _ in texts]

    assert embed_with_retry(["text"], "passage", config, always_succeeds) == [[0.0]]


def test_last_embedding_error_reflects_most_recent_call(config):
    def always_fails(*args, **kwargs):
        raise EmbeddingAPIError("simulated failure")

    def always_succeeds(texts, input_type, model, truncate, api_key, timeout=30.0, dimensions=None):
        return [[0.0] for _ in texts]

    embed_with_retry(["text"], "passage", config, always_fails)
    assert isinstance(last_embedding_error(), EmbeddingAPIError)

    # A subsequent successful call should clear the previous failure record.
    embed_with_retry(["text"], "passage", config, always_succeeds)
    assert last_embedding_error() is None
