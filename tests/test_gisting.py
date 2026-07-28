"""Unit tests for gisting (chunk-summary material) — all run against a mock embed_fn, no API key/network required."""

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
        Chunk(text="The cat looks out the window.", index=0),
        Chunk(
            text="The cat looks out the window. The stock index plunged. The cat takes a nap. "
            "The real estate market froze.",
            index=1,
        ),
        Chunk(text="The cat takes a nap.", index=2),
    ]


def _fake_embed_by_keyword(texts, input_type, model, truncate, api_key, timeout=30.0, dimensions=None):
    """A fake embedding that splits vectors by whether the text contains "cat", to produce a similarity difference."""
    return [[1.0 if "cat" in t else 0.0, 1.0 if "cat" not in t else 0.0, 0.1] for t in texts]


def test_score_chunk_sentences_returns_one_score_per_sentence(chunks, config):
    chunk = chunks[1]  # 4 sentences
    chunk_embedding = [1.0, 0.0, 0.1]  # assume the "cat" vector
    scored = score_chunk_sentences(chunk, chunk_embedding, config=config, embed_fn=_fake_embed_by_keyword)

    assert scored is not None
    assert len(scored) == 4
    sentences_only = [s for s, _ in scored]
    assert sentences_only == [
        "The cat looks out the window.",
        "The stock index plunged.",
        "The cat takes a nap.",
        "The real estate market froze.",
    ]


def test_score_chunk_sentences_higher_score_for_topically_aligned_sentences(chunks, config):
    chunk = chunks[1]
    chunk_embedding = [1.0, 0.0, 0.1]  # the "cat" vector
    scored = score_chunk_sentences(chunk, chunk_embedding, config=config, embed_fn=_fake_embed_by_keyword)
    scores_by_sentence = dict(scored)

    assert scores_by_sentence["The cat looks out the window."] > scores_by_sentence["The stock index plunged."]


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
