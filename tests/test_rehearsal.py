"""Unit tests for rehearsal (maintenance rehearsal A) — all run against a fake
model/tokenizer/embed_fn, no real model download, API key, or network required.
"""

import inspect
import os

import pytest
import torch
from transformers import BatchEncoding

from src.pipeline.embeddings import EmbeddingAPIError, load_config
from src.pipeline.rehearsal import _snap_to_verbatim, novel_ngram_ratio, rehearse_maintenance
from src.pipeline.types import Chunk


@pytest.fixture
def config():
    cfg = load_config("configs/importance_filter.yaml")
    os.environ[cfg["api_key_env"]] = "dummy-key-for-test"
    return cfg


class _EchoTokenizer:
    """A fake tokenizer that registers/looks up chunk.text as a single integer
    ID — with no real subword splitting, this only verifies
    rehearse_maintenance's tokenize->generate->decode control flow.
    """

    def __init__(self):
        self._id_of: dict[str, int] = {}
        self._text_of: dict[int, str] = {}

    def __call__(self, text, return_tensors="pt", truncation=True, max_length=None):
        if text not in self._id_of:
            new_id = len(self._id_of)
            self._id_of[text] = new_id
            self._text_of[new_id] = text
        token_id = self._id_of[text]
        return BatchEncoding(
            {"input_ids": torch.tensor([[token_id]]), "attention_mask": torch.tensor([[1]])}
        )

    def decode(self, ids, skip_special_tokens=True):
        return self._text_of[int(ids[0])]


class _EchoModel:
    """A fake model that returns the input tokens unchanged — makes
    decode(generate(encode(t))) == t hold, so it verifies the "generated text
    is identical to the source" path (where verbatim snapping isn't needed).
    """

    def __init__(self):
        self._param = torch.nn.Parameter(torch.zeros(1))

    def eval(self):
        return self

    def parameters(self):
        yield self._param

    def generate(self, input_ids, attention_mask=None, max_new_tokens=None):
        return input_ids


def _poison_embed_fn(texts, input_type, model, truncate, api_key, timeout=30.0, dimensions=None):
    raise AssertionError("embed_fn was called in a situation where it must not be called")


@pytest.fixture
def sample_chunk():
    return Chunk(
        text="The cat looks out the window. The stock index plunged.",
        index=2,
        paragraph_indices=[5, 6],
        char_start=100,
        char_end=150,
    )


def test_rehearse_maintenance_is_query_agnostic():
    """Guide §2 hard constraint: must not have a question/query-like parameter."""
    params = set(inspect.signature(rehearse_maintenance).parameters)
    forbidden = {p for p in params if "question" in p.lower() or "query" in p.lower()}
    assert not forbidden, f"query-agnostic violation — found forbidden parameter(s): {forbidden}"


def test_rehearse_maintenance_preserves_position_metadata(sample_chunk, config):
    # Echo model means generated == source -> every sentence is already verbatim, so embed_fn must not be called
    result = rehearse_maintenance(
        [sample_chunk],
        model=_EchoModel(),
        tokenizer=_EchoTokenizer(),
        embed_cfg=config,
        embed_fn=_poison_embed_fn,
    )

    assert len(result) == 1
    out = result[0]
    assert out.index == sample_chunk.index
    assert out.paragraph_indices == sample_chunk.paragraph_indices
    assert out.char_start == sample_chunk.char_start
    assert out.char_end == sample_chunk.char_end


def test_rehearse_maintenance_sets_original_text_to_pre_compression_text(sample_chunk, config):
    result = rehearse_maintenance(
        [sample_chunk],
        model=_EchoModel(),
        tokenizer=_EchoTokenizer(),
        embed_cfg=config,
        embed_fn=_poison_embed_fn,
    )
    assert result[0].original_text == sample_chunk.text


def test_rehearse_maintenance_handles_multiple_chunks_independently(config):
    chunks = [
        Chunk(text="This is the first chunk's sentence.", index=0),
        Chunk(text="This is the second chunk's sentence.", index=1),
    ]
    result = rehearse_maintenance(
        chunks, model=_EchoModel(), tokenizer=_EchoTokenizer(), embed_cfg=config, embed_fn=_poison_embed_fn
    )
    assert [c.original_text for c in result] == [c.text for c in chunks]
    assert [c.index for c in result] == [0, 1]


def _fake_embed_fixed_vectors(vectors_by_text: dict):
    def _embed(texts, input_type, model, truncate, api_key, timeout=30.0, dimensions=None):
        return [vectors_by_text[t] for t in texts]

    return _embed


def test_snap_to_verbatim_keeps_exact_matches_without_calling_embed_fn():
    generated = ["The cat looks out the window.", "The stock index plunged."]
    source = ["The cat looks out the window.", "The stock index plunged.", "The real estate market froze."]

    result = _snap_to_verbatim(generated, source, embed_cfg={}, embed_fn=_poison_embed_fn)
    assert result == generated


def test_snap_to_verbatim_replaces_non_matching_sentence_with_closest_by_cosine(config):
    generated = ["The cat stares outside."]  # a paraphrase not present in the source
    source = ["The cat looks out the window.", "The real estate market froze."]

    vectors = {
        "The cat stares outside.": [1.0, 0.0],
        "The cat looks out the window.": [0.9, 0.1],  # higher cosine similarity
        "The real estate market froze.": [0.0, 1.0],
    }
    result = _snap_to_verbatim(generated, source, embed_cfg=config, embed_fn=_fake_embed_fixed_vectors(vectors))
    assert result == ["The cat looks out the window."]


def test_snap_to_verbatim_falls_back_to_generated_on_embedding_failure(config):
    def always_fails(*args, **kwargs):
        raise EmbeddingAPIError("simulated embedding outage")

    config = dict(config)
    config["max_retries"] = 0
    generated = ["A new phrase not in the source."]
    source = ["One source sentence."]

    result = _snap_to_verbatim(generated, source, embed_cfg=config, embed_fn=always_fails)
    assert result == generated  # doesn't die, returns the generated text unchanged


def test_novel_ngram_ratio_is_zero_for_pure_verbatim_text():
    source = "the cat looks out the window the market fell hard today"
    generated = "the cat looks"
    assert novel_ngram_ratio(generated, source, n=3) == 0.0


def test_novel_ngram_ratio_is_positive_for_novel_text():
    source = "the cat looks out the window the market fell today"
    generated = "completely different words forming a brand new sentence"
    assert novel_ngram_ratio(generated, source, n=3) == 1.0


def test_novel_ngram_ratio_short_generation_returns_zero():
    # too short a generation to form an n-gram -> 0 (a safe default instead of an exception)
    assert novel_ngram_ratio("one word", "some source text", n=3) == 0.0
