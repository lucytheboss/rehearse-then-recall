"""Unit tests for the stage-2 teacher client — all against a fake chat_fn, no
API key or network required.
"""

import os

import pytest

from src.pipeline.curation import (
    EMPTY_SUMMARY_PLACEHOLDER,
    TEACHER_SYSTEM_PROMPT_THREADED,
    build_teacher_messages_threaded,
)
from src.pipeline.teacher import (
    TeacherAPIError,
    complete_with_retry,
    curate_document,
    load_curation_config,
)


@pytest.fixture
def config():
    cfg = load_curation_config()
    os.environ[cfg["api_key_env"]] = "dummy-key-for-test"
    return cfg


def _scripted_chat_fn(replies):
    """Returns each reply in turn and records the prompts it was given."""
    calls = []

    def _chat(messages, model, api_key, temperature, max_tokens, timeout):
        calls.append(messages)
        return replies[len(calls) - 1]

    _chat.calls = calls
    return _chat


def _failing_chat_fn(*args, **kwargs):
    raise TeacherAPIError("simulated outage")


def test_curate_document_feeds_each_summary_into_the_next_prompt(config):
    """The rolling objective stage 1 never provided: chunk i's prompt has to
    carry chunk i-1's answer, or this is just one-shot summarization again."""
    chat_fn = _scripted_chat_fn(["summary A", "summary B", "summary C"])
    result = curate_document(["chunk 0", "chunk 1", "chunk 2"], config, chat_fn)

    assert result.running_summaries == ["summary A", "summary B", "summary C"]
    assert EMPTY_SUMMARY_PLACEHOLDER in chat_fn.calls[0][1]["content"]
    assert "summary A" in chat_fn.calls[1][1]["content"]
    assert "summary B" in chat_fn.calls[2][1]["content"]


def test_curate_document_aligns_summaries_with_chunks(config):
    """running_summaries[i] is the summary *after* chunk i — the alignment
    build_stage2_pairs assumes."""
    chat_fn = _scripted_chat_fn([f"s{i}" for i in range(5)])
    result = curate_document([f"c{i}" for i in range(5)], config, chat_fn)
    assert len(result.running_summaries) == 5
    assert result.ok


def test_curate_document_carries_the_summary_forward_on_a_failure(config):
    """Truncating instead would silently shorten the document and bias the
    position axis the collapse diagnostic is plotted on."""
    config = dict(config)
    config["max_retries"] = 0

    calls = {"n": 0}

    def flaky(messages, model, api_key, temperature, max_tokens, timeout):
        calls["n"] += 1
        if calls["n"] == 2:
            raise TeacherAPIError("simulated outage")
        return f"summary {calls['n']}"

    result = curate_document(["c0", "c1", "c2"], config, flaky)

    assert len(result.running_summaries) == 3
    assert result.running_summaries[1] == result.running_summaries[0]
    assert result.failed_positions == [1]
    assert not result.ok


def test_complete_with_retry_returns_none_after_exhausting_retries(config):
    config = dict(config)
    config["max_retries"] = 1
    assert complete_with_retry([{"role": "user", "content": "x"}], config, _failing_chat_fn) is None


class _CountingRateLimiter:
    def __init__(self):
        self.calls = 0

    def acquire(self):
        self.calls += 1


def test_complete_with_retry_acquires_rate_limiter_before_every_attempt(config):
    """Backoff alone only slows a caller down *after* a 429 — the rate
    limiter is what stops the request from being sent too fast in the first
    place, which is the actual fix for 06b's 2026-08-04 429s."""
    config = dict(config)
    config["max_retries"] = 2
    config["retry_base_delay"] = 0.001
    config["retry_max_delay"] = 0.001
    limiter = _CountingRateLimiter()
    complete_with_retry([{"role": "user", "content": "x"}], config, _failing_chat_fn, rate_limiter=limiter)
    assert limiter.calls == 3  # initial attempt + 2 retries


def test_complete_with_retry_works_without_rate_limiter(config):
    """rate_limiter is optional — existing callers must be unaffected."""
    config = dict(config)
    config["max_retries"] = 0
    assert complete_with_retry([{"role": "user", "content": "x"}], config, _failing_chat_fn) is None


def test_complete_with_retry_requires_the_api_key(config):
    config = dict(config)
    config["api_key_env"] = "DEFINITELY_NOT_SET_FOR_TESTS"
    os.environ.pop("DEFINITELY_NOT_SET_FOR_TESTS", None)
    with pytest.raises(TeacherAPIError):
        complete_with_retry([{"role": "user", "content": "x"}], config, _failing_chat_fn)


def test_curation_config_pins_a_deterministic_teacher():
    """A label that changes between runs would make a drop at position 6
    unreadable — sampling noise or real degradation?"""
    assert load_curation_config()["temperature"] == 0.0


# --------------------------------------------------------------------------
# Threaded curation — C-DIC's actual retrieve/generate/write-back shape
# --------------------------------------------------------------------------

from src.pipeline.curation import EMPTY_CONTEXT_PLACEHOLDER
from src.pipeline.embeddings import EmbeddingAPIError, load_config
from src.pipeline.teacher import curate_document_threaded


@pytest.fixture
def embed_cfg():
    cfg = load_config("configs/importance_filter.yaml")
    os.environ[cfg["api_key_env"]] = "dummy-key-for-test"
    return cfg


def _topic_embed_fn(texts, input_type, model, truncate, api_key, timeout=30.0, dimensions=None):
    """One-hot on the `topicN` marker in each chunk, so retrieval decisions
    are deterministic — same fixture idiom as test_rehearsal.py's."""
    vectors = []
    for text in texts:
        vector = [0.0] * 8
        vector[int(text.split("topic")[1][0])] = 1.0
        vectors.append(vector)
    return vectors


def _gist_chat_fn(messages, model, api_key, temperature, max_tokens, timeout):
    """Echoes the current chunk back tagged as a gist, plus a `+SAW` marker
    when retrieved context was actually shown — checkable evidence that
    generation was conditioned on retrieval, without embedding the context's
    own text (which would recursively nest "GIST[" once a revised slot's
    gist becomes someone else's retrieved context)."""
    user_content = messages[1]["content"]
    context_block = user_content.split(
        "Related thread(s) (chosen by retrieval, not by you):\n", 1
    )[1].split("\n\nCurrent chunk:")[0]
    chunk = user_content.split("Current chunk:\n", 1)[1].split("\n\nShortened passage:")[0]
    saw_context = "+SAW" if context_block != EMPTY_CONTEXT_PLACEHOLDER else ""
    return f"GIST[{chunk}]{saw_context}"


def _topic_chunk_texts():
    # chunk 4 returns to chunk 0's topic — the case the mechanism exists to catch.
    return [
        "topic0 the opening scene",
        "topic1 an unrelated aside",
        "topic2 another digression",
        "topic3 more filler material",
        "topic0 the opening scene returns",
    ]


def test_build_teacher_messages_threaded_renders_empty_context_as_placeholder():
    messages = build_teacher_messages_threaded([], "some chunk")
    assert EMPTY_CONTEXT_PLACEHOLDER in messages[1]["content"]
    assert messages[0]["content"] == TEACHER_SYSTEM_PROMPT_THREADED


def test_build_teacher_messages_threaded_includes_retrieved_context():
    messages = build_teacher_messages_threaded(["earlier gist one", "earlier gist two"], "chunk")
    assert "earlier gist one" in messages[1]["content"]
    assert "earlier gist two" in messages[1]["content"]
    assert EMPTY_CONTEXT_PLACEHOLDER not in messages[1]["content"]


def test_teacher_prompt_threaded_uses_shorten_not_summarize():
    """ReadAgent (Lee, Chen, Furuta, Canny & Fischer, ICML 2024, arXiv:2402.09727)
    §3.1: 'shorten' preserves narrative flow across concatenation where
    'summarize' produces a restructured synopsis — exactly our failure mode,
    since every gist gets concatenated with others via retrieval."""
    assert "Shorten" in TEACHER_SYSTEM_PROMPT_THREADED
    assert "do not summarize" in TEACHER_SYSTEM_PROMPT_THREADED


def test_teacher_prompt_threaded_states_the_replace_semantics():
    """Eq. 6's revise branch: the output REPLACES the retrieved thread(s), it
    does not merely append to them."""
    assert "REPLACES" in TEACHER_SYSTEM_PROMPT_THREADED


def test_curate_document_threaded_is_query_agnostic():
    import inspect
    params = set(inspect.signature(curate_document_threaded).parameters)
    forbidden = {p for p in params if "question" in p.lower() or "query" in p.lower()}
    assert not forbidden, f"query-agnostic violation: {forbidden}"


def test_curate_document_threaded_reaches_across_the_whole_document(embed_cfg, config):
    result = curate_document_threaded(
        _topic_chunk_texts(), embed_cfg, config,
        embed_fn=_topic_embed_fn, chat_fn=_gist_chat_fn,
    )
    assert result.context_texts[0] == [], "nothing rehearsed yet"
    assert result.context_texts[4] == ["GIST[topic0 the opening scene]"], (
        "chunk 4 did not retrieve chunk 0's thread across the intervening unrelated chunks"
    )


def test_curate_document_threaded_branches_insert_vs_revise(embed_cfg, config):
    result = curate_document_threaded(
        _topic_chunk_texts(), embed_cfg, config,
        embed_fn=_topic_embed_fn, chat_fn=_gist_chat_fn,
    )
    assert result.write_back_actions == ["insert", "insert", "insert", "insert", "revise"]


def test_curate_document_threaded_replaces_the_matched_slot_not_appends(embed_cfg, config):
    """The bug this mirrors from `rehearse_elaborative`'s history: memory must
    not grow one slot per chunk once a thread is revisited."""
    result = curate_document_threaded(
        _topic_chunk_texts(), embed_cfg, config,
        embed_fn=_topic_embed_fn, chat_fn=_gist_chat_fn,
    )
    final_slot_count = result.memory_states[-1].count("GIST[")
    assert final_slot_count == 4, "5 chunks over 4 topics must end with 4 slots, not 5"


def test_curate_document_threaded_gist_is_conditioned_on_retrieved_context(embed_cfg, config):
    """The whole point: chunk 4's gist is generated *with* chunk 0's thread in
    view, not independently of it — that's what makes this different from the
    linear loop, where there is only ever one state to condition on."""
    result = curate_document_threaded(
        _topic_chunk_texts(), embed_cfg, config,
        embed_fn=_topic_embed_fn, chat_fn=_gist_chat_fn,
    )
    assert "+SAW" in result.gists[4]
    assert "+SAW" not in result.gists[0]


def test_curate_document_threaded_input_output_shape_matches_inference(embed_cfg, config):
    """`format_elaborative_input(chunk, context_texts[i]) -> gists[i]` must be
    the exact shape `rehearse_elaborative` feeds and expects back."""
    from src.pipeline.rehearsal import format_elaborative_input

    chunks = _topic_chunk_texts()
    result = curate_document_threaded(
        chunks, embed_cfg, config, embed_fn=_topic_embed_fn, chat_fn=_gist_chat_fn,
    )
    for i, chunk in enumerate(chunks):
        rebuilt_input = format_elaborative_input(chunk, result.context_texts[i])
        assert f"Current chunk:\n{chunk}" not in rebuilt_input  # sanity: not the teacher prompt itself
        assert chunk in rebuilt_input


def test_curate_document_threaded_tau_write_stricter_than_tau_read(embed_cfg, config):
    result = curate_document_threaded(
        _topic_chunk_texts(), embed_cfg, config,
        embed_fn=_topic_embed_fn, chat_fn=_gist_chat_fn,
        tau_read=0.35, tau_write=1.5,
    )
    assert result.context_texts[4] == ["GIST[topic0 the opening scene]"], "still reads the aligned slot"
    assert all(action == "insert" for action in result.write_back_actions)


def test_curate_document_threaded_records_a_failed_teacher_call(embed_cfg, config):
    config = dict(config)
    config["max_retries"] = 0

    def flaky(messages, model, api_key, temperature, max_tokens, timeout):
        raise TeacherAPIError("simulated outage")

    result = curate_document_threaded(
        _topic_chunk_texts()[:2], embed_cfg, config, embed_fn=_topic_embed_fn, chat_fn=flaky,
    )
    assert result.failed_positions == [0, 1]
    assert result.gists == _topic_chunk_texts()[:2], "falls back to the raw chunk, not empty"
    assert not result.ok


def test_curate_document_threaded_records_a_retrieval_failure(embed_cfg, config):
    def always_fails_embed(*args, **kwargs):
        raise EmbeddingAPIError("simulated embedding outage")

    result = curate_document_threaded(
        _topic_chunk_texts()[:1], embed_cfg, config, embed_fn=always_fails_embed, chat_fn=_gist_chat_fn,
    )
    assert result.retrieval_failed_positions == [0]
    assert result.context_texts == [[]]


def test_curate_document_threaded_ok_property_covers_every_failure_kind(embed_cfg, config):
    result = curate_document_threaded(
        _topic_chunk_texts()[:1], embed_cfg, config, embed_fn=_topic_embed_fn, chat_fn=_gist_chat_fn,
    )
    assert result.ok


def test_curate_document_threaded_fires_on_chunk_done_per_chunk(embed_cfg, config):
    """Without this, a document that is slow under real API load and a
    document that has hung are indistinguishable from the outside — nothing
    else prints between one document finishing and the next."""
    seen = []
    curate_document_threaded(
        _topic_chunk_texts(), embed_cfg, config,
        embed_fn=_topic_embed_fn, chat_fn=_gist_chat_fn,
        on_chunk_done=lambda position, total: seen.append((position, total)),
    )
    assert seen == [(i, 5) for i in range(5)]


def test_curate_document_threaded_works_without_on_chunk_done(embed_cfg, config):
    """The callback is optional — existing callers must be unaffected."""
    result = curate_document_threaded(
        _topic_chunk_texts(), embed_cfg, config, embed_fn=_topic_embed_fn, chat_fn=_gist_chat_fn,
    )
    assert len(result.gists) == 5


def test_curate_document_threaded_shares_rate_limiter_across_all_calls(embed_cfg, config):
    """Every chunk costs 3 real requests (query embed, teacher call, store
    embed) — all 3 must draw from the same limiter, since the whole point is
    one combined budget across everything this document (and, in the
    document-parallel notebook, every concurrently-curated document) sends."""
    limiter = _CountingRateLimiter()
    chunks = _topic_chunk_texts()
    curate_document_threaded(
        chunks, embed_cfg, config, embed_fn=_topic_embed_fn, chat_fn=_gist_chat_fn,
        rate_limiter=limiter,
    )
    assert limiter.calls == 3 * len(chunks)


def test_curate_document_threaded_works_without_rate_limiter(embed_cfg, config):
    """rate_limiter is optional — existing callers must be unaffected."""
    result = curate_document_threaded(
        _topic_chunk_texts(), embed_cfg, config, embed_fn=_topic_embed_fn, chat_fn=_gist_chat_fn,
    )
    assert len(result.gists) == 5
