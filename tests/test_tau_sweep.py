"""Unit tests for Ablation A4's tau sweep — fake model/tokenizer/embed_fn, no
checkpoint, API key, or network required.
"""

import math

import pytest

from src.eval.run_tau_sweep import _load_synthetic_inputs
from src.eval.tau_sweep import (
    CachedEmbedder,
    TauRow,
    answer_recovery,
    degeneration_checks,
    recommend_tau,
    sweep_tau,
    to_markdown,
)


# -- CachedEmbedder: the reason a 5-value sweep is not 5x the budget --------


def _counting_embed_fn():
    def _embed(texts, input_type, model, truncate, api_key, timeout=30.0, dimensions=None):
        _embed.batches.append(list(texts))
        return [[float(len(t)), float(input_type == "query")] for t in texts]

    _embed.batches = []
    return _embed


def test_cached_embedder_only_fetches_misses():
    inner = _counting_embed_fn()
    cached = CachedEmbedder(inner)
    kwargs = dict(input_type="passage", model="m", truncate="NONE", api_key="k")

    cached(["a", "bb"], **kwargs)
    cached(["bb", "ccc"], **kwargs)

    assert inner.batches == [["a", "bb"], ["ccc"]], "the second call must only send its miss"
    assert cached.hits == 1 and cached.misses == 3


def test_cached_embedder_keys_on_input_type():
    """query and passage embeddings of the same text are different vectors —
    the NIM model is trained for that split, so they must not share a cache
    entry."""
    inner = _counting_embed_fn()
    cached = CachedEmbedder(inner)
    base = dict(model="m", truncate="NONE", api_key="k")

    query = cached(["same"], input_type="query", **base)
    passage = cached(["same"], input_type="passage", **base)

    assert query != passage
    assert cached.hits == 0


def test_cached_embedder_preserves_input_order():
    inner = _counting_embed_fn()
    cached = CachedEmbedder(inner)
    kwargs = dict(input_type="passage", model="m", truncate="NONE", api_key="k")

    cached(["bb"], **kwargs)
    result = cached(["a", "bb", "ccc"], **kwargs)

    assert [vector[0] for vector in result] == [1.0, 2.0, 3.0]


def test_cached_embedder_hit_rate_is_zero_before_any_call():
    assert CachedEmbedder(_counting_embed_fn()).hit_rate == 0.0


# -- answer recovery -------------------------------------------------------


def test_answer_recovery_separates_verbatim_from_paraphrase():
    """The EM/F1 gap is informative rather than noise: answers surviving in
    paraphrase is B working as intended, not failing."""
    state = "The suspect Chester Arthur Stiles faced additional charges."
    em, f1 = answer_recovery(state, ["Chester Arthur Stiles"])
    assert em == 1.0 and f1 == 1.0

    em, f1 = answer_recovery(state, ["Chester Stiles"])
    assert em == 0.0, "not a contiguous span in the state"
    assert f1 == 1.0, "but every token survived"


def test_answer_recovery_handles_no_answers():
    assert answer_recovery("anything", []) == (0.0, 0.0)


# -- degeneration checks ---------------------------------------------------


def _row(tau, insert_share, recency_share=0.3, memory_size=10, answer_f1=0.5):
    return TauRow(
        tau=tau, insert_share=insert_share, fire_rate=1 - insert_share,
        recency_share=recency_share, mean_span=2.0, mean_thread_span=3.0,
        memory_size=memory_size, memory_growth=memory_size / 20,
        early_retention=0.6, late_retention=0.4, retention_drop=0.2,
        answer_em=answer_f1, answer_f1=answer_f1,
    )


def test_checks_pass_when_the_sweep_brackets_both_failure_modes():
    rows = [_row(0.3, 0.05), _row(0.6, 0.5), _row(0.9, 0.95)]
    report = degeneration_checks(rows)
    assert report.passed


def test_checks_fail_when_the_range_never_reaches_append_only():
    """A tau chosen from a sweep whose high end never blew up was chosen from
    the flat middle of a curve whose edge was never located."""
    rows = [_row(0.3, 0.05), _row(0.5, 0.2), _row(0.6, 0.3)]
    report = degeneration_checks(rows)
    assert not report.passed
    assert any("append-only" in c.name and not c.passed for c in report.checks)


def test_checks_fail_when_the_range_never_reaches_collapse():
    rows = [_row(0.7, 0.5), _row(0.8, 0.7), _row(0.9, 0.95)]
    report = degeneration_checks(rows)
    assert not report.passed
    assert any("single-thread collapse" in c.name and not c.passed for c in report.checks)


def test_non_monotone_insert_share_is_flagged():
    """Raising the bar for 'on topic' can only move chunks from revise to
    insert, so a decrease means something other than tau changed."""
    rows = [_row(0.3, 0.05), _row(0.6, 0.9), _row(0.9, 0.4)]
    report = degeneration_checks(rows)
    assert not report.passed
    assert any("non-decreasing" in c.name and not c.passed for c in report.checks)


def test_recency_degeneration_is_an_observation_not_a_check():
    """Its absence is a *good* result — thread-aware retrieval beating the
    rolling baseline — so it must never gate `passed`."""
    rows = [_row(0.3, 0.05, recency_share=0.1), _row(0.6, 0.5), _row(0.9, 0.95)]
    report = degeneration_checks(rows)
    assert report.passed
    assert any("no tau degenerated to recency" in o for o in report.observations)

    collapsed = [_row(0.3, 0.05, recency_share=0.95), _row(0.6, 0.5), _row(0.9, 0.95)]
    report = degeneration_checks(collapsed)
    assert report.passed, "still a valid sweep — it just found the degeneration"
    assert any("recency degeneration at tau 0.3" in o for o in report.observations)


def test_degeneration_checks_handle_no_rows():
    assert not degeneration_checks([]).passed


# -- recommendation --------------------------------------------------------


def test_recommend_tau_refuses_when_nothing_sits_between_the_failure_modes():
    rows = [_row(0.3, 0.05), _row(0.9, 0.95)]
    tau, why = recommend_tau(rows)
    assert math.isnan(tau)
    assert "widen the sweep range" in why


def test_recommend_tau_ignores_the_degenerate_high_end():
    """The trap this exists to avoid: answer recoverability rises with tau and
    is highest in the append-only regime, so picking on F1 alone would
    recommend the failure mode."""
    rows = [_row(0.3, 0.05, answer_f1=0.2), _row(0.65, 0.5, answer_f1=0.6),
            _row(0.9, 0.95, answer_f1=0.99)]
    tau, _ = recommend_tau(rows)
    assert tau == 0.65


def test_recommend_tau_breaks_ties_toward_smaller_memory():
    rows = [_row(0.5, 0.3, memory_size=18, answer_f1=0.5),
            _row(0.65, 0.5, memory_size=9, answer_f1=0.5)]
    tau, _ = recommend_tau(rows)
    assert tau == 0.65


# -- end to end ------------------------------------------------------------


def test_sweep_produces_one_row_per_tau_and_reuses_query_embeddings():
    chunks, answers, model, tokenizer, embed_cfg, embed_fn = _load_synthetic_inputs(n_chunks=8)
    taus = (0.3, 0.65, 0.9)

    rows, cached = sweep_tau(
        chunks, answers, model, tokenizer, embed_cfg, embed_fn,
        taus=taus, max_input_length=10_000,
    )

    assert [r.tau for r in rows] == list(taus)
    # Chunk texts are fixed input, so every tau after the first is a pure cache
    # hit on the query side — that is what keeps a 3-value sweep from costing 3x.
    assert cached.hit_rate > 0.5
    assert rows[0].api_calls > rows[-1].api_calls


def test_sweep_shows_memory_growing_with_tau():
    """The mechanism under test: a stricter threshold means fewer revisions,
    so more slots."""
    chunks, answers, model, tokenizer, embed_cfg, embed_fn = _load_synthetic_inputs(n_chunks=16)
    rows, _ = sweep_tau(
        chunks, answers, model, tokenizer, embed_cfg, embed_fn,
        taus=(0.3, 0.9), max_input_length=10_000,
    )
    assert rows[0].memory_size < rows[-1].memory_size
    assert rows[0].insert_share < rows[-1].insert_share


def test_synthetic_fixture_brackets_both_failure_modes():
    """Guards the fixture geometry itself. A regular topic sequence or zero
    drift makes the table a step function, and the sweep would then never reach
    the append-only mode at any tau — see _load_synthetic_inputs."""
    chunks, answers, model, tokenizer, embed_cfg, embed_fn = _load_synthetic_inputs()
    rows, _ = sweep_tau(
        chunks, answers, model, tokenizer, embed_cfg, embed_fn, max_input_length=10_000,
    )
    assert degeneration_checks(rows).passed


def test_markdown_table_marks_em_f1_as_an_upper_bound():
    """EM*/F1* are answer recoverability from the compressed state, not the
    pipeline's QA score, and the table has to say so wherever it is pasted."""
    table = to_markdown([_row(0.3, 0.05), _row(0.9, 0.95)])
    assert table.splitlines()[0].startswith("| τ |")
    assert "upper" in table and "not the pipeline's QA" in table
    assert len([line for line in table.splitlines() if line.startswith("| 0")]) == 2
