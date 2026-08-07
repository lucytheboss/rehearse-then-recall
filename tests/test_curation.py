"""Unit tests for stage-2 curation (elaborative rehearsal B) — prompt assembly
and the collapse diagnostic. No API key or network required.
"""

import pytest

from src.pipeline.curation import (
    EMPTY_SUMMARY_PLACEHOLDER,
    TEACHER_SYSTEM_PROMPT,
    ProbeResult,
    answer_supported,
    build_stage2_pairs,
    build_stage2_pairs_threaded,
    build_teacher_messages,
    probe_accuracy_by_position,
    retention_probes,
    token_f1,
)
from src.pipeline.rehearsal import PASSAGE_FIELD, format_elaborative_input


def test_teacher_prompt_states_both_write_back_branches():
    """C-DIC Eq. 6 branches insert vs revise. A prompt carrying only 'revise'
    is what lets the teacher force an unrelated new topic into an existing
    sentence — the drift the instruction was meant to prevent."""
    assert "CONTINUES" in TEACHER_SYSTEM_PROMPT
    assert "REVISE" in TEACHER_SYSTEM_PROMPT
    assert "INTRODUCES" in TEACHER_SYSTEM_PROMPT
    assert "ADD one new, separate sentence" in TEACHER_SYSTEM_PROMPT
    assert "Never blend two unrelated topics" in TEACHER_SYSTEM_PROMPT


def test_build_teacher_messages_has_system_and_user_roles():
    messages = build_teacher_messages("S_prev text", "C_i text")
    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[0]["content"] == TEACHER_SYSTEM_PROMPT
    assert "S_prev text" in messages[1]["content"]
    assert "C_i text" in messages[1]["content"]


@pytest.mark.parametrize("empty", ["", "   ", "\n"])
def test_build_teacher_messages_routes_the_first_chunk_to_the_insert_branch(empty):
    """No prior summary means nothing to continue, so the first chunk is case
    (b) by definition — said explicitly rather than left as a blank the teacher
    has to interpret."""
    messages = build_teacher_messages(empty, "C_1 text")
    assert EMPTY_SUMMARY_PLACEHOLDER in messages[1]["content"]


def test_build_teacher_messages_keeps_a_real_summary_verbatim():
    messages = build_teacher_messages("The court denied the motion.", "next chunk")
    assert "The court denied the motion." in messages[1]["content"]
    assert EMPTY_SUMMARY_PLACEHOLDER not in messages[1]["content"]


def test_probe_accuracy_by_position_detects_collapse():
    """C-DIC Figure 2(a)'s shape: fine early, falling apart after a few
    accumulated rewrites. A single averaged score would hide this."""
    results = (
        [ProbeResult(chunk_position=i, n_probes=4, n_correct=4) for i in range(3)]
        + [ProbeResult(chunk_position=i, n_probes=4, n_correct=1) for i in range(3, 8)]
    )
    report = probe_accuracy_by_position(results, collapse_after=3)

    assert report["early_accuracy"] == pytest.approx(1.0)
    assert report["late_accuracy"] == pytest.approx(0.25)
    assert report["drop"] == pytest.approx(0.75)
    assert report["by_position"][0] == pytest.approx(1.0)
    assert report["by_position"][7] == pytest.approx(0.25)


def test_probe_accuracy_by_position_reports_no_drop_for_a_stable_loop():
    results = [ProbeResult(chunk_position=i, n_probes=4, n_correct=3) for i in range(8)]
    report = probe_accuracy_by_position(results)
    assert report["drop"] == pytest.approx(0.0)
    assert report["overall_accuracy"] == pytest.approx(0.75)


def test_probe_accuracy_by_position_averages_across_documents():
    """by_position[i] pools the i-th revision over every document curated."""
    results = [
        ProbeResult(chunk_position=0, n_probes=2, n_correct=2),  # doc A
        ProbeResult(chunk_position=0, n_probes=2, n_correct=0),  # doc B
    ]
    report = probe_accuracy_by_position(results)
    assert report["by_position"][0] == pytest.approx(0.5)
    assert report["positions"] == 1


def test_probe_accuracy_handles_empty_input_and_zero_probes():
    assert probe_accuracy_by_position([])["overall_accuracy"] == 0.0
    assert ProbeResult(chunk_position=0, n_probes=0, n_correct=0).accuracy == 0.0


# --------------------------------------------------------------------------
# Stage-2 pairs — the self-conditioned mix standing in for ra-TBPTT's purpose
# --------------------------------------------------------------------------

CHUNKS = [f"chunk {i} text" for i in range(4)]
GOLD = [f"gold summary {i}" for i in range(4)]
SELF = [f"self summary {i}" for i in range(4)]


def test_stage2_pairs_are_rolling_shaped():
    """Chunk i conditions on the summary after chunk i-1 and targets the
    summary after chunk i — the rolling objective stage 1's one-shot XSum
    training never provided."""
    pairs = build_stage2_pairs(CHUNKS, GOLD)
    assert [p.previous_summary for p in pairs] == ["", "gold summary 0", "gold summary 1", "gold summary 2"]
    assert [p.target_text for p in pairs] == GOLD


def test_stage2_input_uses_the_inference_template():
    """A template that drifts between prep and inference is a silent
    train/inference mismatch — t5 absorbs it as degraded output, not an error."""
    pairs = build_stage2_pairs(CHUNKS, GOLD)
    assert pairs[0].input_text == f"{PASSAGE_FIELD} chunk 0 text"
    assert pairs[1].input_text.startswith(f"{PASSAGE_FIELD} chunk 1 text")
    assert "gold summary 0" in pairs[1].input_text


def test_stage2_pairs_default_to_pure_teacher_forcing():
    """ratio 0.0 must reproduce the clean objective exactly, so the mix is a
    one-argument ablation."""
    pairs = build_stage2_pairs(CHUNKS, GOLD, self_summaries=SELF)
    assert all(p.conditioning == "gold" for p in pairs)


def test_self_conditioned_pairs_keep_the_teacher_target():
    """The point is recovery: given a state the model actually produces, emit
    the state it should have. Training toward its own output would instead
    teach it to keep drifting."""
    pairs = build_stage2_pairs(CHUNKS, GOLD, self_summaries=SELF, self_conditioned_ratio=1.0)
    self_pairs = [p for p in pairs if p.conditioning == "self"]
    assert self_pairs, "ratio 1.0 should produce self-conditioned pairs"
    for pair in self_pairs:
        assert pair.previous_summary.startswith("self summary")
        assert pair.target_text.startswith("gold summary")


def test_first_chunk_is_never_self_conditioned():
    """Chunk 0 has no prior state to be conditioned on either way."""
    pairs = build_stage2_pairs(CHUNKS, GOLD, self_summaries=SELF, self_conditioned_ratio=1.0)
    assert pairs[0].conditioning == "gold"
    assert pairs[0].previous_summary == ""


def test_stage2_pairs_are_reproducible_for_a_seed():
    kwargs = dict(self_summaries=SELF, self_conditioned_ratio=0.5)
    first = build_stage2_pairs(CHUNKS, GOLD, seed=7, **kwargs)
    second = build_stage2_pairs(CHUNKS, GOLD, seed=7, **kwargs)
    assert [p.conditioning for p in first] == [p.conditioning for p in second]


def test_stage2_pairs_reject_a_ratio_without_self_summaries():
    with pytest.raises(ValueError):
        build_stage2_pairs(CHUNKS, GOLD, self_conditioned_ratio=0.3)


def test_stage2_pairs_reject_misaligned_inputs():
    with pytest.raises(ValueError):
        build_stage2_pairs(CHUNKS, GOLD[:2])


# --------------------------------------------------------------------------
# Retention probe — the collapse diagnostic's measurement side
# --------------------------------------------------------------------------


def test_token_f1_tolerates_rephrasing_but_not_stopword_overlap():
    """The teacher is instructed to *revise* sentences, so it legitimately
    rephrases spans — exact containment would be too strict, bare token
    presence too loose."""
    assert token_f1("Chester Arthur Stiles", "Police charged Chester Arthur Stiles.") == 1.0
    assert token_f1("Chester Arthur Stiles", "The police made an arrest.") == 0.0
    assert answer_supported("commonplace", "USB became commonplace on devices.", threshold=0.6)
    assert not answer_supported("a variety of earlier interfaces", "USB is a thing.", threshold=0.6)


def test_retention_probes_are_keyed_by_age_not_position():
    """C-DIC Fig. 2(a) is perplexity against *accumulated compressions*. The
    returned chunk_position is an age, so age 0 is 'just read'."""
    summaries = ["alpha here", "alpha here", "gone", "gone"]
    probes = retention_probes(summaries, {0: ["alpha"]})
    assert [p.chunk_position for p in probes] == [0, 1, 2, 3]
    assert [p.accuracy for p in probes] == [1.0, 1.0, 0.0, 0.0]


def test_retention_probes_hold_the_probe_count_constant_across_ages():
    """The confound this design exists to remove: probing every live answer at
    each position makes the count grow with position, so accuracy falls even
    for a compressor that never degrades. Keyed by age, the count at each age
    is a property of the corpus, not of the axis."""
    summaries = ["s0", "s1", "s2"]
    probes = retention_probes(summaries, {0: ["a"], 1: ["b"], 2: ["c"]})
    by_age = {p.chunk_position: p.n_probes for p in probes}
    assert by_age == {0: 3, 1: 2, 2: 1}, "each age pools every chunk old enough to reach it"


def test_retention_probes_feed_the_collapse_report():
    """The two halves have to compose: retention_probes measures,
    probe_accuracy_by_position reads the curve."""
    summaries = ["fact kept", "fact kept", "fact kept", "lost", "lost", "lost"]
    probes = retention_probes(summaries, {0: ["fact"]})
    report = probe_accuracy_by_position(probes, collapse_after=3)
    assert report["early_accuracy"] == pytest.approx(1.0)
    assert report["late_accuracy"] == pytest.approx(0.0)
    assert report["drop"] == pytest.approx(1.0), "a clearly positive drop is the collapse signature"


def test_retention_probes_ignore_chunks_with_no_questions():
    probes = retention_probes(["a", "b"], {0: [], 1: ["b"]})
    assert [(p.chunk_position, p.n_probes) for p in probes] == [(0, 1)]


def test_retention_probes_respect_max_age():
    probes = retention_probes(["x"] * 10, {0: ["x"]}, max_age=2)
    assert [p.chunk_position for p in probes] == [0, 1, 2]


# --------------------------------------------------------------------------
# Threaded pairs — C-DIC's actual retrieve/generate shape
# --------------------------------------------------------------------------

CTX_CHUNKS = [f"chunk {i} text" for i in range(4)]
CTX_CONTEXT = [[], ["gist 0"], [], ["gist 0", "gist 2"]]
CTX_GISTS = [f"gist {i}" for i in range(4)]
CTX_SELF_CONTEXT = [["self ctx a"], [], ["self ctx b"], []]


def test_stage2_pairs_threaded_reproduce_the_curated_shape_exactly():
    """input/target must be exactly format_elaborative_input(chunk, context)
    -> gist — no re-derivation, since curate_document_threaded already
    produced the aligned pair."""
    pairs = build_stage2_pairs_threaded(CTX_CHUNKS, CTX_CONTEXT, CTX_GISTS)
    for i, pair in enumerate(pairs):
        assert pair.input_text == format_elaborative_input(CTX_CHUNKS[i], CTX_CONTEXT[i])
        assert pair.target_text == CTX_GISTS[i]


def test_stage2_pairs_threaded_default_to_pure_teacher_forcing():
    pairs = build_stage2_pairs_threaded(
        CTX_CHUNKS, CTX_CONTEXT, CTX_GISTS, self_context_texts=CTX_SELF_CONTEXT
    )
    assert all(p.conditioning == "gold" for p in pairs)


def test_stage2_pairs_threaded_position_zero_is_eligible_for_self_conditioning():
    """Unlike the linear build_stage2_pairs, position 0 has no structural
    reason to be excluded — retrieval can return something (or nothing) at
    any position, including the first."""
    pairs = build_stage2_pairs_threaded(
        CTX_CHUNKS, CTX_CONTEXT, CTX_GISTS,
        self_context_texts=CTX_SELF_CONTEXT, self_conditioned_ratio=1.0,
    )
    assert pairs[0].conditioning == "self"
    assert pairs[0].input_text == format_elaborative_input(CTX_CHUNKS[0], CTX_SELF_CONTEXT[0])


def test_stage2_pairs_threaded_self_conditioning_keeps_the_teacher_target():
    pairs = build_stage2_pairs_threaded(
        CTX_CHUNKS, CTX_CONTEXT, CTX_GISTS,
        self_context_texts=CTX_SELF_CONTEXT, self_conditioned_ratio=1.0,
    )
    for i, pair in enumerate(pairs):
        assert pair.target_text == CTX_GISTS[i], "self-conditioning must never retarget onto its own drift"


def test_stage2_pairs_threaded_reject_misaligned_inputs():
    with pytest.raises(ValueError):
        build_stage2_pairs_threaded(CTX_CHUNKS, CTX_CONTEXT[:2], CTX_GISTS)
    with pytest.raises(ValueError):
        build_stage2_pairs_threaded(CTX_CHUNKS, CTX_CONTEXT, CTX_GISTS[:2])


def test_stage2_pairs_threaded_reject_a_ratio_without_self_context():
    with pytest.raises(ValueError):
        build_stage2_pairs_threaded(CTX_CHUNKS, CTX_CONTEXT, CTX_GISTS, self_conditioned_ratio=0.3)


def test_stage2_pairs_threaded_reproducible_for_a_seed():
    kwargs = dict(self_context_texts=CTX_SELF_CONTEXT, self_conditioned_ratio=0.5)
    first = build_stage2_pairs_threaded(CTX_CHUNKS, CTX_CONTEXT, CTX_GISTS, seed=3, **kwargs)
    second = build_stage2_pairs_threaded(CTX_CHUNKS, CTX_CONTEXT, CTX_GISTS, seed=3, **kwargs)
    assert [p.conditioning for p in first] == [p.conditioning for p in second]
