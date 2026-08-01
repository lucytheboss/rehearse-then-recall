"""Unit tests for stage-2 curation (elaborative rehearsal B) — prompt assembly
and the collapse diagnostic. No API key or network required.
"""

import pytest

from src.pipeline.curation import (
    EMPTY_SUMMARY_PLACEHOLDER,
    TEACHER_SYSTEM_PROMPT,
    ProbeResult,
    build_teacher_messages,
    probe_accuracy_by_position,
)


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
