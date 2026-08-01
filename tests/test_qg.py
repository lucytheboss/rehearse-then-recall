"""Unit tests for the QG input template (testing effect, condition C)."""

from src.pipeline.qg import QG_PREFIX, format_qg_input


def test_format_qg_input_starts_with_the_task_prefix():
    """The whole reason this lives in one function: `08` trains with the
    prefix, so `09` and the curation loop must generate with it too."""
    assert format_qg_input("123rd", "some context").startswith(QG_PREFIX)


def test_format_qg_input_places_answer_before_context():
    """Order is part of the format the model learns, not a detail."""
    assembled = format_qg_input("Von Miller", "The Broncos' defense ranked first.")
    assert assembled.index("answer:") < assembled.index("context:")
    assert "Von Miller" in assembled
    assert "The Broncos' defense ranked first." in assembled


def test_format_qg_input_is_exactly_reproducible():
    """Guards against whitespace drift between the prep and inference sites —
    the failure mode is silent, so it has to be pinned by a literal."""
    assert format_qg_input("A", "B") == "generate question: answer: A context: B"
