"""Question generation (testing effect, condition C) — input template.

One function, defined here rather than inline in the notebooks, because the
QG input format has to be **identical** at training time (`08`) and at
generation time (`09`, and later the stage-2 curation loop in `curation.py`
that uses QG-generated probes). A format that drifts between the two does not
raise anything — t5 simply produces worse output, which reads as a model
quality problem and gets debugged in the wrong place.

That drift already happened once: `08` was updated to prepend
`generate question:` after a pilot produced declarative sentences instead of
questions, but `09`'s qualitative cell kept building its input without the
prefix, and the saved dataset on disk predated the change too. Both were
silently feeding the model a form it was not trained on.
"""

from __future__ import annotations

# T5 was pretrained with task prefixes ("summarize:", "translate English to
# German:"). Single-task fine-tuning does not strictly require one, but the
# pilot without it echoed the context as a declarative sentence rather than
# asking anything — undertraining is the likelier cause, and the prefix is
# free insurance that costs three tokens.
QG_PREFIX = "generate question:"


def format_qg_input(answer: str, context: str) -> str:
    """The reverse of normal QA: given a context and a target answer span,
    the model must produce a question whose answer is that span.

    This is what the testing-effect stage needs — pick a fact inside a chunk,
    get a quiz question aimed at it.
    """
    return f"{QG_PREFIX} answer: {answer} context: {context}"
