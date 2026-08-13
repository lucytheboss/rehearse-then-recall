"""Stage-2 offline curation for elaborative rehearsal (B) — teacher prompting
and the collapse diagnostic.

Everything here runs **offline, to build training data**. It is not part of the
inference pipeline, so the "no generative prompts before the final QA step"
principle in `03_실험 설계` §핵심 원칙 is untouched: at evaluation time B is
still a single seq2seq forward pass.

Why stage 2 cannot be skipped
-----------------------------
C-DIC (Jung, Kim, Jung, Wang, Zhang, Cheung, See & Chen, ICML 2026,
arXiv:2606.12411) Table 1 measures exactly the shortcut we would be taking.
ICAE trained one-shot and then applied incrementally scores PPL 513.774 on MSC,
against 27.656 for the same model used one-shot as intended and 8.431 for
C-DIC — a ~19x blow-up from reusing a one-shot compressor in a rolling loop.
Appendix F attributes it to the structural cause, not to tuning: a compressor
whose objective never trained it to consume and update its *own* output drifts
and compounds error when made to do so.

Stage 1 (`06`/`07`) trains B on XSum `(document -> one-sentence summary)`,
which is a one-shot compression objective. Running that checkpoint in the
rolling `(S_{i-1}, C_i) -> S_i` loop at evaluation time is the same structural
mismatch, so the failure is a predicted outcome rather than a risk. The
curation loop here exists to produce the rolling-shaped targets stage 2 needs.
"""

from __future__ import annotations

import random
import string
from dataclasses import dataclass

from src.pipeline.rehearsal import format_elaborative_input

# The write-back rule (C-DIC Eq. 6, Algorithm 1) is not "always revise" — it
# branches on similarity. When the new content matches an existing thread it
# replaces that slot (revise); when it matches nothing above the threshold it
# opens a new slot (insert), so unrelated topics are never blended. The prompt
# below is the text-level transposition of that branch: our shortened version
# has no addressable slots, so the branch has to be stated as an instruction
# and the sentence plays the role of the slot.
#
# Without this branch, "고쳐써라" alone pushes the teacher to force every new
# chunk into the existing paragraph — which is the semantic drift / losing the
# thread failure C-DIC §1 describes, reintroduced by the very instruction meant
# to prevent staleness.
#
# "Shorten", not "summarize" (2026-08-02). ReadAgent (Lee, Chen, Furuta, Canny
# & Fischer, ICML 2024, arXiv:2402.09727) §3.1 names the exact effect this
# prompt is built to exploit: "We use the word 'shorten' ... as it tends to
# help preserve the narrative flow, making it more natural to concatenate.
# Using the word 'summarize' tended to produce a restructured summary." That
# concatenation failure is precisely ours — every chunk's shortened text is
# stitched into one running document, so a verb that invites restructuring
# invites drift at every concatenation boundary, stacking on top of whatever
# the insert/revise branch below is already guarding against. ReadAgent's own
# gisting has no revise step (each page is shortened once, independently); the
# (a)/(b) branch is C-DIC's contribution, layered on top of ReadAgent's verb
# choice rather than replacing it.
TEACHER_SYSTEM_PROMPT = """You are maintaining a running shortened version of a document as it is read chunk by chunk. Shorten — do not summarize. Give back less of the passage's own wording, in the same order, rather than a restructured synopsis, so that pieces from different chunks read as one continuous passage when placed next to each other.

You will be given the current shortened version (may be empty on the first
chunk) and the next chunk of the document.

Do NOT simply append the new chunk's shortened content to the end. First
decide, for the new chunk, whether it:

(a) CONTINUES a topic/entity/event already present in the shortened version —
    then REVISE the existing sentence(s) that are now incomplete, outdated, or
    contradicted. Integrate the new details into them. Do not add a redundant
    new sentence for the same thing.

(b) INTRODUCES a topic/entity/event not covered by the shortened version —
    then ADD one new, separate sentence for it. Do not force it into an
    unrelated existing sentence.

Never blend two unrelated topics into a single sentence just to keep it short.
Never repeat information already correctly stated.

Give me only the updated shortened version. DO NOT explain your reason, no labels."""

TEACHER_USER_TEMPLATE = """Shortened version so far:
{previous_summary}

Next chunk:
{chunk_text}

Updated shortened version:"""

# What an empty running summary is rendered as. The first chunk has nothing to
# continue, so it is by definition case (b) — the placeholder says that in the
# prompt's own vocabulary instead of leaving a blank the teacher has to guess
# at.
EMPTY_SUMMARY_PLACEHOLDER = "(empty — this is the first chunk)"


def build_teacher_messages(previous_summary: str, chunk_text: str) -> list[dict]:
    """Chat messages for one curation step, in NIM/OpenAI format.

    `previous_summary` empty (the first chunk) renders as
    `EMPTY_SUMMARY_PLACEHOLDER`, which routes the teacher down the (b) branch:
    there is no prior topic to continue, so the first sentence is a new slot.
    """
    rendered_summary = previous_summary.strip() or EMPTY_SUMMARY_PLACEHOLDER
    return [
        {"role": "system", "content": TEACHER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": TEACHER_USER_TEMPLATE.format(
                previous_summary=rendered_summary, chunk_text=chunk_text
            ),
        },
    ]


# --------------------------------------------------------------------------
# Threaded curation — the teacher standing in for B inside C-DIC's actual
# retrieve / generate / write-back loop, not the linear summary above.
# --------------------------------------------------------------------------
#
# `build_teacher_messages` above trains a different task than the one B is
# asked to perform at inference. `rehearse_elaborative` (`rehearsal.py`) does
# not maintain one growing text — it keeps a `ThreadMemory` of separate
# topic threads, retrieves the ones related to the current chunk (Eq. 3),
# generates a *new gist for that chunk's thread* conditioned on them, and the
# result *replaces* the matched slot (Eq. 6) rather than being appended to a
# document-wide summary. The linear loop above only ever has one state to
# condition on; it never has to choose among several unrelated retrieved
# threads and decide which one this chunk continues. Training B on the linear
# shape and then asking it, at inference, to pick one of up to `top_k`
# retrieved slots and replace just that one is a second, shape-level version
# of the exact train/inference mismatch stage 2 exists to close (C-DIC
# Table 1) — this time in what the rolling task looks like, not merely in
# whether it is rolling.
#
# `curate_document_threaded` (`teacher.py`) drives the same `ThreadMemory`
# class `rehearse_elaborative` uses, with the teacher chat model standing in
# for the local seq2seq model at the generation step, so a pair built from its
# output is `format_elaborative_input(chunk, retrieved_context) -> gist` —
# the literal shape of one inference step, with the teacher's answer as the
# label instead of the small model's own.

TEACHER_SYSTEM_PROMPT_THREADED = """You are shortening a long document as it is read chunk by chunk, keeping separate topics as separate threads rather than folding everything into one continuous summary.

You will be given the current chunk and, if any exist, the thread(s) already shortened from earlier in the document that a retrieval step — not you — judged most related to it.

If related thread(s) are given: they cover the same topic, entity, or event as this chunk. Produce ONE shortened passage that folds the current chunk's new information into them — revise whatever they said that is now incomplete, outdated, or contradicted, and keep whatever they said that this chunk does not touch. Your output REPLACES the thread(s) you were shown; nothing survives from them except what you carry into your output.

If no related thread is given: this chunk opens a new thread. Shorten it on its own — do not invent a connection to anything unrelated just because none was given.

Shorten — do not summarize. Give back less of the passage's own wording, in the same order, rather than a restructured synopsis, so that pieces of the same thread read as one continuous passage when placed next to each other.

Give me only the shortened passage for this thread. DO NOT explain your reason, no labels."""

TEACHER_USER_TEMPLATE_THREADED = """Related thread(s) (chosen by retrieval, not by you):
{context}

Current chunk:
{chunk_text}

Shortened passage:"""

# What "nothing was retrieved" is rendered as. Generalises
# `EMPTY_SUMMARY_PLACEHOLDER`'s role above — that placeholder only ever fires
# on the first chunk (nothing exists yet to continue); this one fires
# whenever retrieval comes back empty, which can happen at any position once
# a chunk's topic has no earlier match above tau.
EMPTY_CONTEXT_PLACEHOLDER = "(none — this chunk starts a new thread)"


TEACHER_SYSTEM_PROMPT_THREADED_CORRECTIVE = """You are shortening a long document as it is read chunk by chunk, keeping separate topics as separate threads rather than folding everything into one continuous summary.

You already produced a shortened passage for this chunk, but it lost information: it no longer supports answering a question whose evidence is in this chunk. Revise your shortened passage so it explicitly preserves the fact(s) listed below, using the same "shorten, don't summarize" style as before — give back less of the passage's own wording, in the same order, not a restructured synopsis.

Do not add anything not supported by the current chunk or the related thread(s) below. Do not simply append the missing fact as an extra clause; integrate it naturally.

Give me only the revised shortened passage. DO NOT explain your reason, no labels."""

TEACHER_USER_TEMPLATE_THREADED_CORRECTIVE = """Related thread(s) (chosen by retrieval, not by you):
{context}

Current chunk:
{chunk_text}

Your previous shortened passage (lost information):
{previous_attempt}

Fact(s) it must still support (each is an answer span from a real question about this chunk):
{missing_facts}

Revised shortened passage:"""


def build_teacher_messages_threaded_corrective(
    context_texts: list[str], chunk_text: str, previous_attempt: str, missing_facts: list[str],
) -> list[dict]:
    """Chat messages for a corrective retry — same shape as
    `build_teacher_messages_threaded`, plus the failed attempt and the answer
    span(s) that must survive the revision. See
    `curate_document_threaded_with_testing`'s docstring for why this exists
    and what "missing_facts" actually are (answer strings, not questions)."""
    rendered_context = "\n\n".join(context_texts) if context_texts else EMPTY_CONTEXT_PLACEHOLDER
    return [
        {"role": "system", "content": TEACHER_SYSTEM_PROMPT_THREADED_CORRECTIVE},
        {
            "role": "user",
            "content": TEACHER_USER_TEMPLATE_THREADED_CORRECTIVE.format(
                context=rendered_context,
                chunk_text=chunk_text,
                previous_attempt=previous_attempt,
                missing_facts="\n".join(f"- {fact}" for fact in missing_facts),
            ),
        },
    ]


def build_teacher_messages_threaded(context_texts: list[str], chunk_text: str) -> list[dict]:
    """Chat messages for one *threaded* curation step.

    `context_texts` are not "the previous chunk's state" — they are whatever
    `ThreadMemory.retrieve` found related to `chunk_text` by similarity,
    which may come from many chunks back, may be several slots, or may be
    empty. Empty renders as `EMPTY_CONTEXT_PLACEHOLDER`, which routes the
    teacher toward opening a new thread rather than leaving it to guess at a
    blank.
    """
    rendered_context = "\n\n".join(context_texts) if context_texts else EMPTY_CONTEXT_PLACEHOLDER
    return [
        {"role": "system", "content": TEACHER_SYSTEM_PROMPT_THREADED},
        {
            "role": "user",
            "content": TEACHER_USER_TEMPLATE_THREADED.format(
                context=rendered_context, chunk_text=chunk_text
            ),
        },
    ]


# --------------------------------------------------------------------------
# Anti-collapse retry (2026-08-13) — TEACHER_SYSTEM_PROMPT_THREADED's revise
# branch was found to systematically collapse: given a related thread, both
# the teacher (70B) and the trained student reproduce the thread near-
# verbatim and drop the current chunk's own content, regardless of which
# text sits in which field (confirmed with the fields swapped). Since the
# student's stage-2 targets are literally this function's output
# (`build_stage2_pairs_threaded`), the student learned the collapse
# faithfully from the teacher's own collapsed labels — retraining the
# student cannot fix this; the teacher-side generation has to stop
# collapsing first. A stronger system prompt alone (below) fixes some
# positions but not all in a 6-chunk trial (2/5 non-failed positions still
# collapsed) — this pairs it with the same probe-and-retry shape
# `curate_document_threaded_with_testing` already uses, but probing for
# collapse (token_f1 against the given context) instead of missing answer
# spans.
# --------------------------------------------------------------------------

TEACHER_SYSTEM_PROMPT_THREADED_ANTICOLLAPSE = """You are shortening a long document as it is read chunk by chunk, keeping separate topics as separate threads rather than folding everything into one continuous summary.

You will be given the current chunk and, if any exist, the thread(s) already shortened from earlier in the document that a retrieval step — not you — judged most related to it.

If related thread(s) are given: they cover the same topic, entity, or event as this chunk. Produce ONE shortened passage that folds the current chunk's new information into them — revise whatever they said that is now incomplete, outdated, or contradicted, and keep whatever they said that this chunk does not touch. Your output REPLACES the thread(s) you were shown; nothing survives from them except what you carry into your output.

CRITICAL: your output must contain specific wording drawn from the CURRENT CHUNK, not only from the related thread(s). A shortened passage that is the related thread(s) unchanged -- with nothing added from the current chunk -- is WRONG, even if the current chunk's topic overlaps the thread. The current chunk always has at least one fact not already in the thread(s); find it and include it. Never output the related thread(s) verbatim or near-verbatim.

If no related thread is given: this chunk opens a new thread. Shorten it on its own — do not invent a connection to anything unrelated just because none was given.

Shorten — do not summarize. Give back less of the passage's own wording, in the same order, rather than a restructured synopsis, so that pieces of the same thread read as one continuous passage when placed next to each other.

Give me only the shortened passage for this thread. DO NOT explain your reason, no labels."""

TEACHER_USER_TEMPLATE_THREADED_ANTICOLLAPSE_CORRECTIVE = """Related thread(s) (chosen by retrieval, not by you):
{context}

Current chunk:
{chunk_text}

Your previous shortened passage (this is WRONG — it is nearly identical to the related thread(s) above and contains nothing from the current chunk):
{previous_attempt}

Try again. Your revised passage MUST include specific wording that appears in the current chunk but not in the related thread(s) above.

Revised shortened passage:"""


def build_teacher_messages_threaded_anticollapse(context_texts: list[str], chunk_text: str) -> list[dict]:
    """Same shape as `build_teacher_messages_threaded`, with
    `TEACHER_SYSTEM_PROMPT_THREADED_ANTICOLLAPSE`'s explicit constraint
    against reproducing the given thread(s) unchanged."""
    rendered_context = "\n\n".join(context_texts) if context_texts else EMPTY_CONTEXT_PLACEHOLDER
    return [
        {"role": "system", "content": TEACHER_SYSTEM_PROMPT_THREADED_ANTICOLLAPSE},
        {
            "role": "user",
            "content": TEACHER_USER_TEMPLATE_THREADED.format(context=rendered_context, chunk_text=chunk_text),
        },
    ]


def build_teacher_messages_threaded_anticollapse_corrective(
    context_texts: list[str], chunk_text: str, previous_attempt: str,
) -> list[dict]:
    """Corrective retry for a detected collapse (see module note above) —
    same shape as `build_teacher_messages_threaded_corrective`, but the
    "what's wrong" signal is collapse (output ~= context), not a missing
    answer span."""
    rendered_context = "\n\n".join(context_texts) if context_texts else EMPTY_CONTEXT_PLACEHOLDER
    return [
        {"role": "system", "content": TEACHER_SYSTEM_PROMPT_THREADED_ANTICOLLAPSE},
        {
            "role": "user",
            "content": TEACHER_USER_TEMPLATE_THREADED_ANTICOLLAPSE_CORRECTIVE.format(
                context=rendered_context, chunk_text=chunk_text, previous_attempt=previous_attempt,
            ),
        },
    ]


def collapse_score(context_texts: list[str], gist: str) -> float:
    """How much of `gist` is just the given context, token-F1 style (reuses
    `token_f1`'s normalize+overlap, with context as the "answer" being
    checked for survival in `gist`). High score = gist mostly reproduces
    context; the collapse this module's retry loop exists to catch. Empty
    context always scores 0.0 -- there is nothing to have collapsed into."""
    if not context_texts:
        return 0.0
    return token_f1(" ".join(context_texts), gist)


# --------------------------------------------------------------------------
# Self-conditioned pairs — the text-level stand-in for C-DIC's ra-TBPTT.
# --------------------------------------------------------------------------


@dataclass
class Stage2Pair:
    """One (input, target) example for stage-2 training.

    `conditioning` records which running summary the input was built from:

      - "gold" — the teacher's own previous summary. Clean, and what plain
        teacher forcing produces.
      - "self" — the *stage-1 model's* previous summary, paired with the
        teacher's target anyway. The example therefore teaches recovery: given
        a state the model actually produces, emit the state it should have.
    """

    input_text: str
    target_text: str
    previous_summary: str
    chunk_text: str
    chunk_position: int
    conditioning: str


def build_stage2_pairs(
    chunk_texts: list[str],
    gold_summaries: list[str],
    self_summaries: list[str] | None = None,
    self_conditioned_ratio: float = 0.0,
    seed: int = 0,
) -> list[Stage2Pair]:
    """Rolling-shaped training pairs, optionally conditioned on the model's own
    prior output.

    Why this exists
    ---------------
    C-DIC's third component is retrieval-aware TBPTT: the per-turn loss
    `l_t = -log P(r_t | q_t, R_t)` backpropagates one hop into the retrieved
    slot `Z_{j_t}`, with every other slot stop-gradiented. Its ablation is not
    cosmetic — removing it costs PPL 12.295 against 9.356 on REALTALK, the
    second-largest drop in Table 4 after incremental compression itself.

    It cannot be ported literally. C-DIC's memory holds latent slots, so a
    gradient can flow into one; ours holds *text*, and text is discrete, so
    there is no path from `l_t` back into a slot at all. Any claim to have
    implemented ra-TBPTT here would be false.

    What is portable is the problem it solves. ra-TBPTT exists because a
    compressor trained only on clean states never learns what to do with the
    states it will actually be handed — the same exposure bias that makes
    Table 1's ICAE-incremental blow up to PPL 513.774. Stage 2's curated data
    closes half of that gap by making the *task* rolling; it leaves the other
    half open, because every training input still carries the teacher's
    immaculate previous summary while inference carries the model's own.

    So: run the stage-1 checkpoint over the same corpus in the rolling loop,
    keep its running summaries as `self_summaries`, and mix a fraction of them
    in as conditioning while keeping the teacher's target. That is DAgger, and
    it is a training-data property rather than a gradient trick, which is
    exactly what a text-level memory can support. Report it as an analogue of
    ra-TBPTT's *purpose*, never as an implementation of its mechanism.

    Arguments
    ---------
    `gold_summaries[i]` is the teacher's running summary **after** chunk i, so
    chunk i's input conditions on `gold_summaries[i-1]` and targets
    `gold_summaries[i]`. `self_summaries[i]` is the stage-1 model's running
    summary after chunk i, on the same document, and is used only as
    conditioning — never as a target, or the model would be trained toward its
    own drift.

    `self_conditioned_ratio` is the fraction of eligible pairs (position > 0)
    drawn from `self_summaries`. 0.0 reproduces pure teacher forcing, so the
    ablation is one argument. Sample it low first: every self-conditioned pair
    is a pair not spent on the clean objective, and the point is to make the
    model robust to its own states, not to train it on them predominantly.
    """
    if len(chunk_texts) != len(gold_summaries):
        raise ValueError(
            f"chunk_texts and gold_summaries must align: {len(chunk_texts)} vs {len(gold_summaries)}"
        )
    if self_summaries is not None and len(self_summaries) != len(chunk_texts):
        raise ValueError(
            f"self_summaries must align with chunk_texts: {len(self_summaries)} vs {len(chunk_texts)}"
        )
    if not 0.0 <= self_conditioned_ratio <= 1.0:
        raise ValueError(f"self_conditioned_ratio must be in [0, 1]: {self_conditioned_ratio}")
    if self_conditioned_ratio and self_summaries is None:
        raise ValueError("self_conditioned_ratio > 0 requires self_summaries")

    rng = random.Random(seed)
    pairs: list[Stage2Pair] = []

    for position, (chunk_text, target) in enumerate(zip(chunk_texts, gold_summaries)):
        # Chunk 0 has no prior state to be conditioned on either way, so it is
        # never eligible for self-conditioning.
        use_self = (
            position > 0
            and self_summaries is not None
            and rng.random() < self_conditioned_ratio
        )
        source = self_summaries if use_self else gold_summaries
        previous = source[position - 1] if position > 0 else ""

        pairs.append(
            Stage2Pair(
                # Built with the inference-time template so stage 2 trains on
                # the exact shape `rehearse_elaborative` feeds. A template that
                # drifts between the two is a silent train/inference mismatch,
                # and t5 absorbs it as degraded output rather than an error.
                input_text=format_elaborative_input(
                    chunk_text, [previous] if previous else []
                ),
                target_text=target,
                previous_summary=previous,
                chunk_text=chunk_text,
                chunk_position=position,
                conditioning="self" if use_self else "gold",
            )
        )

    return pairs


def build_stage2_pairs_threaded(
    chunk_texts: list[str],
    context_texts: list[list[str]],
    gists: list[str],
    self_context_texts: list[list[str]] | None = None,
    self_conditioned_ratio: float = 0.0,
    seed: int = 0,
) -> list[Stage2Pair]:
    """Threaded counterpart to `build_stage2_pairs` — pairs built from C-DIC's
    actual retrieve/generate shape (`teacher.curate_document_threaded`)
    rather than the linear running-summary shape above.

    `context_texts[i]` / `gists[i]` are `ThreadedCurationResult.context_texts`
    / `.gists`: what the teacher retrieved before generating chunk i's gist,
    and the gist itself. The pair is then exactly
    `format_elaborative_input(chunk_texts[i], context_texts[i]) -> gists[i]`
    — the literal shape `rehearse_elaborative` feeds and expects back at
    inference. The linear version had to reconstruct this alignment itself
    (`gold_summaries[i-1]` as context); here it is already the alignment the
    curation loop produced, so there is nothing to look up.

    `self_context_texts[i]`, if given, is what the *stage-1 model's own*
    `rehearse_elaborative` rollout retrieved at position i — get it by running
    that checkpoint over the same `chunk_texts` with
    `record_context_texts=True` and reading each
    `RetrievalRecord.retrieved_context_texts`. Mixing a fraction of these in
    as the *conditioning*, while keeping the teacher's `gists[i]` as the
    *target*, is this shape's version of the DAgger mix `build_stage2_pairs`
    performs: same purpose (train recovery from the states the model will
    actually reach — the exposure bias ra-TBPTT targets), same disclaimer
    (not an implementation of ra-TBPTT's gradient mechanism — text slots are
    discrete, so no gradient reaches them).

    Position 0 is eligible for self-conditioning here, unlike in
    `build_stage2_pairs`. There the first chunk had no prior state under
    either source, so mixing was a no-op there and excluded by construction.
    Here retrieval can return something at *any* position (a document's first
    chunk still queries an empty memory, same as any other query into an
    as-yet-unpopulated pool), so `context[0]` and `self_context[0]` are both
    just "whatever retrieval returned" — no structural reason to special-case
    it.
    """
    if not (len(chunk_texts) == len(context_texts) == len(gists)):
        raise ValueError(
            "chunk_texts, context_texts, and gists must align: "
            f"{len(chunk_texts)}, {len(context_texts)}, {len(gists)}"
        )
    if self_context_texts is not None and len(self_context_texts) != len(chunk_texts):
        raise ValueError(
            f"self_context_texts must align with chunk_texts: {len(self_context_texts)} vs {len(chunk_texts)}"
        )
    if not 0.0 <= self_conditioned_ratio <= 1.0:
        raise ValueError(f"self_conditioned_ratio must be in [0, 1]: {self_conditioned_ratio}")
    if self_conditioned_ratio and self_context_texts is None:
        raise ValueError("self_conditioned_ratio > 0 requires self_context_texts")

    rng = random.Random(seed)
    pairs: list[Stage2Pair] = []

    for position, (chunk_text, context, target) in enumerate(zip(chunk_texts, context_texts, gists)):
        use_self = self_context_texts is not None and rng.random() < self_conditioned_ratio
        used_context = self_context_texts[position] if use_self else context

        pairs.append(
            Stage2Pair(
                input_text=format_elaborative_input(chunk_text, used_context),
                target_text=target,
                previous_summary="\n\n".join(used_context),
                chunk_text=chunk_text,
                chunk_position=position,
                conditioning="self" if use_self else "gold",
            )
        )

    return pairs


@dataclass
class ProbeResult:
    """One self-test outcome from the curation loop's §2-3 step.

    `chunk_position` is the axis the collapse check is plotted against. Two
    readings are in use and they are not interchangeable:

      - **position** — 0-based order within the document, i.e. how many
        revisions the running summary has been through. This is the raw
        reading, and `probe_accuracy_by_position` assumes it.
      - **age** — how many revisions have happened *since the probed fact was
        read*, which is what `retention_probes` produces. Use this one for the
        retention curve; see that function for why position alone is
        confounded.
    """

    chunk_position: int
    n_probes: int
    n_correct: int

    @property
    def accuracy(self) -> float:
        return self.n_correct / self.n_probes if self.n_probes else 0.0


def probe_accuracy_by_position(
    results: list[ProbeResult],
    collapse_after: int = 3,
) -> dict:
    """Collapse diagnostic for the rolling loop — B's counterpart to A's
    expected-failure list.

    C-DIC Figure 2(a) plots perplexity against the number of accumulated
    compressions and finds static compressors rising sharply **after 3-4
    consecutive compressions** while C-DIC stays flat. The same shape should be
    checked here, with probe accuracy standing in for perplexity: if accuracy
    on chunk 8's probes is far below chunk 1's, the running summary is
    degrading with each rewrite and a single averaged score would hide it
    completely.

    `collapse_after=3` splits early from late at C-DIC's observed knee. `drop`
    is early minus late accuracy, so a clearly positive value is the collapse
    signature. Returns `by_position` as well, which is the series to plot —
    the average alone cannot distinguish a steady decline from a cliff.

    Positions are averaged across documents, so `by_position[i]` is "accuracy
    at the i-th revision" pooled over every document curated.
    """
    by_position: dict[int, list[float]] = {}
    for result in results:
        by_position.setdefault(result.chunk_position, []).append(result.accuracy)

    mean_by_position = {
        position: sum(values) / len(values)
        for position, values in sorted(by_position.items())
    }

    early = [r.accuracy for r in results if r.chunk_position < collapse_after]
    late = [r.accuracy for r in results if r.chunk_position >= collapse_after]
    early_mean = sum(early) / len(early) if early else 0.0
    late_mean = sum(late) / len(late) if late else 0.0

    return {
        "by_position": mean_by_position,
        "positions": len(mean_by_position),
        "collapse_after": collapse_after,
        "early_accuracy": early_mean,
        "late_accuracy": late_mean,
        "drop": early_mean - late_mean,
        "overall_accuracy": sum(r.accuracy for r in results) / len(results) if results else 0.0,
    }


# --------------------------------------------------------------------------
# Retention probe — the collapse diagnostic's measurement side.
# --------------------------------------------------------------------------

_PUNCT = str.maketrans("", "", string.punctuation)
_ARTICLES = {"a", "an", "the"}


def normalize_answer(text: str) -> str:
    """SQuAD-style normalisation: lowercase, drop punctuation and articles,
    collapse whitespace."""
    lowered = text.lower().translate(_PUNCT)
    return " ".join(word for word in lowered.split() if word not in _ARTICLES)


def token_f1(answer: str, summary: str) -> float:
    """Token overlap between an answer span and the running summary, as F1.

    Precision is measured against the *answer's* tokens rather than the
    summary's, so a long summary is not penalised for containing anything
    besides this one answer — the question is whether the answer survived, not
    whether the summary is about it. That makes this recall-dominated by
    design, which is the right bias for a retention check.
    """
    answer_tokens = normalize_answer(answer).split()
    if not answer_tokens:
        return 0.0
    summary_tokens = set(normalize_answer(summary).split())
    if not summary_tokens:
        return 0.0

    matched = sum(token in summary_tokens for token in answer_tokens)
    if matched == 0:
        return 0.0
    recall = matched / len(answer_tokens)
    precision = matched / len(set(answer_tokens))
    return 2 * precision * recall / (precision + recall)


def answer_supported(answer: str, summary: str, threshold: float = 0.6) -> bool:
    """Whether the running summary still carries enough of an answer span.

    Exact containment would be too strict — the teacher is instructed to
    *revise* sentences, so it legitimately rephrases spans — and bare token
    presence too loose, since a long summary shares stopwords with everything.
    """
    return token_f1(answer, summary) >= threshold


def retention_probes(
    running_summaries: list[str],
    answers_by_chunk: dict[int, list[str]],
    threshold: float = 0.6,
    max_age: int | None = None,
) -> list[ProbeResult]:
    """Does a fact read at chunk j survive k more revisions?

    This is C-DIC Figure 2(a) — perplexity against the number of *accumulated*
    compressions — with answer retention standing in for perplexity. The
    returned `ProbeResult.chunk_position` is therefore an **age** (`i - j`),
    not a document position, and feeding these into
    `probe_accuracy_by_position` gives the retention curve.

    Why age rather than position
    ----------------------------
    The obvious probe — at each position, test every answer read so far — is
    confounded. The number of live answers grows with position while the
    running summary stays a fixed-length summary, so accuracy falls with
    position even for a compressor that never degrades at all. A downward
    curve would then be guaranteed in advance and would prove nothing about
    collapse.

    Keying on age fixes it: for every source chunk j the same answers are
    tracked forward, so the count at age k is a property of the corpus rather
    than of k, and the only thing varying along the axis is how many rewrites
    the summary has been through since those answers entered it.

    `answers_by_chunk` maps a chunk position to the answer spans whose evidence
    lies in that chunk. Chunks with no questions contribute nothing.
    """
    by_age: dict[int, list[int]] = {}

    for source_position, answers in answers_by_chunk.items():
        if not answers or source_position >= len(running_summaries):
            continue
        for target_position in range(source_position, len(running_summaries)):
            age = target_position - source_position
            if max_age is not None and age > max_age:
                break
            summary = running_summaries[target_position]
            correct = sum(answer_supported(a, summary, threshold) for a in answers)
            by_age.setdefault(age, []).append((correct, len(answers)))

    return [
        ProbeResult(
            chunk_position=age,
            n_probes=sum(total for _, total in pairs),
            n_correct=sum(correct for correct, _ in pairs),
        )
        for age, pairs in sorted(by_age.items())
    ]


# --------------------------------------------------------------------------
# Self-summaries — the model's own rolling states, for the DAgger mix.
# --------------------------------------------------------------------------


def rolling_self_summaries(
    chunk_texts: list[str],
    model,
    tokenizer,
    max_input_length: int = 512,
    max_new_tokens: int = 64,
) -> list[str]:
    """Runs a checkpoint through the rolling loop and keeps what it produces.

    These become `build_stage2_pairs(self_summaries=...)`: the states the model
    actually reaches, to be paired with the states the teacher says it should
    have. Driving this with the *stage-1* checkpoint is the point — its drift
    is precisely what stage 2 needs to learn to recover from.

    Deliberately retrieval-free. Stage-2 training inputs carry exactly one
    prior summary as context, so the conditioning generated here has to have
    the same shape; running the full `rehearse_elaborative` memory loop would
    produce inputs with up to `top_k` retrieved slots and a train/inference
    mismatch in the opposite direction. It also costs no embedding calls.
    """
    import torch

    model.eval()
    device = next(model.parameters()).device

    summaries: list[str] = []
    previous = ""

    for chunk_text in chunk_texts:
        model_input = format_elaborative_input(chunk_text, [previous] if previous else [])
        inputs = tokenizer(
            model_input, return_tensors="pt", truncation=True, max_length=max_input_length
        ).to(device)

        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

        # An empty generation would blank the conditioning for every later
        # chunk, turning the rest of the document into unconditioned pairs.
        previous = generated or previous
        summaries.append(previous)

    return summaries
