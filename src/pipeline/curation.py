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

from dataclasses import dataclass

# The write-back rule (C-DIC Eq. 6, Algorithm 1) is not "always revise" — it
# branches on similarity. When the new content matches an existing thread it
# replaces that slot (revise); when it matches nothing above the threshold it
# opens a new slot (insert), so unrelated topics are never blended. The prompt
# below is the text-level transposition of that branch: our summary has no
# addressable slots, so the branch has to be stated as an instruction and the
# sentence plays the role of the slot.
#
# Without this branch, "고쳐써라" alone pushes the teacher to force every new
# chunk into the existing paragraph — which is the semantic drift / losing the
# thread failure C-DIC §1 describes, reintroduced by the very instruction meant
# to prevent staleness.
TEACHER_SYSTEM_PROMPT = """You are maintaining a running summary of a document as it is read chunk by chunk.
You will be given the current running summary (may be empty on the first chunk)
and the next chunk of the document.

Do NOT simply append new sentences to the old summary. First decide, for the new
chunk, whether it:

(a) CONTINUES a topic/entity/event already present in the running summary —
    then REVISE the existing sentence(s) that are now incomplete, outdated, or
    contradicted. Integrate the new details into them. Do not add a redundant
    new sentence for the same thing.

(b) INTRODUCES a topic/entity/event not covered by the running summary —
    then ADD one new, separate sentence for it. Do not force it into an
    unrelated existing sentence.

Never blend two unrelated topics into a single sentence just to keep the
summary short. Never repeat information already correctly stated.

Output ONLY the updated running summary. No explanation, no labels."""

TEACHER_USER_TEMPLATE = """Running summary so far:
{previous_summary}

Next chunk:
{chunk_text}

Updated running summary:"""

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


@dataclass
class ProbeResult:
    """One chunk's self-test outcome from the curation loop's §2-3 step.

    `chunk_position` is 0-based order within the document — i.e. how many
    revisions the running summary has already been through, which is the axis
    the collapse check is measured against.
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
