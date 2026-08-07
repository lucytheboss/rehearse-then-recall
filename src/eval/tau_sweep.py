"""Ablation A4 — sweeping tau, the retrieval / write-back threshold.

Why a sweep rather than a cited value
-------------------------------------
tau governs both halves of C-DIC's loop: which prior slots are retrieved
(Eq. 3) and whether a chunk revises an existing thread or opens a new one
(Eq. 6). C-DIC App. K Table 16 sweeps it and finds MSC performance similar for
tau in {0.6, 0.7, 0.8} with a sharp failure at tau = 0.9 — total slots 3.5 ->
30.0, PPL 8.427 -> 12.202; on LongMemEval, 18.8 -> 160.7 slots. Their stated
cause: "overly strict retrieval prevents the model from revising existing
memory states and instead encourages adding new ones."

So the *failure mode* transfers, but the *value* does not. Their similarities
are ICAE latent slots pooled to a vector; ours are `llama-nemotron-embed` over
natural text, and two similarity distributions on different representations
have no reason to share a threshold. App. K tells us what the high end looks
like when it breaks; it does not tell us where our own break point sits. That
is what this sweep measures.

The two failure modes the sweep has to expose
---------------------------------------------
- **tau too low** — everything clears the bar, so every chunk is judged
  on-topic. `insert_share -> 0`, memory collapses toward a single slot that is
  rewritten over and over, and retrieval ranks by raw similarity with no floor,
  which in practice returns whatever is nearest — usually the most recent
  thread. If `recency_share -> 1` alongside, thread-aware retrieval has
  rediscovered recency and bought nothing over the rolling baseline.
- **tau too high** — nothing clears the bar, so nothing is ever revised.
  `insert_share -> 1` and memory grows one slot per chunk. The text-level
  symptom is a running summary that only ever gains sentences.

`degeneration_checks` states both as explicit predicates rather than leaving
them to be eyeballed off a plot.

Cost
----
Re-running the loop per tau would multiply embedding calls by the number of
tau values. `CachedEmbedder` removes most of that: the *query* embedding of a
chunk depends only on the chunk text, which is fixed input, so every tau after
the first is a pure cache hit on the query side. Only the gist embeddings vary,
and only where the generated gist itself differs.

QA scoring is re-scored, not re-run. `answer_recovery` asks whether an answer
span is still recoverable from the compressed state, which needs no QA model
and no API call. Read it as an **upper bound** on what any downstream reader
could achieve at that tau — an answer absent from the state cannot be produced
by a reader, but an answer present in it is not automatically one a reader
would find. It is not the QA EM/F1 of the full pipeline and must not be
reported as such.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

from src.pipeline.curation import (
    normalize_answer,
    probe_accuracy_by_position,
    retention_probes,
    token_f1,
)
from src.pipeline.rehearsal import rehearse_elaborative, retrieval_span_report

DEFAULT_TAUS = (0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.9)
"""Widened 2026-08-02 after the original (0.3, 0.5, 0.65, 0.8, 0.9) — a guess
patterned after C-DIC App. K's own band — failed `degeneration_checks` on real
corpora. Driving the stage-1 (`07`) checkpoint's gists through
llama-nemotron-embed on `news_train/corpus_00` put insert_share at 0.909
already at tau=0.3, i.e. the *entire* original range sat inside the
append-only regime; `wiki_train/corpus_00` needed tau <= 0.05 to see anything
but append-only, and its insert_share jumped from 0.09 at tau=0.03 straight to
1.0 at tau=0.08 — a much sharper transition than news, not a smooth gradient.

This range is itself only evidence from the stage-1 checkpoint, which
`rehearse_elaborative`'s own docs call an invalid configuration (one-shot
XSum training driven in the rolling loop — the exact C-DIC Table 1 mismatch).
Re-sweep after `07b` produces a stage-2 checkpoint; do not assume this range
still brackets both failure modes once the gists it's measuring change.
0.35 — the current `min_similarity` default in `rehearse_elaborative` — is
kept in the range so every sweep run shows where the production default
actually sits relative to the two regimes, not just where the paper's band
sat."""


class CachedEmbedder:
    """Memoises `embed_fn` on (text, input_type) across the whole sweep.

    Wraps an `embed_texts`-shaped callable and keeps its signature, so it drops
    straight into `rehearse_elaborative(embed_fn=...)`. A partially cached batch
    only sends its misses.

    `calls` / `hits` are the audit: with N chunks and T tau values the query
    side should show N misses total rather than N*T, and the report prints the
    ratio so a claim about saved budget is measured rather than asserted.
    """

    def __init__(self, embed_fn):
        self._embed_fn = embed_fn
        self._cache: dict[tuple[str, str], list[float]] = {}
        self.calls = 0  # underlying API calls actually issued
        self.hits = 0
        self.misses = 0

    def __call__(self, texts, input_type, model, truncate, api_key, timeout=30.0, dimensions=None):
        missing = [t for t in texts if (t, input_type) not in self._cache]
        self.hits += len(texts) - len(missing)
        self.misses += len(missing)

        if missing:
            self.calls += 1
            fetched = self._embed_fn(
                missing, input_type=input_type, model=model, truncate=truncate,
                api_key=api_key, timeout=timeout, dimensions=dimensions,
            )
            for text, embedding in zip(missing, fetched):
                self._cache[(text, input_type)] = embedding

        return [self._cache[(t, input_type)] for t in texts]

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


def answer_recovery(state: str, answers: list[str], f1_threshold: float = 0.6) -> tuple[float, float]:
    """(EM, F1) of answer spans against a compressed state.

    EM is normalised substring containment — the answer survived compression
    intact. F1 is `token_f1`, which tolerates the rewriting elaborative
    rehearsal exists to perform, so the EM/F1 gap is itself informative: a
    large gap means answers survive in paraphrase rather than verbatim, which
    is B working as intended rather than failing.
    """
    if not answers:
        return 0.0, 0.0

    normalized_state = normalize_answer(state)
    exact = sum(normalize_answer(a) in normalized_state for a in answers)
    f1 = sum(token_f1(a, state) for a in answers)
    return exact / len(answers), f1 / len(answers)


@dataclass
class TauRow:
    """One tau's outcome. Field order is the table's column order."""

    tau: float
    insert_share: float
    fire_rate: float
    recency_share: float
    mean_span: float
    mean_thread_span: float
    memory_size: int
    memory_growth: float
    early_retention: float
    late_retention: float
    retention_drop: float
    answer_em: float
    answer_f1: float
    api_calls: int = 0
    notes: str = ""


def sweep_tau(
    chunks,
    answers_by_chunk: dict[int, list[str]],
    model,
    tokenizer,
    embed_cfg: dict,
    embed_fn,
    taus=DEFAULT_TAUS,
    top_k: int = 3,
    max_input_length: int = 512,
    max_new_tokens: int = 64,
    memory_capacity: int | None = None,
    on_overflow: str = "grow",
    probe_f1_threshold: float = 0.6,
    collapse_after: int = 3,
) -> tuple[list[TauRow], CachedEmbedder]:
    """Runs `rehearse_elaborative` over one fixed chunk set, varying only tau.

    Same chunks, same model, same seed-free greedy decoding at every tau, so
    any difference in the table is attributable to the threshold and nothing
    else. `tau_read` and `tau_write` are deliberately tied together here —
    A4 is about the single-knob behaviour the pipeline currently has; whether
    decoupling them helps is a separate ablation.

    `on_overflow="grow"` by default so the unbounded behaviour stays visible:
    the point of the sweep is to watch memory blow up at high tau, and a
    capacity bound would hide exactly that. Pass a capacity to sweep tau under
    bounded memory instead.

    `answers_by_chunk` maps a chunk position to the answer spans whose evidence
    lies in it — the same structure `06b` builds, reused here so scoring costs
    no new questions and no QA model.
    """
    cached = CachedEmbedder(embed_fn)
    rows: list[TauRow] = []

    for tau in taus:
        before = cached.calls

        _, records = rehearse_elaborative(
            chunks, model=model, tokenizer=tokenizer,
            embed_cfg=embed_cfg, embed_fn=cached,
            max_input_length=max_input_length, max_new_tokens=max_new_tokens,
            top_k=top_k, min_similarity=tau,
            memory_capacity=memory_capacity, on_overflow=on_overflow,
            record_memory_state=True,
        )

        report = retrieval_span_report(records)

        # The memory state after each chunk is what a downstream reader would
        # see at that position, so retention is scored against it rather than
        # against the rehearsed chunk texts (which accumulate and would make
        # retention trivially high).
        states = [r.memory_state for r in records]
        probes = retention_probes(states, answers_by_chunk, probe_f1_threshold)
        retention = probe_accuracy_by_position(probes, collapse_after=collapse_after)

        final_state = states[-1] if states else ""
        all_answers = [a for answers in answers_by_chunk.values() for a in answers]
        em, f1 = answer_recovery(final_state, all_answers, probe_f1_threshold)

        rows.append(
            TauRow(
                tau=tau,
                insert_share=round(report["insert_share"], 4),
                fire_rate=round(report["fire_rate"], 4),
                recency_share=round(report["recency_share"], 4),
                mean_span=round(report["mean_span"], 4),
                mean_thread_span=round(report["mean_thread_span"], 4),
                memory_size=report["memory_size"],
                memory_growth=round(report["memory_growth"], 4),
                early_retention=round(retention["early_accuracy"], 4),
                late_retention=round(retention["late_accuracy"], 4),
                retention_drop=round(retention["drop"], 4),
                answer_em=round(em, 4),
                answer_f1=round(f1, 4),
                api_calls=cached.calls - before,
                notes="pool write failures" if report["pool_write_failures"] else "",
            )
        )

    return rows, cached


# --------------------------------------------------------------------------
# The failure modes, as predicates rather than as a plot to be eyeballed.
# --------------------------------------------------------------------------


@dataclass
class DegenerationCheck:
    name: str
    passed: bool
    detail: str


@dataclass
class DegenerationReport:
    """Two kinds of statement, kept apart on purpose.

    `checks` are about the **sweep**: did the tau range actually bracket both
    failure modes? A failure here means the range was too narrow and no value
    should be chosen from it yet, so these gate `passed`.

    `observations` are about the **pipeline**: what the sweep found. These must
    never gate anything. "No tau degenerated to recency" is a *good* result —
    it means thread-aware retrieval is beating the rolling baseline — and an
    earlier version of this class marked it FAIL, which would have pressured the
    sweep toward reporting a failure that isn't there.
    """

    checks: list[DegenerationCheck] = field(default_factory=list)
    observations: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def __str__(self) -> str:
        lines = [
            f"[{'ok ' if c.passed else 'FAIL'}] {c.name}: {c.detail}" for c in self.checks
        ]
        if self.observations:
            lines.append("")
            lines += [f"  - {o}" for o in self.observations]
        return "\n".join(lines)


def degeneration_checks(
    rows: list[TauRow],
    low_insert_share: float = 0.2,
    high_insert_share: float = 0.8,
    recency_collapse: float = 0.8,
) -> DegenerationReport:
    """Verifies the sweep actually spans both failure modes.

    These are checks on the *sweep*, not on the pipeline: a failing check means
    the tau range was too narrow to see the boundary, so the range should be
    widened before a value is chosen from it. Choosing tau from a sweep that
    never reached either failure mode is choosing from the flat middle of a
    curve whose edges were never located.

    1. `insert_share` must be non-decreasing in tau. It is the fraction of
       chunks that opened a new thread, and raising the bar for "on topic" can
       only ever move chunks from revise to insert. A violation means something
       other than tau changed between runs.
    2. The low end must actually reach collapse-into-one-thread
       (`insert_share <= low_insert_share`).
    3. The high end must actually reach append-only
       (`insert_share >= high_insert_share`), which is the unbounded-slot
       growth the paper concedes as Limitation 1.
    Recency degeneration is recorded as an *observation*, not a check. Whether
    a low tau collapses to recency is the question the sweep is asking; making
    its absence a failure would be requiring the answer in advance.
    """
    report = DegenerationReport()
    if not rows:
        report.checks.append(DegenerationCheck("non-empty", False, "no rows"))
        return report

    ordered = sorted(rows, key=lambda r: r.tau)
    shares = [r.insert_share for r in ordered]

    monotone = all(b >= a - 1e-9 for a, b in zip(shares, shares[1:]))
    report.checks.append(
        DegenerationCheck(
            "insert_share is non-decreasing in tau",
            monotone,
            " -> ".join(f"{r.tau}:{r.insert_share:.2f}" for r in ordered),
        )
    )

    low = ordered[0]
    report.checks.append(
        DegenerationCheck(
            f"low tau reaches single-thread collapse (insert_share <= {low_insert_share})",
            low.insert_share <= low_insert_share,
            f"tau={low.tau} insert_share={low.insert_share:.2f} memory_size={low.memory_size}",
        )
    )

    high = ordered[-1]
    report.checks.append(
        DegenerationCheck(
            f"high tau reaches append-only growth (insert_share >= {high_insert_share})",
            high.insert_share >= high_insert_share,
            f"tau={high.tau} insert_share={high.insert_share:.2f} "
            f"memory_growth={high.memory_growth:.2f}",
        )
    )

    recency = [
        r for r in ordered
        if r.recency_share >= recency_collapse and r.insert_share <= low_insert_share
    ]
    if recency:
        report.observations.append(
            "recency degeneration at tau "
            + ", ".join(f"{r.tau} (recency_share={r.recency_share:.2f})" for r in recency)
            + " — at these values thread-aware retrieval is only rediscovering "
            "recency and the embedding cost buys nothing"
        )
    else:
        report.observations.append(
            f"no tau degenerated to recency (max recency_share "
            f"{max(r.recency_share for r in ordered):.2f} < {recency_collapse}) — "
            "retrieval is reaching past the previous chunk at every threshold tested"
        )

    peak = max(ordered, key=lambda r: r.answer_f1)
    report.observations.append(
        f"answer recoverability peaks at tau={peak.tau} "
        f"(F1={peak.answer_f1:.3f}, EM={peak.answer_em:.3f}, {peak.memory_size} slots)"
    )

    return report


def recommend_tau(rows: list[TauRow]) -> tuple[float, str]:
    """The tau furthest from both failure modes, by retention.

    Deliberately simple and stated rather than hidden: among the tau values
    that are in neither degenerate regime (`0.2 < insert_share < 0.8`), pick
    the one with the highest `answer_f1`, breaking ties toward the smaller
    memory. If no tau sits in that band the sweep has not bracketed a usable
    value and this says so instead of returning a number.
    """
    usable = [r for r in rows if 0.2 < r.insert_share < 0.8]
    if not usable:
        return math.nan, (
            "no tau sits between the two failure modes — widen the sweep range "
            "before choosing a value"
        )
    best = max(usable, key=lambda r: (r.answer_f1, -r.memory_size))
    return best.tau, (
        f"tau={best.tau}: insert_share={best.insert_share:.2f}, "
        f"recency_share={best.recency_share:.2f}, answer_f1={best.answer_f1:.3f}, "
        f"memory {best.memory_size} slots"
    )


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------

_TABLE_COLUMNS = [
    ("tau", "τ"),
    ("insert_share", "insert_share"),
    ("recency_share", "recency_share"),
    ("mean_span", "mean_span"),
    ("memory_size", "slots"),
    ("retention_drop", "ret_drop"),
    ("answer_em", "EM*"),
    ("answer_f1", "F1*"),
]


def to_rows(rows: list[TauRow]) -> list[dict]:
    return [asdict(row) for row in rows]


def to_markdown(rows: list[TauRow]) -> str:
    """The A4 table, narrowed to the columns the slide needs.

    EM*/F1* carry the asterisk on purpose — they are answer *recoverability*
    from the compressed state, an upper bound on downstream QA, not the
    pipeline's QA score.
    """
    header = "| " + " | ".join(label for _, label in _TABLE_COLUMNS) + " |"
    rule = "|" + "|".join("---" for _ in _TABLE_COLUMNS) + "|"
    lines = [header, rule]
    for row in sorted(rows, key=lambda r: r.tau):
        values = []
        for key, _ in _TABLE_COLUMNS:
            value = getattr(row, key)
            values.append(f"{value}" if isinstance(value, int) else f"{value:.3f}")
        lines.append("| " + " | ".join(values) + " |")
    lines.append("")
    lines.append("\\* answer recoverability from the final compressed state — an upper")
    lines.append("bound on downstream QA, not the pipeline's QA EM/F1.")
    return "\n".join(lines)
