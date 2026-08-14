# Rehearse, Then Recall — Progress Report

_Last updated: 2026-08-15 (paper framing decided — §8: B (verbatim survives, rewritten collapses) + C (Transfer-Appropriate Compression) layered, implemented in both short-paper drafts with a restructured intro/RQ2a-RQ2b split/Table 8/conclusion; literature citations spot-checked (EXIT, LongLLMLingua, RAPTOR, C-DIC confirmed accurate). 2026-08-14 findings: seam padding closes maintenance_extractive_dynamic's novel gap to RAG completely, confirmed at full scale n=79 — now matches or beats RAG on 2 of 4 genres; no-merge thread clustering confirmed at full scale — recovers B/gist from collapse (0.121→0.264) but hurts already-good A content (0.428→0.390); soft/attention-weighted extensions of thread selection tried and confirmed not competitive with RAG's efficiency frontier; structural_map_extractive confirmed at full scale, underperformed)_

## 1. What this project is testing

Long-context LLM QA under a memory-rehearsal framing, combining two lines of
prior work:

- **C-DIC** (Jung et al., ICML 2026, arXiv:2606.12411) — a retrieve/revise/
  write-back loop over a multi-slot `ThreadMemory`, used here for
  **elaborative rehearsal (B)**: a rolling pass that rewrites each chunk
  while conditioned on related prior material.
- **ReadAgent** (Lee et al., ICML 2024, arXiv:2402.09727) — "shorten, don't
  summarize" gisting, and specifically **ReadAgent-P's interactive lookup**:
  answer from gists if possible, otherwise point at a page/chunk and get its
  raw text for a follow-up call.

The central question this report covers: **can a compressed, gist-based
reading strategy match or beat giving the model the full raw document
(`full_context`), and at what cost?**

## 2. Pipeline summary

1. **Stage 1** (`06`/`07`): B trained one-shot on XSum-style compression.
   Running it in a rolling loop at inference is a known train/inference
   mismatch (C-DIC Table 1) — confirmed here too (see §4).
2. **Stage 2** (`06b`/`07b`): B retrained on rolling-shaped targets, curated
   by a teacher LLM (`meta/llama-3.1-70b-instruct`) driving the same
   `ThreadMemory` loop B uses at inference (`curate_document_threaded`,
   `src/pipeline/teacher.py`). 336 train + 48 val/test documents curated,
   7.1% teacher-call failure rate.
3. **Evaluation** (`10`, `11`): the trained stage-2 checkpoint is used to
   gist 4 eval genres — **wiki**, **news**, **novel** (narrativeqa),
   **caselaw** (CaseHOLD, multiple-choice) — and tested under several
   reading strategies against two baselines from `03`: `closed_book` and
   `full_context`.

## 3. Stage-2 checkpoint quality (`07b`)

`early`/`late` are probe accuracy pooled over ages 0-2 / ages 3+ respectively
(the split the collapse diagnostic actually uses — see
`probe_accuracy_by_position`, `collapse_after=3`); `drop` is early minus late,
so positive means collapse.

| stage-2 checkpoint | early | late | drop |
|---|---|---|---|
| t5-small, self_conditioned_ratio=0.3 (original) | 0.355 | 0.244 | +0.111 |
| t5-small, self_conditioned_ratio=0.0 (ablation) | 0.312 | 0.263 | +0.050 |
| **flan-t5-base, self_conditioned_ratio=0.0** | **0.395** | **0.308** | **+0.088** |
| *(for reference)* stage 1, one-shot driven incrementally | 0.065 | 0.087 | -0.022 |
| *(for reference)* teacher, upper bound | 0.383 | 0.417 | -0.034 |

![Figure 4 — Stage-2 checkpoint retention vs. teacher](docs/report_assets/fig4_retention_curve.png)

*(Figure is from the original t5-small run — regenerate from
`results/elaborative_stage2_retention_by_age_flant5base_ratio0.csv` before
using this version in the writeup.)*

Two separate levers were tried, and neither one fully closes the gap to the
teacher on its own:

- **Removing self-conditioning** (`ratio=0.0`) is the more effective lever
  for the *drop* metric specifically — it roughly halves it (+0.111 →
  +0.050), the intended effect of guarding against exposure bias.
- **Scaling the base model** (t5-small → flan-t5-base, still at
  `ratio=0.0`) raises retention at essentially every age in absolute terms
  (early 0.312 → 0.395, late 0.263 → 0.308 — closer to the teacher's 0.383 /
  0.417 than either t5-small config gets) but does **not** stack additively
  with the ratio fix on the drop metric itself (+0.088, worse than +0.050).
  Bigger model, higher ceiling, same-shaped curve.

Every downstream condition in §4 was evaluated against the **t5-small,
ratio=0.0** checkpoint, not the flan-t5-base one — `10`'s
`STAGE2_CHECKPOINT_RATIO0` toggle hasn't been extended to point at the new
checkpoint yet, so §4's numbers still carry that smaller compressor's
ceiling. Whether the flan-t5-base checkpoint's higher absolute retention
translates into higher downstream QA accuracy is an open, unanswered
question, not yet tested.

## 4. Evaluation results

![Figure 1 — Accuracy by genre x condition, full scale](docs/report_assets/fig1_accuracy_full_scale.png)

### 4.1 Full-scale, validated (n = 579: wiki 100, news 200, novel 79, caselaw 200)

| condition | wiki | news | novel | caselaw | avg tokens (wiki/news/novel/caselaw) |
|---|---|---|---|---|---|
| closed_book | 0.280 | 0.040 | 0.025 | 0.335 | ~110 (all genres) |
| **full_context** | **0.810** | **0.395** | **0.494** | **0.360** | 19,566 / 65,035 / 81,139 / 87,423 |
| chunked_sequential (raw, 4-chunk budget) | 0.190 | 0.010 | 0.063 | 0.225 | — |
| chunked_sequential_rehearsal (B alone, 4-chunk budget) | 0.230 | 0.010 | 0.051 | 0.185 | — |
| full_context_rehearsal_lookup (gist map + 1-chunk lookup, cap 20) | 0.400 | 0.080 | 0.063 | 0.250 | 8,812 / 25,771 / 31,914 / 25,887 |
| full_context_rehearsal_lookup_retrieval (+ embedding-similarity targeting) | 0.660 | 0.255 | 0.215 | 0.250 | 9,877 / 26,745 / 32,227 / 26,978 |
| **full_context_rehearsal_lookup_adaptive** (Adaptive-k, no per-genre tuning) | **0.760** | **0.255** | 0.177 | **0.300** | 9,738 / 26,962 / 33,252 / 30,625 |

**Reading this table (updated 2026-08-12 with adaptive-k now confirmed at
full scale — previously pilot-only)**: `chunked_sequential_rehearsal` tests
B in a regime where it structurally can't win (same 4-raw-chunk budget as
the uncompressed baseline). The real test starts at
`full_context_rehearsal_lookup`: show every chunk's gist at once, let the
model ask for raw text on 1+ specific chunks. **Adaptive-k is now the best
of the three gist-based conditions overall** (pooled: full_context 0.515,
lookup 0.192, lookup+retrieval 0.318, lookup+adaptive 0.347) and gets
closest to `full_context` on wiki (0.760 vs 0.810, a 0.05 gap) and caselaw
(0.300 vs 0.360) — but **novel is the one genre where retrieval still beats
adaptive** (0.215 vs 0.177), so "adaptive-k wins" is not uniform across
genres. No condition here beats `full_context` outright at full scale.
Token savings are real and large regardless: adaptive-k runs at 50-65%
fewer tokens than `full_context` in every genre (e.g. wiki 9,738 vs 19,566).

*(Note: the lookup/retrieval pooled numbers above differ slightly from an
earlier reading of this table — 0.232/0.344 vs the current 0.192/0.318 —
after a full re-run on 2026-08-12. Same checkpoint, same code; the gap is
consistent with this project's own documented `full_context`
non-reproducibility at temperature=0 (~3.5% row flips on rerun, §2 of `02`).
The numbers above are from the complete, most recent n=579 run.)*

![Figure 2 — Token cost by genre x condition, full scale](docs/report_assets/fig2_tokens_full_scale.png)

### 4.1.1 Accuracy per token

![Figure 5 — Accuracy per token, full scale](docs/report_assets/fig5_accuracy_per_token.png)

| genre | full_context | chunked_seq | chunked_seq+rehearsal | lookup | lookup+retrieval | lookup+adaptive |
|---|---|---|---|---|---|---|
| wiki | 0.414 | 0.426 | **1.317** | 0.454 | 0.668 | 0.781 |
| news | 0.061 | 0.033 | 0.054 | 0.031 | 0.095 | **0.095** |
| novel | 0.061 | 0.247 | **0.286** | 0.020 | 0.067 | 0.053 |
| caselaw | 0.041 | 0.600 | **0.775** | 0.097 | 0.093 | 0.098 |

_(accuracy per 10,000 tokens; `closed_book` excluded — near-zero tokens makes its ratio a meaningless outlier)_

**Read this one carefully — it rewards being cheap more than it rewards
being good.** `chunked_seq+rehearsal` tops every genre here, including
caselaw at 0.775, but caselaw's *absolute* accuracy for that condition is
0.185 (§4.1) — worse than `closed_book`'s 0.335, i.e. worse than guessing
from the question alone. A condition that answers badly using very few
tokens still scores well on this metric, because the denominator collapses
faster than the numerator does. Use this figure to compare *among
conditions that already clear a usable accuracy bar* (e.g. `lookup+retrieval`
and `lookup+adaptive` both beat `full_context` on this metric in news,
novel, and caselaw while also being reasonably accurate in absolute
terms) — not as a standalone ranking.

### 4.2 Pilot-scale, provisional (n = 20/genre — **not yet confirmed at full scale**)

![Figure 3 — Per-genre cap and Adaptive-k vs. full-scale baselines](docs/report_assets/fig3_pilot_comparison.png)

Adaptive-k is now confirmed at full scale (§4.1: caselaw 0.300) — kept here
only for the direct pilot-scale comparison against the per-genre cap, which
is still pilot-only:

| condition | wiki | news | novel | caselaw |
|---|---|---|---|---|
| full_context_rehearsal_lookup_retrieval_pergenre (cap 5 for caselaw, 20 elsewhere) | 0.750 | 0.300 | 0.450 | **0.450** |
| full_context_rehearsal_lookup_adaptive (pilot, n=20) | 0.750 | 0.300 | 0.450 | 0.300 |
| full_context_rehearsal_lookup_adaptive (**full scale, n=200, §4.1**) | 0.760 | 0.255 | 0.177 | 0.300 |

The full-scale adaptive-k number landed close to its own pilot on caselaw
(0.300 both) but drifted on news (0.300→0.255) and novel (0.450→0.177) —
another reminder that pilot numbers move at scale, in either direction, not
just downward.

`full_context_with_map` (`11`, full raw text + gist map, single call) —
pilot deltas over `full_context`, pre-shortening-fix, **abandoned, not
directly comparable to the table above** (see §5): wiki +0.140, news
-0.045, novel +0.006, caselaw +0.040.

**Read with real caution.** This project already has one confirmed example
of a pilot-scale result reversing at scale: `full_context_rehearsal_lookup_retrieval`'s
wiki accuracy measured **0.850** (beating `full_context`'s 0.810) at n=20,
then dropped to **0.620** at the full n=100 — noise, not signal. Every
number in this table needs the same full-scale confirmation before it goes
in a paper as a finding rather than a lead.

That said, caselaw's per-genre-cap jump (0.245 -> 0.450 at n=20) is the
largest effect size measured anywhere in this project, and lines up with a
consistent pattern across three separate experiments (the `12` sweep, the
base lookup condition, and this one) that caselaw specifically does worse
with more retrieved context. Worth prioritizing for full-scale
confirmation over the other pending cells.

**Adaptive-k does not match the per-genre cap on caselaw specifically**
(0.300 vs 0.450 at n=20), the one genre the whole per-genre-vs-adaptive
question was actually about. `ADAPTIVE_RELATIVE_THRESHOLD` was swept over
{0.75, 0.85, 0.90, 0.95, 0.98} (n=10/genre/value, 2026-08-11) to check
whether this was simply an untuned default rather than a real mechanism
gap — it wasn't a tuning problem: caselaw stayed at 0.4 across every value
except 0.98 (0.5, on a 1-question swing at n=10 — not distinguishable from
noise), while wiki (0.9) and news (0.1) showed *zero* sensitivity to the
threshold at all. Token cost fell slightly as the threshold tightened
(23,857 → 22,778, mechanical — fewer chunks pass a stricter cutoff), with
no accompanying accuracy gain. Read together with the flat per-genre
numbers, this points at a **targeting-quality gap, not a chunk-count
gap** — the per-genre cap's caselaw advantage likely isn't "fewer chunks,"
it's *which* chunks, something a relative-score-margin rule over the same
similarity ranking can't fix by moving its one threshold. Concluding this
line of tuning here; see §7.

### 4.3 Pure RAG vs. rehearsal — full investigation (confirmed at full scale, n=579)

Every condition in §4.1 compresses first (B's stage-2 rehearsal), whether
or not it also retrieves — nothing in §4.1-4.2 isolates whether the
*rehearsal* step itself is contributing, versus the retrieval half doing
all the work. This section reports a same-day investigation (2026-08-13)
that started as that one missing control and expanded into eight further
conditions once the control's result turned out to be decisive.

#### 4.3.1 `raw_retrieval_adaptive` — the missing control, confirmed

**No rehearsal checkpoint, no teacher, no compression anywhere** — embed
the question, select chunks with the identical Adaptive-k mechanism
`full_context_rehearsal_lookup_adaptive` uses (same threshold, same
embeddings), answer from their **raw** text in one call. The §4.2-era pilot
(n=15/genre) showed this leading on 3 of 4 genres; per this report's own
standing caution about pilot-vs-full-scale reversals (§4.2), that result
was held pending confirmation. It is now confirmed at full scale:

| condition | wiki | news | novel | caselaw | overall | avg tokens |
|---|---|---|---|---|---|---|
| `raw_retrieval_adaptive` (RAG) | **0.850** | **0.420** | **0.316** | **0.300** | **0.439** | 3,588 |
| `full_context_rehearsal_lookup_adaptive` (§4.1's best gist-based condition) | 0.760 | 0.255 | 0.177 | 0.275 | 0.339 | 25,551 |

RAG wins every genre on accuracy and uses 7x fewer tokens. (Caselaw here is
post-`<HOLDING>`-fix, §5 — both rows in this table are on the corrected
corpus.)

#### 4.3.2 Isolating architecture from content

`full_context_rehearsal_lookup_adaptive` always shows the *full document's*
gist map on its first call, whether or not lookup ever triggers — a
structural cost/architecture confound independent of whether rehearsed
*content* helps. `gist_retrieval_adaptive` (`10` §13d) holds RAG's
single-call architecture and chunk selection fixed and swaps only the
content shown (gist instead of raw text):

| condition | wiki | news | novel | caselaw | overall | avg tokens |
|---|---|---|---|---|---|---|
| `gist_retrieval_adaptive` | 0.130 | 0.055 | 0.051 | 0.210 | 0.121 | 1,105 |

Loses to RAG **and** to `full_context_rehearsal_lookup_adaptive` on every
genre. The gist-based lookup condition's partial competitiveness in §4.1
comes from its raw-text lookup escape hatch, not from the gist content
itself — gist content shows no advantage under any architecture tested so
far.

A follow-up, `gist_retrieval_gistembed` (`10` §13e), reselects using
*gist*-embedding similarity instead of raw-text embedding similarity (same
content) to rule out a selection-mismatch confound: overall accuracy is
essentially unchanged (0.121 → 0.119), confirming the loss is about gist
content quality, not which chunks get selected.

A second follow-up, `hybrid_gistselect_rawanswer` (`10` §13f) — select via
gist embeddings, answer from raw text — tests whether gist embeddings are a
*better* retriever even though gist *content* isn't a better answer
source:

| condition | wiki | news | novel | caselaw | overall | avg tokens |
|---|---|---|---|---|---|---|
| `hybrid_gistselect_rawanswer` | 0.480 | 0.175 | 0.152 | 0.295 | 0.266 | 5,364 |

Loses to RAG on **both** accuracy and tokens — gist embeddings are a worse
retriever than raw-text embeddings, not a better one.

#### 4.3.3 Teacher-gist ceiling test → thread-memory collapse, confirmed root cause

To test whether t5-small's compressor quality (not compression as such) is
the bottleneck, `meta/llama-3.1-70b-instruct` was run through the identical
thread-retrieve-replace mechanism (`curate_document_threaded`) that built
every gist above, on a 30-chunks-per-genre prefix pilot:

| genre | raw→gist word ratio | failed positions |
|---|---|---|
| wiki | 4,137 → 7,959 (**192%**) | 5/30 |
| novel | 4,123 → 6,096 (**148%**) | 0/30 |
| news | 3,829 → 2,597 (68%) | 0/30 |
| caselaw | 4,172 → 2,871 (69%) | 0/30 |

Wiki and novel *expanded* rather than compressed. Inspecting all 6 gists
for wiki side by side (not just one, as the initial spot-check did) showed
they were **near-identical** — chunks covering boroughs, founding history,
tourism, Wall Street, and glacial geology all produced essentially the same
sentence as chunk 0's intro. This is not "topic drift" on one example; it
is total collapse. The same pattern was independently confirmed in the
*student*'s own output: `full_gists_by_genre_flant5base.json` (the
precomputed, already-on-disk gists used throughout this project's earlier
evaluation) shows the identical pattern for wiki, news, novel, and case
law alike — every chunk's gist beyond position 0 reproduces position 0's
content almost verbatim, regardless of genre.

**Controlled minimal test isolates the cause.** Calling the student
(t5-small) and the teacher (70B) directly with two topically distinct
sentences alternately placed in the "current chunk" and "related thread"
slots of their respective prompts: both models reproduced whichever
content sat in the "related thread" slot near-verbatim and ignored
"current chunk" entirely — regardless of which actual content occupied
which slot (confirmed by swapping them). This rules out both a code bug
(the prompt-building and write-back logic in `rehearsal.py`/
`thread_memory.py` were independently inspected and are correct — they
faithfully pass whatever text the model returns into memory) and a
positional artifact (the student's prompt puts "current chunk" first,
the teacher's puts "related thread" first, yet both collapse toward the
"related thread" slot specifically).

**Root cause: the student learned this from the teacher.** The Stage-2
training target ($S_i$ in Eq. 3, §2.2.2 methodology) *is* the teacher's
`curate_document_threaded` output — and that output was already collapsed,
per the finding above. The student was never trained on clean rolling
summaries; it was trained on a large volume of examples whose "correct"
answer, whenever a related thread existed, was essentially "reproduce the
thread." Supervised training on that signal produces exactly what was
observed. **Retraining the student would not fix this** — the flaw is
upstream, in what `curate_document_threaded` produces, which flows into
`06b`'s corpus of training pairs. This also revises §3's earlier
"compressor scale / training data volume" explanation for Table 2's
residual gap (§3): scaling the compressor (flan-t5-base, §3) raised
absolute retention without improving the age-related *decay* itself,
which is now legible as the same collapse pattern degrading a bigger
model too, not a capacity-gap symptom curable by more parameters or more
documents.

**Mitigation attempt (2026-08-13, in progress).** A collapse check
(`collapse_score` — token-F1 overlap between a regenerated gist and its
given context) plus corrective retry was added
(`curate_document_threaded_anticollapse`, `src/pipeline/teacher.py`): on
detecting collapse, retry up to twice with explicit corrective feedback,
then regenerate with no context at all (treating the chunk as opening a
fresh thread) if still collapsed. A 6-chunk sequential trial found the
corrective retries **never succeeded** (0/10 across 5 context-bearing
chunks) — every case that resolved did so via the no-context fallback,
meaning the final gists are now accurate per-chunk but the mechanism is
functionally closer to independent per-chunk summarization than genuine
cross-chunk thread integration. A 20-chunk, retry-skipped re-run
(`max_retries=0`, since retries never helped) is in progress to confirm
this holds at slightly larger scale before deciding whether to pursue a
full corpus re-curation. Full-scale teacher re-curation (real API cost;
~1,395 chunks, ~2.9h at the configured rate limit even before any retry
overhead) has not been run.

#### 4.3.4 Why: literature grounding (2026-08-13)

- **RAPTOR** (Sarthi et al., ICLR 2024) — a hierarchical tree of
  progressively more abstract summaries, with retrieval choosing which
  *level* fits a given query; +20pp on QuALITY. Every gist condition tested
  here forces one uniform compression level regardless of question.
- **EXIT** (Kim et al., ACL 2025 Findings, arXiv:2412.12559) — extractive
  compression (selecting existing sentences) beats abstractive
  summarization for RAG (EM 41.4 vs. 36.9), matching §4.3.3's directly
  observed failure mode (exact terms dropped, content rewritten).
- **LongLLMLingua / SmartChunk-style compression** — query-*aware*
  compression (done after the question is known) beats query-agnostic
  compression. B's design is query-agnostic by construction (the rehearsal
  step never sees the eventual question) — unfavorable on this axis too.

Both axes point the same direction: gist_retrieval's loss is not "lossy
compression can't work here," but that this project's specific compression
choice (abstractive, query-agnostic) sits on the disadvantaged side of two
axes the 2025 literature has already characterized.

#### 4.3.5 A condition built on that literature: extractive, query-aware pruning

`extractive_query_aware_adaptive` (`10` §13g) holds RAG's chunk selection
fixed, then — *after* the question is known — scores individual sentences
within the selected chunks against the question and keeps only the
highest-scoring ones, verbatim (no rewriting):

| condition | wiki | news | novel | caselaw | overall | avg tokens | avg latency |
|---|---|---|---|---|---|---|---|
| RAG (`raw_retrieval_adaptive`) | 0.850 | 0.420 | 0.316 | 0.300 | 0.439 | 3,588 | 4.75s |
| `extractive_query_aware_adaptive` | 0.780 | 0.370 | **0.114** | 0.255 | 0.366 | **680** | **3.17s** |

This is the first condition in this investigation with a genuine,
defensible trade-off rather than a strict loss: **5.3x fewer tokens, ~34%
lower latency, for a 7.3pp accuracy cost** — within 5-7pp of RAG on 3 of 4
genres. Accuracy-per-10k-tokens: 5.38 vs. RAG's 1.22 (4.4x). The exception
is novel, which drops sharply (-20.2pp) — plausibly because narrative
continuity depends on surrounding sentences that isolated extractive
pruning discards. Not yet investigated: a sentence-window variant (±1
sentence around each selected one) that might recover novel's loss without
giving up the token/latency advantage.

#### 4.3.6 Length-stress test — does document length change the calculus?

Motivated by 2026-era context-window reality: advertised windows are large
(Claude/GPT 1M tokens, Gemini 2-10M), but usable reliability is not —
NVIDIA's RULER puts effective capacity at 50-65% of advertised, and Adobe's
NoLiMa (2025) found most models score below half their short-context
accuracy past **32K tokens**. `full_context_rehearsal_lookup_adaptive`'s
own token usage (26K-33K/question in 3 of 4 genres, §4.1) already sits at
that threshold — raising the question of whether a longer-document regime
would favor compression (bounded size) over RAG (unbounded selection size)
by construction.

Tested with `01_length_stress_prep.ipynb`'s per-document truncation ladder
(1,000-10,000 words, pilot range of a 50,000-word ladder) via a standalone
script (`length_stress_rag_vs_gist.py`): at every tested length, RAG's
*selected* (retrieved) token budget stays small and grows slowly (338 →
826 tokens, 1K→10K words) because Adaptive-k selects a roughly constant
number of relevant chunks regardless of total document length. Gist size,
by contrast, scales with the *whole document* (every chunk gets gisted,
selectively or not) and grows far faster — already exceeding RAG's budget
at the shortest tested rung, even at the tightest compression setting
(`max_new_tokens=64`: 704 tokens at 1,000 words vs. RAG's 338; the gap
widens with length). Loosening the compression cap (128/256 tokens) makes
this worse, not better. **The length-stress hypothesis does not hold in the
tested range — if anything it reverses**: RAG's relevance-gated retrieval
scales sub-linearly with corpus size; query-agnostic whole-document
compression scales linearly with it. A separate finding from the same
script: RAG's own recall (whether Adaptive-k selects the chunk actually
containing the evidence) degrades sharply with length (0.6 at 1K words →
0.1 at 10K) — a real problem, but one for RAG's own tuning, not evidence
that compression fixes it (accuracy at length was not measured for either
approach — only token/size proxies).

#### 4.3.7 Does less-aggressive compression recover accuracy? No — confirmed, rejected

`gist_retrieval_adaptive_mnt128` (`10` §13h) tests a narrower question than
§4.3.6: at nb10's actual eval scale (not the length-stress ladder's single
long documents), does regenerating the gist with `max_new_tokens=128`
instead of 64 (the student's inference cap vs. the 512 its teacher targets
were built with — a real train/inference mismatch, not evidence 64 is
optimal) recover some of §4.3.2's accuracy loss without costing much more
than its 1,105 avg tokens?

**No — it makes both axes worse.** Pilot (n=20/genre, same slice as
`gist_retrieval_adaptive`'s mnt=64 baseline for a matched comparison):

| genre | mnt=64 | mnt=128 |
|---|---|---|
| caselaw | 0.35 | 0.30 |
| news | 0.05 | 0.05 |
| novel | 0.15 | 0.10 |
| wiki | 0.30 | **0.00** |
| overall | 0.212 | 0.112 |
| avg tokens | 902 | 1,539 |

Accuracy nearly halves and tokens grow 1.7x. Consistent with §4.3.3: the
problem is not that 64 tokens is too little room to fit the *right*
content — it's that the collapse mechanism produces the *wrong* content
regardless of how much room it's given, so a bigger budget only lets it
reproduce more of the wrong (collapsed) thing.

#### 4.3.8 `structural_map_extractive` — confirmed at full scale, underperformed

§4.3.14's cognitive-localization hypothesis (split the one overloaded
mechanism into three specialized pieces — structural map / executive chunk
selection / extractive detail) was piloted, built, and run to full scale
(`10` §13i, n=579). It underperformed expectations: **0.142 overall**,
4,929 avg tokens, 12.91s latency — worse than `extractive_query_aware_adaptive`
(0.366) on both accuracy and tokens, and not competitive with RAG. The
per-chunk topic-label map (query-agnostic by construction, so it cannot
collapse the way §4.3.3's gists do) does not appear to give the LLM enough
signal to select the right chunks — the executive-selection step reasoning
over labels loses more than the collapse-avoidance gains back. Recorded as
a negative result: splitting the mechanism into specialized pieces did not,
on its own, recover accuracy — see §4.3.9 for the piece that did.

#### 4.3.9 Maintenance rehearsal (A) revived — the strongest compression-family result yet

§4.3.2's isolation left one cell of the extractive/abstractive ×
query-aware/agnostic design unfilled: extractive **and** query-agnostic.
Maintenance rehearsal (A) — dropped from scope on 2026-08-12 (§6) — fills
exactly that cell: a RoBERTa `SentenceScorer` selects the top sentences
per chunk verbatim, with no rewriting and no cross-chunk conditioning. Because
nothing is regenerated, §4.3.3's collapse mechanism cannot occur
structurally. The already-trained checkpoint (`experiments/rehearsal_maintenance_roberta/`)
was combined with RAG's own Adaptive-k chunk selection and re-evaluated at
full scale (n=579) in four variants (`10` §13j-m):

| variant | wiki | news | novel | caselaw | overall | avg tokens |
|---|---|---|---|---|---|---|
| `maintenance_extractive_adaptive` (fixed R=0.3) | 0.440 | 0.360 | 0.241 | 0.265 | 0.325 | 1,567 |
| `maintenance_extractive_r50` (fixed R=0.5) | 0.520 | 0.410 | 0.278 | 0.285 | 0.368 | 2,011 |
| `maintenance_extractive_combined` (A's pool + §4.3.5's query-aware re-selection) | 0.510 | 0.335 | 0.101 | 0.235 | 0.299 | 467 |
| `maintenance_extractive_dynamic` (dependent-ratio, 32K-token target) | **0.880** | 0.415 | 0.278 | 0.275 | **0.428** | 1,862 |
| `raw_retrieval_adaptive` (RAG, reference) | 0.850 | 0.420 | 0.316 | 0.300 | 0.439 | 3,588 |

Raising the fixed ratio alone helps (0.325→0.368) — consistent with `05`'s
own recall@k measurement that R=0.3 loses 43% of evidence sentences while
R=0.5 loses only 26%. This is the *opposite* direction from gist under a
relaxed budget (§4.3.7's mnt128, which got worse with more room) — since A's
content is verbatim, more budget means strictly more real information, not
more room to reproduce collapsed content.

The dynamic-ratio variant (`dependent_ratio()`, Sie et al. NLLP 2024 §3.2 —
compression ratio set per document from `target_context_tokens=32,000`,
matching the NoLiMa reliability threshold cited in §4.3.4) is the standout:
**0.428 overall vs. RAG's 0.439 — a 0.011 gap — at roughly half RAG's
tokens (1,862 vs. 3,588), and it beats RAG outright on wiki (0.880 vs.
0.850)**, the only condition in the entire investigation to exceed RAG's
absolute accuracy on any genre. The combined variant (A prefilter +
extractive_query_aware_adaptive's query-aware re-selection within it) is a
clear negative result by contrast: cutting tokens further (467) came at a
real accuracy cost (0.299), driven mostly by a collapse on novel (0.101
vs. 0.241 for base A) — two lossy selection stages compounded rather than
complementing each other.

Read against §4.3.14's cognitive-psychology framing, this is a genuine
tension worth naming rather than smoothing over: levels-of-processing
theory (Craik & Lockhart, 1972) predicts *elaborative* processing should
beat *maintenance* processing for humans, because deeper semantic encoding
produces better long-term retention than surface repetition. This project's
result inverts that — because the "deeper" processing (elaborative
rehearsal's rewriting) is exactly where thread-memory collapse lives
(§4.3.3), the surface-level strategy (maintenance's verbatim selection)
wins instead. Porting a human memory strategy to an LLM pipeline does not
guarantee the same strategy stays best; which strategy wins can depend on
where the failure mode lives in the *implementation*, not just which
strategy is "deeper" in the cognitive-psychology sense.

#### 4.3.11 Seam padding closes novel's remaining gap — confirmed at full scale

§4.3.9 left one open question: `maintenance_extractive_dynamic` closed most
of RAG's advantage but still trailed it on novel (0.278 vs. 0.316, a 0.038
gap), hypothesized as the "seam" cost documented in `extractive.py`'s
`seam_report` — joining non-adjacent selected sentences drops the local
continuity between them, plausibly hurting a narrative-continuity genre more
than others. Tested by padding every selected sentence with its immediate
neighbor on each side (`window=1`, `select_sentences_windowed` in
`extractive.py`, `10` §13n) — run on the **full novel corpus (n=79)**, not a
pilot subset (the window=0 baseline for the same 79 questions was already on
disk from §13m's full-scale run, so only the window=1 arm needed a fresh
QA pass).

| variant | novel accuracy | avg tokens |
|---|---|---|
| `maintenance_extractive_dynamic`, window=0 (§13m) | 0.278 (22/79) | 538 |
| `maintenance_extractive_dynamic`, window=1 (§13n) | **0.316 (25/79)** | 858 |
| `raw_retrieval_adaptive` (RAG, reference) | 0.316 (25/79) | 977 |

Padding **exactly closes the gap to RAG** — same accuracy, same underlying
count (25/79), at 858 avg tokens vs. RAG's 977 (12% fewer). Question-level
detail: 6 questions flipped correct, 3 flipped incorrect (net +3), consistent
with a real but partial fix rather than a uniform improvement — some
individual answers still depend on content the ±1 window doesn't reach. With
this result, `maintenance_extractive_dynamic` now matches or exceeds RAG on
**two** of four genres (wiki: 0.880 vs. 0.850; novel: 0.316 vs. 0.316) while
remaining close on the other two (news, caselaw) — confirming the seam
hypothesis was the right diagnosis for most, though not all, of novel's
remaining gap.

#### 4.3.12 No-merge thread clustering — confirmed at full scale: fixes B, doesn't help A

A second 2026-08-14 idea, from reviewing `ThreadMemory`'s mechanics directly:
the "revise" branch that collapses (§4.3.3) replaces a thread's text with a
model's regeneration of "retrieved thread + current chunk" — but the
*clustering decision* behind it (C-DIC Eq. 4-5, which thread a chunk belongs
to) is pure cosine similarity and calls no model at all. It is not what
collapses. This motivated keeping the decision and discarding the
regeneration: `thread_grouping.cluster_into_threads` (new) groups chunks by
embedding similarity into threads, but a thread's content is just its
members' independently-produced texts concatenated — never merged or
rewritten. Two content sources feed this same clustering function:

- **`gist_threaded_nomerge`** — B's compressor run per chunk via
  `rehearse_elaborative_independent` (new) with *no* retrieved context at
  all, i.e. `format_elaborative_input(chunk, [])` — structurally cannot
  collapse, since there is nothing to collapse toward. Deliberately reuses
  the stage-2 checkpoint (not stage-1/one-shot): this is exactly the
  "no-context fallback" path §4.3.3's 20-chunk anti-collapse pilot already
  validated empirically (0 collapse, every chunk accurate), not a new,
  unvalidated one.
- **`maintenance_threaded_nomerge`** — A's already-computed dynamic-ratio
  extraction (§4.3.9), clustered the same way. Free to build: the content
  already existed; this only adds grouping plus a thread-level QA runner
  (Adaptive-k over each thread's anchor embedding, same mechanism every
  other condition already uses at chunk granularity).

Piloted across all 4 genres at n=15/genre (`10` §13o), then run to full
scale (n=579) once the pilot signal looked strong enough to justify it —
deliberately not novel-only, since this targets the general collapse
mechanism rather than novel's seam issue specifically:

| condition | wiki | news | novel | caselaw | overall | avg tokens |
|---|---|---|---|---|---|---|
| `gist_threaded_nomerge` (full scale) | 0.330 | 0.220 | 0.228 | 0.290 | **0.264** | 886 |
| `maintenance_threaded_nomerge` (full scale) | 0.650 | 0.410 | 0.241 | 0.300 | **0.390** | 1,519 |
| *(reference)* `gist_retrieval_adaptive` (full scale, collapsed) | 0.130 | 0.055 | 0.051 | 0.210 | 0.121 | 1,105 |
| *(reference)* `maintenance_extractive_dynamic` (full scale, §4.3.9/11) | 0.880 | 0.415 | 0.316 | 0.275 | 0.428 | 1,862 |
| *(reference)* `raw_retrieval_adaptive` (RAG, full scale) | 0.850 | 0.420 | 0.316 | 0.300 | 0.439 | 3,588 |

The pilot (n=15/genre: 0.317 / 0.467) overstated both conditions — consistent
with this project's repeated pattern of optimistic pilot signals (wiki
`lookup+retrieval` n=20→n=100, §4.2; the testing-effect n=1→n=25 reversal,
§6) — but the two conditions' full-scale results diverge in an informative
way rather than both simply regressing toward the mean:

**`gist_threaded_nomerge` is a confirmed, genuine win.** Every one of the 4
genres improved over `gist_retrieval_adaptive`'s collapsed baseline (wiki
+0.20, news +0.165, novel +0.177, caselaw +0.08), overall 0.264 vs. 0.121 —
roughly 2.2x, at 886 vs. 1,105 tokens (20% fewer). The core hypothesis
survives full-scale testing: the destructive merge step, not gisting itself,
was what caused §4.3.3's collapse. This is not competitive with RAG (0.439)
or `maintenance_extractive_dynamic` (0.428), but it is the first condition
in this project to meaningfully recover *generative* (rewritten) content
from collapse rather than side-stepping generation entirely.

**`maintenance_threaded_nomerge` is a genuine negative result.** It
underperforms the simpler `maintenance_extractive_dynamic` it clusters
(0.390 vs. 0.428) despite starting from the exact same per-chunk content.
Per genre: wiki drops sharply (0.880 → 0.650, −0.23), novel drops slightly
(0.278 → 0.241, −0.038), news is flat (−0.005), caselaw improves marginally
(+0.025). The wiki drop is the most informative data point — wiki's
dynamic-ratio variant already approaches R≈1 (documents shorter than the
32K-token target, so almost nothing gets cut, §4.3.9), meaning
Adaptive-k-over-chunks was already close to showing everything relevant.
Grouping into threads and then selecting *threads* instead adds a coarser,
lossier selection step on top of an already-near-optimal one — for A
specifically, thread-level grouping has nothing to fix and only a selection
step to lose accuracy through. The asymmetry between the two conditions is
itself the finding: no-merge thread clustering helps content that was
otherwise unusable (B) and hurts content that otherwise wasn't (A) — grouping
is a fix for collapse, not a general-purpose improvement.

#### 4.3.13 Extending attention past scoring — soft selection, confirmed not to beat RAG's frontier

§4.3.12's thread-level Adaptive-k is, mechanically, a *hard* cutoff on top of a
cosine-similarity score — the same QK-dot-product scoring step attention
uses, without the softmax-weighted combination step that follows it in a
real attention layer. A 2026-08-14 discussion asked whether carrying that
idea further — soft, weighted selection instead of a hard yes/no — would
recover more of the gap to RAG. Two designs were tried, both scoped to
`gist_threaded_nomerge`'s threads only (§4.3.12 found grouping itself hurts
A, so a smarter selection rule on top of a hurtful grouping wasn't expected
to help there):

**Design 1 — softmax-weighted word budget** (`attention_weighted_budget`):
every thread gets a temperature-scaled softmax weight over its similarity to
the query, and a total word budget is allocated across threads proportional
to that weight; each included thread's text is truncated to its share.

- First attempt used an absolute weight floor (`min_weight=0.02`) to zero out
  irrelevant threads. Bug: with N threads, a roughly uniform softmax gives
  each ~1/N weight — past N≈50 that already undercuts a fixed 0.02 floor, so
  *every* thread got zeroed and the prompt shipped with an effectively empty
  passage. Confirmed directly: prompt tokens for news/novel/caselaw (all
  hundreds of source chunks, correspondingly many threads) collapsed to
  ~110-260 (matching `closed_book`'s token count), and 37-63% of answers were
  literally "no passage provided." Fixed by making the floor relative to the
  top thread's weight instead of absolute (`min_weight_ratio`), which can
  never zero out every thread regardless of N.
- With that fixed, the *budget size* itself was still wrong: it reused
  `maintenance_extractive_dynamic`'s K=32,000-token target, which sizes a
  one-time whole-document compression pass, not a per-question context
  budget — a different layer. Result: 15,210 avg tokens/question (17.9x
  `gist_threaded_nomerge`'s 852, 5.8x RAG's 2,642 on the same paired
  subset), for accuracy still below RAG (0.417 vs. 0.517). Recalibrating to
  K=2,500 (RAG's own scale) fixed the token blowup but then **accuracy fell
  below the hard-cutoff baseline it was supposed to improve on** (0.25 vs.
  0.317) — truncating each included thread's content to fit its proportional
  slice cost more than the smoother selection boundary gained, most visibly
  on news and novel.

**Design 2 — nucleus (top-p) selection** (`nucleus_thread_selection`):
same softmax weighting, but instead of allocating a word budget per thread,
threads are ranked by weight and included **in full** (never truncated)
until cumulative weight crosses `top_p` — directly targeting Design 1's
diagnosed failure mode (truncation of the most-relevant thread).

| variant | overall accuracy | avg tokens | vs. hard cutoff (same 60-q subset) |
|---|---|---|---|
| `gist_threaded_nomerge` (hard cutoff, Adaptive-k) | 0.317 | 852 | — |
| weighted budget, K=32,000 (bug) | 0.417 | 15,210 | +0.10 acc, 17.9x tokens |
| weighted budget, K=2,500 | 0.25 | 2,834 | −0.07 acc, 3.3x tokens |
| nucleus, top_p=0.8 | 0.467 | 16,578 | +0.15 acc, 19.5x tokens |
| nucleus, top_p=0.3 | 0.367 | 6,415 | +0.05 acc, 7.5x tokens |
| *(reference)* `raw_retrieval_adaptive` (RAG) | 0.517 | 2,642 | +0.20 acc, 3.1x tokens |

*(All rows above are the same paired 60-question pilot subset — n=15/genre
— for a controlled comparison; not yet run to full scale.)*

Nucleus selection did remove Design 1's truncation problem and genuinely
improved accuracy over the hard cutoff at every `top_p` tested — the
mechanism itself is sound. But the improvement is driven almost entirely by
sharply higher inclusion counts (news/novel/caselaw have hundreds of source
chunks and correspondingly many threads with a *flat* similarity
distribution, so reaching even 30% cumulative softmax weight requires
including a large fraction of all threads), which drives tokens up faster
than accuracy: lowering `top_p` from 0.8 to 0.3 cut tokens by more than half
(16,578 → 6,415) but also gave back two-thirds of the accuracy gain (0.467 →
0.367). **At every point tested along this curve, RAG's own point (0.517
accuracy, 2,642 tokens) sits strictly above and to the left of it** — no
tested configuration of either soft-selection design reaches RAG's
accuracy-per-token frontier, let alone beats it.

**Honest read**: the attention-scoring analogy was directionally right (the
same QK-similarity score does drive every selection mechanism in this
project) but extending it into *soft, weighted* selection did not pay off
here, specifically because independently-generated gist threads produce a
similarity distribution too flat over too many candidates for a softmax-based
rule to stay both selective and cheap. Hard cutoffs (Adaptive-k's
biggest-gap heuristic) sidestep this because they key off the *shape* of the
sorted score curve rather than an absolute or cumulative threshold, and that
turns out to matter more than the smoothness of the selection boundary
itself. `gist_threaded_nomerge` (§4.3.12) remains the confirmed, full-scale
result from this whole thread-clustering line of investigation; neither
soft-selection variant is recommended for a full-scale run given this
pilot-scale pattern.

#### 4.3.14 A cognitive-psychology reading of the whole condition set

§4.3.3's diagnosis — one mechanism (`ThreadMemory`'s retrieve-and-replace)
was asked to do three functionally distinct jobs (track structure,
integrate new information, preserve verbatim detail) and collapsed under
the load — has a direct parallel in how human memory is organized: these
are not one system either. Hippocampal indexing/structural mapping,
prefrontal executive control over *when* to retrieve, and sensory/verbatim
detail are functionally separate, coordinating systems (a parallel
HippoRAG, Jimenez Gutierrez et al. 2024, arXiv:2405.14831, draws explicitly
for RAG). Reading each condition through this lens, rather than only as an
ablation, clarifies what each one is actually testing:

| condition | cognitive-psychology reading | why |
|---|---|---|
| `closed_book` | semantic memory alone | answers from parametric knowledge, no new encoding at all |
| `full_context` | working memory, unbounded | not a real human strategy — an oracle upper bound on what *would* be possible without capacity limits |
| `chunked_sequential` | primacy-limited working memory, no strategy | sees only the first few raw chunks, no compression or integration |
| `full_context_rehearsal_lookup[_retrieval/_adaptive]` | elaborative rehearsal, hippocampal-integration pathway | the pathway §4.3.3 found collapses — "revise a related thread with new information" is precisely the hippocampal binding operation that failed |
| `raw_retrieval_adaptive` (RAG) | cue-dependent recognition | matches a cue directly against the environment (the raw document) every time; builds no internal representation at all |
| `gist_retrieval_adaptive` / `_gistembed` / `_mnt128` | reconstructive recall | Bartlett's (1932) sense specifically — recall as an active reconstruction from a schema, prone to drift and distortion, not a verbatim trace; the collapse is that distortion made visible |
| `hybrid_gistselect_rawanswer` | schema-cued, veridical retrieval | uses a reconstructed schema only to decide *where* to look, then retrieves the original verbatim |
| `extractive_query_aware_adaptive` | retrieval practice (testing effect) | active retrieval at the moment of need, from the original material, per Roediger & Karpicke (2006) — this project's RQ1 answer |
| `structural_map_extractive` (§4.3.8) | specialized systems, not one mechanism | structural map (hippocampal indexing, built with no cross-chunk conditioning so it cannot collapse) + executive chunk selection (prefrontal) + extractive detail (sensory/verbatim) as three separate, coordinating pieces instead of one mechanism carrying all three — did not, on its own, recover accuracy (§4.3.8) |
| `maintenance_extractive_*` family (§4.3.9) | maintenance rehearsal | Craik & Lockhart's (1972) "shallow"/surface pole of levels-of-processing — verbatim sentence selection, no semantic reconstruction, no cross-chunk integration; the dynamic-ratio variant is this project's second RQ1 answer, and the strongest compression-family result overall |

This reading also sharpens what this project's headline result actually
says: it is not "compression loses to retrieval" in the abstract — it is
that *reconstructive* recall (gist) loses to both *recognition*-based
search (RAG) and *retrieval-practice*-style active recall (extractive
query-aware), across every genre tested, and the reconstructive pathway's
specific failure mode (collapse toward whatever was retrieved as context)
is now mechanistically understood rather than merely observed.

#### Summary

![Figure 6 — Accuracy by genre × condition, all tested conditions](docs/report_assets/fig6_accuracy_all_conditions.png)

![Figure 7 — Token cost by condition (log scale, mean over 4 genres)](docs/report_assets/fig7_tokens_all_conditions.png)

![Figure 8 — Efficiency frontier: accuracy vs. token cost](docs/report_assets/fig8_efficiency_frontier.png)

| condition | cognitive reading | overall accuracy | avg tokens | vs. RAG |
|---|---|---|---|---|
| `raw_retrieval_adaptive` (RAG) | recognition retrieval | 0.439 | 3,588 | — |
| `full_context_rehearsal_lookup_adaptive` (best §4.1 gist condition) | elaborative rehearsal | 0.339 | 25,551 | loses accuracy, 7x the tokens |
| `gist_retrieval_adaptive` | reconstructive recall | 0.121 | 1,105 | loses badly on every genre |
| `gist_retrieval_gistembed` | reconstructive recall (gist-cued) | 0.119 | ~1,100 | confirms selection wasn't the issue |
| `hybrid_gistselect_rawanswer` | schema-cued, veridical retrieval | 0.266 | 5,364 | loses accuracy *and* tokens |
| `gist_retrieval_adaptive_mnt128` | reconstructive recall, looser budget | 0.112 (pilot) | 1,539 (pilot) | worse on both axes |
| `extractive_query_aware_adaptive` | retrieval practice / testing effect | 0.366 | **680** | close accuracy (3/4 genres), 5.3x fewer tokens |
| `structural_map_extractive` | specialized systems | 0.142 | 4,929 | underperformed — worse than extractive on both axes |
| `maintenance_extractive_adaptive` (fixed R=0.3) | maintenance rehearsal | 0.325 | 1,567 | loses accuracy, but 2.3x fewer tokens |
| `maintenance_extractive_r50` (fixed R=0.5) | maintenance rehearsal | 0.368 | 2,011 | closer, still 1.8x fewer tokens |
| `maintenance_extractive_combined` (A + query-aware re-selection) | maintenance rehearsal, re-pruned | 0.299 | 467 | negative result — two lossy stages compound |
| `maintenance_extractive_dynamic` (dependent-ratio, 32K target) | maintenance rehearsal | **0.428** | **1,862** | **0.011 below RAG overall, ~half the tokens; beats RAG on wiki (0.880 vs 0.850), ties RAG on novel with §4.3.11's seam padding (0.316 vs 0.316)** |
| `gist_threaded_nomerge` (no-merge thread clustering) | reconstructive recall, thread-grouped | 0.264 | 886 | confirmed 2.2x `gist_retrieval_adaptive`'s collapsed baseline on every genre — merge, not gisting, was the failure |
| `maintenance_threaded_nomerge` (no-merge thread clustering) | maintenance rehearsal, thread-grouped | 0.390 | 1,519 | negative result — underperforms `maintenance_extractive_dynamic` it clusters (0.428), esp. wiki (−0.23) |

![Figure 9 — Maintenance-rehearsal (A) variants vs. RAG and extractive query-aware](docs/report_assets/fig9_maintenance_variants_accuracy.png)

![Figure 10 — Efficiency frontier including maintenance (A) variants](docs/report_assets/fig10_efficiency_frontier_with_maintenance.png)

Figure 8 makes the shape of the pre-maintenance result legible in one view: RAG and
extractive query-aware sit on the accuracy-cost frontier (extractive
dominating on cost for a modest accuracy concession); every
*reconstructive-recall* condition — regardless of which model built the
gist, which embeddings selected the chunk, or how loosely it was
compressed — sits strictly below and to the right of that frontier,
worse on accuracy despite in some cases (`hybrid_gistselect_rawanswer`)
costing *more* tokens than RAG itself. Figure 10 adds the maintenance
family to that same frontier: `maintenance_extractive_dynamic` sits closer
to RAG than any other tested alternative, at roughly half the token cost.

**Current honest read**: no condition built on *generative* (rewritten,
gisted) content **at full scale** beats RAG on accuracy, at any tested
compression aggressiveness, model size, or selection mechanism — §4.3.3
explains why with a specific, identified mechanism (thread-memory collapse)
rather than a general "compression is hard" appeal. Two conditions now have
a genuine, defensible advantage, and both are extractive:
`extractive_query_aware_adaptive` (RAG's own chunk selection, pruned
query-aware, kept verbatim — a cost/latency argument, 5.3x fewer tokens for
a 7.3pp accuracy cost) and `maintenance_extractive_dynamic` (verbatim
sentence selection at a document-length-dependent ratio, no query awareness
at all — the closest any tested condition comes to RAG's accuracy, 0.011
apart overall, at ~half the tokens, and now the only condition to match or
beat RAG outright on more than one genre — wiki and, with §4.3.11's seam
padding, novel too). Read together, the two extractive alternatives bracket
RAG from below in accuracy while asking for a fraction of its tokens; the
through-line across this entire investigation is that **verbatim content
survives, rewritten content collapses** — `structural_map_extractive`
(§4.3.8), the one attempt to add generative reasoning (topic labels) back
into an otherwise extractive pipeline, underperformed both, consistent with
that reading. §4.3.12 pushes on this same line from the *generative* side,
now confirmed at full scale: collapse is specifically caused by the
destructive merge step rather than generation itself, so removing only the
merge (not the generation) recovers gisted content — `gist_threaded_nomerge`
scores 0.264 vs. `gist_retrieval_adaptive`'s 0.121, improving on every genre
— without becoming an exception to "no generative condition beats RAG" (RAG
is still 0.439). §4.3.12 also found the same mechanism actively hurts
already-good extractive content (`maintenance_threaded_nomerge` underperforms
`maintenance_extractive_dynamic`, 0.390 vs. 0.428) — thread clustering is a
fix for collapse specifically, not a general-purpose accuracy improvement.

## 5. Methodology notes worth keeping

- **A real bug was found and fixed in the lookup mechanism.** The
  second-call ("show raw text, answer again") step originally reused the
  first call's full message history, which resent the entire per-chunk
  gist block (tens of thousands of tokens) a second time — burying the one
  useful addition (raw text) behind duplicate noise. Measured effect:
  accuracy on lookup-triggered questions was 0.0-0.30 pre-fix, *below* both
  `full_context` and the non-triggered path — i.e., asking for help made
  answers worse. Fixed by making the second call a fresh, short prompt
  (system + raw text + question only). See `10` §8-9 for the full account.
- **Per-genre tuning was tried, worked, and was deliberately abandoned** in
  favor of Adaptive-k (arXiv:2506.08479) — a hardcoded `{genre: cap}` table
  doesn't generalize to a genre outside the 4 tested. The adaptive version
  replaces every fixed-count expansion signal (position window, embedding
  top-K, per-genre cap) with one mechanism: keep whichever chunks score
  within a relative margin of the top similarity score for *this*
  question, decided from the score distribution's own shape, not a lookup
  table.
- **`full_context_with_map` (`11`) was explored and set aside on cost
  grounds**, not accuracy grounds — it showed a real pilot-scale win in 3 of
  4 genres, but is structurally more expensive than `full_context` itself
  (full raw text + a gist map, every call), which conflicts with this
  project's efficiency framing. Not deleted — kept as a documented
  alternative for a cost-insensitive setting.
- **Fixed (2026-08-12): the CaseHOLD *eval* corpus's `<HOLDING>` artifact.**
  Previously only the *train* corpora were stripped (`06b`); the eval
  corpus was flagged but left unfixed. Verified the exact removal pattern
  (`r" ?\(<HOLDING>\)"`) against 20 train corpora before/after backups
  (17/20 exact match; the other 3 differ only at a document-separator
  edge case that doesn't occur in the eval file), applied it to
  `caselaw_eval_corpus.txt` (395 occurrences removed, 4,733 chars), and
  recomputed every question's `evidence_char_pos`/`evidence_word_pos` in
  `caselaw_eval_questions.csv` by tracking exactly how much text was
  removed before each position — not re-deriving it from excerpt
  boundaries. All 200 questions validated to land exactly at a document
  separator post-strip; a spot-check of 4 questions across the corpus
  confirmed identical surrounding text at the shifted position. Old
  positions kept as `evidence_char_pos_old`/`evidence_word_pos_old` for
  audit. **Caselaw was re-evaluated on all three lookup conditions
  post-fix — accuracy barely moved anywhere**
  (`lookup` 0.250→0.255, `+retrieval` 0.250→0.235,
  `+adaptive` 0.300→0.275, all within noise) **— the `<HOLDING>` artifact
  does not explain caselaw's "more retrieved context hurts" anomaly
  (§4.2).** All three re-evaluations are complete as of 2026-08-13; §4.1's
  and §4.3's tables use the corrected corpus throughout.

## 6. Scope: B fully tested, A revived 2026-08-14, A→B and full C out of scope

Per the experimental design doc's full grid (strategy {A maintenance, B
elaborative, A->B} x testing-effect {no-C, +C} x lookup {no, yes}, 2
baselines + 12 intervention cells x 4 genres): **B**, with and without
lookup variants, has been fully tested (§4). **A (maintenance rehearsal)
was dropped as of 2026-08-12, then explicitly revived on 2026-08-14** once
§4.3.2's extractive/query-agnostic diagnosis identified it as the one cell
of that 2x2 the project hadn't tested — see §4.3.9 for the full result
(its dynamic-ratio variant is this project's second RQ1-supporting
finding, and the strongest compression-family result overall). The A→B
sequential combination and C's full 336-document scale-up remain out of
scope. The lookup mechanism itself is written strategy-agnostic
(`_lookup_window_indices` / the adaptive selection don't assume B
specifically), so combining A with lookup/Adaptive-k required no new
mechanism — only reusing the already-trained checkpoint from `04`/`05`.

**C (testing effect) — implemented and piloted at n=25, reverses the n=1 signal.**
The original design trained a standalone QG model (`08`/`09`, SQuAD
reverse-QA) to generate self-test questions. As actually implemented
(`06c_testing_effect_stage2_prep.ipynb`, `src/pipeline/teacher.py`'s
`curate_document_threaded_with_testing`), C instead reuses the real
evidence-linked questions `06b`'s curation already has (`answers_by_chunk`)
as the in-loop test material, folded directly into stage-2 curation as a
probe → corrective-retry → verbatim-fallback loop — grounded in Roediger &
Karpicke (2006) and Karpicke & Roediger (2008)'s testing-effect literature,
not a synthesized-question approach. `08`/`09` are effectively superseded
by this approach.

A single-document pilot (news, 12 chunks) initially looked strong: drop
went from +0.040 (plain) to -0.005 (with testing). **That reversed at
n=25 (7 news, 6 caselaw, 6 narrativeqa, 6 wiki), same documents both ways:**

| | early | late | drop |
|---|---|---|---|
| plain curation (`06b`) | 0.387 | 0.458 | **-0.071** (no collapse) |
| **testing-effect curation** | **0.863** | 0.676 | **+0.186** (collapse) |

The mechanism clearly *does* something — of 142 tested chunks, only 21.1%
passed their probe on the first try; 64.8% passed after a corrective retry
(133 retry calls spent), 14.1% exhausted retries and fell back to verbatim.
That effort buys much higher **absolute** retention early (0.863 vs 0.387)
but a **steeper decline** with age than plain curation shows — the opposite
of the intended effect on the metric this project actually cares about.
Read together, this looks like the corrective-retry loop optimizing each
gist for *passing its own probe right now* at some cost to how well that
gist survives being read again several revisions later — a plausible
cognitive-science parallel is cramming: strong immediate recall, weaker
durability. This project has one prior confirmed case of a pilot number
reversing at scale (§4.2's wiki `lookup+retrieval`, n=20→n=100); this is a
second, at 12 chunks→25 documents. **Treat the n=25 numbers as the current
best estimate, not final** — not yet run at `06b`'s full 336-document
scale.

## 7. Status (updated 2026-08-14)

- **Headline: pure RAG (no rehearsal, no teacher, no compression) beats
  every *generative* rehearsal-family condition on every genre**, confirmed
  at full scale (n=579) — 0.439 overall accuracy vs. 0.339 for the best
  gist-based condition, at 7x fewer tokens. See §4.3.1. **But an extractive
  (verbatim, no-rewriting) form of rehearsal comes within 0.011 of matching
  it overall, at ~half the tokens, and now matches or beats it outright on
  2 of 4 genres** — `maintenance_extractive_dynamic`, §4.3.9 (beats RAG on
  wiki, 0.880 vs. 0.850) + §4.3.11 (ties RAG exactly on novel once seam
  padding is added, 0.316 vs. 0.316, confirmed at full scale n=79). The
  headline is no longer "compression loses to retrieval" in the abstract;
  it's "*rewritten* content collapses, *verbatim* content doesn't"
  (§4.3.9, §4.3.14).
- **The loss is content, not architecture, not selection, and confirmed not
  model size — root cause identified: thread-memory collapse** (§4.3.2-3):
  swapping only the content shown (gist vs. raw) at RAG's own single-call
  architecture loses on every genre; reselecting via gist embeddings
  instead of raw-text embeddings changes almost nothing; answering from
  raw text after selecting via gist embeddings loses on both accuracy and
  tokens. A 70B teacher run through the identical thread-retrieve-replace
  mechanism collapses the same way the student does — not on one
  cherry-picked example but systematically (all 6 chunks checked side by
  side for one genre converge on near-identical text). A controlled
  minimal test with the "current chunk"/"related thread" fields swapped
  confirms both model sizes reproduce whichever content sits in "related
  thread" and ignore "current chunk," regardless of which actual content
  occupies which slot. Since the student's training target *is* the
  teacher's (already-collapsed) output, the student learned this failure
  faithfully from its training data — retraining it will not fix this;
  the flaw is upstream, in teacher curation.
- **2025 literature (RAPTOR, EXIT, LongLLMLingua) explains why** (§4.3.4):
  this project's B is abstractive and query-agnostic by design — both
  properties the literature already documents as disadvantaged relative to
  extractive, query-aware alternatives for this kind of task.
- **One condition shows a genuine, defensible advantage**:
  `extractive_query_aware_adaptive` (§4.3.5) — RAG's own chunk selection,
  pruned to query-relevant sentences *after* the question is known, kept
  verbatim. 5.3x fewer tokens and lower latency than RAG for a 7.3pp
  accuracy cost, competitive within 5-7pp on 3 of 4 genres; novel is a
  clear remaining weak point.
- **Length-stress test rules out the context-window-relief hypothesis in
  the tested range** (§4.3.6) — gist size scales with whole-document
  length; RAG's retrieved-token budget scales with relevant content only,
  which stays roughly flat regardless of corpus size. The hypothesized
  advantage for compression at longer documents did not appear at
  1,000-10,000 words; if anything it reverses.
- **Relaxing the compression cap does not recover accuracy — confirmed,
  rejected** (§4.3.7): `gist_retrieval_adaptive_mnt128` (64→128 tokens)
  scored *worse* on both accuracy (0.212→0.112) and tokens (902→1,539) at
  nb10's actual eval scale. Consistent with the collapse diagnosis above —
  more budget just lets the mechanism reproduce more of the wrong content.
- **Anti-collapse mitigation: partial, still being validated**
  (`curate_document_threaded_anticollapse`, §4.3.3) — a collapse-detection
  + corrective-retry loop was built; in a 6-chunk trial, the corrective
  retries themselves never succeeded (0/10), but a final no-context
  fallback did recover accurate per-chunk content every time. This fixes
  *correctness* but likely gives up genuine cross-chunk integration in the
  process. A 20-chunk, retry-skipped re-run is in progress to confirm this
  holds before deciding whether a full corpus re-curation is worth the
  cost.
- **Adaptive-k vs. per-genre cap on caselaw** (§4.2): tuning
  `ADAPTIVE_RELATIVE_THRESHOLD` did not close the gap — a targeting-quality
  limitation, not a chunk-count one.
- **Bigger compressor (flan-t5-base)**: raised absolute retention but not
  downstream QA accuracy in a pilot (§3). Reported negative result, not
  pursued further.
- **C (testing effect)**: implemented, piloted at n=25 across all 4
  genres (§6) — trades higher early retention for steeper age-related
  decline versus plain curation. Real finding, not yet run at full scale;
  deprioritized behind §4.3's investigation.
- **`<HOLDING>` eval-corpus artifact — fixed and fully checked (§5)**: all
  three lookup conditions re-evaluated post-fix, accuracy moved <0.03
  everywhere — the artifact does not explain caselaw's "more context
  hurts" pattern.
- **20-chunk anti-collapse check: correctness fixed, integration not**
  (§4.3.3) — with corrective retries skipped (0/15 succeeded across two
  trials) and going straight to the no-context fallback on detected
  collapse, all 20 chunks' gists came back accurate and topically distinct
  (0 collapse in the final output). But 15/19 context-bearing chunks only
  got there via the fallback — i.e., by giving up on integration and
  regenerating independently. The mechanism now reliably produces *correct*
  gists; it does not yet reliably produce *elaborative* ones. Whether that
  distinction matters for downstream QA accuracy is untested — would need a
  real QA run on gists built this way, not yet done (real API cost, and at
  ~2h/20 chunks under today's endpoint conditions, a full corpus run is a
  multi-hour commitment).
- **`structural_map_extractive` confirmed at full scale — underperformed**
  (§4.3.8): splitting the job §4.3.3 found one mechanism collapsing under
  into three specialized pieces (structural map / executive chunk
  selection / extractive detail) scored 0.142 overall at 4,929 avg
  tokens — worse than `extractive_query_aware_adaptive` on both axes.
  Negative result: cognitive-localization motivation alone did not recover
  accuracy; see the next bullet for the piece that did.
- **Maintenance rehearsal (A) revived and confirmed at full scale — the
  strongest compression-family result in the project** (§4.3.9): dropped
  from scope 2026-08-12, revived once §4.3.2's diagnosis flagged it as the
  untested extractive+query-agnostic cell. Best variant
  (`maintenance_extractive_dynamic`, document-length-dependent compression
  ratio) scores 0.428 overall vs. RAG's 0.439 — a 0.011 gap — at roughly
  half RAG's tokens (1,862 vs 3,588), and beats RAG outright on wiki (0.880
  vs 0.850), the only condition in the project to do so on any genre. A
  combined variant (A + query-aware re-selection) is a clear negative
  result by contrast (0.299, driven by a novel-genre collapse to 0.101) —
  two lossy selection stages compounded rather than complementing each
  other. This is this project's second RQ1-supporting finding, alongside
  §4.3.5's extractive_query_aware_adaptive — and the through-line across
  both is the same: verbatim content survives, rewritten content collapses.
- **Seam padding closes novel's remaining gap to RAG — confirmed at full
  scale** (§4.3.11): padding every selected sentence with ±1 neighbor
  (`window=1`) took `maintenance_extractive_dynamic`'s novel accuracy from
  0.278 to 0.316, exactly matching RAG's novel accuracy (25/79 both), at
  858 avg tokens vs. RAG's 977. Run on the full 79-question novel corpus,
  not a pilot. `maintenance_extractive_dynamic` now matches or beats RAG on
  2 of 4 genres (wiki, novel).
- **No-merge thread clustering — confirmed at full scale, asymmetric result**
  (§4.3.12): keeps `ThreadMemory`'s clustering *decision* (pure cosine
  similarity, not itself the collapse source) while discarding the
  destructive merge step that causes collapse. Two conditions —
  `gist_threaded_nomerge` (B gisted per-chunk with no retrieved context, so
  it cannot collapse, then clustered) and `maintenance_threaded_nomerge` (A's
  existing dynamic-ratio content, clustered the same way) — piloted at
  n=15/genre, then run to full scale (n=579). Pilot numbers (0.317 / 0.467)
  both overstated the effect, as this project's pilots often do, but the two
  conditions diverged rather than both simply regressing: `gist_threaded_nomerge`
  confirmed at 0.264 — every genre improved over `gist_retrieval_adaptive`'s
  collapsed baseline (0.121), a genuine, if partial, recovery from collapse.
  `maintenance_threaded_nomerge` confirmed at 0.390 — *worse* than the
  `maintenance_extractive_dynamic` content it clusters (0.428), driven mainly
  by a sharp wiki drop (0.880 → 0.650) where the ungrouped version was
  already near-optimal (R≈1). Reading: thread clustering fixes collapsed
  content and actively hurts content that wasn't broken to begin with — a
  fix for collapse specifically, not a general-purpose accuracy lever.
- **Soft (attention-style) selection tried on top of `gist_threaded_nomerge`
  — confirmed not competitive with RAG, not pursued further** (§4.3.13):
  replacing the thread-level hard cutoff with a softmax-weighted budget or
  nucleus (top-p) selection improved accuracy over the hard cutoff in every
  configuration tested, but always at a token cost RAG itself doesn't need
  to pay — at every point tested (weighted budget K=2,500: 0.25/2,834 tok;
  nucleus top_p=0.3: 0.367/6,415 tok; top_p=0.8: 0.467/16,578 tok), RAG's own
  point (0.517/2,642 tok) sits strictly above and to the left. Two real bugs
  were found and fixed along the way (an absolute weight floor that zeroed
  every thread once thread count passed ~50; a budget target sized for
  whole-document compression reused where a per-question budget was needed)
  — worth recording since both are easy mistakes to repeat, not just this
  project's. `gist_threaded_nomerge`'s hard cutoff remains the best
  confirmed result on this branch.
- **Not yet run to full scale**: the teacher-gist ceiling test with a
  *working* (non-collapsing) curation mechanism (real API cost, ~2.9h+
  estimated, more with retry overhead), and testing-effect's full
  336-document curation (§6) — both legitimate follow-ups. The two
  soft-selection variants above were deliberately *not* promoted to full
  scale given the pilot-scale pattern (§4.3.13).

## 8. Framing candidates for the writeup

**Decided (2026-08-15): B + C combined**, layered rather than either alone —
B (verbatim survives, rewritten collapses) as the precondition layer, C
(Transfer-Appropriate Compression) as the layer explaining which
verbatim-preserving strategy wins where. Reasoning: B alone explains
*whether* a strategy works but reads as a systems/mechanism paper with the
cognitive-psychology framing reduced to vocabulary; C alone is the
strongest theoretical hook (a real, citable theoretical dispute — Craik &
Lockhart 1972 vs. Morris, Bransford & Franks 1977 — resolved by data, not
just cited) but on its own leaves the collapse-mechanism work (the
project's most concrete, hard-won result) without a home. Layering both
keeps the mechanism work as necessary groundwork and lets the
genre-dependent TAP story carry the theoretical payload, which the
portfolio goal (a cognitive-psychology + AI lab audience) weighted heavily.
§4.3.13's soft-selection result stayed in as a deliberate contrast at the
end of the TAP discussion — it shows that *matching* selection to structure
matters, but the smoothness of the selection boundary on its own does not.

**Title decided**: "Fit Over Depth: Transfer-Appropriate Compression for
Long-Context LLM QA" (retires the earlier "Rehearse, Then Recall" — that
title centers elaborative rehearsal, the specific strategy RQ2a rejects, so
keeping it would point the title at the paper's negative result rather than
its positive one). "Fit Over Depth" was chosen over "Deeper Isn't Better"
deliberately: the paper's actual claim is that there is no universal
depth/surface ranking at all (TAP), not that shallow beats deep — a
directional title would misstate the conclusion.

This is implemented in **`11_숏페이퍼 초안.md`/`11_숏페이퍼 초안 English.md`**
(Obsidian): a new opening paragraph reframes the introduction around
inconsistent results in prior work (C-DIC/ReadAgent/RAPTOR positive within
their own designs vs. EXIT/LongLLMLingua showing other configurations of
the same strategy family losing to plain retrieval), RQ2 splits into RQ2a
("does it work" — decided by rewriting vs. preservation) and RQ2b ("which is
better" — decided by TAP), §2.4.4 gained a genre × strategy table (Table 8)
and the Morris/Bransford/Franks resolution, and the conclusion closes the
loop back to the introduction's literature-inconsistency framing. See
**`14_프레이밍 재구성 — 붕괴(1층)+전이적합압축(2층).md`** (Obsidian) for the full
design blueprint this was built from, including exactly which existing
result maps to which layer.

The candidates below are kept as the record of that decision, not
superseded — B and D remain accurate readings of the same data if the
audience or venue changes. The project's central claim was reframed several
times as evidence accumulated (query-agnostic gisting → RQ1/RQ2 → the
asymmetric thread-clustering + soft-selection results → RQ2a/RQ2b), and it
grew big enough that different subsets of findings would have supported
genuinely different papers, not just different titles for the same one.
Each candidate below names what it would put at the center, what it would
foreground from §4, and what it would need to cut or demote to a footnote
to stay inside a 3-6p short paper.

**A. RQ1/RQ2 (current draft's framing).** Two nested questions: does
applying a human memory strategy help (RQ1), does the specific strategy
first hypothesized — elaborative rehearsal's gist — help (RQ2). RQ2
rejected, RQ1 supported by two alternate strategies (testing effect,
maintenance rehearsal). *Foregrounds*: §4.3.5, §4.3.9, §4.3.11. *Cuts*:
§4.3.12's asymmetric result and §4.3.13's soft-selection tangent don't map
cleanly onto either RQ and would stay as brief forward-looking notes.
*Effort*: lowest — abstract/intro/conclusion already written this way.
*Risk*: RQ1's support is now three unrelated strategies (testing effect,
maintenance, no-merge gist) succeeding for three different reasons — reads
increasingly like "something worked" rather than one sharp claim.

**B. Verbatim survives, rewritten collapses.** One mechanistic claim,
argued from every angle in §4: any mechanism that *regenerates* content —
elaborative rehearsal's rolling merge, thread-memory's revise branch —
collapses regardless of model scale (§4.3.3); any mechanism that keeps
content verbatim or generates it context-free — extraction (§4.3.5, §4.3.9),
no-merge clustering (§4.3.12), context-free gisting (§4.3.12) — doesn't,
independent of *how* it selects that content. §4.3.13's soft-selection
result even strengthens this reading: it kept the no-regeneration property
and never collapsed, it just wasn't efficient — a different, non-fatal kind
of failure. *Foregrounds*: §4.3.3 (the mechanism), §4.3.9/11/12 (four
independent confirmations from different angles). *Cuts*: little — this is
the framing most of §4 already supports without forcing. *Effort*: medium —
needs the intro/abstract rewritten around the mechanism rather than the two
RQs, but no new experiments. *Risk*: reads more like a systems/mechanism
paper than a cognitive-psychology one; the human-memory-strategy framing
becomes motivation/vocabulary rather than the paper's own claim.

**C. Transfer-Appropriate Compression.** No single strategy is universally
best; which one wins depends on whether its selection granularity and
criterion match the document's structure and the task's retrieval demands —
Transfer-Appropriate Processing (Morris, Bransford & Franks 1977) rather
than Craik & Lockhart's (1972) uniform depth-of-processing hierarchy, which
§4.3.9's own tension (maintenance beats elaborative here, opposite the
human-cognition prediction) already gestures at. Short docs favor near-zero
compression (wiki, §4.3.9); long narrative docs favor coarse whole-chunk
retrieval over fine-grained extraction unless local continuity is restored
(novel, §4.3.5 vs. §4.3.11); selection *criterion* (query-similarity vs.
importance) matters as much as *granularity*. *Foregrounds*: the per-genre
breakdown across §4.3.5/9/11, read together rather than condition-by-
condition. *Cuts*: §4.3.13's soft-selection result isn't genre-specific and
would need to be a secondary note, not central evidence. *Effort*:
medium-high — needs a genre-comparison table/figure as the paper's central
evidence, not yet built as such (draft only sketched this in chat, not
written into the paper). *Risk*: the theoretical hook is strong but
underspecified without dedicated experiments designed to isolate criterion
from granularity rather than reading it post hoc off condition ablations
built for other purposes.

**D. Diagnosis-first / methods framing: finding and fixing thread-memory
collapse.** Centers the diagnostic *process* rather than any single winning
strategy: noticing an implausible result, isolating architecture from
content from model scale via controlled ablation (§4.3.2-3, already
visualized as the diagnosis funnel, Fig. 3), identifying one exact
mechanism, then testing that diagnosis from three independent angles
(extraction, context-free generation, no-merge clustering) that all point
the same way. §4.3.13's bug-hunting inside the soft-selection experiment
(an absolute weight floor silently failing at scale, a budget target reused
across a semantic-layer mismatch) is genuinely on-theme here rather than a
tangent. *Foregrounds*: §4.3.2-3 as the paper's centerpiece, §4.3.9/11/12/13
as confirmatory follow-through. *Cuts*: needs less genre-by-genre nuance
than C, less "which strategy is cognitively X" framing than A/B. *Effort*:
medium — the diagnostic content already exists in full (§2.4.3 of the
current draft covers most of it) but isn't currently the paper's spine.
*Risk*: strongest for a systems/ML audience, weaker pitch for a venue that
wants the cognitive-psychology angle to be load-bearing rather than
motivating color.

**Not a standalone candidate, but worth keeping regardless of which is
chosen**: §4.3.13's specific negative result (hard cutoffs beat every
softmax-weighted selection rule tried, because independently-generated
content produces similarity scores too flat over too many candidates for a
cumulative or proportional threshold to stay both selective and cheap) is a
clean, self-contained finding on its own and fits as a subsection under B or
D without much adaptation.
