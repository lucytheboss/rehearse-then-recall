# Rehearse, Then Recall — Progress Report

_Last updated: 2026-08-07_

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

| | early (age 0-2) | late (age 9-11) | drop |
|---|---|---|---|
| stage 1 (one-shot, driven incrementally) | 0.055 | 0.070 | -0.015 |
| **stage 2 (rolling, threaded)** | **0.355** | **0.244** | **+0.111** |
| teacher (upper bound) | 0.383 | 0.417 | -0.034 |

![Figure 4 — Stage-2 checkpoint retention vs. teacher](docs/report_assets/fig4_retention_curve.png)

Stage 2 is 5-6x better than stage 1 at every age, but still has a real,
positive collapse signature the teacher doesn't — a quality ceiling every
downstream condition below inherits, since none of them retrain the
compressor itself.

## 4. Evaluation results

![Figure 1 — Accuracy by genre x condition, full scale](docs/report_assets/fig1_accuracy_full_scale.png)

### 4.1 Full-scale, validated (n = 579: wiki 100, news 200, novel 79, caselaw 200)

| condition | wiki | news | novel | caselaw | avg tokens (wiki/news/novel/caselaw) |
|---|---|---|---|---|---|
| closed_book | 0.280 | 0.040 | 0.025 | 0.335 | ~110 (all genres) |
| **full_context** | **0.810** | **0.395** | **0.494** | **0.360** | 19,566 / 65,035 / 81,139 / 87,423 |
| chunked_sequential (raw, 4-chunk budget) | 0.190 | 0.010 | 0.063 | 0.225 | — |
| chunked_sequential_rehearsal (B alone, 4-chunk budget) | 0.230 | 0.010 | 0.051 | 0.185 | — |
| full_context_rehearsal_lookup (gist map + 1-chunk lookup, cap 20) | 0.370 | 0.105 | 0.203 | 0.250 | 7,940 / 26,627 / — / 25,652 |
| **full_context_rehearsal_lookup_retrieval** (+ embedding-similarity targeting) | **0.620** | **0.220** | **0.291** | **0.245** | ~50-70% fewer tokens than full_context in every genre |

**Reading this table**: `chunked_sequential_rehearsal` tests B in a regime
where it structurally can't win (same 4-raw-chunk budget as the
uncompressed baseline — compression has nothing to gain there). The real
test starts at `full_context_rehearsal_lookup`: show every chunk's gist at
once, let the model ask for raw text on 1+ specific chunks. Adding
embedding-similarity retrieval as a second targeting signal
(`+retrieval`) is a clear, validated win over plain lookup in every genre
except caselaw — but no condition here beats `full_context` at full scale.
Pooled across all 579 questions: full_context 0.515, lookup 0.232,
lookup+retrieval 0.344. Token savings (50-70%) are real and robust
regardless of the accuracy gap.

![Figure 2 — Token cost by genre x condition, full scale](docs/report_assets/fig2_tokens_full_scale.png)

### 4.1.1 Accuracy per token

![Figure 5 — Accuracy per token, full scale](docs/report_assets/fig5_accuracy_per_token.png)

| genre | full_context | chunked_seq | chunked_seq+rehearsal | lookup | lookup+retrieval |
|---|---|---|---|---|---|
| wiki | 0.414 | 0.426 | **1.317** | 0.425 | 0.636 |
| news | 0.061 | 0.033 | 0.054 | 0.040 | **0.082** |
| novel | 0.061 | 0.247 | **0.286** | 0.065 | 0.093 |
| caselaw | 0.041 | 0.600 | **0.775** | 0.098 | 0.093 |

_(accuracy per 10,000 tokens; `closed_book` excluded — near-zero tokens makes its ratio a meaningless outlier)_

**Read this one carefully — it rewards being cheap more than it rewards
being good.** `chunked_seq+rehearsal` tops every genre here, including
caselaw at 0.775, but caselaw's *absolute* accuracy for that condition is
0.185 (§4.1) — worse than `closed_book`'s 0.335, i.e. worse than guessing
from the question alone. A condition that answers badly using very few
tokens still scores well on this metric, because the denominator collapses
faster than the numerator does. Use this figure to compare *among
conditions that already clear a usable accuracy bar* (e.g. `lookup+retrieval`
beats `full_context` on this metric in news, novel, and caselaw while also
being reasonably accurate in absolute terms) — not as a standalone ranking.

### 4.2 Pilot-scale, provisional (n = 20/genre — **not yet confirmed at full scale**)

![Figure 3 — Per-genre cap and Adaptive-k vs. full-scale baselines](docs/report_assets/fig3_pilot_comparison.png)

| condition | wiki | news | novel | caselaw |
|---|---|---|---|---|
| full_context_rehearsal_lookup_retrieval_pergenre (cap 5 for caselaw, 20 elsewhere) | 0.750 | 0.300 | 0.450 | **0.450** |
| full_context_rehearsal_lookup_adaptive (Adaptive-k, no per-genre tuning) | 0.750 | 0.300 | 0.450 | 0.300 |

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

**Adaptive-k does not yet match the per-genre cap on caselaw specifically**
(0.300 vs 0.450, same n=20) — the one genre the whole per-genre-vs-adaptive
question was actually about. Elsewhere the two are identical, which makes
sense: `ADAPTIVE_RELATIVE_THRESHOLD` hasn't been tuned at all yet (first
value tried), while the per-genre cap came from three rounds of evidence.
This is not a reason to prefer the per-genre table — it's a reason to tune
Adaptive-k's one threshold before concluding it can't reach the same
result generalizably. See §7.

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
- **Known open caveat**: the CaseHOLD *eval* corpus still contains the
  literal `<HOLDING>` placeholder token (stripped from the *train* corpora
  in `06b` only — fixing the eval side was explicitly deferred). All
  caselaw numbers above are read from text with that artifact present.

## 6. What's not built yet

Per the experimental design doc's full grid (strategy {A maintenance, B
elaborative, A->B} x testing-effect {no-C, +C} x lookup {no, yes}, 2
baselines + 12 intervention cells x 4 genres): only **B alone**, with and
without lookup variants, has been tested. A (maintenance), A->B, and C
(testing-effect/QG) are unbuilt. The lookup mechanism itself is written to
be strategy-agnostic (`_lookup_window_indices` / the adaptive selection
don't assume B specifically), so extending to A/A->B should mean feeding a
different chunk representation in, not rebuilding the mechanism.

## 7. Next steps, in priority order

1. Confirm `full_context_rehearsal_lookup_retrieval_pergenre` and
   `_adaptive`'s caselaw result at full scale (200 questions) — the single
   most promising unconfirmed number in this report.
2. If Adaptive-k holds up, retire the per-genre-cap condition from any
   paper claim (kept only as a "this doesn't generalize, here's what does"
   contrast) and adopt adaptive selection as the default mechanism.
3. Consider using teacher-generated gists (`curate_document_threaded`,
   already built, never repurposed for this) in place of the collapse-prone
   stage-2 checkpoint's gists for the map/lookup layer specifically — a
   one-time precompute cost that could raise every condition's ceiling at
   once, since none of the tuning done so far touches gist quality itself.
4. Fix the caselaw eval corpus's `<HOLDING>` artifact before any final
   reported caselaw number.
