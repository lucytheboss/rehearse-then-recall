# rehearse-then-recall

## Project overview

Does rehearsing a long document as you read it — compressing and
periodically revising a running memory of it, the way spaced-repetition/
testing-effect research suggests humans retain material — help an LLM
answer questions about that document better than either reading it raw
(`full_context`) or reading nothing (`closed_book`)?

Combines two prior methods: **C-DIC** (arXiv:2606.12411)'s retrieve/revise/
write-back loop over a multi-slot thread memory, and **ReadAgent**
(arXiv:2402.09727)'s gist-then-lookup reading strategy. A small seq2seq
model is trained to rewrite each chunk of a document into a running,
threaded gist as it goes; at QA time, the model reads the gists (cheap) and
can ask for a specific chunk's raw text (ReadAgent-P-style lookup) before
answering.

Evaluated across 4 genres — wiki, news, novel (narrativeqa), caselaw
(CaseHOLD) — comparing gist-based reading strategies against
`closed_book`/`full_context` baselines on accuracy, token cost, API calls,
and accuracy-per-token (an efficiency-adjusted view — see
[ANALYSIS_REPORT.md](ANALYSIS_REPORT.md) §4.1.1 for why it needs to be read
alongside absolute accuracy, not alone).

**Current headline finding (2026-08-15)**: prior work applying human
memory-reinforcement strategies to LLM pipelines reports inconsistent
results, and this project reproduced that inconsistency internally before
explaining it. A plain retrieval-augmented baseline with no rehearsal, no
compression, and no teacher model anywhere (`raw_retrieval_adaptive`) beats
every *rewriting*-based rehearsal condition on every genre, at a fraction of
the token cost — traced to a **thread-memory collapse**: given a retrieved
"related" thread, both the trained student (t5-small) and a 70B teacher
reproduce that thread near-verbatim and ignore the new chunk's own content,
confirmed with a controlled test that rules out a code bug or positional
artifact and reproduces independent of model size. Since the student's
training target *is* the teacher's own (already-collapsed) output,
retraining the student cannot fix this — the flaw is upstream, in teacher
curation itself.

Two strategies that never rewrite content sidestep the collapse entirely
and close most or all of the gap to RAG: **extractive, query-aware sentence
pruning** (retrieval practice / testing-effect framing — 5.3x fewer tokens,
34% lower latency than RAG for a modest accuracy cost) and **maintenance
rehearsal with a document-length-adaptive compression ratio** (verbatim
sentence selection — ties or *beats* RAG outright on the wiki genre, at
roughly half the tokens). Which of these two wins is itself genre-dependent
rather than universal — consistent with transfer-appropriate processing
(Morris, Bransford & Franks, 1977) rather than a single depth-of-processing
hierarchy. A follow-up mechanism (no-merge thread clustering) recovers
collapsed *generative* content substantially (2.2x the collapsed
baseline) but actively hurts already-good extractive content; a further
attempt to extend attention-style soft weighting into the selection rule
itself was tested and confirmed **not** to beat RAG's accuracy-per-token
frontier at any tested configuration — a clean negative result kept in the
record rather than discarded. See
**[ANALYSIS_REPORT.md](ANALYSIS_REPORT.md) §4.3** for the full diagnostic
account and **§8** for how this is being framed for the paper.

<p align="center">
  <img src="docs/report_assets/fig10_efficiency_frontier_with_maintenance.png" width="600" alt="Efficiency frontier: accuracy vs. tokens consumed, RAG vs. rehearsal-family conditions including maintenance rehearsal variants">
</p>
<p align="center"><sub>Accuracy vs. token cost (log scale, mean over 4 genres). Every <em>generative</em> (rewritten) condition sits below RAG's frontier; the maintenance-rehearsal family (extractive, verbatim) sits closest to it, with the dynamic-ratio variant nearly matching RAG at about half the tokens.</sub></p>

<p align="center">
  <img src="docs/report_assets/fig6_accuracy_all_conditions.png" width="700" alt="Accuracy by genre x condition, all tested conditions">
</p>
<p align="center"><sub>Accuracy by genre × condition (full scale, n=579). Rewriting-based (gist) conditions collapse across every genre; extractive alternatives track RAG closely, with the winner among them varying by genre.</sub></p>

## Paper / presentation links
- Paper: TBD (working title: "Fit Over Depth: Transfer-Appropriate Compression for Long-Context LLM QA")
- Slides: TBD
- Progress report: [ANALYSIS_REPORT.md](ANALYSIS_REPORT.md)

## Structure

```
rehearse-then-recall/
├── README.md
├── ANALYSIS_REPORT.md              # current results, findings, open questions
├── LICENSE
├── .gitignore
├── pyproject.toml / requirements.txt
│
├── configs/
│   ├── chunking.yaml               # chunk min/max words, semantic pagination
│   ├── curation.yaml                # stage-2 teacher curation (model, cost, doc shaping)
│   └── importance_filter.yaml       # embedding config shared across chunking/rehearsal/retrieval
│
├── data/
│   ├── raw/                        # raw data (gitignored)
│   ├── processed/                  # train/eval corpora per genre (caselaw, news, wiki, narrativeqa)
│   ├── sample/                     # eval question sets (CSV) per genre
│   └── scripts/                    # corpus builders + synthetic QA generation (per genre)
│
├── src/
│   ├── pipeline/
│   │   ├── chuncking.py            # semantic pagination into chunks
│   │   ├── rehearsal.py            # inference-time A (maintenance) / B (elaborative) rehearsal
│   │   ├── curation.py             # stage-2 offline curation + collapse diagnostic
│   │   ├── teacher.py              # teacher LLM API calls (curate_document, curate_document_threaded)
│   │   ├── thread_memory.py        # C-DIC's multi-slot ThreadMemory (the collapsing retrieve-and-replace mechanism)
│   │   ├── thread_grouping.py      # no-merge thread clustering — same similarity-based grouping, never regenerates content
│   │   ├── extractive.py           # sentence scorer for maintenance rehearsal (+ windowed/no-truncation selection variants)
│   │   ├── embeddings.py           # NIM embedding calls + retry/rate-limit wrapper
│   │   ├── gisting.py              # sentence splitting / verbatim-snap helpers
│   │   ├── qg.py                   # question-generation input template (testing effect, C)
│   │   ├── rate_limit.py           # shared token-bucket rate limiter
│   │   └── types.py                # Chunk dataclass
│   ├── eval/                       # tau sweep, maintenance-model comparison
│   └── train/                      # (reserved — training currently driven from notebooks)
│
├── experiments/                    # trained checkpoints (maintenance, elaborative stage 1/2, QG)
│
├── results/                        # QA run outputs (CSV) — one file per condition, checkpointed
│
├── notebooks/
│   ├── 00-02                       # pipeline smoke test, length-stress prep/test
│   ├── 03_qa_baseline_3conditions   # closed_book / full_context / chunked_sequential baselines
│   ├── 04-05                       # maintenance rehearsal (A) prep + train
│   ├── 06-07                       # elaborative rehearsal (B) stage 1 prep + train
│   ├── 06b-07b                     # elaborative rehearsal (B) stage 2 (rolling-shaped) prep + train
│   ├── 08-09                       # question generation (testing effect, C) prep + train
│   ├── 10_pipeline_6conditions      # main eval notebook — name is legacy; grew from 6 to 20+ conditions (B, lookup variants, pure RAG, gist-retrieval trilogy, extractive query-aware pruning, compression-cap sweep, structural map, maintenance-rehearsal revival + variants, no-merge thread clustering, soft-selection experiments)
│   └── 11_map_augmented_full_context  # full text + gist map in one call (cost-heavy, exploratory)
│
├── tests/                          # unit tests for src/pipeline
│
└── docs/
    ├── paper/                      # materials for the paper/presentation
    └── report_assets/              # figures embedded in ANALYSIS_REPORT.md
```

> Folders containing only a `.gitkeep` don't have real files yet (structure reserved in advance). Update this tree as they get filled in.
