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

**Current headline finding (2026-08-13)**: a plain retrieval-augmented
baseline with no rehearsal, no compression, and no teacher model anywhere
(`raw_retrieval_adaptive`) beats every rehearsal-based condition on every
genre, at a fraction of the token cost. A follow-up investigation
(9 further conditions, isolating architecture/content/selection/model size,
plus 2025 literature on extractive vs. abstractive and query-aware vs.
query-agnostic compression) narrows down why, and identifies one condition
— extractive, query-aware sentence pruning over RAG's own retrieved chunks
— with a genuine, defensible accuracy-for-tokens trade-off. See
**[ANALYSIS_REPORT.md](ANALYSIS_REPORT.md) §4.3** for the full account.

## Paper / presentation links
- Paper: TBD
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
│   │   ├── thread_memory.py        # C-DIC's multi-slot ThreadMemory
│   │   ├── extractive.py           # sentence scorer for maintenance rehearsal
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
│   ├── 10_pipeline_6conditions      # main eval notebook — grew from 6 to 10 conditions (B, lookup variants, pure RAG, gist-retrieval trilogy, extractive query-aware pruning, compression-cap sweep)
│   └── 11_map_augmented_full_context  # full text + gist map in one call (cost-heavy, exploratory)
│
├── tests/                          # unit tests for src/pipeline
│
└── docs/
    ├── paper/                      # materials for the paper/presentation
    └── report_assets/              # figures embedded in ANALYSIS_REPORT.md
```

> Folders containing only a `.gitkeep` don't have real files yet (structure reserved in advance). Update this tree as they get filled in.
