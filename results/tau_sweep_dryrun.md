# Ablation A4 — tau sweep

**DRY RUN — synthetic inputs, not results.**

| τ | insert_share | recency_share | mean_span | slots | ret_drop | EM* | F1* |
|---|---|---|---|---|---|---|---|
| 0.300 | 0.130 | 0.130 | 3.696 | 4 | 0.882 | 0.167 | 0.167 |
| 0.500 | 0.217 | 0.130 | 3.696 | 6 | 0.790 | 0.250 | 0.250 |
| 0.650 | 0.522 | 0.130 | 3.696 | 13 | 0.343 | 0.542 | 0.542 |
| 0.800 | 0.609 | 0.130 | 3.696 | 15 | 0.306 | 0.625 | 0.625 |
| 0.900 | 0.913 | 0.130 | 3.696 | 22 | 0.031 | 0.917 | 0.917 |

\* answer recoverability from the final compressed state — an upper
bound on downstream QA, not the pipeline's QA EM/F1.

## Failure-mode checks

```
[ok ] insert_share is non-decreasing in tau: 0.3:0.13 -> 0.5:0.22 -> 0.65:0.52 -> 0.8:0.61 -> 0.9:0.91
[ok ] low tau reaches single-thread collapse (insert_share <= 0.2): tau=0.3 insert_share=0.13 memory_size=4
[ok ] high tau reaches append-only growth (insert_share >= 0.8): tau=0.9 insert_share=0.91 memory_growth=0.92

  - no tau degenerated to recency (max recency_share 0.13 < 0.8) — retrieval is reaching past the previous chunk at every threshold tested
  - answer recoverability peaks at tau=0.9 (F1=0.917, EM=0.917, 22 slots)
```

## Recommendation

tau=0.8: insert_share=0.61, recency_share=0.13, answer_f1=0.625, memory 15 slots
