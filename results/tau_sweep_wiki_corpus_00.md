# Ablation A4 — tau sweep



| τ | insert_share | recency_share | mean_span | slots | ret_drop | EM* | F1* |
|---|---|---|---|---|---|---|---|
| 0.020 | 0.091 | 0.579 | 2.263 | 2 | 0.000 | 0.000 | 0.000 |
| 0.050 | 0.273 | 0.360 | 3.080 | 4 | 0.000 | 0.000 | 0.000 |
| 0.100 | 1.000 | 0.091 | 6.000 | 12 | 0.000 | 0.000 | 0.000 |
| 0.200 | 1.000 | 0.091 | 6.000 | 12 | 0.000 | 0.000 | 0.000 |
| 0.350 | 1.000 | 0.091 | 6.000 | 12 | 0.000 | 0.000 | 0.000 |
| 0.500 | 1.000 | 0.091 | 6.000 | 12 | 0.000 | 0.000 | 0.000 |
| 0.700 | 1.000 | 0.091 | 6.000 | 12 | 0.000 | 0.000 | 0.000 |
| 0.900 | 1.000 | 0.091 | 6.000 | 12 | 0.000 | 0.000 | 0.000 |

\* answer recoverability from the final compressed state — an upper
bound on downstream QA, not the pipeline's QA EM/F1.

## Failure-mode checks

```
[ok ] insert_share is non-decreasing in tau: 0.02:0.09 -> 0.05:0.27 -> 0.1:1.00 -> 0.2:1.00 -> 0.35:1.00 -> 0.5:1.00 -> 0.7:1.00 -> 0.9:1.00
[ok ] low tau reaches single-thread collapse (insert_share <= 0.2): tau=0.02 insert_share=0.09 memory_size=2
[ok ] high tau reaches append-only growth (insert_share >= 0.8): tau=0.9 insert_share=1.00 memory_growth=1.00

  - no tau degenerated to recency (max recency_share 0.58 < 0.8) — retrieval is reaching past the previous chunk at every threshold tested
  - answer recoverability peaks at tau=0.02 (F1=0.000, EM=0.000, 2 slots)
```

## Recommendation

tau=0.05: insert_share=0.27, recency_share=0.36, answer_f1=0.000, memory 4 slots
