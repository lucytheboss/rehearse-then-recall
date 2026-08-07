# Ablation A4 — tau sweep



| τ | insert_share | recency_share | mean_span | slots | ret_drop | EM* | F1* |
|---|---|---|---|---|---|---|---|
| 0.020 | 0.091 | 0.909 | 1.091 | 2 | 0.165 | 0.095 | 0.141 |
| 0.050 | 0.182 | 0.643 | 1.643 | 3 | 0.128 | 0.068 | 0.117 |
| 0.100 | 0.364 | 0.538 | 3.538 | 5 | 0.153 | 0.054 | 0.098 |
| 0.200 | 0.545 | 0.636 | 2.636 | 7 | 0.140 | 0.149 | 0.161 |
| 0.350 | 0.909 | 0.545 | 2.727 | 11 | 0.083 | 0.203 | 0.263 |
| 0.500 | 1.000 | 0.545 | 2.727 | 12 | 0.133 | 0.203 | 0.370 |
| 0.700 | 1.000 | 0.545 | 2.727 | 12 | 0.133 | 0.203 | 0.370 |
| 0.900 | 1.000 | 0.545 | 2.727 | 12 | 0.133 | 0.203 | 0.370 |

\* answer recoverability from the final compressed state — an upper
bound on downstream QA, not the pipeline's QA EM/F1.

## Failure-mode checks

```
[ok ] insert_share is non-decreasing in tau: 0.02:0.09 -> 0.05:0.18 -> 0.1:0.36 -> 0.2:0.55 -> 0.35:0.91 -> 0.5:1.00 -> 0.7:1.00 -> 0.9:1.00
[ok ] low tau reaches single-thread collapse (insert_share <= 0.2): tau=0.02 insert_share=0.09 memory_size=2
[ok ] high tau reaches append-only growth (insert_share >= 0.8): tau=0.9 insert_share=1.00 memory_growth=1.00

  - recency degeneration at tau 0.02 (recency_share=0.91) — at these values thread-aware retrieval is only rediscovering recency and the embedding cost buys nothing
  - answer recoverability peaks at tau=0.5 (F1=0.370, EM=0.203, 12 slots)
```

## Recommendation

tau=0.2: insert_share=0.55, recency_share=0.64, answer_f1=0.161, memory 7 slots
