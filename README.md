# rehearse-then-recall

## Project overview


## Paper / presentation links
- Paper: TBD
- Slides: TBD

## Structure

```
rehearse-then-recall/
├── README.md
├── LICENSE
├── .gitignore
├── pyproject.toml / requirements.txt
│
├── configs/                       # per-experiment hyperparameters (YAML)
│   └── chunking.yaml              # chunking min/max words, needle-drop post-hoc position config
│
├── data/
│   ├── raw/                       # raw data (gitignored)
│   ├── processed/                 # preprocessed training data
│   ├── eval_texts/                # evaluation texts
│   └── scripts/                   # data preprocessing scripts
│
├── src/
│   ├── models/                    # maintenance/elaborative rehearsal, QG model definitions
│   ├── pipeline/                  # chunking, importance filter, pipeline orchestration
│   ├── train/                     # per-model training scripts
│   └── eval/                      # evaluation metrics (EM/F1 etc.)
│
├── experiments/
│   ├── ablation_extraction/       # ablation results with/without the extraction stage
│   └── logs/
│
├── notebooks/                     # Colab training/experiment notebooks
│
└── docs/
    └── paper/                     # materials for the paper/presentation
```

> Folders containing only a `.gitkeep` don't have real files yet (structure reserved in advance). Update this tree as they get filled in.
