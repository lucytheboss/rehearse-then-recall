"""Builds N training/reward case-law corpora, disjoint from the eval corpus.

`build_caselaw_eval_corpus.py` uses `lex_glue/case_hold` `train[:400]`. This
takes the *next* `n_corpora * 400` examples, so disjointness holds by
construction (verified against the eval corpus text anyway). CaseHOLD's train
split has 45,000 examples, so 20 corpora consume under a fifth of it.

Keeps CaseHOLD's native 5-option multiple-choice format, as the eval builder
does. Note this makes case-law scores incomparable to the other genres'
short-answer accuracy — chance is 0.20 here and ~0 there.

Unlike the eval builder, questions are NOT subsampled: one item per excerpt is
kept, because the reward signal (evidence retention) is free to compute and a
denser signal is strictly better for training.

Usage:
    python data/scripts/build_caselaw_train_corpora.py [--n-corpora 20]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset

root = Path(__file__).resolve().parent
while not (root / "src").exists() and root != root.parent:
    root = root.parent

EVAL_N_EXAMPLES = 400  # what build_caselaw_eval_corpus.py consumed, from the front
EXAMPLES_PER_CORPUS = 400  # ~135 words each -> ~54k words, matching the eval corpus
SEPARATOR = "\n\n***\n\n"  # matches chuncking.py's structural-break detection


def _word_index(text: str, char_pos: int) -> int:
    return len(text[:char_pos].split()) + 1


def build_one(examples) -> tuple[str, list[dict]]:
    text_parts: list[str] = []
    questions: list[dict] = []
    cumulative_chars = 0

    for i, ex in enumerate(examples):
        context = ex["context"]
        text_parts.append(context)
        questions.append(
            {
                "question_id": i + 1,
                "question": context,  # CaseHOLD's prompt *is* the excerpt with a citation placeholder
                "evidence_char_pos": cumulative_chars,
                "correct_option_index": ex["label"],
                "type": "case_hold_mc",
                **{f"option_{j}": opt for j, opt in enumerate(ex["endings"])},
            }
        )
        cumulative_chars += len(context)
        if i < len(examples) - 1:
            cumulative_chars += len(SEPARATOR)

    full_text = SEPARATOR.join(text_parts)
    for q in questions:
        q["evidence_word_pos"] = _word_index(full_text, q["evidence_char_pos"])
        # The anchor is the excerpt start, so it must land exactly there.
        assert full_text.startswith(q["question"], q["evidence_char_pos"]), "excerpt anchor mismatch"
    return full_text, questions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-corpora", type=int, default=20)
    args = parser.parse_args()

    lo = EVAL_N_EXAMPLES
    hi = lo + args.n_corpora * EXAMPLES_PER_CORPUS
    print(f"Loading lex_glue/case_hold train[{lo}:{hi}] (eval used train[:{lo}])...")
    ds = load_dataset("lex_glue", "case_hold", split=f"train[{lo}:{hi}]")
    print(f"  loaded {len(ds)} examples")

    eval_text = (root / "data" / "processed" / "caselaw_eval" / "caselaw_eval_corpus.txt").read_text(
        encoding="utf-8"
    )

    out_root = root / "data" / "processed" / "caselaw_train"
    out_root.mkdir(parents=True, exist_ok=True)

    import pandas as pd

    index = []
    for i in range(args.n_corpora):
        chunk = ds.select(range(i * EXAMPLES_PER_CORPUS, min((i + 1) * EXAMPLES_PER_CORPUS, len(ds))))
        if len(chunk) < EXAMPLES_PER_CORPUS:
            print(f"  ! pool exhausted at corpus_{i:02d} ({len(chunk)} examples) — stopping")
            break

        text, questions = build_one(list(chunk))
        overlap = sum(1 for q in questions if q["question"] in eval_text)
        assert overlap == 0, f"corpus_{i:02d} shares {overlap} excerpts with the eval corpus"

        corpus_dir = out_root / f"corpus_{i:02d}"
        corpus_dir.mkdir(exist_ok=True)
        (corpus_dir / "corpus.txt").write_text(text, encoding="utf-8")

        cols = ["question_id", "question", "evidence_char_pos", "evidence_word_pos",
                "correct_option_index", "type"] + [f"option_{j}" for j in range(5)]
        pd.DataFrame(questions)[cols].to_csv(
            corpus_dir / "questions.csv", index=False, encoding="utf-8-sig"
        )

        row = {"corpus": f"corpus_{i:02d}", "n_excerpts": len(chunk),
               "words": len(text.split()), "n_questions": len(questions)}
        (corpus_dir / "manifest.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
        index.append(row)
        print(f"  corpus_{i:02d}: {row['n_excerpts']} excerpts, {row['words']:,} words, "
              f"{row['n_questions']} questions")

    (out_root / "index.json").write_text(
        json.dumps({"genre": "caselaw", "format": "5-option multiple choice (chance = 0.20)",
                    "n_corpora": len(index),
                    "disjoint_from": f"lex_glue/case_hold train[:{EVAL_N_EXAMPLES}] (eval)",
                    "corpora": index}, indent=2),
        encoding="utf-8",
    )
    print(f"\nSaved {len(index)} corpora to {out_root} "
          f"({sum(r['n_questions'] for r in index):,} questions total)")


if __name__ == "__main__":
    main()
