"""Builds the legal-case-law-genre evaluation corpus from `lex_glue`'s
`case_hold` config — English replacement for a Korean case-law source.

Unlike wiki/news (extractive QA: answer is a verbatim span in a long
document) or statute (LLM-generated QA grounded in fetched text), CaseHOLD's
native task is genuinely different in shape: given a short excerpt from a
judicial opinion with a citation placeholder, pick the correct "holding"
statement out of 5 candidates. There's no verbatim answer span to
position-track the way the other genres do — so this keeps CaseHOLD's own
native multiple-choice format (matching this project's original 4-option
approach from `eval_sample01.csv`/`02_qa_baseline_3conditions.ipynb`, just
with 5 options here) rather than forcing it into the extractive-QA shape.

Individual excerpts (~135 words median) are far too short to be a long
multi-chunk document on their own, so — same pattern as the wiki/news
builders — many excerpts are concatenated into one corpus, each contributing
one multiple-choice item anchored to the word position where its excerpt
starts (this position anchor is what lets a length/position-sensitive
analysis be run on this genre too, even without a verbatim-span answer).
"""

from __future__ import annotations

import json
from pathlib import Path

from datasets import load_dataset

root = Path(__file__).resolve().parent
while not (root / "src").exists() and root != root.parent:
    root = root.parent

N_EXAMPLES = 100
SEPARATOR = "\n\n***\n\n"  # matches chuncking.py's structural-break detection


def _word_index(text: str, char_pos: int) -> int:
    return len(text[:char_pos].split()) + 1


def main() -> None:
    print("Loading lex_glue/case_hold...")
    ds = load_dataset("lex_glue", "case_hold", split=f"train[:{N_EXAMPLES}]")

    text_parts: list[str] = []
    questions: list[dict] = []
    cumulative_chars = 0

    for i, ex in enumerate(ds):
        context = ex["context"]
        char_pos = cumulative_chars
        text_parts.append(context)

        questions.append(
            {
                "question_id": i + 1,
                "question": "Which of the following best states the holding for this passage?",
                "options": ex["endings"],
                "correct_option_index": ex["label"],
                "evidence_char_pos": char_pos,
                "type": "caselaw_multiple_choice",
            }
        )

        cumulative_chars += len(context)
        if i < len(ds) - 1:
            cumulative_chars += len(SEPARATOR)

    full_text = SEPARATOR.join(text_parts)

    for q in questions:
        q["evidence_word_pos"] = _word_index(full_text, q["evidence_char_pos"])
        assert full_text[q["evidence_char_pos"] : q["evidence_char_pos"] + 20].strip(), (
            f"empty context at question {q['question_id']}"
        )

    print(f"\ncorpus: {len(full_text)} chars, {len(full_text.split())} words, {len(questions)} questions")

    out_dir = root / "data" / "processed" / "caselaw_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_dir = root / "data" / "sample"
    sample_dir.mkdir(parents=True, exist_ok=True)

    text_path = out_dir / "caselaw_eval_corpus.txt"
    text_path.write_text(full_text, encoding="utf-8")

    import pandas as pd

    flat_rows = []
    for q in questions:
        row = {
            "question_id": q["question_id"],
            "question": q["question"],
            "evidence_char_pos": q["evidence_char_pos"],
            "evidence_word_pos": q["evidence_word_pos"],
            "correct_option_index": q["correct_option_index"],
            "type": q["type"],
        }
        for j, option in enumerate(q["options"]):
            row[f"option_{j}"] = option
        flat_rows.append(row)

    questions_df = pd.DataFrame(flat_rows)
    csv_path = sample_dir / "caselaw_eval_questions.csv"
    questions_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    manifest_path = out_dir / "caselaw_eval_manifest.json"
    manifest_path.write_text(
        json.dumps({"n_examples": len(ds), "n_questions": len(questions)}, indent=2), encoding="utf-8"
    )

    print(f"\nSaved:\n  {text_path}\n  {csv_path}\n  {manifest_path}")


if __name__ == "__main__":
    main()
