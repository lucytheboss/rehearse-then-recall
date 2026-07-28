"""Builds the wiki-genre evaluation corpus from SQuAD — the English
replacement for the deleted KorQuAD-based `build_length_stress_corpus.py`.

SQuAD's `context` field already gives the exact Wikipedia paragraph text and
`answers.answer_start` gives the exact character offset of each answer
*within its own paragraph* — unlike narrativeqa (free-form, no position
info) or cnn_dailymail (abstractive-only), so no oracle extraction or
verbatim-search step is needed here. The only real work is: pick one article
with enough paragraphs/questions to be a genuinely long multi-chunk document
(scanned ~subset of `train`; "New_York_City" won by a wide margin — 148
paragraphs, 817 questions, ~14,900 words), reassemble its paragraphs in
original order, and remap each answer's per-paragraph character offset to a
position in the concatenated document.
"""

from __future__ import annotations

import json
from pathlib import Path

from datasets import load_dataset

root = Path(__file__).resolve().parent
while not (root / "src").exists() and root != root.parent:
    root = root.parent

TITLE = "New_York_City"
QC_MAX_OCCURRENCE = 3  # answer must appear this many times or fewer in the full doc to be position-reliable
MAX_QUESTIONS = 100  # SQuAD gives 817 raw questions for this title — far more than an eval set needs


def _word_index(text: str, char_pos: int) -> int:
    """1-indexed word position: word count before char_pos, plus 1.
    Same definition used throughout this project's other corpus scripts."""
    return len(text[:char_pos].split()) + 1


def collect_paragraphs_and_questions(ds, title: str) -> tuple[list[str], list[dict]]:
    """Dedupes contexts for `title`, preserving first-occurrence order (SQuAD
    groups a paragraph's questions consecutively, so this recovers the
    original article order), and pairs each question with its paragraph index."""
    paragraphs: list[str] = []
    context_to_index: dict[str, int] = {}
    raw_questions: list[dict] = []

    for row in ds:
        if row["title"] != title:
            continue
        context = row["context"]
        if context not in context_to_index:
            context_to_index[context] = len(paragraphs)
            paragraphs.append(context)

        if not row["answers"]["text"]:
            continue
        raw_questions.append(
            {
                "paragraph_index": context_to_index[context],
                "question": row["question"],
                "answer": row["answers"]["text"][0],
                "local_char_pos": row["answers"]["answer_start"][0],
            }
        )

    return paragraphs, raw_questions


def build_corpus(paragraphs: list[str], raw_questions: list[dict]) -> tuple[str, list[dict]]:
    full_text = "\n\n".join(paragraphs)

    paragraph_offsets = [0]
    for p in paragraphs:
        paragraph_offsets.append(paragraph_offsets[-1] + len(p) + len("\n\n"))

    questions = []
    for q in raw_questions:
        global_char_pos = paragraph_offsets[q["paragraph_index"]] + q["local_char_pos"]
        assert full_text[global_char_pos : global_char_pos + len(q["answer"])] == q["answer"], (
            f"position mismatch for question {q['question']!r}"
        )
        occurrence = full_text.count(q["answer"])
        if not (1 <= occurrence <= QC_MAX_OCCURRENCE):
            continue
        questions.append(
            {
                "question": q["question"],
                "answer": q["answer"],
                "evidence_char_pos": global_char_pos,
                "evidence_word_pos": _word_index(full_text, global_char_pos),
                "type": "squad_extractive",
            }
        )

    return full_text, questions


def main() -> None:
    print(f"Loading squad train split, filtering to title={TITLE!r}...")
    ds = load_dataset("squad", split="train")

    paragraphs, raw_questions = collect_paragraphs_and_questions(ds, TITLE)
    print(f"paragraphs: {len(paragraphs)}, raw questions: {len(raw_questions)}")

    full_text, questions = build_corpus(paragraphs, raw_questions)
    print(f"corpus: {len(full_text)} chars, {len(full_text.split())} words")
    print(f"questions passing occurrence-count QC (<= {QC_MAX_OCCURRENCE}x): {len(questions)}")

    if len(questions) > MAX_QUESTIONS:
        # Spread the subsample across the document rather than taking a prefix,
        # so the eval set still covers early/middle/late positions.
        step = len(questions) / MAX_QUESTIONS
        questions = [questions[int(i * step)] for i in range(MAX_QUESTIONS)]
        print(f"subsampled to {len(questions)} questions (spread across document)")

    for i, q in enumerate(questions, start=1):
        q["question_id"] = i

    out_dir = root / "data" / "processed" / "wiki_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_dir = root / "data" / "sample"
    sample_dir.mkdir(parents=True, exist_ok=True)

    text_path = out_dir / "wiki_eval_corpus.txt"
    text_path.write_text(full_text, encoding="utf-8")

    import pandas as pd

    questions_df = pd.DataFrame(questions)[
        ["question_id", "question", "answer", "evidence_char_pos", "evidence_word_pos", "type"]
    ]
    csv_path = sample_dir / "wiki_eval_questions.csv"
    questions_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    manifest_path = out_dir / "wiki_eval_manifest.json"
    manifest_path.write_text(
        json.dumps({"title": TITLE, "n_paragraphs": len(paragraphs), "n_questions": len(questions)}, indent=2),
        encoding="utf-8",
    )

    print(f"\nSaved:\n  {text_path}\n  {csv_path}\n  {manifest_path}")


if __name__ == "__main__":
    main()
