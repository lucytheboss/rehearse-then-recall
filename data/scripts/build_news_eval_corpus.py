"""Builds the news-genre evaluation corpus from NewsQA — English replacement
for the KorQuAD-based approach in the deleted `build_length_stress_corpus.py`.

A single NewsQA article (median ~560 words) is too short to be a genuine
multi-chunk long document on its own, so — following the same pattern the
deleted script used for KorQuAD (concatenating 26 separate short documents
into one corpus) — this concatenates multiple distinct articles, chosen
greedily by "words per native question" (lower = more QA-dense = fewer
articles needed to reach the word budget). Unlike cnn_dailymail (abstractive
summaries only) or narrativeqa (free-form answers, no position), NewsQA
ships exact answer character offsets directly, so no oracle extraction or
verbatim search is needed.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset

root = Path(__file__).resolve().parent
while not (root / "src").exists() and root != root.parent:
    root = root.parent

N_CANDIDATE_ROWS = 30_000  # rows scanned to build the candidate article pool
# Sized to match the longest novel rung (50,000 words) so every genre can be
# tested on the same ladder rather than topping out at one context window.
TARGET_MIN_WORDS = 50_000
TARGET_MAX_WORDS = 55_000
QC_MAX_OCCURRENCE = 3
MAX_QUESTIONS = 200  # far more pass QC than an eval set needs; subsample, spread across the document
SCENE_BREAK_SEPARATOR = "\n\n***\n\n"  # matches chuncking.py's structural-break detection


def _word_index(text: str, char_pos: int) -> int:
    return len(text[:char_pos].split()) + 1


def collect_candidates(ds) -> list[dict]:
    by_context: dict[str, list] = defaultdict(list)
    for row in ds:
        by_context[row["context"]].append(row)

    candidates = []
    for context, rows in by_context.items():
        word_count = len(context.split())
        if word_count < 200:  # essentially empty/too-short articles
            continue

        qa_items = []
        for row in rows:
            if not row["answers"]:
                continue
            answer = row["answers"][0]
            char_pos = row["labels"][0]["start"][0]
            if context[char_pos : char_pos + len(answer)] != answer:
                continue  # position doesn't line up exactly, skip rather than guess
            qa_items.append({"question": row["question"], "answer": answer, "local_char_pos": char_pos})

        if not qa_items:
            continue

        candidates.append(
            {
                "context": context,
                "word_count": word_count,
                "qa_items": qa_items,
                "density": word_count / len(qa_items),  # lower = more efficient
            }
        )

    return candidates


def greedy_select(candidates: list[dict]) -> list[dict]:
    candidates_sorted = sorted(candidates, key=lambda c: c["density"])
    selected = []
    cumulative_words = 0
    for cand in candidates_sorted:
        if cumulative_words >= TARGET_MIN_WORDS:
            break
        if cumulative_words > 0 and cumulative_words + cand["word_count"] > TARGET_MAX_WORDS:
            continue
        selected.append(cand)
        cumulative_words += cand["word_count"]
    return selected


def build_corpus(selected: list[dict]) -> tuple[str, list[dict]]:
    text_parts: list[str] = []
    questions: list[dict] = []
    cumulative_chars = 0

    for doc_index, doc in enumerate(selected):
        text_parts.append(doc["context"])
        doc_start_char = cumulative_chars

        for item in doc["qa_items"]:
            global_char_pos = doc_start_char + item["local_char_pos"]
            questions.append(
                {
                    "question": item["question"],
                    "answer": item["answer"],
                    "evidence_char_pos": global_char_pos,
                    "type": "newsqa_extractive",
                }
            )

        cumulative_chars += len(doc["context"])
        if doc_index < len(selected) - 1:
            cumulative_chars += len(SCENE_BREAK_SEPARATOR)

    full_text = SCENE_BREAK_SEPARATOR.join(text_parts)

    final_questions = []
    for q in questions:
        occurrence = full_text.count(q["answer"])
        if not (1 <= occurrence <= QC_MAX_OCCURRENCE):
            continue
        assert full_text[q["evidence_char_pos"] : q["evidence_char_pos"] + len(q["answer"])] == q["answer"], (
            f"position mismatch for question {q['question']!r}"
        )
        q["evidence_word_pos"] = _word_index(full_text, q["evidence_char_pos"])
        final_questions.append(q)

    return full_text, final_questions


def main() -> None:
    print(f"Loading lucadiliello/newsqa (first {N_CANDIDATE_ROWS} train rows)...")
    ds = load_dataset("lucadiliello/newsqa", split=f"train[:{N_CANDIDATE_ROWS}]")

    print("Collecting candidate articles...")
    candidates = collect_candidates(ds)
    print(f"candidate articles: {len(candidates)}")

    selected = greedy_select(candidates)
    total_words = sum(c["word_count"] for c in selected)
    total_qa = sum(len(c["qa_items"]) for c in selected)
    print(f"\nselected: {len(selected)} articles, {total_words} words, {total_qa} raw questions")

    full_text, questions = build_corpus(selected)
    print(f"\nfinal corpus: {len(full_text)} chars, {len(full_text.split())} words")
    print(f"questions passing occurrence-count QC (<= {QC_MAX_OCCURRENCE}x): {len(questions)}")

    if len(questions) > MAX_QUESTIONS:
        questions.sort(key=lambda q: q["evidence_char_pos"])
        step = len(questions) / MAX_QUESTIONS
        questions = [questions[int(i * step)] for i in range(MAX_QUESTIONS)]
        print(f"subsampled to {len(questions)} questions (spread across document)")

    for i, q in enumerate(questions, start=1):
        q["question_id"] = i

    out_dir = root / "data" / "processed" / "news_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_dir = root / "data" / "sample"
    sample_dir.mkdir(parents=True, exist_ok=True)

    text_path = out_dir / "news_eval_corpus.txt"
    text_path.write_text(full_text, encoding="utf-8")

    import pandas as pd

    questions_df = pd.DataFrame(questions)[
        ["question_id", "question", "answer", "evidence_char_pos", "evidence_word_pos", "type"]
    ]
    csv_path = sample_dir / "news_eval_questions.csv"
    questions_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    manifest_path = out_dir / "news_eval_manifest.json"
    manifest_path.write_text(
        json.dumps({"n_articles": len(selected), "n_questions": len(questions)}, indent=2), encoding="utf-8"
    )

    print(f"\nSaved:\n  {text_path}\n  {csv_path}\n  {manifest_path}")


if __name__ == "__main__":
    main()
