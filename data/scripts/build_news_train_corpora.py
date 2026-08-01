"""Builds N *training/reward* news corpora, disjoint from the eval corpus.

Why this exists
---------------
`build_news_eval_corpus.py` produces a single 50k-word corpus. That is correct
for evaluation, but it leaves the project with one news "document" — so a
learned compression policy has one training context per genre and cannot
support any claim of generalisation, and there is no document-level train/test
split at all.

This script reuses that script's candidate collection and QC (position
verification + occurrence uniqueness) unchanged, and only changes how articles
are grouped: instead of one corpus, the pool is dealt into `N_CORPORA`
disjoint corpora of the same 50k-word size.

Two constraints it enforces
---------------------------
1. **Disjoint from the eval corpus.** Any article whose text appears in
   `news_eval_corpus.txt` is dropped. Checked against the actual corpus text
   rather than by re-deriving the selection, so it holds even if the upstream
   dataset ordering shifts.
2. **No density gradient.** Candidates are sorted by words-per-question
   (the eval script's greedy criterion) and then dealt **round-robin** into the
   corpora. Filling corpora sequentially instead would give corpus 0 the most
   QA-dense articles and corpus 19 the least, making the corpora systematically
   unequal — a confound in anything trained on them.

Questions are NOT subsampled here (the eval script caps at 200 to keep the API
bill down). Reward signal is free to compute via evidence positions, so more
questions per corpus is strictly better.

Usage:
    python data/scripts/build_news_train_corpora.py [--n-corpora 20] [--rows 150000]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from datasets import load_dataset

root = Path(__file__).resolve().parent
while not (root / "src").exists() and root != root.parent:
    root = root.parent
sys.path.insert(0, str(root / "data" / "scripts"))

from build_news_eval_corpus import (  # noqa: E402
    QC_MAX_OCCURRENCE,
    SCENE_BREAK_SEPARATOR,
    TARGET_MAX_WORDS,
    TARGET_MIN_WORDS,
    _word_index,
    collect_candidates,
)


def load_eval_corpus_text() -> str:
    path = root / "data" / "processed" / "news_eval" / "news_eval_corpus.txt"
    if not path.exists():
        raise SystemExit(f"eval corpus not found at {path} — run build_news_eval_corpus.py first")
    return path.read_text(encoding="utf-8")


def drop_eval_articles(candidates: list[dict], eval_text: str) -> list[dict]:
    """Removes any article already used in the eval corpus.

    Substring containment against the eval text, not a re-run of the eval
    script's selection: the selection depends on dataset ordering and tie
    breaks, and a silent overlap here would leak eval documents into training.
    """
    kept = [c for c in candidates if c["context"] not in eval_text]
    print(f"  dropped {len(candidates) - len(kept)} articles already in the eval corpus")
    return kept


def deal_round_robin(candidates: list[dict], n_corpora: int) -> list[list[dict]]:
    """Deals density-sorted candidates into `n_corpora` buckets, round-robin.

    Each bucket stops accepting once it reaches TARGET_MIN_WORDS, and never
    exceeds TARGET_MAX_WORDS, matching the eval corpus's size band so the
    genres stay on the same length ladder.
    """
    candidates_sorted = sorted(candidates, key=lambda c: c["density"])
    buckets: list[list[dict]] = [[] for _ in range(n_corpora)]
    words = [0] * n_corpora

    cursor = 0
    for cand in candidates_sorted:
        # Find the next bucket that still has room, starting from the cursor so
        # consecutive (similar-density) articles land in different corpora.
        for offset in range(n_corpora):
            i = (cursor + offset) % n_corpora
            if words[i] >= TARGET_MIN_WORDS:
                continue
            if words[i] > 0 and words[i] + cand["word_count"] > TARGET_MAX_WORDS:
                continue
            buckets[i].append(cand)
            words[i] += cand["word_count"]
            cursor = (i + 1) % n_corpora
            break

        if all(w >= TARGET_MIN_WORDS for w in words):
            break

    return [b for b, w in zip(buckets, words) if w >= TARGET_MIN_WORDS]


def build_one(selected: list[dict]) -> tuple[str, list[dict]]:
    """Same assembly + QC as the eval script, minus the question subsampling."""
    text_parts: list[str] = []
    questions: list[dict] = []
    cumulative_chars = 0

    for doc_index, doc in enumerate(selected):
        text_parts.append(doc["context"])
        doc_start_char = cumulative_chars
        for item in doc["qa_items"]:
            questions.append(
                {
                    "question": item["question"],
                    "answer": item["answer"],
                    "evidence_char_pos": doc_start_char + item["local_char_pos"],
                    "type": "newsqa_extractive",
                }
            )
        cumulative_chars += len(doc["context"])
        if doc_index < len(selected) - 1:
            cumulative_chars += len(SCENE_BREAK_SEPARATOR)

    full_text = SCENE_BREAK_SEPARATOR.join(text_parts)

    final: list[dict] = []
    for q in questions:
        occurrence = full_text.count(q["answer"])
        if not (1 <= occurrence <= QC_MAX_OCCURRENCE):
            continue
        assert full_text[q["evidence_char_pos"] : q["evidence_char_pos"] + len(q["answer"])] == q["answer"], (
            f"position mismatch for question {q['question']!r}"
        )
        q["evidence_word_pos"] = _word_index(full_text, q["evidence_char_pos"])
        final.append(q)

    final.sort(key=lambda q: q["evidence_char_pos"])
    for i, q in enumerate(final, start=1):
        q["question_id"] = i
    return full_text, final


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-corpora", type=int, default=20)
    parser.add_argument("--rows", type=int, default=150_000,
                        help="NewsQA train rows to scan for the candidate pool")
    args = parser.parse_args()

    print(f"Loading lucadiliello/newsqa (first {args.rows:,} train rows)...")
    ds = load_dataset("lucadiliello/newsqa", split=f"train[:{args.rows}]")

    print("Collecting candidate articles...")
    candidates = collect_candidates(ds)
    print(f"  candidate articles: {len(candidates):,}  ({sum(c['word_count'] for c in candidates):,} words)")

    candidates = drop_eval_articles(candidates, load_eval_corpus_text())

    available = sum(c["word_count"] for c in candidates)
    needed = args.n_corpora * TARGET_MIN_WORDS
    print(f"  usable: {len(candidates):,} articles / {available:,} words "
          f"(need ~{needed:,} for {args.n_corpora} corpora)")
    if available < needed:
        print(f"  ! not enough — raise --rows. Will build as many as the pool allows.")

    buckets = deal_round_robin(candidates, args.n_corpora)
    print(f"\nbuilt {len(buckets)} corpora (requested {args.n_corpora})")

    out_root = root / "data" / "processed" / "news_train"
    out_root.mkdir(parents=True, exist_ok=True)

    import pandas as pd

    index = []
    for i, bucket in enumerate(buckets):
        text, questions = build_one(bucket)
        corpus_dir = out_root / f"corpus_{i:02d}"
        corpus_dir.mkdir(exist_ok=True)

        (corpus_dir / "corpus.txt").write_text(text, encoding="utf-8")
        pd.DataFrame(questions)[
            ["question_id", "question", "answer", "evidence_char_pos", "evidence_word_pos", "type"]
        ].to_csv(corpus_dir / "questions.csv", index=False, encoding="utf-8-sig")

        row = {"corpus": f"corpus_{i:02d}", "n_articles": len(bucket),
               "words": len(text.split()), "n_questions": len(questions)}
        (corpus_dir / "manifest.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
        index.append(row)
        print(f"  corpus_{i:02d}: {row['n_articles']:3} articles, "
              f"{row['words']:,} words, {row['n_questions']:4} questions")

    (out_root / "index.json").write_text(
        json.dumps({"genre": "news", "n_corpora": len(index),
                    "disjoint_from": "data/processed/news_eval/news_eval_corpus.txt",
                    "corpora": index}, indent=2),
        encoding="utf-8",
    )
    total_q = sum(r["n_questions"] for r in index)
    print(f"\nSaved {len(index)} corpora to {out_root}  ({total_q:,} questions total)")


if __name__ == "__main__":
    main()
