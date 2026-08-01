"""Builds N training/reward wiki corpora from SQuAD, and partitions the
remaining titles for QG training.

Two problems solved at once
---------------------------
1. **Wiki has one document.** `build_wiki_eval_corpus.py` uses a single title
   (`New_York_City`, 148 paragraphs, ~14.9k words). This packs *other* titles
   into `n_corpora` corpora of ~50k words each, matching the other genres'
   length ladder.

2. **QG trains on 9 Wikipedia articles.** `08_qg_testing_effect_prep.ipynb`
   slices `squad train[:3000]`, and SQuAD is ordered by title — so those 3,000
   rows come from only **9** of the 442 titles (Beyoncé, Chopin, Antibiotics,
   iPod, Montana, ...). That is a far narrower training distribution than
   "SQuAD is Wikipedia" suggests, and a likely contributor to the QG model's
   poor out-of-domain behaviour.

Both are fixed by a single **title-level partition**, written to
`title_partition.json`:

  - `New_York_City`            -> wiki eval corpus (untouched, already measured)
  - `corpus_titles`            -> the wiki train corpora built here
  - `qg_titles`                -> everything left over, for QG training

Because the partition is disjoint by construction, QG can be retrained on a
few hundred topically varied titles without ever seeing a wiki corpus it will
later be asked to generate probes for.

Usage:
    python data/scripts/build_wiki_train_corpora.py [--n-corpora 20] [--seed 20260801]
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset

root = Path(__file__).resolve().parent
while not (root / "src").exists() and root != root.parent:
    root = root.parent

EVAL_TITLE = "New_York_City"
TARGET_MIN_WORDS = 50_000
TARGET_MAX_WORDS = 55_000
QC_MAX_OCCURRENCE = 3
PARAGRAPH_SEP = "\n\n"
TITLE_SEP = "\n\n***\n\n"  # matches chuncking.py's structural-break detection


def _word_index(text: str, char_pos: int) -> int:
    return len(text[:char_pos].split()) + 1


def group_by_title(ds) -> dict[str, dict]:
    """Per title: paragraphs in first-occurrence order (SQuAD groups a
    paragraph's questions consecutively, so this recovers article order) plus
    each question tagged with its paragraph index."""
    titles: dict[str, dict] = defaultdict(lambda: {"paragraphs": [], "index": {}, "questions": []})
    for row in ds:
        entry = titles[row["title"]]
        context = row["context"]
        if context not in entry["index"]:
            entry["index"][context] = len(entry["paragraphs"])
            entry["paragraphs"].append(context)
        if not row["answers"]["text"]:
            continue
        entry["questions"].append(
            {
                "paragraph_index": entry["index"][context],
                "question": row["question"],
                "answer": row["answers"]["text"][0],
                "local_char_pos": row["answers"]["answer_start"][0],
            }
        )
    for entry in titles.values():
        entry["words"] = sum(len(p.split()) for p in entry["paragraphs"])
    return dict(titles)


def pack_titles(titles: dict[str, dict], order: list[str], n_corpora: int) -> list[list[str]]:
    """Greedily fills one corpus at a time to the 50k-55k band.

    Sequential rather than round-robin (unlike the news builder): titles here
    are whole Wikipedia articles chosen at random, so there is no density
    ordering that would bias early corpora.
    """
    buckets: list[list[str]] = []
    current: list[str] = []
    words = 0
    for title in order:
        w = titles[title]["words"]
        if words > 0 and words + w > TARGET_MAX_WORDS:
            if words >= TARGET_MIN_WORDS:
                buckets.append(current)
                if len(buckets) == n_corpora:
                    return buckets
                current, words = [], 0
            else:
                continue  # this title would overshoot; try a smaller one
        current.append(title)
        words += w
        if words >= TARGET_MIN_WORDS:
            buckets.append(current)
            if len(buckets) == n_corpora:
                return buckets
            current, words = [], 0
    return buckets


def build_one(titles: dict[str, dict], group: list[str]) -> tuple[str, list[dict]]:
    text_parts: list[str] = []
    offsets: list[tuple[str, int, list[int]]] = []
    cumulative = 0

    for i, title in enumerate(group):
        paragraphs = titles[title]["paragraphs"]
        para_offsets = []
        local = 0
        for p in paragraphs:
            para_offsets.append(cumulative + local)
            local += len(p) + len(PARAGRAPH_SEP)
        block = PARAGRAPH_SEP.join(paragraphs)
        text_parts.append(block)
        offsets.append((title, cumulative, para_offsets))
        cumulative += len(block)
        if i < len(group) - 1:
            cumulative += len(TITLE_SEP)

    full_text = TITLE_SEP.join(text_parts)

    questions: list[dict] = []
    for title, _, para_offsets in offsets:
        for q in titles[title]["questions"]:
            pos = para_offsets[q["paragraph_index"]] + q["local_char_pos"]
            if full_text[pos : pos + len(q["answer"])] != q["answer"]:
                continue  # offset did not survive assembly; drop rather than guess
            if not (1 <= full_text.count(q["answer"]) <= QC_MAX_OCCURRENCE):
                continue
            questions.append(
                {
                    "question": q["question"],
                    "answer": q["answer"],
                    "evidence_char_pos": pos,
                    "evidence_word_pos": _word_index(full_text, pos),
                    "source_title": title,
                    "type": "squad_extractive",
                }
            )

    questions.sort(key=lambda q: q["evidence_char_pos"])
    for i, q in enumerate(questions, start=1):
        q["question_id"] = i
    return full_text, questions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-corpora", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260801)
    args = parser.parse_args()

    print("Loading squad train split...")
    ds = load_dataset("squad", split="train")
    titles = group_by_title(ds)
    print(f"  {len(titles)} titles, {sum(t['words'] for t in titles.values()):,} words")

    pool = sorted(t for t in titles if t != EVAL_TITLE)  # sorted first => seed fully determines order
    random.Random(args.seed).shuffle(pool)

    groups = pack_titles(titles, pool, args.n_corpora)
    print(f"\nbuilt {len(groups)} corpora (requested {args.n_corpora})")

    used = {t for g in groups for t in g}
    leftover = [t for t in pool if t not in used]

    out_root = root / "data" / "processed" / "wiki_train"
    out_root.mkdir(parents=True, exist_ok=True)

    import pandas as pd

    index = []
    for i, group in enumerate(groups):
        text, questions = build_one(titles, group)
        corpus_dir = out_root / f"corpus_{i:02d}"
        corpus_dir.mkdir(exist_ok=True)
        (corpus_dir / "corpus.txt").write_text(text, encoding="utf-8")
        pd.DataFrame(questions)[
            ["question_id", "question", "answer", "evidence_char_pos",
             "evidence_word_pos", "source_title", "type"]
        ].to_csv(corpus_dir / "questions.csv", index=False, encoding="utf-8-sig")

        row = {"corpus": f"corpus_{i:02d}", "n_titles": len(group),
               "words": len(text.split()), "n_questions": len(questions), "titles": group}
        (corpus_dir / "manifest.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
        index.append(row)
        print(f"  corpus_{i:02d}: {row['n_titles']:2} titles, {row['words']:,} words, "
              f"{row['n_questions']:4} questions")

    partition = {
        "seed": args.seed,
        "eval_title": EVAL_TITLE,
        "corpus_titles": sorted(used),
        "qg_titles": sorted(leftover),
        "note": "qg_titles is disjoint from eval_title and corpus_titles — "
                "use it for 08_qg_testing_effect_prep so QG never trains on a "
                "wiki corpus it will later be asked to probe.",
    }
    (out_root / "title_partition.json").write_text(json.dumps(partition, indent=2), encoding="utf-8")
    (out_root / "index.json").write_text(
        json.dumps({"genre": "wiki", "n_corpora": len(index),
                    "corpora": [{k: v for k, v in r.items() if k != "titles"} for r in index]}, indent=2),
        encoding="utf-8",
    )

    print(f"\ntitle partition: eval 1 / corpora {len(used)} / QG {len(leftover)}")
    print(f"Saved {len(index)} corpora to {out_root} "
          f"({sum(r['n_questions'] for r in index):,} questions total)")


if __name__ == "__main__":
    main()
