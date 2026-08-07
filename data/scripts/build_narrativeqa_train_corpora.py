"""Builds N *training* narrativeqa corpora, disjoint from the length-stress /
novel-eval document set, for `06b`'s stage-2 curation.

Why this exists
----------------
`03`'s baseline shows `novel` with the largest relative context-underuse gap
of any genre — full_context 0.494 vs chunked_sequential 0.063 (n=79) — a much
bigger drop-off than wiki's 0.81 vs 0.19. wiki was dropped from `06b`'s
training genres for exactly that reason: its full_context ceiling is already
high, so there is less headroom for elaborative rehearsal to demonstrate
recovering toward it. novel showing the opposite pattern is the argument for
adding it instead of wiki, not just for variety.

Unlike news/wiki/caselaw, narrativeqa corpora are **one book per corpus**,
not several short articles round-robined together. A book is one continuous
narrative — concatenating unrelated books the way `build_news_train_corpora`
concatenates unrelated articles would reintroduce exactly the kind of
cross-document seam `rehearse_elaborative`'s thread memory is not meant to
track, and would undercut the genre's whole reason for being added (testing
whether B's rehearsal helps track a *single* narrative across chunks).

Reuses `01_length_stress_prep.ipynb`'s exact selection logic (boilerplate
stripping, verbatim-answer matching against crowd-written answer variants) —
same rules, applied to books that notebook did not already select, so the two
document sets stay disjoint.

Usage:
    python data/scripts/build_narrativeqa_train_corpora.py [--n-corpora 20]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
from datasets import load_dataset

root = Path(__file__).resolve().parent
while not (root / "src").exists() and root != root.parent:
    root = root.parent

DOC_KIND = "gutenberg"
TARGET_MIN_WORDS = 50_000
TARGET_MAX_WORDS = 55_000
MIN_QUESTIONS_PER_CORPUS = 5
MAX_DOCS_TO_SCAN = 200


def strip_gutenberg_boilerplate(raw_text: str) -> str:
    """Same as `01_length_stress_prep.ipynb` — license text and transcriber
    notes outside the START/END markers are not part of the narrative and
    would skew every answer position computed against the document."""
    start_idx = raw_text.find("*** START")
    end_idx = raw_text.find("*** END")
    if start_idx == -1 or end_idx == -1:
        return raw_text.strip()
    return raw_text[raw_text.find("\n", start_idx) : end_idx].strip()


def word_index(text: str, char_pos: int) -> int:
    return len(text[:char_pos].split()) + 1


def find_verbatim_answer(answer_variants: list[str], text: str) -> tuple[str, int] | None:
    """First answer variant that appears verbatim in text, with its char
    position — narrativeqa answers are crowd-written free text, so most are
    paraphrases; only the verbatim-findable subset has a position we can
    trust as training/probe evidence."""
    for candidate in answer_variants:
        pos = text.find(candidate)
        if pos >= 0:
            return candidate, pos
    return None


def truncate_to_word_count(paragraphs: list, full_text: str, target_words: int) -> str:
    """Cut at the nearest paragraph boundary without exceeding target_words —
    same rule as `01`'s ladder, so a corpus never ends mid-paragraph."""
    cumulative_words = 0
    cut_char_pos = 0
    for i, p in enumerate(paragraphs):
        p_words = len(p.text.split())
        if cumulative_words > 0 and cumulative_words + p_words > target_words:
            break
        cumulative_words += p_words
        next_offset = paragraphs[i + 1].char_offset if i + 1 < len(paragraphs) else len(full_text)
        cut_char_pos = next_offset
    return full_text[:cut_char_pos]


def load_already_used_doc_ids() -> set[str]:
    path = root / "data" / "sample" / "narrativeqa_length_stress.csv"
    if not path.exists():
        return set()
    return set(pd.read_csv(path)["doc_id"].unique())


def main() -> None:
    import sys

    sys.path.insert(0, str(root))
    from src.pipeline.chuncking import plain_text_to_paragraphs

    parser = argparse.ArgumentParser()
    parser.add_argument("--n-corpora", type=int, default=20)
    args = parser.parse_args()

    exclude_doc_ids = load_already_used_doc_ids()
    print(f"excluding {len(exclude_doc_ids)} documents already used by 01_length_stress_prep")

    print("Loading deepmind/narrativeqa (train split)...")
    ds = load_dataset("deepmind/narrativeqa", split="train")
    flat = ds.flatten()

    questions_by_doc: dict[str, list[dict]] = defaultdict(list)
    first_row_index: dict[str, int] = {}
    for i, (doc_id, kind, question, answers) in enumerate(
        zip(flat["document.id"], flat["document.kind"], flat["question.text"], flat["answers"])
    ):
        if kind != DOC_KIND or doc_id in exclude_doc_ids:
            continue
        first_row_index.setdefault(doc_id, i)
        questions_by_doc[doc_id].append(
            {"question": question, "answer_variants": [a["text"] for a in answers]}
        )

    print(f"candidate {DOC_KIND} documents (excluding already-used): {len(questions_by_doc)}")

    ranked = sorted(questions_by_doc.items(), key=lambda kv: len(kv[1]), reverse=True)

    out_root = root / "data" / "processed" / "narrativeqa_train"
    out_root.mkdir(parents=True, exist_ok=True)
    index = []

    for doc_id, rows in ranked[:MAX_DOCS_TO_SCAN]:
        if len(index) >= args.n_corpora:
            break

        text = strip_gutenberg_boilerplate(ds[first_row_index[doc_id]]["document"]["text"])
        if len(text.split()) < TARGET_MIN_WORDS:
            continue

        paragraphs = plain_text_to_paragraphs(text)
        corpus_text = truncate_to_word_count(paragraphs, text, TARGET_MAX_WORDS)
        if len(corpus_text.split()) < TARGET_MIN_WORDS:
            continue  # paragraph boundaries landed short of the band this time

        questions = []
        for row in rows:
            found = find_verbatim_answer(row["answer_variants"], corpus_text)
            if found is None:
                continue
            answer_text, char_pos = found
            questions.append(
                {
                    "question": row["question"],
                    "answer": answer_text,
                    "evidence_char_pos": char_pos,
                    "evidence_word_pos": word_index(corpus_text, char_pos),
                    "type": "narrativeqa_extractive_subset",
                }
            )

        if len(questions) < MIN_QUESTIONS_PER_CORPUS:
            continue

        questions.sort(key=lambda q: q["evidence_char_pos"])
        for i, q in enumerate(questions, start=1):
            q["question_id"] = i

        corpus_dir = out_root / f"corpus_{len(index):02d}"
        corpus_dir.mkdir(exist_ok=True)
        (corpus_dir / "corpus.txt").write_text(corpus_text, encoding="utf-8")
        pd.DataFrame(questions)[
            ["question_id", "question", "answer", "evidence_char_pos", "evidence_word_pos", "type"]
        ].to_csv(corpus_dir / "questions.csv", index=False, encoding="utf-8-sig")

        row_summary = {
            "corpus": f"corpus_{len(index):02d}", "doc_id": doc_id,
            "words": len(corpus_text.split()), "n_questions": len(questions),
        }
        (corpus_dir / "manifest.json").write_text(json.dumps(row_summary, indent=2), encoding="utf-8")
        index.append(row_summary)
        print(f"  corpus_{row_summary['corpus'][-2:]}: {doc_id[:8]}  "
              f"{row_summary['words']:,} words, {row_summary['n_questions']:3} questions")

    if len(index) < args.n_corpora:
        print(f"\n! only found {len(index)}/{args.n_corpora} usable books "
              f"(scanned {min(MAX_DOCS_TO_SCAN, len(ranked))} candidates) — "
              "raise --n-corpora's caller expectations or MAX_DOCS_TO_SCAN")

    (out_root / "index.json").write_text(
        json.dumps({"genre": "narrativeqa", "n_corpora": len(index),
                    "excluded_doc_ids": sorted(exclude_doc_ids), "corpora": index}, indent=2),
        encoding="utf-8",
    )
    total_q = sum(r["n_questions"] for r in index)
    print(f"\nSaved {len(index)} corpora to {out_root}  ({total_q:,} questions total)")


if __name__ == "__main__":
    main()
