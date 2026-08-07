"""Generates synthetic extractive QA probes for `caselaw_train` corpora.

Why this exists
----------------
CaseHOLD ships multiple-choice holdings (`correct_option_index` + 5 options),
with no answer span at all — `06b`'s retention probe
(`src/pipeline/curation.py`'s `retention_probes`) needs a literal answer
string to F1-score against the rolling summary, which the native format
cannot supply. This asks an LLM to extract short factual questions (party
names, courts, citations, dates, statutes, dollar amounts, holdings stated
in the passage) whose answers are exact verbatim spans in the passage — the
same "trust only verbatim matches" rule `01_length_stress_prep.ipynb` uses
for narrativeqa's crowd-written answers, for the same reason: an answer this
script cannot locate in the source text is not usable evidence for the probe.

Output is a **separate** file (`questions_extractive.csv`) next to the
existing `questions.csv`, not a replacement — the native CaseHOLD
multiple-choice questions stay untouched for anything else that reads them
(e.g. `03`'s eval baseline). `06b`'s `build_documents()` prefers
`questions_extractive.csv` when present, falling back to `questions.csv`.

Uses the smaller/faster `llama-3.1-8b-instruct`, not the stage-2 teacher
model — this is basic factual extraction, not elaborative rehearsal, and
does not need the larger model's quality (already validated elsewhere in
this project at ~0.74s/call).

Usage:
    python data/scripts/generate_caselaw_extractive_probes.py [--corpora 20]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

root = Path(__file__).resolve().parent
while not (root / "src").exists() and root != root.parent:
    root = root.parent
sys.path.insert(0, str(root))

from src.pipeline.teacher import chat_completion, complete_with_retry  # noqa: E402

MODEL = "meta/llama-3.1-8b-instruct"
WINDOW_WORDS = 2500
PAIRS_PER_WINDOW = 2

QG_SYSTEM_PROMPT = (
    "You extract simple factual questions from legal case text. Given a passage, "
    "write short factual questions (party names, courts, citations, dates, statutes, "
    "dollar amounts, or holdings explicitly stated) whose answers are SHORT PHRASES "
    "copied EXACTLY, word-for-word, from the passage. Do not paraphrase the answer. "
    'Respond with ONLY a JSON array: [{"question": "...", "answer": "..."}, ...]. '
    "No other text."
)


def _config() -> dict:
    return {
        "model": MODEL, "temperature": 0.0, "max_tokens": 512, "timeout": 60.0,
        "max_retries": 3, "retry_base_delay": 1.0, "retry_max_delay": 15.0,
        "api_key_env": "NVIDIA_NIM_API_KEY",
    }


def _windows(text: str, window_words: int) -> list[str]:
    """Word-boundary slices. These only exist to prompt the QG model — unlike
    a training target, they are never shown verbatim to anything position
    sensitive, so normalized whitespace from `" ".join` is fine here."""
    words = text.split()
    return [" ".join(words[i : i + window_words]) for i in range(0, len(words), window_words)]


def _parse_pairs(raw: str) -> list[dict]:
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    return [d for d in data if isinstance(d, dict) and "question" in d and "answer" in d]


def generate_for_corpus(corpus_text: str, config: dict) -> list[dict]:
    questions = []
    for window in _windows(corpus_text, WINDOW_WORDS):
        messages = [
            {"role": "system", "content": QG_SYSTEM_PROMPT},
            {"role": "user", "content": f"Passage:\n{window}\n\nGenerate {PAIRS_PER_WINDOW} questions."},
        ]
        raw = complete_with_retry(messages, config, chat_completion)
        if raw is None:
            continue
        for pair in _parse_pairs(raw):
            answer = str(pair["answer"]).strip()
            if not answer:
                continue
            pos = corpus_text.find(answer)
            if pos < 0:
                continue  # not verbatim-findable — discarded, same rule narrativeqa uses
            questions.append(
                {
                    "question": str(pair["question"]).strip(),
                    "answer": answer,
                    "evidence_char_pos": pos,
                    "evidence_word_pos": len(corpus_text[:pos].split()) + 1,
                    "type": "caselaw_synthetic_extractive",
                }
            )
    return questions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpora", type=int, default=20)
    args = parser.parse_args()

    config = _config()
    out_root = root / "data" / "processed" / "caselaw_train"
    corpus_dirs = sorted(out_root.glob("corpus_*"))[: args.corpora]
    if not corpus_dirs:
        raise SystemExit(f"no corpora under {out_root} — run build_caselaw_train_corpora.py first")

    total = 0
    for corpus_dir in corpus_dirs:
        text = (corpus_dir / "corpus.txt").read_text(encoding="utf-8")
        questions = generate_for_corpus(text, config)
        questions.sort(key=lambda q: q["evidence_char_pos"])
        for i, q in enumerate(questions, start=1):
            q["question_id"] = i

        pd.DataFrame(questions)[
            ["question_id", "question", "answer", "evidence_char_pos", "evidence_word_pos", "type"]
        ].to_csv(corpus_dir / "questions_extractive.csv", index=False, encoding="utf-8-sig")

        total += len(questions)
        print(f"  {corpus_dir.name}: {len(questions)} verbatim-verified questions")

    print(f"\ntotal: {total} questions across {len(corpus_dirs)} corpora")


if __name__ == "__main__":
    main()
