"""Compares A_generic (04, cnn_dailymail + ROUGE-oracle labels) against
A_matched (04b, this project's own corpora + real evidence-position labels)
on one question: at a given compression ratio, which model's extracted
sentences more often still contain the *real* answer evidence?

This is a free, offline proxy for the eventual full experiment (feed the
compressed context to the actual QA LLM and score real accuracy, per
`03_qa_baseline_3conditions.ipynb`'s method) — it costs zero API calls
because it never calls a QA model, it only checks whether the evidence
*sentence* survives compression. Read it as a first pass to pick a likely
winner; confirm that winner (not both) with the real QA-accuracy method
before finalizing, since sentence survival is necessary but not sufficient
for the downstream model to actually answer correctly.

Deliberately reuses the *fallback* chunker (`_paginate_by_word_count_only`),
not `paginate_semantic` — this needs no embedding API calls, so it costs
nothing to re-run while deciding between the two models. It is not the exact
chunking `rehearse_maintenance_extractive` uses at inference (semantic
pagination), but boundary choice affects *which* sentences land together far
less than it affects whether a sentence is salient at all (04b's own
reasoning for using fixed windows to build A_matched's labels), so it should
not change which model wins.

Usage (from the project root, in the project's own venv):
    python -m src.eval.compare_maintenance_models
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml
from transformers import AutoTokenizer

from src.pipeline.chuncking import (
    _paginate_by_word_count_only,
    chunk_containing_position,
    plain_text_to_paragraphs,
)
from src.pipeline.extractive import SentenceScorer, score_sentences, select_sentences
from src.pipeline.gisting import split_into_sentences
from src.pipeline.types import Chunk

RATIOS = [0.3, 0.5]  # same two points 05_rehearsal_maintenance_train.ipynb reports

# caselaw is excluded for the same reason 04b excludes it from A_matched's
# training data: CaseHOLD's answer is one of five holding statements that do
# not appear in the text, so there is no "sentence containing the evidence"
# to check.
GENRES = {
    "news": (
        "data/processed/news_eval/news_eval_corpus.txt",
        "data/sample/news_eval_questions.csv",
    ),
    "wiki": (
        "data/processed/wiki_eval/wiki_eval_corpus.txt",
        "data/sample/wiki_eval_questions.csv",
    ),
}

MODEL_DIRS = {
    "A_generic": "experiments/rehearsal_maintenance_roberta",
    "A_matched": "experiments/rehearsal_maintenance_evidence",
}


@dataclass
class Question:
    evidence_char_pos: int
    answer: str


def load_questions(path: str) -> list[Question]:
    with open(path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return [
            Question(evidence_char_pos=int(row["evidence_char_pos"]), answer=row["answer"])
            for row in reader
        ]


def chunk_corpus(text: str, max_words: int, min_words: int) -> list[Chunk]:
    """No-API chunking — see module docstring for why this is the right
    trade-off for this comparison specifically."""
    paragraphs = plain_text_to_paragraphs(text)
    return _paginate_by_word_count_only(paragraphs, min_words, max_words)


def evidence_sentence(chunk: Chunk, evidence_char_pos: int) -> str | None:
    """Which of the chunk's sentences contains the corpus-absolute
    evidence_char_pos, or None if it falls in a gap (whitespace normalization
    can shift positions by a few characters — a miss here is logged as
    'not locatable', not scored against either model)."""
    local_pos = evidence_char_pos - chunk.char_start
    if not (0 <= local_pos < len(chunk.text)):
        return None
    cursor = 0
    for sentence in split_into_sentences(chunk.text):
        start = chunk.text.find(sentence, cursor)
        if start == -1:
            continue
        end = start + len(sentence)
        if start <= local_pos < end:
            return sentence
        cursor = end
    return None


def load_model(experiment_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(experiment_dir)
    model = SentenceScorer("roberta-base")
    state_dict = torch.load(Path(experiment_dir) / "sentence_scorer.pt", map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model, tokenizer


def main() -> None:
    chunk_cfg = yaml.safe_load(open("configs/chunking.yaml", encoding="utf-8"))
    max_words, min_words = chunk_cfg["max_words"], chunk_cfg["min_words"]

    models = {name: load_model(path) for name, path in MODEL_DIRS.items()}

    # {(genre, model_name, ratio): [n_survived, n_total]}
    results: dict[tuple[str, str, float], list[int]] = {
        (genre, name, ratio): [0, 0]
        for genre in GENRES
        for name in MODEL_DIRS
        for ratio in RATIOS
    }
    unlocatable = 0

    for genre, (corpus_path, questions_path) in GENRES.items():
        text = Path(corpus_path).read_text(encoding="utf-8")
        chunks = chunk_corpus(text, max_words, min_words)
        questions = load_questions(questions_path)
        print(f"[{genre}] {len(chunks)} chunks, {len(questions)} questions")

        # Score each chunk once per model, reused across ratios and questions
        # that land in the same chunk — select_sentences is cheap, scoring is not.
        chunk_sentences: dict[int, list[str]] = {}
        chunk_scores: dict[tuple[str, int], list[float]] = {}
        for chunk in chunks:
            sentences = [s for s in split_into_sentences(chunk.text) if s.strip()]
            chunk_sentences[chunk.index] = sentences
            if not sentences:
                continue
            for name, (model, tokenizer) in models.items():
                chunk_scores[(name, chunk.index)] = score_sentences(sentences, model, tokenizer)

        for q in questions:
            chunk = chunk_containing_position(chunks, q.evidence_char_pos)
            if chunk is None:
                unlocatable += 1
                continue
            target_sentence = evidence_sentence(chunk, q.evidence_char_pos)
            if target_sentence is None:
                unlocatable += 1
                continue

            sentences = chunk_sentences[chunk.index]
            for name in MODEL_DIRS:
                scores = chunk_scores.get((name, chunk.index))
                if scores is None:
                    continue
                for ratio in RATIOS:
                    kept = select_sentences(sentences, scores, ratio)
                    survived = target_sentence in kept
                    counts = results[(genre, name, ratio)]
                    counts[1] += 1
                    counts[0] += int(survived)

    print(f"\nquestions with no locatable evidence sentence (skipped): {unlocatable}")
    print("\nevidence-sentence retention rate — real answer position vs. extracted text")
    print(f"{'genre':<8}{'model':<12}{'ratio':<8}{'survived/total':<18}{'rate':<8}")
    for genre in GENRES:
        for ratio in RATIOS:
            for name in MODEL_DIRS:
                survived, total = results[(genre, name, ratio)]
                rate = survived / total if total else 0.0
                print(f"{genre:<8}{name:<12}{ratio:<8}{f'{survived}/{total}':<18}{rate:<8.3f}")
        print()

    print("overall (pooled across genres)")
    for ratio in RATIOS:
        for name in MODEL_DIRS:
            survived = sum(results[(g, name, ratio)][0] for g in GENRES)
            total = sum(results[(g, name, ratio)][1] for g in GENRES)
            rate = survived / total if total else 0.0
            print(f"  ratio={ratio}  {name:<12}{survived}/{total} = {rate:.3f}")


if __name__ == "__main__":
    main()
