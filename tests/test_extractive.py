"""Unit tests for extractive (maintenance rehearsal A, RoBERTa sentence scorer) —
all run against a fake tokenizer/encoder, no real model download or network required.

`SentenceScorer.__init__` pulls roberta-base from the hub, so the forward-pass
tests build the module without it and swap in `_PositionEncoder`. That keeps the
tested surface on this file's own logic — the `<s>`-position gather and the
masked loss — rather than on the pretrained weights.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import BatchEncoding

from src.pipeline.extractive import (
    SentenceScorer,
    collate,
    encode_sentences,
    score_sentences,
    seam_report,
    select_sentences,
    select_sentences_windowed,
)

BOS, PAD, EOS = 0, 1, 2


class _WordTokenizer:
    """A fake RoBERTa-style tokenizer: one id per whitespace-separated word.

    No subword splitting, so token counts are predictable and the length-budget
    behaviour of `encode_sentences` can be asserted exactly.
    """

    bos_token_id = BOS
    pad_token_id = PAD
    eos_token_id = EOS

    def __init__(self):
        self._id_of: dict[str, int] = {}

    def __call__(self, text, add_special_tokens=False):
        ids = [self._id_of.setdefault(w, len(self._id_of) + 10) for w in text.split()]
        return BatchEncoding({"input_ids": ids})


class _PositionEncoder(nn.Module):
    """Stands in for RoBERTa: every token's hidden state is its own position
    index, repeated across the hidden dim.

    Paired with a sum-classifier this makes each logit equal to the position it
    was gathered from, so the tests can read off *which* token the head scored.
    """

    def __init__(self, hidden: int):
        super().__init__()
        self.hidden = hidden

    def forward(self, input_ids, attention_mask):
        positions = torch.arange(input_ids.size(1), dtype=torch.float)
        states = positions.view(1, -1, 1).repeat(input_ids.size(0), 1, self.hidden)
        return SimpleNamespace(last_hidden_state=states)


def _make_scorer(hidden: int = 4) -> SentenceScorer:
    """A SentenceScorer whose __init__ (and its roberta-base download) is skipped."""
    scorer = SentenceScorer.__new__(SentenceScorer)
    nn.Module.__init__(scorer)
    scorer.encoder = _PositionEncoder(hidden)
    scorer.dropout = nn.Dropout(0.0)
    scorer.classifier = nn.Linear(hidden, 1)
    with torch.no_grad():
        # sum(state)/hidden == the position index -> logit reveals the gathered slot
        scorer.classifier.weight.fill_(1.0 / hidden)
        scorer.classifier.bias.zero_()
    return scorer.eval()


@pytest.fixture
def tokenizer():
    return _WordTokenizer()


# --- encode_sentences ---------------------------------------------------------


def test_encode_sentences_wraps_each_sentence_in_bos_eos(tokenizer):
    encoded = encode_sentences(["the cat sat", "it rained"], tokenizer)

    ids = encoded["input_ids"]
    assert len(ids) == (1 + 3 + 1) + (1 + 2 + 1)
    assert ids[0] == BOS and ids[4] == EOS
    assert ids[5] == BOS and ids[8] == EOS


def test_encode_sentences_cls_positions_point_at_each_bos(tokenizer):
    encoded = encode_sentences(["the cat sat", "it rained", "a b c d"], tokenizer)

    assert encoded["cls_positions"] == [0, 5, 9]
    assert all(encoded["input_ids"][p] == BOS for p in encoded["cls_positions"])
    assert encoded["n_kept"] == 3


def test_encode_sentences_attention_mask_matches_input_length(tokenizer):
    encoded = encode_sentences(["the cat sat", "it rained"], tokenizer)

    assert encoded["attention_mask"] == [1] * len(encoded["input_ids"])


def test_encode_sentences_drops_overflowing_sentences_instead_of_truncating(tokenizer):
    # Each sentence costs 2 words + bos/eos = 4 tokens; a budget of 9 fits two.
    encoded = encode_sentences(["a b", "c d", "e f"], tokenizer, max_tokens=9)

    assert encoded["n_kept"] == 2
    assert encoded["cls_positions"] == [0, 4]
    # the third sentence is absent entirely, not cut in half
    assert len(encoded["input_ids"]) == 8


def test_encode_sentences_keeps_nothing_when_first_sentence_alone_overflows(tokenizer):
    encoded = encode_sentences(["a b c d e"], tokenizer, max_tokens=4)

    assert encoded["n_kept"] == 0
    assert encoded["cls_positions"] == []
    assert encoded["input_ids"] == []


def test_encode_sentences_handles_empty_input(tokenizer):
    encoded = encode_sentences([], tokenizer)

    assert encoded == {"input_ids": [], "attention_mask": [], "cls_positions": [], "n_kept": 0}


# --- collate ------------------------------------------------------------------


@pytest.fixture
def ragged_batch(tokenizer):
    """Two items ragged in both dimensions: sequence length and sentence count."""
    return [
        encode_sentences(["the cat sat", "it rained"], tokenizer),
        encode_sentences(["a b c d e"], tokenizer),
    ]


def test_collate_pads_sequences_to_the_longest_item(ragged_batch):
    batch = collate(ragged_batch, pad_token_id=PAD)

    assert batch["input_ids"].shape == (2, 9)
    assert batch["input_ids"][1, 7:].tolist() == [PAD, PAD]
    assert batch["attention_mask"][1].tolist() == [1] * 7 + [0, 0]


def test_collate_marks_padded_sentence_slots_in_cls_mask(ragged_batch):
    batch = collate(ragged_batch, pad_token_id=PAD)

    assert batch["cls_mask"].tolist() == [[1, 1], [1, 0]]
    # padded slots point at position 0, which cls_mask keeps out of the loss
    assert batch["cls_positions"].tolist() == [[0, 5], [0, 0]]


def test_collate_pads_labels_and_casts_them_to_float(ragged_batch):
    batch = collate(
        [{**ragged_batch[0], "labels": [1, 0]}, {**ragged_batch[1], "labels": [1]}],
        pad_token_id=PAD,
    )

    assert batch["labels"].dtype == torch.float
    assert batch["labels"].tolist() == [[1.0, 0.0], [1.0, 0.0]]


def test_collate_omits_labels_when_the_batch_has_none(ragged_batch):
    assert "labels" not in collate(ragged_batch, pad_token_id=PAD)


# --- SentenceScorer.forward ---------------------------------------------------


def test_forward_returns_one_logit_per_sentence_slot(ragged_batch):
    batch = collate(ragged_batch, pad_token_id=PAD)
    out = _make_scorer()(**batch)

    assert out["logits"].shape == (2, 2)
    assert out["loss"] is None  # no labels -> no loss


def test_forward_gathers_the_state_at_each_cls_position(ragged_batch):
    batch = collate(ragged_batch, pad_token_id=PAD)
    logits = _make_scorer()(**batch)["logits"]

    # _PositionEncoder + sum-classifier -> logit == the gathered position index
    assert logits.tolist() == batch["cls_positions"].float().tolist()


def test_forward_loss_ignores_padded_sentence_slots(ragged_batch):
    batch = collate(
        [{**ragged_batch[0], "labels": [1, 0]}, {**ragged_batch[1], "labels": [1]}],
        pad_token_id=PAD,
    )
    scorer = _make_scorer()
    loss = scorer(**batch)["loss"]

    flipped = {**batch, "labels": batch["labels"].clone()}
    flipped["labels"][1, 1] = 1.0  # the padded slot, whose label must not matter
    assert torch.allclose(scorer(**flipped)["loss"], loss)


def test_forward_loss_is_the_mean_over_real_slots_only(ragged_batch):
    batch = collate(
        [{**ragged_batch[0], "labels": [1, 0]}, {**ragged_batch[1], "labels": [1]}],
        pad_token_id=PAD,
    )
    out = _make_scorer()(**batch)

    per_sentence = nn.functional.binary_cross_entropy_with_logits(
        out["logits"], batch["labels"], reduction="none"
    )
    real = per_sentence[batch["cls_mask"].bool()]
    assert torch.allclose(out["loss"], real.mean())


# --- score_sentences ----------------------------------------------------------


def test_score_sentences_returns_one_score_per_sentence(tokenizer):
    sentences = ["the cat sat", "it rained", "a b c d"]
    scores = score_sentences(sentences, _make_scorer(), tokenizer)

    assert len(scores) == len(sentences)
    assert all(s != float("-inf") for s in scores)


def test_score_sentences_gives_minus_inf_to_sentences_dropped_for_length(tokenizer):
    # 200 words + bos/eos = 202 tokens each; two fit under 512, the third does not.
    sentences = [" ".join(f"w{i}_{j}" for j in range(200)) for i in range(3)]
    scores = score_sentences(sentences, _make_scorer(), tokenizer)

    assert len(scores) == 3
    assert scores[2] == float("-inf")
    assert scores[0] != float("-inf") and scores[1] != float("-inf")


def test_score_sentences_returns_all_minus_inf_when_nothing_fits(tokenizer):
    sentences = [" ".join(f"w{j}" for j in range(600))]
    assert score_sentences(sentences, _make_scorer(), tokenizer) == [float("-inf")]


def test_score_sentences_is_query_agnostic(tokenizer):
    """Module contract: nothing about a downstream question enters this model."""
    import inspect

    params = set(inspect.signature(score_sentences).parameters)
    forbidden = {p for p in params if "question" in p.lower() or "query" in p.lower()}
    assert not forbidden, f"query-agnostic violation — found forbidden parameter(s): {forbidden}"


# --- select_sentences ---------------------------------------------------------


def test_select_sentences_keeps_the_top_scoring_ones_in_source_order():
    sentences = ["first", "second", "third", "fourth"]
    scores = [0.1, 0.9, 0.2, 0.8]

    # ceil(0.5 * 4) = 2 -> "second" and "fourth", but returned in source order
    assert select_sentences(sentences, scores, ratio=0.5) == ["second", "fourth"]


def test_select_sentences_rounds_the_budget_up():
    sentences = ["a", "b", "c"]
    # ceil(0.34 * 3) = 2, not 1
    assert len(select_sentences(sentences, [0.3, 0.2, 0.1], ratio=0.34)) == 2


def test_select_sentences_always_keeps_at_least_one():
    sentences = ["a", "b", "c"]
    assert select_sentences(sentences, [0.1, 0.9, 0.2], ratio=0.0) == ["b"]


def test_select_sentences_with_ratio_one_returns_everything_in_order():
    sentences = ["a", "b", "c"]
    assert select_sentences(sentences, [0.1, 0.9, 0.2], ratio=1.0) == sentences


def test_select_sentences_ranks_minus_inf_sentences_last():
    sentences = ["dropped", "kept", "also dropped"]
    scores = [float("-inf"), 0.5, float("-inf")]
    # ceil(0.1 * 3) = 1 -> only the scored sentence fits
    assert select_sentences(sentences, scores, ratio=0.1) == ["kept"]


def test_select_sentences_fills_leftover_budget_with_minus_inf_sentences():
    """The budget is unconditional: if it exceeds the number of scored sentences,
    length-dropped ones fill the rest. They are still verbatim source sentences,
    so the chunk stays faithful — it just contains sentences the scorer never saw.
    """
    sentences = ["dropped", "kept", "also dropped"]
    scores = [float("-inf"), 0.5, float("-inf")]

    selected = select_sentences(sentences, scores, ratio=0.34)  # ceil -> budget of 2
    assert len(selected) == 2
    assert "kept" in selected
    assert selected == [s for s in sentences if s in selected]  # still source order


def test_select_sentences_handles_empty_input():
    assert select_sentences([], [], ratio=0.5) == []


# --- select_sentences_windowed -------------------------------------------------


def test_select_sentences_windowed_with_window_zero_matches_select_sentences():
    sentences = ["first", "second", "third", "fourth"]
    scores = [0.1, 0.9, 0.2, 0.8]
    assert select_sentences_windowed(sentences, scores, ratio=0.5, window=0) == select_sentences(
        sentences, scores, ratio=0.5
    )


def test_select_sentences_windowed_pulls_in_one_neighbor_each_side():
    sentences = ["a", "b", "c", "d", "e"]
    scores = [0.1, 0.1, 0.9, 0.1, 0.1]  # only "c" (index 2) is top-scored
    # ceil(0.2 * 5) = 1 -> budget is just "c", window=1 pads to b, c, d
    assert select_sentences_windowed(sentences, scores, ratio=0.2, window=1) == ["b", "c", "d"]


def test_select_sentences_windowed_clamps_at_document_edges():
    sentences = ["a", "b", "c"]
    scores = [0.9, 0.1, 0.1]  # ceil(0.1 * 3) = 1 -> only "a" (index 0) is top-scored
    # window=1 would reach index -1; must clamp to the start of the document instead
    assert select_sentences_windowed(sentences, scores, ratio=0.1, window=1) == ["a", "b"]


def test_select_sentences_windowed_deduplicates_overlapping_neighborhoods():
    sentences = ["a", "b", "c", "d", "e"]
    scores = [0.9, 0.1, 0.1, 0.1, 0.9]  # "a" and "e" both top-scored, ratio picks both
    # window=1 around each: {0,1} and {3,4} -- no overlap, but must not duplicate "e"
    assert select_sentences_windowed(sentences, scores, ratio=0.4, window=1) == [
        "a",
        "b",
        "d",
        "e",
    ]


def test_select_sentences_windowed_handles_empty_input():
    assert select_sentences_windowed([], [], ratio=0.5, window=1) == []


# --- seam_report --------------------------------------------------------------

SOURCE = "The cat looks out the window. The stock index plunged. The cat takes a nap."


def test_seam_report_clean_for_adjacent_verbatim_sentences():
    kept = ["The cat looks out the window.", "The stock index plunged."]
    report = seam_report(kept, SOURCE)

    assert report["ok"] is True
    assert report["not_verbatim"] == []
    assert report["novel_total"] == 0
    assert report["novel_ratio"] == 0.0
    assert report["seams"] == 1


def test_seam_report_tolerates_novel_ngrams_that_span_a_seam():
    """Skipping a sentence creates n-grams across the gap — unavoidable, not a defect."""
    kept = ["The cat looks out the window.", "The cat takes a nap."]  # non-adjacent
    report = seam_report(kept, SOURCE)

    assert report["novel_total"] > 0  # the seam produced novel trigrams
    assert report["novel_inside_sentence"] == []  # but none inside a sentence
    assert report["ok"] is True
    assert report["not_verbatim"] == []


def test_seam_report_flags_a_sentence_that_was_altered():
    kept = ["The cat stares out the window."]  # a paraphrase, not in the source
    report = seam_report(kept, SOURCE)

    assert report["ok"] is False
    assert report["not_verbatim"] == kept
    assert report["novel_inside_sentence"]  # the novel n-grams sit inside the sentence


def test_seam_report_normalizes_whitespace_before_comparing():
    kept = ["The  cat\nlooks   out the window."]
    report = seam_report(kept, SOURCE)

    assert report["not_verbatim"] == []
    assert report["ok"] is True


def test_seam_report_counts_seams_as_gaps_between_kept_sentences():
    assert seam_report([], SOURCE)["seams"] == 0
    assert seam_report(["The stock index plunged."], SOURCE)["seams"] == 0
    assert seam_report(["The cat takes a nap.", "The stock index plunged."], SOURCE)["seams"] == 1


def test_seam_report_handles_empty_selection():
    report = seam_report([], SOURCE)

    assert report["ok"] is True
    assert report["novel_ratio"] == 0.0  # no n-grams -> no division by zero
    assert report["novel_total"] == 0
