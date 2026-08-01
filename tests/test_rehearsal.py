"""Unit tests for rehearsal (maintenance rehearsal A) — all run against a fake
model/tokenizer/embed_fn, no real model download, API key, or network required.
"""

import inspect
import os

import pytest
import torch
from transformers import BatchEncoding

from src.pipeline.embeddings import EmbeddingAPIError, load_config
from src.pipeline.rehearsal import (
    PASSAGE_FIELD,
    RetrievalRecord,
    _snap_to_verbatim,
    format_elaborative_input,
    novel_ngram_ratio,
    rehearse_elaborative,
    rehearse_maintenance,
    rehearse_maintenance_extractive,
    retrieval_span_report,
    retrieve_aligned_context,
)
from src.pipeline.types import Chunk


@pytest.fixture
def config():
    cfg = load_config("configs/importance_filter.yaml")
    os.environ[cfg["api_key_env"]] = "dummy-key-for-test"
    return cfg


class _EchoTokenizer:
    """A fake tokenizer that registers/looks up chunk.text as a single integer
    ID — with no real subword splitting, this only verifies
    rehearse_maintenance's tokenize->generate->decode control flow.
    """

    def __init__(self):
        self._id_of: dict[str, int] = {}
        self._text_of: dict[int, str] = {}

    def __call__(self, text, return_tensors="pt", truncation=True, max_length=None):
        if text not in self._id_of:
            new_id = len(self._id_of)
            self._id_of[text] = new_id
            self._text_of[new_id] = text
        token_id = self._id_of[text]
        return BatchEncoding(
            {"input_ids": torch.tensor([[token_id]]), "attention_mask": torch.tensor([[1]])}
        )

    def decode(self, ids, skip_special_tokens=True):
        return self._text_of[int(ids[0])]


class _EchoModel:
    """A fake model that returns the input tokens unchanged — makes
    decode(generate(encode(t))) == t hold, so it verifies the "generated text
    is identical to the source" path (where verbatim snapping isn't needed).
    """

    def __init__(self):
        self._param = torch.nn.Parameter(torch.zeros(1))

    def eval(self):
        return self

    def parameters(self):
        yield self._param

    def generate(self, input_ids, attention_mask=None, max_new_tokens=None):
        return input_ids


def _poison_embed_fn(texts, input_type, model, truncate, api_key, timeout=30.0, dimensions=None):
    raise AssertionError("embed_fn was called in a situation where it must not be called")


@pytest.fixture
def sample_chunk():
    return Chunk(
        text="The cat looks out the window. The stock index plunged.",
        index=2,
        paragraph_indices=[5, 6],
        char_start=100,
        char_end=150,
    )


def test_rehearse_maintenance_is_query_agnostic():
    """Guide §2 hard constraint: must not have a question/query-like parameter."""
    params = set(inspect.signature(rehearse_maintenance).parameters)
    forbidden = {p for p in params if "question" in p.lower() or "query" in p.lower()}
    assert not forbidden, f"query-agnostic violation — found forbidden parameter(s): {forbidden}"


def test_rehearse_maintenance_preserves_position_metadata(sample_chunk, config):
    # Echo model means generated == source -> every sentence is already verbatim, so embed_fn must not be called
    result = rehearse_maintenance(
        [sample_chunk],
        model=_EchoModel(),
        tokenizer=_EchoTokenizer(),
        embed_cfg=config,
        embed_fn=_poison_embed_fn,
    )

    assert len(result) == 1
    out = result[0]
    assert out.index == sample_chunk.index
    assert out.paragraph_indices == sample_chunk.paragraph_indices
    assert out.char_start == sample_chunk.char_start
    assert out.char_end == sample_chunk.char_end


def test_rehearse_maintenance_sets_original_text_to_pre_compression_text(sample_chunk, config):
    result = rehearse_maintenance(
        [sample_chunk],
        model=_EchoModel(),
        tokenizer=_EchoTokenizer(),
        embed_cfg=config,
        embed_fn=_poison_embed_fn,
    )
    assert result[0].original_text == sample_chunk.text


def test_rehearse_maintenance_handles_multiple_chunks_independently(config):
    chunks = [
        Chunk(text="This is the first chunk's sentence.", index=0),
        Chunk(text="This is the second chunk's sentence.", index=1),
    ]
    result = rehearse_maintenance(
        chunks, model=_EchoModel(), tokenizer=_EchoTokenizer(), embed_cfg=config, embed_fn=_poison_embed_fn
    )
    assert [c.original_text for c in result] == [c.text for c in chunks]
    assert [c.index for c in result] == [0, 1]


# --------------------------------------------------------------------------
# Maintenance rehearsal (A), extractive path — the canonical one
# --------------------------------------------------------------------------


def _fixed_score_fn(scores: list[float]):
    """Stands in for the RoBERTa scorer so selection can be checked exactly."""

    def _score(sentences, model, tokenizer):
        assert len(scores) == len(sentences), f"expected {len(scores)} sentences, got {len(sentences)}"
        return scores

    return _score


@pytest.fixture
def four_sentence_chunk():
    return Chunk(
        text="Alpha opens the report. Beta carries the finding. Gamma is an aside. Delta closes it.",
        index=3,
        paragraph_indices=[7, 8],
        char_start=200,
        char_end=290,
    )


def test_rehearse_maintenance_extractive_is_query_agnostic():
    params = set(inspect.signature(rehearse_maintenance_extractive).parameters)
    forbidden = {p for p in params if "question" in p.lower() or "query" in p.lower()}
    assert not forbidden, f"query-agnostic violation: {forbidden}"


def test_rehearse_maintenance_extractive_needs_no_embedding_api():
    """The point of replacing the seq2seq path: verbatim retention is
    structural, so there is no embedding call to fail and no degraded mode."""
    params = set(inspect.signature(rehearse_maintenance_extractive).parameters)
    assert "embed_cfg" not in params
    assert "embed_fn" not in params


def test_rehearse_maintenance_extractive_keeps_top_k_in_source_order(four_sentence_chunk):
    # ratio 0.5 over 4 sentences -> budget 2 -> the two highest-scoring ones,
    # restored to the order they appear in the source.
    result = rehearse_maintenance_extractive(
        [four_sentence_chunk],
        model=_EchoModel(),
        tokenizer=_TopicTokenizer(),
        compression_ratio=0.5,
        score_fn=_fixed_score_fn([0.1, 0.9, 0.2, 0.8]),
    )
    assert result[0].text == "Beta carries the finding. Delta closes it."


def test_rehearse_maintenance_extractive_output_is_verbatim(four_sentence_chunk):
    """Structural guarantee: every kept sentence appears in the source."""
    result = rehearse_maintenance_extractive(
        [four_sentence_chunk], model=_EchoModel(), tokenizer=_TopicTokenizer(),
        compression_ratio=0.5, score_fn=_fixed_score_fn([0.1, 0.9, 0.2, 0.8]),
    )
    for sentence in result[0].text.split(". "):
        assert sentence.strip(". ") in four_sentence_chunk.text


def test_rehearse_maintenance_extractive_preserves_position_metadata(four_sentence_chunk):
    result = rehearse_maintenance_extractive(
        [four_sentence_chunk], model=_EchoModel(), tokenizer=_TopicTokenizer(),
        compression_ratio=0.5, score_fn=_fixed_score_fn([0.1, 0.9, 0.2, 0.8]),
    )
    out = result[0]
    assert (out.index, out.paragraph_indices, out.char_start, out.char_end) == (
        four_sentence_chunk.index, four_sentence_chunk.paragraph_indices,
        four_sentence_chunk.char_start, four_sentence_chunk.char_end,
    )
    assert out.original_text == four_sentence_chunk.text


def test_rehearse_maintenance_extractive_without_a_ratio_keeps_everything(four_sentence_chunk):
    result = rehearse_maintenance_extractive(
        [four_sentence_chunk], model=_EchoModel(), tokenizer=_TopicTokenizer(),
        score_fn=_fixed_score_fn([0.1, 0.9, 0.2, 0.8]),
    )
    assert result[0].text == four_sentence_chunk.text


def test_rehearse_maintenance_extractive_uses_the_dependent_ratio(four_sentence_chunk):
    """K/|D| with the word-counting fake tokenizer: 15 words in the chunk, so
    K=7 gives R=0.467 -> ceil(0.467 * 4) = 2 of 4 sentences."""
    result = rehearse_maintenance_extractive(
        [four_sentence_chunk], model=_EchoModel(), tokenizer=_TopicTokenizer(),
        target_context_tokens=7, score_fn=_fixed_score_fn([0.1, 0.9, 0.2, 0.8]),
    )
    assert result[0].text == "Beta carries the finding. Delta closes it."


def test_rehearse_maintenance_extractive_keeps_a_blank_chunk_intact():
    """An empty output would break the position bookkeeping later stages read."""
    blank = Chunk(text="   ", index=1, char_start=5, char_end=8)
    result = rehearse_maintenance_extractive(
        [blank], model=_EchoModel(), tokenizer=_TopicTokenizer(),
        compression_ratio=0.5, score_fn=_fixed_score_fn([]),
    )
    assert result[0].text == "   "
    assert result[0].index == 1


def _fake_embed_fixed_vectors(vectors_by_text: dict):
    def _embed(texts, input_type, model, truncate, api_key, timeout=30.0, dimensions=None):
        return [vectors_by_text[t] for t in texts]

    return _embed


def test_snap_to_verbatim_keeps_exact_matches_without_calling_embed_fn():
    generated = ["The cat looks out the window.", "The stock index plunged."]
    source = ["The cat looks out the window.", "The stock index plunged.", "The real estate market froze."]

    result = _snap_to_verbatim(generated, source, embed_cfg={}, embed_fn=_poison_embed_fn)
    assert result == generated


def test_snap_to_verbatim_replaces_non_matching_sentence_with_closest_by_cosine(config):
    generated = ["The cat stares outside."]  # a paraphrase not present in the source
    source = ["The cat looks out the window.", "The real estate market froze."]

    vectors = {
        "The cat stares outside.": [1.0, 0.0],
        "The cat looks out the window.": [0.9, 0.1],  # higher cosine similarity
        "The real estate market froze.": [0.0, 1.0],
    }
    result = _snap_to_verbatim(generated, source, embed_cfg=config, embed_fn=_fake_embed_fixed_vectors(vectors))
    assert result == ["The cat looks out the window."]


def test_snap_to_verbatim_falls_back_to_generated_on_embedding_failure(config):
    def always_fails(*args, **kwargs):
        raise EmbeddingAPIError("simulated embedding outage")

    config = dict(config)
    config["max_retries"] = 0
    generated = ["A new phrase not in the source."]
    source = ["One source sentence."]

    result = _snap_to_verbatim(generated, source, embed_cfg=config, embed_fn=always_fails)
    assert result == generated  # doesn't die, returns the generated text unchanged


def test_novel_ngram_ratio_is_zero_for_pure_verbatim_text():
    source = "the cat looks out the window the market fell hard today"
    generated = "the cat looks"
    assert novel_ngram_ratio(generated, source, n=3) == 0.0


def test_novel_ngram_ratio_is_positive_for_novel_text():
    source = "the cat looks out the window the market fell today"
    generated = "completely different words forming a brand new sentence"
    assert novel_ngram_ratio(generated, source, n=3) == 1.0


def test_novel_ngram_ratio_short_generation_returns_zero():
    # too short a generation to form an n-gram -> 0 (a safe default instead of an exception)
    assert novel_ngram_ratio("one word", "some source text", n=3) == 0.0


# --------------------------------------------------------------------------
# Elaborative rehearsal (B) — retrieval over the already-rehearsed history
# --------------------------------------------------------------------------


class _TopicTokenizer(_EchoTokenizer):
    """`_EchoTokenizer` plus the word-level counting path `_fit_context_to_budget`
    needs. The two call sites are distinguished by `return_tensors`: the
    generation path wants one tensor id for the whole string, the budget path
    wants a length proportional to the text.
    """

    def __call__(self, text, return_tensors=None, truncation=True, max_length=None, add_special_tokens=True):
        if return_tensors == "pt":
            return super().__call__(text, return_tensors, truncation, max_length)
        return BatchEncoding({"input_ids": list(range(len(text.split())))})


def _topic_embed_fn(texts, input_type, model, truncate, api_key, timeout=30.0, dimensions=None):
    """One-hot embedding on the `topicN` marker in each text, so similarity is
    exactly topic identity and retrieval decisions are deterministic. Reads the
    first marker, which is the current passage's — `format_elaborative_input`
    puts it first.
    """
    vectors = []
    for text in texts:
        vector = [0.0] * 8
        vector[int(text.split("topic")[1][0])] = 1.0
        vectors.append(vector)
    return vectors


def _always_fails_embed_fn(texts, input_type, model, truncate, api_key, timeout=30.0, dimensions=None):
    raise EmbeddingAPIError("simulated outage")


def test_elaborative_functions_are_query_agnostic():
    """Guide §2 again, for B. Retrieval is keyed on the current chunk; a
    question-shaped parameter here would make B question-aware and collapse the
    A/B contrast."""
    for fn in (rehearse_elaborative, retrieve_aligned_context, format_elaborative_input):
        params = set(inspect.signature(fn).parameters)
        forbidden = {p for p in params if "question" in p.lower() or "query" in p.lower()}
        assert not forbidden, f"{fn.__name__} query-agnostic violation: {forbidden}"


def test_retrieve_aligned_context_reaches_past_the_previous_chunk(config):
    """The claim the mechanism exists to make: an early chunk wins over a
    recent one when it is the semantically aligned one."""
    pool_texts = ["topic0 alpha", "topic1 beta", "topic2 gamma", "topic3 delta"]
    pool_embeddings = _topic_embed_fn(pool_texts, "passage", "m", "END", "k")

    hits = retrieve_aligned_context(
        "topic0 revisited", pool_texts, [0, 1, 2, 3], pool_embeddings,
        embed_cfg=config, embed_fn=_topic_embed_fn, top_k=3, min_similarity=0.35,
    )
    assert [index for index, _, _ in hits] == [0]


def test_retrieve_aligned_context_respects_the_relevance_floor(config):
    """Nothing aligned -> retrieve nothing, rather than the least-unrelated
    candidates. Unrelated context is what makes a small seq2seq invent links."""
    pool_texts = ["topic0 alpha", "topic1 beta"]
    pool_embeddings = _topic_embed_fn(pool_texts, "passage", "m", "END", "k")

    hits = retrieve_aligned_context(
        "topic5 unrelated", pool_texts, [0, 1], pool_embeddings,
        embed_cfg=config, embed_fn=_topic_embed_fn, min_similarity=0.35,
    )
    assert hits == []


def test_retrieve_aligned_context_skips_the_api_on_an_empty_pool(config):
    assert retrieve_aligned_context("topic0 x", [], [], [], config, _poison_embed_fn) == []


def test_retrieve_aligned_context_returns_none_on_api_failure(config):
    """None, not [] — the caller has to tell 'nothing relevant' apart from
    'could not look'."""
    pool_texts = ["topic0 alpha"]
    pool_embeddings = [[1.0] + [0.0] * 7]
    result = retrieve_aligned_context(
        "topic0 x", pool_texts, [0], pool_embeddings, config, _always_fails_embed_fn
    )
    assert result is None


def _topic_chunks():
    # chunk 4 returns to chunk 0's topic — unreachable for a recency-only state
    return [
        Chunk(text="topic0 the opening scene", index=0, paragraph_indices=[0], char_start=0, char_end=10),
        Chunk(text="topic1 an unrelated aside", index=1, paragraph_indices=[1], char_start=10, char_end=20),
        Chunk(text="topic2 another digression", index=2, paragraph_indices=[2], char_start=20, char_end=30),
        Chunk(text="topic3 more filler material", index=3, paragraph_indices=[3], char_start=30, char_end=40),
        Chunk(text="topic0 the opening scene returns", index=4, paragraph_indices=[4], char_start=40, char_end=50),
    ]


def test_rehearse_elaborative_retrieves_across_the_whole_history(config):
    _, records = rehearse_elaborative(
        _topic_chunks(), model=_EchoModel(), tokenizer=_TopicTokenizer(),
        embed_cfg=config, embed_fn=_topic_embed_fn, max_input_length=10_000,
    )
    assert records[4].retrieved_indices == [0], "chunk 4 did not reach back to chunk 0"
    assert records[0].retrieved_indices == []  # nothing rehearsed yet


def test_rehearse_elaborative_branches_insert_vs_revise_on_tau(config):
    """C-DIC Eq. 6 read off the retrieval that already ran: chunk 4 continues
    chunk 0's topic (revise), while chunks 1-3 each open a new one (insert)."""
    _, records = rehearse_elaborative(
        _topic_chunks(), model=_EchoModel(), tokenizer=_TopicTokenizer(),
        embed_cfg=config, embed_fn=_topic_embed_fn, max_input_length=10_000,
    )
    assert [r.write_back_action for r in records] == [
        "insert", "insert", "insert", "insert", "revise",
    ]


def test_retrieval_span_report_flags_a_tau_that_never_revises():
    """C-DIC App. K's failure mode: too strict a tau means nothing is ever
    judged on-topic, nothing is revised, and the summary only grows. The
    aggregate has to make that visible."""
    records = [RetrievalRecord(chunk_index=i, pool_size=i) for i in range(1, 6)]
    report = retrieval_span_report(records)
    assert report["insert_share"] == 1.0
    assert report["fire_rate"] == 0.0


def test_rehearse_elaborative_inserts_when_no_topic_recurs(config):
    """A document whose chunks share no topic should be all-insert — the
    branch must not collapse to 'always revise'."""
    chunks = [Chunk(text=f"topic{i} a distinct subject", index=i) for i in range(5)]
    _, records = rehearse_elaborative(
        chunks, model=_EchoModel(), tokenizer=_TopicTokenizer(),
        embed_cfg=config, embed_fn=_topic_embed_fn, max_input_length=10_000,
    )
    assert all(r.write_back_action == "insert" for r in records)


def test_retrieve_aligned_context_falls_back_to_the_best_match(config):
    """Eq. 3's other half: below tau, generation still conditions on the
    nearest neighbour rather than on nothing. The returned hit sits below the
    floor, which is how the caller tells it apart from a real match."""
    pool_texts = ["topic0 alpha", "topic1 beta"]
    pool_embeddings = _topic_embed_fn(pool_texts, "passage", "m", "END", "k")

    hits = retrieve_aligned_context(
        "topic5 unrelated", pool_texts, [0, 1], pool_embeddings,
        embed_cfg=config, embed_fn=_topic_embed_fn, min_similarity=0.35,
        fallback_top1=True,
    )
    assert len(hits) == 1
    assert hits[0][2] < 0.35


def test_rehearse_elaborative_keeps_context_on_a_topic_shift(config):
    """A topic-shift chunk is still elaborated against its nearest neighbour —
    B must not silently degenerate into A's independent per-chunk processing
    now that carry_previous defaults off."""
    chunks = [Chunk(text=f"topic{i} a distinct subject", index=i) for i in range(4)]
    _, records = rehearse_elaborative(
        chunks, model=_EchoModel(), tokenizer=_TopicTokenizer(),
        embed_cfg=config, embed_fn=_topic_embed_fn, max_input_length=10_000,
    )
    # Every chunk after the first had a non-empty pool and still got context,
    # while the write-back branch correctly recorded a topic shift.
    for record in records[1:]:
        assert record.retrieved_indices, "topic-shift chunk was left with no context"
        assert record.write_back_action == "insert"


def test_rehearse_elaborative_preserves_position_metadata_and_original_text(config):
    chunks = _topic_chunks()
    result, _ = rehearse_elaborative(
        chunks, model=_EchoModel(), tokenizer=_TopicTokenizer(),
        embed_cfg=config, embed_fn=_topic_embed_fn, max_input_length=10_000,
    )
    for out, source in zip(result, chunks):
        assert (out.index, out.paragraph_indices, out.char_start, out.char_end) == (
            source.index, source.paragraph_indices, source.char_start, source.char_end
        )
        assert out.original_text == source.text


def test_rehearse_elaborative_degrades_to_recency_when_the_api_fails(config):
    """An embedding outage must not abort the run — it costs retrieval, and the
    record says so."""
    result, records = rehearse_elaborative(
        _topic_chunks(), model=_EchoModel(), tokenizer=_TopicTokenizer(),
        embed_cfg=config, embed_fn=_always_fails_embed_fn, max_input_length=10_000,
    )
    assert len(result) == 5
    assert all(r.retrieved_indices == [] for r in records)
    # Nothing reaches the pool, so later chunks see an empty history and never
    # even attempt a lookup — the outage has to stay visible in the report
    # rather than reading as "this document had nothing to retrieve".
    assert all(r.pool_write_failed for r in records)
    assert retrieval_span_report(records)["pool_write_failures"] == 5


def test_format_elaborative_input_puts_the_current_passage_first():
    """Right-truncation cuts the tail, so the current passage must not be there."""
    assembled = format_elaborative_input("CURRENT", ["ctx one", "ctx two"])
    assert assembled.startswith(f"{PASSAGE_FIELD} CURRENT")
    assert assembled.index("CURRENT") < assembled.index("ctx one")
    assert format_elaborative_input("CURRENT", []) == f"{PASSAGE_FIELD} CURRENT"


def test_retrieval_span_report_separates_reaching_back_from_recency():
    records = [
        RetrievalRecord(chunk_index=0, pool_size=0),  # not eligible
        RetrievalRecord(chunk_index=1, retrieved_indices=[0], similarities=[0.9],
                        pool_size=1, write_back_action="revise"),
        RetrievalRecord(chunk_index=9, retrieved_indices=[0], similarities=[0.8],
                        pool_size=9, write_back_action="revise"),
        RetrievalRecord(chunk_index=10, pool_size=10),  # eligible, stayed below tau
    ]
    report = retrieval_span_report(records)
    assert report["eligible"] == 3
    assert report["fire_rate"] == pytest.approx(2 / 3)
    assert report["max_span"] == 9
    assert report["mean_span"] == pytest.approx(5.0)  # (1 + 9) / 2
    assert report["recency_share"] == pytest.approx(0.5)  # only the span-1 hit


def test_retrieval_span_report_flags_degeneration_to_recency():
    """The failure this report exists to catch: similarity ranking that only
    ever returns the previous chunk means the mechanism bought nothing."""
    records = [RetrievalRecord(chunk_index=i, retrieved_indices=[i - 1], similarities=[0.9],
                               pool_size=i, write_back_action="revise") for i in range(1, 6)]
    report = retrieval_span_report(records)
    assert report["mean_span"] == 1.0
    assert report["recency_share"] == 1.0
