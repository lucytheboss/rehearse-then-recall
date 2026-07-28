"""Unit tests for chuncking — runs with no external API calls other than nltk.

The paginate_semantic-related tests inject a mock embed_fn, so they run
without any real NIM API calls (the embedding/retry settings still come
from configs/importance_filter.yaml as-is).
"""

import os

import pytest

from src.pipeline.chuncking import (
    Paragraph,
    _dialogue_groups,
    _paginate_by_word_count_only,
    chunk_containing_position,
    html_to_paragraphs,
    paginate_semantic,
    plain_text_to_paragraphs,
)
from src.pipeline.embeddings import EmbeddingAPIError, load_config


def test_plain_text_to_paragraphs_splits_on_blank_lines():
    text = "First paragraph.\n\nSecond paragraph."
    paragraphs = plain_text_to_paragraphs(text)
    assert [p.text for p in paragraphs] == ["First paragraph.", "Second paragraph."]


def test_plain_text_to_paragraphs_char_offset_matches_original():
    text = "First paragraph.\n\nSecond paragraph."
    paragraphs = plain_text_to_paragraphs(text)
    for p in paragraphs:
        assert text[p.char_offset : p.char_offset + len(p.text)] == p.text


def test_plain_text_to_paragraphs_normalizes_internal_newline_to_space():
    text = "One line.\nContinues to the next line."
    paragraphs = plain_text_to_paragraphs(text)
    assert paragraphs[0].text == "One line. Continues to the next line."


def test_plain_text_to_paragraphs_scene_break_and_quote_flags():
    text = "Narrative paragraph.\n\n* * *\n\nNext paragraph."
    paragraphs = plain_text_to_paragraphs(text)
    assert [p.is_scene_break for p in paragraphs] == [False, True, False]
    assert all(p.is_quote is False for p in paragraphs)


def test_html_to_paragraphs_extracts_p_tags_in_document_order():
    html = "<html><body><p>First paragraph.</p><div>ignored div</div><p>Second paragraph.</p></body></html>"
    paragraphs = html_to_paragraphs(html)
    assert [p.text for p in paragraphs] == ["First paragraph.", "Second paragraph."]


def test_html_to_paragraphs_ignores_table_content():
    """Assumes tables have already been converted to <p> tags — any leftover
    <table> is ignored (checks that this function doesn't wrongly absorb
    table data as a paragraph if convert_korquad_tables.py left it as-is
    instead of replacing it with <p>)."""
    html = "<body><p>Body sentence.</p><table><tr><td>table data</td></tr></table></body>"
    paragraphs = html_to_paragraphs(html)
    assert [p.text for p in paragraphs] == ["Body sentence."]


def test_html_to_paragraphs_normalizes_whitespace():
    html = "<p>  multiple   spaces and\nline breaks   in this paragraph  </p>"
    paragraphs = html_to_paragraphs(html)
    assert paragraphs[0].text == "multiple spaces and line breaks in this paragraph"


def test_html_to_paragraphs_skips_empty_p_tags():
    html = "<p>Has content.</p><p>   </p><p>Next content.</p>"
    paragraphs = html_to_paragraphs(html)
    assert [p.text for p in paragraphs] == ["Has content.", "Next content."]
    assert [p.index for p in paragraphs] == [0, 1]  # index stays contiguous even though the empty paragraph is skipped


def test_html_to_paragraphs_char_offset_matches_joined_text():
    """char_offset is based on the paragraphs joined with '\\n\\n', not the
    original HTML (the same coordinate system as plain_text_to_paragraphs)."""
    html = "<p>First paragraph.</p><p>Second paragraph.</p>"
    paragraphs = html_to_paragraphs(html)
    joined = "\n\n".join(p.text for p in paragraphs)
    for p in paragraphs:
        assert joined[p.char_offset : p.char_offset + len(p.text)] == p.text


def test_html_to_paragraphs_scene_break_flag():
    html = "<p>Narrative paragraph.</p><p>* * *</p><p>Next paragraph.</p>"
    paragraphs = html_to_paragraphs(html)
    assert [p.is_scene_break for p in paragraphs] == [False, True, False]


def test_dialogue_groups_merges_quote_spanning_multiple_sentences():
    """A line of dialogue split across multiple sentence fragments by nltk
    should be merged back into a single group."""
    sentences = ["Background sentence.", "“First line of dialogue.", "Second line of dialogue?”", "Continuing narration."]
    groups = _dialogue_groups(sentences, max_words=100)
    assert groups == [
        ["Background sentence."],
        ["“First line of dialogue.", "Second line of dialogue?”"],
        ["Continuing narration."],
    ]


def test_dialogue_groups_falls_back_to_sentences_when_group_exceeds_max_words():
    """If a dialogue group's word count exceeds max_words, fall back to
    sentence-level splitting to respect the chunk cap."""
    long_dialogue = ["“" + ("word " * 30).strip() + ".", ("other " * 30).strip() + ".”"]
    groups = _dialogue_groups(long_dialogue, max_words=50)
    assert groups == [[long_dialogue[0]], [long_dialogue[1]]]


def test_dialogue_groups_ignores_english_apostrophe():
    """A ’ following a letter, as in 'I’m', is treated as an apostrophe, not a quote mark."""
    groups = _dialogue_groups(["I’m fine.", "Next sentence."], max_words=100)
    assert groups == [["I’m fine."], ["Next sentence."]]


def test_dialogue_groups_self_closed_quote_stays_single_sentence():
    groups = _dialogue_groups(["“A line of dialogue opened and closed within one sentence.”", "Next sentence."], max_words=100)
    assert groups == [["“A line of dialogue opened and closed within one sentence.”"], ["Next sentence."]]


def test_dialogue_groups_merges_straight_single_quote_thought_spanning_sentences():
    """A thought expressed with straight single quotes ('...') spanning multiple sentences should still merge."""
    sentences = ["He thought.", "'Who is that person.", "Why is he here.'", "He walked on again."]
    groups = _dialogue_groups(sentences, max_words=100)
    assert groups == [
        ["He thought."],
        ["'Who is that person.", "Why is he here.'"],
        ["He walked on again."],
    ]


def test_dialogue_groups_ignores_english_straight_apostrophe():
    """A straight apostrophe (') following a letter, as in 'don't'/'it's', is not mistaken for a quote mark either."""
    groups = _dialogue_groups(["I don't know.", "It's fine.", "Next sentence."], max_words=100)
    assert groups == [["I don't know."], ["It's fine."], ["Next sentence."]]


def _paragraph(text: str, index: int, char_offset: int, is_scene_break: bool = False) -> Paragraph:
    return Paragraph(text=text, index=index, char_offset=char_offset, is_scene_break=is_scene_break)


# --- _paginate_by_word_count_only (internal fallback for when paginate_semantic's API call fails) ---


def test_paginate_by_word_count_only_never_splits_a_sentence():
    """Every sentence returned by nltk falls entirely within exactly one chunk's char range."""
    text = " ".join(f"This is sentence number {i}." for i in range(1, 41))
    paragraphs = plain_text_to_paragraphs(text)
    chunks = _paginate_by_word_count_only(paragraphs, min_words=10, max_words=30)

    search_pos = 0
    for i in range(1, 41):
        sentence = f"This is sentence number {i}."
        offset = text.find(sentence, search_pos)
        search_pos = offset + len(sentence)
        containing = [c for c in chunks if c.char_start <= offset < c.char_end]
        assert len(containing) == 1, f"sentence must fall within exactly one chunk's range: {sentence!r}"
        end_offset = offset + len(sentence)
        assert containing[0].char_end >= end_offset, f"sentence was cut at a chunk boundary: {sentence!r}"


def test_paginate_by_word_count_only_does_not_split_dialogue_across_chunks():
    """Even if a dialogue line straddles the min/max words boundary, it stays whole within one chunk."""
    filler_a = ("word " * 60).strip() + "."
    dialogue = "“There are two rooms here. That one is used by someone else. What do you study?”"
    filler_b = ("other " * 60).strip() + "."
    text = f"{filler_a}\n\n{dialogue} The landlord asked.\n\n{filler_b}"

    paragraphs = plain_text_to_paragraphs(text)
    chunks = _paginate_by_word_count_only(paragraphs, min_words=10, max_words=70)

    for c in chunks:
        assert c.text.count("“") == c.text.count("”")  # dialogue stays complete within a single chunk


def test_paginate_by_word_count_only_respects_min_max_words():
    text = "word " * 400
    paragraphs = plain_text_to_paragraphs(text)
    chunks = _paginate_by_word_count_only(paragraphs, min_words=100, max_words=200)
    for c in chunks[:-1]:  # the last chunk may be under min since it's just the remainder
        assert 100 <= len(c.text.split()) <= 200


def test_paginate_by_word_count_only_scene_break_is_hard_boundary():
    text = "First paragraph.\n\n***\n\nSecond paragraph."
    paragraphs = plain_text_to_paragraphs(text)
    chunks = _paginate_by_word_count_only(paragraphs, min_words=1, max_words=1000)
    assert len(chunks) == 2
    assert "***" in chunks[0].text
    assert chunks[1].text == "Second paragraph."


# --- chunk_containing_position ---


@pytest.fixture
def _cc_config():
    cfg = load_config("configs/importance_filter.yaml")
    os.environ[cfg["api_key_env"]] = "dummy-key-for-test"
    return cfg


def test_chunk_containing_position_finds_matching_chunk(_cc_config):
    text = "First paragraph content.\n\nSecond paragraph content."
    paragraphs = plain_text_to_paragraphs(text)
    chunks = paginate_semantic(
        paragraphs, min_words=1, max_words=5, granularity="paragraph", config=_cc_config, embed_fn=_fake_embed_generic
    )
    pos = text.find("Second")
    hit = chunk_containing_position(chunks, pos)
    assert hit is not None
    assert "Second" in hit.text


def test_chunk_containing_position_out_of_range_returns_none(_cc_config):
    chunks = paginate_semantic(
        plain_text_to_paragraphs("Paragraph."),
        min_words=1,
        max_words=5,
        granularity="paragraph",
        config=_cc_config,
        embed_fn=_fake_embed_generic,
    )
    assert chunk_containing_position(chunks, 10_000) is None


# --- paginate_semantic ---


@pytest.fixture
def semantic_config():
    cfg = load_config("configs/importance_filter.yaml")
    os.environ[cfg["api_key_env"]] = "dummy-key-for-test"
    return cfg


def _fake_embed_generic(texts, input_type, model, truncate, api_key, timeout=30.0, dimensions=None):
    """Returns a slightly different vector per text with no real meaning — for structural-constraint tests
    only (the similarity values themselves aren't used)."""
    return [[len(t) % 7, hash(t) % 11, 1.0] for t in texts]


def test_paginate_semantic_never_splits_a_sentence(semantic_config):
    """Every sentence returned by nltk falls entirely within exactly one chunk's char range."""
    text = " ".join(f"This is sentence number {i}." for i in range(1, 41))
    paragraphs = plain_text_to_paragraphs(text)
    chunks = paginate_semantic(
        paragraphs, min_words=10, max_words=30, granularity="paragraph", config=semantic_config, embed_fn=_fake_embed_generic
    )

    search_pos = 0
    for i in range(1, 41):
        sentence = f"This is sentence number {i}."
        offset = text.find(sentence, search_pos)
        search_pos = offset + len(sentence)
        containing = [c for c in chunks if c.char_start <= offset < c.char_end]
        assert len(containing) == 1, f"sentence must fall within exactly one chunk's range: {sentence!r}"
        end_offset = offset + len(sentence)
        assert containing[0].char_end >= end_offset, f"sentence was cut at a chunk boundary: {sentence!r}"


def test_paginate_semantic_does_not_split_dialogue_across_chunks(semantic_config):
    """Even if a dialogue line straddles the min/max words boundary, it stays whole within one chunk."""
    filler_a = ("word " * 60).strip() + "."
    dialogue = "“There are two rooms here. That one is used by someone else. What do you study?”"
    filler_b = ("other " * 60).strip() + "."
    text = f"{filler_a}\n\n{dialogue} The landlord asked.\n\n{filler_b}"

    paragraphs = plain_text_to_paragraphs(text)
    chunks = paginate_semantic(
        paragraphs, min_words=10, max_words=70, granularity="sentence", config=semantic_config, embed_fn=_fake_embed_generic
    )

    for c in chunks:
        assert c.text.count("“") == c.text.count("”")  # dialogue stays complete within a single chunk


def test_paginate_semantic_respects_min_max_words(semantic_config):
    text = "word " * 400
    paragraphs = plain_text_to_paragraphs(text)
    chunks = paginate_semantic(
        paragraphs, min_words=100, max_words=200, granularity="paragraph", config=semantic_config, embed_fn=_fake_embed_generic
    )
    for c in chunks[:-1]:  # the last chunk may be under min since it's just the remainder
        assert 100 <= len(c.text.split()) <= 200


def test_paginate_semantic_scene_break_is_hard_boundary(semantic_config):
    text = "First paragraph.\n\n***\n\nSecond paragraph."
    paragraphs = plain_text_to_paragraphs(text)
    chunks = paginate_semantic(
        paragraphs, min_words=1, max_words=1000, granularity="paragraph", config=semantic_config, embed_fn=_fake_embed_generic
    )
    assert len(chunks) == 2
    assert "***" in chunks[0].text
    assert chunks[1].text == "Second paragraph."


def _numbered_paragraph_text(i: int) -> str:
    return f"Paragraph number {i}."


def test_paginate_semantic_cuts_at_lowest_similarity_seam(semantic_config):
    """Verifies the cut actually happens at the topic-shift point (the lowest-similarity seam).

    Gives a fake embedding where paragraphs 0-2 are "topic A" and 3-5 are
    "topic B", and checks that the lowest-similarity seam (between
    paragraphs 2 and 3) becomes exactly the chunk boundary.
    """
    topic_a_texts = {_numbered_paragraph_text(i) for i in range(3)}

    def fake_embed_by_topic(texts, input_type, model, truncate, api_key, timeout=30.0, dimensions=None):
        return [[1.0, 0.0, 0.0] if t in topic_a_texts else [0.0, 1.0, 0.0] for t in texts]

    text = "\n\n".join(_numbered_paragraph_text(i) for i in range(6))
    paragraphs = plain_text_to_paragraphs(text)

    # each paragraph is 3 words (Paragraph/number/N.) — min_words=6 (2 paragraphs), max_words=12 (4 paragraphs)
    chunks = paginate_semantic(
        paragraphs, min_words=6, max_words=12, granularity="paragraph", config=semantic_config, embed_fn=fake_embed_by_topic
    )

    assert [c.paragraph_indices for c in chunks] == [[0, 1, 2], [3, 4, 5]]


def test_paginate_semantic_cuts_at_lowest_similarity_seam_sentence_granularity(semantic_config):
    """Verifies the topic-boundary cut also happens with granularity="sentence" — based on sentence embeddings."""
    topic_a_texts = {_numbered_paragraph_text(i) for i in range(3)}

    def fake_embed_by_topic(texts, input_type, model, truncate, api_key, timeout=30.0, dimensions=None):
        return [[1.0, 0.0, 0.0] if t in topic_a_texts else [0.0, 1.0, 0.0] for t in texts]

    text = "\n\n".join(_numbered_paragraph_text(i) for i in range(6))
    paragraphs = plain_text_to_paragraphs(text)

    chunks = paginate_semantic(
        paragraphs, min_words=6, max_words=12, granularity="sentence", config=semantic_config, embed_fn=fake_embed_by_topic
    )

    assert [c.paragraph_indices for c in chunks] == [[0, 1, 2], [3, 4, 5]]


def test_paginate_semantic_falls_back_to_word_count_only_on_api_failure(semantic_config):
    """If embed_fn keeps failing, behaves identically to _paginate_by_word_count_only when on_error=pass_through."""

    def always_fails(*args, **kwargs):
        raise EmbeddingAPIError("simulated failure")

    text = " ".join(f"This is sentence number {i}." for i in range(1, 20))
    paragraphs = plain_text_to_paragraphs(text)

    semantic_config["max_retries"] = 0
    semantic_config["on_error"] = "pass_through"

    fallback = paginate_semantic(
        paragraphs, min_words=10, max_words=30, granularity="paragraph", config=semantic_config, embed_fn=always_fails
    )
    expected = _paginate_by_word_count_only(paragraphs, min_words=10, max_words=30)
    assert [c.text for c in fallback] == [c.text for c in expected]


def test_paginate_semantic_raises_on_api_failure_when_configured(semantic_config):
    def always_fails(*args, **kwargs):
        raise EmbeddingAPIError("simulated failure")

    text = " ".join(f"This is sentence number {i}." for i in range(1, 20))
    paragraphs = plain_text_to_paragraphs(text)

    semantic_config["max_retries"] = 0
    semantic_config["on_error"] = "raise"

    with pytest.raises(EmbeddingAPIError):
        paginate_semantic(
            paragraphs, min_words=10, max_words=30, granularity="paragraph", config=semantic_config, embed_fn=always_fails
        )


def test_paginate_semantic_raise_includes_underlying_failure_reason(semantic_config):
    """When on_error=raise, the raised exception's message must include the real underlying cause (the original exception message)."""

    def always_fails(*args, **kwargs):
        raise EmbeddingAPIError("simulated failure detail xyz")

    text = " ".join(f"This is sentence number {i}." for i in range(1, 20))
    paragraphs = plain_text_to_paragraphs(text)

    semantic_config["max_retries"] = 0
    semantic_config["on_error"] = "raise"

    with pytest.raises(EmbeddingAPIError, match="simulated failure detail xyz"):
        paginate_semantic(
            paragraphs, min_words=10, max_words=30, granularity="paragraph", config=semantic_config, embed_fn=always_fails
        )


def test_paginate_semantic_empty_paragraphs_returns_empty_list(semantic_config):
    assert paginate_semantic([], min_words=10, max_words=30, granularity="paragraph", config=semantic_config) == []


def test_paginate_semantic_invalid_granularity_raises(semantic_config):
    paragraphs = plain_text_to_paragraphs("Paragraph.")
    with pytest.raises(ValueError):
        paginate_semantic(
            paragraphs, min_words=1, max_words=10, granularity="invalid", config=semantic_config, embed_fn=_fake_embed_generic
        )
