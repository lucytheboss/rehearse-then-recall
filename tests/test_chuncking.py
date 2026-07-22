"""chuncking 단위 테스트 — kss 외 외부 API 호출 없이 동작.

paginate_semantic 관련 테스트는 embed_fn을 mock으로 주입해 실제 NIM API
호출 없이 동작한다(임베딩/재시도 설정은 configs/importance_filter.yaml 그대로 사용).
"""

import os

import pytest

from src.pipeline.chuncking import (
    Paragraph,
    _dialogue_groups,
    _paginate_by_word_count_only,
    chunk_containing_position,
    paginate_semantic,
    plain_text_to_paragraphs,
)
from src.pipeline.embeddings import EmbeddingAPIError, load_config


def test_plain_text_to_paragraphs_splits_on_blank_lines():
    text = "첫 문단.\n\n둘째 문단."
    paragraphs = plain_text_to_paragraphs(text)
    assert [p.text for p in paragraphs] == ["첫 문단.", "둘째 문단."]


def test_plain_text_to_paragraphs_char_offset_matches_original():
    text = "첫 문단.\n\n둘째 문단."
    paragraphs = plain_text_to_paragraphs(text)
    for p in paragraphs:
        assert text[p.char_offset : p.char_offset + len(p.text)] == p.text


def test_plain_text_to_paragraphs_normalizes_internal_newline_to_space():
    text = "한 줄.\n다음 줄로 이어짐."
    paragraphs = plain_text_to_paragraphs(text)
    assert paragraphs[0].text == "한 줄. 다음 줄로 이어짐."


def test_plain_text_to_paragraphs_scene_break_and_quote_flags():
    text = "서술 문단.\n\n* * *\n\n다음 문단."
    paragraphs = plain_text_to_paragraphs(text)
    assert [p.is_scene_break for p in paragraphs] == [False, True, False]
    assert all(p.is_quote is False for p in paragraphs)


def test_dialogue_groups_merges_quote_spanning_multiple_sentences():
    """대사가 kss 문장 조각 여러 개에 걸쳐 있으면 하나의 묶음으로 합쳐진다."""
    sentences = ["배경 설명 문장.", "“첫 대사 부분.", "두번째 대사 부분?”", "이어지는 서술."]
    groups = _dialogue_groups(sentences, max_words=100)
    assert groups == [
        ["배경 설명 문장."],
        ["“첫 대사 부분.", "두번째 대사 부분?”"],
        ["이어지는 서술."],
    ]


def test_dialogue_groups_falls_back_to_sentences_when_group_exceeds_max_words():
    """대사 묶음의 단어 수가 max_words를 넘으면 문장 단위로 되돌려 청크 상한을 지킨다."""
    long_dialogue = ["“" + ("가나다 " * 30).strip() + ".", ("라마바 " * 30).strip() + ".”"]
    groups = _dialogue_groups(long_dialogue, max_words=50)
    assert groups == [[long_dialogue[0]], [long_dialogue[1]]]


def test_dialogue_groups_ignores_english_apostrophe():
    """'I’m'처럼 알파벳 뒤에 오는 ’는 인용부호가 아니라 아포스트로피로 취급한다."""
    groups = _dialogue_groups(["I’m fine.", "다음 문장."], max_words=100)
    assert groups == [["I’m fine."], ["다음 문장."]]


def test_dialogue_groups_self_closed_quote_stays_single_sentence():
    groups = _dialogue_groups(["“한 문장 안에서 열리고 닫히는 대사.”", "다음 문장."], max_words=100)
    assert groups == [["“한 문장 안에서 열리고 닫히는 대사.”"], ["다음 문장."]]


def test_dialogue_groups_merges_straight_single_quote_thought_spanning_sentences():
    """한국어 생각 표현이 곧은 작은따옴표('...')로 여러 문장에 걸쳐 있어도 병합된다."""
    sentences = ["그는 생각했다.", "'저 사람은 누구지.", "왜 여기 있는 걸까.'", "다시 걸음을 옮겼다."]
    groups = _dialogue_groups(sentences, max_words=100)
    assert groups == [
        ["그는 생각했다."],
        ["'저 사람은 누구지.", "왜 여기 있는 걸까.'"],
        ["다시 걸음을 옮겼다."],
    ]


def test_dialogue_groups_ignores_english_straight_apostrophe():
    """'don't', 'it's'처럼 알파벳 뒤에 오는 곧은 작은따옴표(')도 인용부호로 오인하지 않는다."""
    groups = _dialogue_groups(["I don't know.", "It's fine.", "다음 문장."], max_words=100)
    assert groups == [["I don't know."], ["It's fine."], ["다음 문장."]]


def _paragraph(text: str, index: int, char_offset: int, is_scene_break: bool = False) -> Paragraph:
    return Paragraph(text=text, index=index, char_offset=char_offset, is_scene_break=is_scene_break)


# --- _paginate_by_word_count_only (paginate_semantic의 API 실패 시 내부 fallback) ---


def test_paginate_by_word_count_only_never_splits_a_sentence():
    """kss가 반환한 문장 하나하나가 정확히 한 청크의 char 범위 안에 온전히 들어 있다."""
    text = " ".join(f"이것은 문장 번호 {i}입니다." for i in range(1, 41))
    paragraphs = plain_text_to_paragraphs(text)
    chunks = _paginate_by_word_count_only(paragraphs, min_words=10, max_words=30)

    search_pos = 0
    for i in range(1, 41):
        sentence = f"이것은 문장 번호 {i}입니다."
        offset = text.find(sentence, search_pos)
        search_pos = offset + len(sentence)
        containing = [c for c in chunks if c.char_start <= offset < c.char_end]
        assert len(containing) == 1, f"문장이 정확히 하나의 청크 범위에 있어야 함: {sentence!r}"
        end_offset = offset + len(sentence)
        assert containing[0].char_end >= end_offset, f"문장이 청크 경계에서 잘림: {sentence!r}"


def test_paginate_by_word_count_only_does_not_split_dialogue_across_chunks():
    """대사가 min/max words 경계에 걸려도 대사 전체가 한 청크 안에 남는다."""
    filler_a = ("가나다 " * 60).strip() + "."
    dialogue = "“여기 방이 두 칸. 저쪽엔 다른 애가 써. 학생은 뭐 공부해?”"
    filler_b = ("라마바 " * 60).strip() + "."
    text = f"{filler_a}\n\n{dialogue} 집주인이 물었다.\n\n{filler_b}"

    paragraphs = plain_text_to_paragraphs(text)
    chunks = _paginate_by_word_count_only(paragraphs, min_words=10, max_words=70)

    for c in chunks:
        assert c.text.count("“") == c.text.count("”")  # 대사가 청크 하나 안에서 완결됨


def test_paginate_by_word_count_only_respects_min_max_words():
    text = "가나다 " * 400
    paragraphs = plain_text_to_paragraphs(text)
    chunks = _paginate_by_word_count_only(paragraphs, min_words=100, max_words=200)
    for c in chunks[:-1]:  # 마지막 청크는 나머지 분량이라 min 미만일 수 있음
        assert 100 <= len(c.text.split()) <= 200


def test_paginate_by_word_count_only_scene_break_is_hard_boundary():
    text = "첫 문단.\n\n***\n\n둘째 문단."
    paragraphs = plain_text_to_paragraphs(text)
    chunks = _paginate_by_word_count_only(paragraphs, min_words=1, max_words=1000)
    assert len(chunks) == 2
    assert "***" in chunks[0].text
    assert chunks[1].text == "둘째 문단."


# --- chunk_containing_position ---


@pytest.fixture
def _cc_config():
    cfg = load_config("configs/importance_filter.yaml")
    os.environ[cfg["api_key_env"]] = "dummy-key-for-test"
    return cfg


def test_chunk_containing_position_finds_matching_chunk(_cc_config):
    text = "첫 문단 내용.\n\n둘째 문단 내용."
    paragraphs = plain_text_to_paragraphs(text)
    chunks = paginate_semantic(
        paragraphs, min_words=1, max_words=5, granularity="paragraph", config=_cc_config, embed_fn=_fake_embed_generic
    )
    pos = text.find("둘째")
    hit = chunk_containing_position(chunks, pos)
    assert hit is not None
    assert "둘째" in hit.text


def test_chunk_containing_position_out_of_range_returns_none(_cc_config):
    chunks = paginate_semantic(
        plain_text_to_paragraphs("문단."),
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
    """의미 없이 텍스트마다 조금씩 다른 벡터만 반환 — 구조 제약 테스트용(유사도 값 자체는 안 씀)."""
    return [[len(t) % 7, hash(t) % 11, 1.0] for t in texts]


def test_paginate_semantic_never_splits_a_sentence(semantic_config):
    """kss가 반환한 문장 하나하나가 정확히 한 청크의 char 범위 안에 온전히 들어 있다."""
    text = " ".join(f"이것은 문장 번호 {i}입니다." for i in range(1, 41))
    paragraphs = plain_text_to_paragraphs(text)
    chunks = paginate_semantic(
        paragraphs, min_words=10, max_words=30, granularity="paragraph", config=semantic_config, embed_fn=_fake_embed_generic
    )

    search_pos = 0
    for i in range(1, 41):
        sentence = f"이것은 문장 번호 {i}입니다."
        offset = text.find(sentence, search_pos)
        search_pos = offset + len(sentence)
        containing = [c for c in chunks if c.char_start <= offset < c.char_end]
        assert len(containing) == 1, f"문장이 정확히 하나의 청크 범위에 있어야 함: {sentence!r}"
        end_offset = offset + len(sentence)
        assert containing[0].char_end >= end_offset, f"문장이 청크 경계에서 잘림: {sentence!r}"


def test_paginate_semantic_does_not_split_dialogue_across_chunks(semantic_config):
    """대사가 min/max words 경계에 걸려도 대사 전체가 한 청크 안에 남는다."""
    filler_a = ("가나다 " * 60).strip() + "."
    dialogue = "“여기 방이 두 칸. 저쪽엔 다른 애가 써. 학생은 뭐 공부해?”"
    filler_b = ("라마바 " * 60).strip() + "."
    text = f"{filler_a}\n\n{dialogue} 집주인이 물었다.\n\n{filler_b}"

    paragraphs = plain_text_to_paragraphs(text)
    chunks = paginate_semantic(
        paragraphs, min_words=10, max_words=70, granularity="sentence", config=semantic_config, embed_fn=_fake_embed_generic
    )

    for c in chunks:
        assert c.text.count("“") == c.text.count("”")  # 대사가 청크 하나 안에서 완결됨


def test_paginate_semantic_respects_min_max_words(semantic_config):
    text = "가나다 " * 400
    paragraphs = plain_text_to_paragraphs(text)
    chunks = paginate_semantic(
        paragraphs, min_words=100, max_words=200, granularity="paragraph", config=semantic_config, embed_fn=_fake_embed_generic
    )
    for c in chunks[:-1]:  # 마지막 청크는 나머지 분량이라 min 미만일 수 있음
        assert 100 <= len(c.text.split()) <= 200


def test_paginate_semantic_scene_break_is_hard_boundary(semantic_config):
    text = "첫 문단.\n\n***\n\n둘째 문단."
    paragraphs = plain_text_to_paragraphs(text)
    chunks = paginate_semantic(
        paragraphs, min_words=1, max_words=1000, granularity="paragraph", config=semantic_config, embed_fn=_fake_embed_generic
    )
    assert len(chunks) == 2
    assert "***" in chunks[0].text
    assert chunks[1].text == "둘째 문단."


def _numbered_paragraph_text(i: int) -> str:
    return f"이것은 문단 {i}입니다."


def test_paginate_semantic_cuts_at_lowest_similarity_seam(semantic_config):
    """토픽이 바뀌는 지점(유사도가 가장 낮은 이음매)에서 실제로 끊기는지 확인.

    문단 0~2는 "topic A", 3~5는 "topic B"인 가짜 임베딩을 줘서, 유사도가
    가장 낮은 이음매(문단 2와 3 사이)가 정확히 청크 경계가 되는지 검증한다.
    """
    topic_a_texts = {_numbered_paragraph_text(i) for i in range(3)}

    def fake_embed_by_topic(texts, input_type, model, truncate, api_key, timeout=30.0, dimensions=None):
        return [[1.0, 0.0, 0.0] if t in topic_a_texts else [0.0, 1.0, 0.0] for t in texts]

    text = "\n\n".join(_numbered_paragraph_text(i) for i in range(6))
    paragraphs = plain_text_to_paragraphs(text)

    # 문단 하나당 3단어(이것은/문단/N입니다.) — min_words=6(문단 2개), max_words=12(문단 4개)
    chunks = paginate_semantic(
        paragraphs, min_words=6, max_words=12, granularity="paragraph", config=semantic_config, embed_fn=fake_embed_by_topic
    )

    assert [c.paragraph_indices for c in chunks] == [[0, 1, 2], [3, 4, 5]]


def test_paginate_semantic_cuts_at_lowest_similarity_seam_sentence_granularity(semantic_config):
    """granularity="sentence"에서도 토픽 경계에서 끊기는지 확인 — 문장 임베딩 기반."""
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
    """embed_fn이 계속 실패하면 on_error=pass_through일 때 _paginate_by_word_count_only와 동일하게 동작한다."""

    def always_fails(*args, **kwargs):
        raise EmbeddingAPIError("simulated failure")

    text = " ".join(f"이것은 문장 번호 {i}입니다." for i in range(1, 20))
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

    text = " ".join(f"이것은 문장 번호 {i}입니다." for i in range(1, 20))
    paragraphs = plain_text_to_paragraphs(text)

    semantic_config["max_retries"] = 0
    semantic_config["on_error"] = "raise"

    with pytest.raises(EmbeddingAPIError):
        paginate_semantic(
            paragraphs, min_words=10, max_words=30, granularity="paragraph", config=semantic_config, embed_fn=always_fails
        )


def test_paginate_semantic_raise_includes_underlying_failure_reason(semantic_config):
    """on_error=raise일 때 발생하는 예외 메시지에 실제 실패 원인(원본 예외 메시지)이 포함돼야 한다."""

    def always_fails(*args, **kwargs):
        raise EmbeddingAPIError("simulated failure detail xyz")

    text = " ".join(f"이것은 문장 번호 {i}입니다." for i in range(1, 20))
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
    paragraphs = plain_text_to_paragraphs("문단.")
    with pytest.raises(ValueError):
        paginate_semantic(
            paragraphs, min_words=1, max_words=10, granularity="invalid", config=semantic_config, embed_fn=_fake_embed_generic
        )
