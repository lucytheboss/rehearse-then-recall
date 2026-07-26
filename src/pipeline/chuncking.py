"""청킹(chunking) — plain text를 Paragraph로 분할 후 Chunk로 페이지네이션.

소스 어댑터(plain_text_to_paragraphs)와 페이지네이션 로직을 분리한다.
페이지네이션 공개 함수는 paginate_semantic 하나뿐 — 이웃 문단/문장 간
임베딩 코사인 유사도가 가장 낮은 지점(TextTiling, Hearst 1997 계열)에서
끊는다. 임베딩 API 실패 시에만 내부 fallback(_paginate_by_word_count_only)을 쓴다.

청크 경계 우선순위:
1. 문장(kss) 내부는 절대 자르지 않는다.
2. 인용부호가 안 닫힌 대화 묶음은 하나로 취급해 중간에 경계를 넣지 않는다.
3. 단, 묶음이 max_words를 넘으면(대화가 너무 길면) 문장 단위로 풀어 경계를 허용한다.
4. 문단 경계는 하드 경계가 아닌 후보 지점 — 후보 중 유사도가 가장 낮은 곳에서 끊는다.
5. scene_break 문단만 예외적으로 하드 경계.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, cast

import kss
from bs4 import BeautifulSoup

from src.pipeline.embeddings import (
    EmbeddingAPIError,
    cosine_similarity,
    embed_texts,
    embed_with_retry,
    last_embedding_error,
)
from src.pipeline.types import Chunk, Paragraph

# 장면 구분용 구분 기호들.
_SCENE_BREAK_CHARS = set("*※•○●◇◆-·~ \t\n\r")


def _is_scene_break_text(text: str) -> bool:
    """텍스트가 전부 장면 구분 기호로만 이루어졌는지 판정."""
    stripped = text.strip()
    return bool(stripped) and all(ch in _SCENE_BREAK_CHARS for ch in stripped)


def plain_text_to_paragraphs(text: str) -> list[Paragraph]:
    """plain text 소스 어댑터: 빈 줄 기준 문단 분할 → 정규화된 Paragraph 리스트.

    빈 줄(공백만 있는 줄 포함, 정규식 \\n\\s*\\n+)로 구분된 블록 하나가 문단
    하나다 — 블록 내부의 단일 개행은 문단 경계가 아니라 줄바꿈이므로 공백
    1개로 정규화한다. char_offset은 문단 시작의 원문 내 실제 위치를
    가리킨다 (개행→공백 정규화는 1:1 치환이라 문단 내부 위치도 원문과 거의
    어긋나지 않음). is_quote는 항상 False — plain text에는 이탤릭 같은
    서식 정보가 없어 인용구 여부를 구조적으로 판별할 수 없다.
    """
    paragraphs: list[Paragraph] = []
    index = 0
    pos = 0
    separators = [(m.start(), m.end()) for m in re.finditer(r"\n\s*\n+", text)]
    for sep_start, sep_end in separators + [(len(text), len(text))]:
        block = text[pos:sep_start]
        normalized = re.sub(r"\s+", " ", block).strip()
        if normalized:
            paragraphs.append(
                Paragraph(
                    text=normalized,
                    index=index,
                    char_offset=pos + (len(block) - len(block.lstrip())),
                    is_scene_break=_is_scene_break_text(normalized),
                    is_quote=False,
                )
            )
            index += 1
        pos = sep_end

    return paragraphs


def html_to_paragraphs(html: str) -> list[Paragraph]:
    """HTML 소스 어댑터 — plain_text_to_paragraphs와 나란히 쓰는 두 번째
    소스 어댑터다. `<table>`은 이미 자연어 문장으로 변환되어 `<p>`로 치환된
    상태라고 가정한다(예: `data/scripts/convert_korquad_tables.py` 결과물) —
    이 함수 자체는 표를 해석하지 않는다.

    `<p>` 태그를 문서 순서대로 순회해 Paragraph로 변환한다. char_offset은
    원본 HTML 문자열 위치가 아니라, 문단들을 "\\n\\n"로 이어붙였을 때의
    누적 위치다 — plain_text_to_paragraphs의 char_offset과 같은 좌표계라
    이후 단계(paginate_semantic, chunk_containing_position)가 어댑터 종류를
    몰라도 동일하게 동작한다. paginate_semantic 이후 로직은 전혀 건드리지 않는다.
    """
    soup = BeautifulSoup(html, "lxml")
    paragraphs: list[Paragraph] = []
    index = 0
    pos = 0
    for p_tag in soup.find_all("p"):
        text = re.sub(r"\s+", " ", p_tag.get_text(separator=" ", strip=True)).strip()
        if not text:
            continue
        paragraphs.append(
            Paragraph(
                text=text,
                index=index,
                char_offset=pos,
                is_scene_break=_is_scene_break_text(text),
                is_quote=False,
            )
        )
        index += 1
        pos += len(text) + 2  # "\n\n" 구분자 기준 누적 오프셋

    return paragraphs


def _paragraph_sentences(paragraph: Paragraph) -> list[str]:
    """문단을 문장 단위로 세분화. scene_break 문단은 그대로 둔다.

    backend="fast" 고정: "auto"(형태소 분석)는 긴 문단에서 문장 내부 띄어쓰기를
    깨뜨리는 버그가 있다(예: "더 싸게" → "더싸 게"). "fast"는 구두점 기준으로만
    분리해 이 문제가 없다.
    """
    if paragraph.is_scene_break:
        return [paragraph.text]
    sentences = kss.split_sentences(paragraph.text, backend="fast")
    if sentences and isinstance(sentences[0], list):
        return [sentence for sentence_group in sentences for sentence in sentence_group]
    if sentences:
        return cast(list[str], sentences)
    return [paragraph.text]


# 굽은따옴표(여닫는 기호가 다름): 왼쪽 +1, 오른쪽 -1로 깊이 추적(중첩 인용 대비)
_CURLY_QUOTE_DEPTH = {"‘": 1, "“": 1, "’": -1, "”": -1}
# 곧은따옴표(여닫는 기호가 같음): 등장할 때마다 열림/닫힘 토글
_STRAIGHT_QUOTE_TOGGLE_CHARS = ('"', "'")
# 아포스트로피로 오인되기 쉬운 문자 — _is_apostrophe로 걸러낸다
_APOSTROPHE_LIKE_CHARS = ("’", "'")


def _is_apostrophe(text: str, idx: int) -> bool:
    """영어 축약형(`I'm`, `don't`)의 아포스트로피는 인용부호가 아님."""
    return idx > 0 and text[idx - 1].isascii() and text[idx - 1].isalnum()


def _dialogue_groups(sentences: list[str], max_words: int) -> list[list[str]]:
    """문장들을 인용부호가 닫히지 않은 구간끼리 하나의 묶음으로 합친다.
    묶음이 max_words를 넘으면(대화가 너무 길면) 문장 단위로 되돌린다.
    """
    groups: list[list[str]] = []
    buffer: list[str] = []
    curly_depth = 0
    straight_open = {ch: False for ch in _STRAIGHT_QUOTE_TOGGLE_CHARS}

    for sentence in sentences:
        buffer.append(sentence)
        for i, ch in enumerate(sentence):
            if ch in _APOSTROPHE_LIKE_CHARS and _is_apostrophe(sentence, i):
                continue
            if ch in _CURLY_QUOTE_DEPTH:
                curly_depth = max(0, curly_depth + _CURLY_QUOTE_DEPTH[ch])
            elif ch in _STRAIGHT_QUOTE_TOGGLE_CHARS:
                straight_open[ch] = not straight_open[ch]
        if curly_depth == 0 and not any(straight_open.values()):
            groups.append(buffer)
            buffer = []
    if buffer:  # 인용부호가 안 닫힌 채로 끝난 경우
        groups.append(buffer)

    exploded: list[list[str]] = []
    for group in groups:
        if len(group) > 1 and sum(len(s.split()) for s in group) > max_words:
            exploded.extend([s] for s in group)  # 문장 단위로 되돌림
        else:
            exploded.append(group)
    return exploded


def _paginate_by_word_count_only(
    paragraphs: list[Paragraph],
    min_words: int,
    max_words: int,
) -> list[Chunk]:
    """문장 단위로 누적해 min/max words 안에서 청크로 묶는다.
    paginate_semantic이 임베딩 API 실패 시(on_error="pass_through")에만
    쓰는 내부 fallback.
    """
    chunks: list[Chunk] = []
    current: list[tuple[str, int, int]] = []  # (sentence, paragraph_index, char_offset)
    current_words = 0

    def flush() -> None:
        nonlocal current, current_words
        if not current:
            return
        chunks.append(
            Chunk(
                text=" ".join(s for s, _, _ in current),
                index=len(chunks),
                paragraph_indices=sorted({p_idx for _, p_idx, _ in current}),
                char_start=current[0][2],
                char_end=current[-1][2] + len(current[-1][0]),
            )
        )
        current = []
        current_words = 0

    for p in paragraphs:
        search_pos = 0
        for group in _dialogue_groups(_paragraph_sentences(p), max_words):
            group_items: list[tuple[str, int, int]] = []
            for sentence in group:
                found_at = p.text.find(sentence, search_pos)
                offset = p.char_offset + (found_at if found_at >= 0 else search_pos)
                if found_at >= 0:
                    search_pos = found_at + len(sentence)
                group_items.append((sentence, p.index, offset))

            group_words = sum(len(s.split()) for s, _, _ in group_items)
            if current and current_words >= min_words and current_words + group_words > max_words:
                flush()

            current.extend(group_items)
            current_words += group_words

        if p.is_scene_break:
            flush()

    flush()
    return chunks


@dataclass
class _SemanticUnit:
    """paginate_semantic의 최소 취급 단위 — 문장 하나 또는 대사 묶음 하나.
    이 단위 내부로는 청크 경계가 지나가지 않는다 — 이음매는 항상 서로
    다른 유닛 사이에서만 정의된다.
    """

    items: list[tuple[str, int, int]]  # (문장, 문단 index, char offset)
    word_count: int
    is_scene_break: bool
    paragraph_index: int  # granularity="paragraph"일 때 임베딩 조회용
    sentence_start: int  # granularity="sentence"일 때 첫 문장의 전역 인덱스
    sentence_end: int  # 마지막 문장의 전역 인덱스 (포함)


def _build_semantic_units(paragraphs: list[Paragraph], max_words: int) -> tuple[list[_SemanticUnit], list[str]]:
    """문단들을 순회해 _SemanticUnit 리스트(+ 전역 문장 리스트)로 펼친다."""
    units: list[_SemanticUnit] = []
    flat_sentences: list[str] = []
    for p in paragraphs:
        search_pos = 0
        for group in _dialogue_groups(_paragraph_sentences(p), max_words):
            group_items: list[tuple[str, int, int]] = []
            start_flat = len(flat_sentences)
            for sentence in group:
                found_at = p.text.find(sentence, search_pos)
                offset = p.char_offset + (found_at if found_at >= 0 else search_pos)
                if found_at >= 0:
                    search_pos = found_at + len(sentence)
                group_items.append((sentence, p.index, offset))
                flat_sentences.append(sentence)
            units.append(
                _SemanticUnit(
                    items=group_items,
                    word_count=sum(len(s.split()) for s, _, _ in group_items),
                    is_scene_break=p.is_scene_break,
                    paragraph_index=p.index,
                    sentence_start=start_flat,
                    sentence_end=len(flat_sentences) - 1,
                )
            )
    return units, flat_sentences


def _batched_embed_texts(texts: list[str], config: dict, embed_fn) -> list[list[float]] | None:
    """텍스트 리스트를 batch_size 단위로 잘라 배치 임베딩한다."""
    embeddings: list[list[float]] = []
    batch_size = config["batch_size"]
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        batch_embeddings = embed_with_retry(batch, "passage", config, embed_fn)
        if batch_embeddings is None:
            return None
        embeddings.extend(batch_embeddings)
    return embeddings


def _embed_semantic_units(
    paragraphs: list[Paragraph],
    flat_sentences: list[str],
    granularity: Literal["paragraph", "sentence"],
    config: dict,
    embed_fn,
) -> tuple[Literal["paragraph", "sentence"], dict[int, list[float]] | list[list[float]]] | None:
    """granularity에 따라 문단 단위 또는 문장 단위로 배치 임베딩한다."""
    if granularity == "paragraph":
        embeddings = _batched_embed_texts([p.text for p in paragraphs], config, embed_fn)
        if embeddings is None:
            return None
        return "paragraph", {p.index: emb for p, emb in zip(paragraphs, embeddings)}
    if granularity == "sentence":
        embeddings = _batched_embed_texts(flat_sentences, config, embed_fn)
        if embeddings is None:
            return None
        return "sentence", embeddings
    raise ValueError(f'granularity는 "paragraph" 또는 "sentence"여야 함: {granularity!r}')


def _compute_seam_scores(
    units: list[_SemanticUnit],
    embeddings: tuple[Literal["paragraph", "sentence"], dict[int, list[float]] | list[list[float]]],
) -> list[float]:
    """인접한 두 _SemanticUnit 사이의 코사인 유사도(이음매 점수) 리스트.

    길이는 len(units) - 1, 낮을수록 좋은 청크 경계 후보. granularity="paragraph"면
    문단 임베딩끼리, "sentence"면 앞 유닛의 마지막 문장과 뒤 유닛의 첫 문장을 비교한다.
    """
    kind, data = embeddings
    scores: list[float] = []
    for k in range(len(units) - 1):
        if kind == "paragraph":
            emb_a = data[units[k].paragraph_index]
            emb_b = data[units[k + 1].paragraph_index]
        else:
            emb_a = data[units[k].sentence_end]
            emb_b = data[units[k + 1].sentence_start]
        scores.append(cosine_similarity(emb_a, emb_b))
    return scores


def _emit_semantic_chunk(unit_group: list[_SemanticUnit], index: int) -> Chunk:
    items = [item for u in unit_group for item in u.items]
    return Chunk(
        text=" ".join(s for s, _, _ in items),
        index=index,
        paragraph_indices=sorted({p_idx for _, p_idx, _ in items}),
        char_start=items[0][2],
        char_end=items[-1][2] + len(items[-1][0]),
    )


def paginate_semantic(
    paragraphs: list[Paragraph],
    min_words: int,
    max_words: int,
    granularity: Literal["paragraph", "sentence"],
    config: dict,
    embed_fn=embed_texts,
) -> list[Chunk]:
    """이웃 단위 간 임베딩 유사도가 가장 낮아지는 지점에서 끊는다.

    TextTiling(Hearst, 1997) 계열의 임베딩 버전 — min_words~max_words 구간 안의
    이음매 후보 중 유사도가 가장 낮은 지점을 청크 경계로 고른다. scene_break
    문단은 유사도와 무관하게 항상 하드 경계.

    embed_fn 실패 시 config["on_error"]에 따라 raise하거나
    _paginate_by_word_count_only로 대체한다(기본값).
    """
    if not paragraphs:
        return []

    units, flat_sentences = _build_semantic_units(paragraphs, max_words)
    if not units:
        return []

    embeddings = _embed_semantic_units(paragraphs, flat_sentences, granularity, config, embed_fn)
    if embeddings is None:
        if config.get("on_error", "pass_through") == "raise":
            reason = last_embedding_error()
            detail = f" 원인: {type(reason).__name__}: {reason}" if reason else ""
            raise EmbeddingAPIError(
                f"임베딩 API 호출이 재시도 후에도 실패해 의미 기반 페이지네이션을 진행할 수 없습니다.{detail}"
            )
        return _paginate_by_word_count_only(paragraphs, min_words, max_words)

    seam_scores = _compute_seam_scores(units, embeddings)

    prefix = [0]
    for u in units:
        prefix.append(prefix[-1] + u.word_count)

    chunks: list[Chunk] = []
    i = 0
    n = len(units)
    while i < n:
        end = i
        scene_break_end: int | None = None
        for j in range(i, n):
            words_so_far = prefix[j + 1] - prefix[i]
            if words_so_far > max_words and j > i:
                break
            end = j
            if units[j].is_scene_break:
                scene_break_end = j
                break

        if scene_break_end is not None:
            cut_at = scene_break_end
        elif end == n - 1:
            cut_at = end  # 마지막 청크는 남은 유닛 전부 포함
        else:
            candidates = [k for k in range(i, end + 1) if prefix[k + 1] - prefix[i] >= min_words]
            cut_at = min(candidates, key=lambda k: seam_scores[k]) if candidates else end

        chunks.append(_emit_semantic_chunk(units[i : cut_at + 1], len(chunks)))
        i = cut_at + 1

    return chunks


def chunk_containing_position(chunks: list[Chunk], char_position: int) -> Chunk | None:
    """원문 char offset이 주어졌을 때, 그 위치를 포함하는 청크를 반환한다.
    """
    for chunk in chunks:
        if chunk.char_start <= char_position < chunk.char_end:
            return chunk
    return None