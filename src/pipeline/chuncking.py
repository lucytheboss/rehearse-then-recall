"""Chunking — splits plain text into Paragraphs, then paginates into Chunks.

Separates the source adapter (plain_text_to_paragraphs) from the pagination
logic. There's exactly one public pagination function, paginate_semantic —
it cuts at the point where embedding cosine similarity between neighboring
paragraphs/sentences is lowest (TextTiling, Hearst 1997 family). The
internal fallback (_paginate_by_word_count_only) is only used when the
embedding API fails.

Chunk-boundary priority:
1. Never cut inside a sentence (nltk).
2. Treat a dialogue group with unclosed quotation marks as one unit — never
   place a boundary in the middle of it.
3. Unless the group exceeds max_words (dialogue too long), in which case
   fall back to sentence-level splitting.
4. Paragraph boundaries are candidate points, not hard boundaries — cut at
   whichever candidate has the lowest similarity.
5. Only a structurally-marked paragraph is a hard boundary, exceptionally —
   not just novel-style scene-break markers ("***" etc.), but as genres
   diversified to wiki/news/legal-statute/case-law, markdown headings,
   section numbers (Section 1, §101, (a), roman numerals), and short
   all-caps headings ("FACTS", "OPINION") are now recognized the same way
   as hard boundaries (`_is_structural_break_text`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize

from src.pipeline.embeddings import (
    EmbeddingAPIError,
    cosine_similarity,
    embed_texts,
    embed_with_retry,
    last_embedding_error,
)
from src.pipeline.types import Chunk, Paragraph

# Scene-break marker characters (narrative-style genres).
_SCENE_BREAK_CHARS = set("*※•○●◇◆-·~ \t\n\r")

# Structural-heading patterns for non-narrative genres (wiki/news/legal
# statute/case-law) — a separate path from the novel scene-break markers
# above. Recognizes markdown headings, section numbers, and roman-numeral
# headings.
_HEADING_PATTERN = re.compile(
    r"^(#{1,6}\s|(section|article|chapter)\s+\d|§\s*\d|[ivxlcdm]{1,6}\.\s|\d+(\.\d+)*\.?\s|\([a-z0-9]{1,4}\)\s)",
    re.IGNORECASE,
)
_MAX_HEADING_WORDS = 12  # longer than this is treated as a body paragraph, not a heading


def _looks_like_heading(text: str) -> bool:
    """True for a markdown heading / section number / roman-numeral heading,
    or a short all-caps heading. Heading-length is capped at
    _MAX_HEADING_WORDS to distinguish this from an ordinary paragraph that
    just happens to start with a capital letter."""
    if len(text.split()) > _MAX_HEADING_WORDS:
        return False
    if _HEADING_PATTERN.match(text):
        return True
    letters = [ch for ch in text if ch.isalpha()]
    return bool(letters) and all(ch.isupper() for ch in letters)


def _is_structural_break_text(text: str) -> bool:
    """True if the whole paragraph is scene-break markers or looks like a
    structural heading — both cases are always treated as a hard boundary,
    regardless of similarity scoring."""
    stripped = text.strip()
    if not stripped:
        return False
    if all(ch in _SCENE_BREAK_CHARS for ch in stripped):
        return True
    return _looks_like_heading(stripped)


def plain_text_to_paragraphs(text: str) -> list[Paragraph]:
    """Plain text source adapter: splits into paragraphs on blank lines ->
    normalized Paragraph list.

    One block separated by a blank line (including a line with only
    whitespace, regex \\n\\s*\\n+) is one paragraph — a single newline inside
    a block is a line wrap, not a paragraph boundary, so it's normalized to
    a single space. char_offset points to the paragraph's actual start
    position in the source text (the newline->space normalization is a 1:1
    substitution, so positions inside the paragraph barely drift from the
    source either). is_quote is always False — plain text has no formatting
    info like italics, so there's no structural way to tell whether
    something is a quotation.
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
                    is_scene_break=_is_structural_break_text(normalized),
                    is_quote=False,
                )
            )
            index += 1
        pos = sep_end

    return paragraphs


def html_to_paragraphs(html: str) -> list[Paragraph]:
    """HTML source adapter — the second source adapter, used alongside
    plain_text_to_paragraphs. Assumes any `<table>` has already been
    converted to natural-language sentences and replaced with `<p>` tags
    (e.g. the output of `data/scripts/convert_korquad_tables.py`) — this
    function itself doesn't interpret tables.

    Walks `<p>` tags in document order and converts them to Paragraphs.
    char_offset is not a position in the original HTML string — it's the
    cumulative position as if the paragraphs were joined with "\\n\\n". This
    uses the same coordinate system as plain_text_to_paragraphs's
    char_offset, so downstream steps (paginate_semantic,
    chunk_containing_position) behave identically regardless of which
    adapter produced the paragraphs. Nothing downstream of paginate_semantic
    is touched at all.
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
                is_scene_break=_is_structural_break_text(text),
                is_quote=False,
            )
        )
        index += 1
        pos += len(text) + 2  # cumulative offset assuming a "\n\n" separator

    return paragraphs


def _paragraph_sentences(paragraph: Paragraph) -> list[str]:
    """Splits a paragraph into sentences. A scene_break paragraph is left as-is."""
    if paragraph.is_scene_break:
        return [paragraph.text]
    sentences = sent_tokenize(paragraph.text)
    return sentences if sentences else [paragraph.text]


# Curly quotes (open/close differ): track depth, +1 for open / -1 for close (handles nesting)
_CURLY_QUOTE_DEPTH = {"‘": 1, "“": 1, "’": -1, "”": -1}
# Straight quotes (open/close are the same character): toggle open/closed on each occurrence
_STRAIGHT_QUOTE_TOGGLE_CHARS = ('"', "'")
# Characters easily mistaken for apostrophes — filtered out by _is_apostrophe
_APOSTROPHE_LIKE_CHARS = ("’", "'")


def _is_apostrophe(text: str, idx: int) -> bool:
    """An apostrophe in an English contraction (`I'm`, `don't`) is not a quote mark."""
    return idx > 0 and text[idx - 1].isascii() and text[idx - 1].isalnum()


def _dialogue_groups(sentences: list[str], max_words: int) -> list[list[str]]:
    """Merges sentences into groups wherever quotation marks stay unclosed
    across sentence boundaries. If a group exceeds max_words (dialogue too
    long), falls back to sentence-level splitting.
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
    if buffer:  # ended with an unclosed quote
        groups.append(buffer)

    exploded: list[list[str]] = []
    for group in groups:
        if len(group) > 1 and sum(len(s.split()) for s in group) > max_words:
            exploded.extend([s] for s in group)  # fall back to sentence-level
        else:
            exploded.append(group)
    return exploded


def _paginate_by_word_count_only(
    paragraphs: list[Paragraph],
    min_words: int,
    max_words: int,
) -> list[Chunk]:
    """Accumulates sentence-by-sentence and groups into chunks within
    min/max words. The internal fallback paginate_semantic uses only when
    the embedding API fails (on_error="pass_through").
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
    """The smallest unit paginate_semantic works with — one sentence or one
    dialogue group. A chunk boundary never passes through the interior of a
    unit — a seam is always defined only between two different units.
    """

    items: list[tuple[str, int, int]]  # (sentence, paragraph index, char offset)
    word_count: int
    is_scene_break: bool
    paragraph_index: int  # for embedding lookup when granularity="paragraph"
    sentence_start: int  # global index of the first sentence, when granularity="sentence"
    sentence_end: int  # global index of the last sentence (inclusive)


def _build_semantic_units(paragraphs: list[Paragraph], max_words: int) -> tuple[list[_SemanticUnit], list[str]]:
    """Walks the paragraphs and flattens them into a _SemanticUnit list (+ a
    global sentence list)."""
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
    """Splits a text list into batch_size-sized chunks and embeds them in batches."""
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
    """Batch-embeds at either paragraph or sentence granularity."""
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
    raise ValueError(f'granularity must be "paragraph" or "sentence": {granularity!r}')


def _compute_seam_scores(
    units: list[_SemanticUnit],
    embeddings: tuple[Literal["paragraph", "sentence"], dict[int, list[float]] | list[list[float]]],
) -> list[float]:
    """List of cosine similarity (seam score) between each pair of adjacent
    _SemanticUnits.

    Length is len(units) - 1; lower is a better chunk-boundary candidate.
    With granularity="paragraph", compares paragraph embeddings directly;
    with "sentence", compares the last sentence of the earlier unit against
    the first sentence of the later one.
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
    """Cuts at the point where embedding similarity between neighboring units
    is lowest.

    An embedding-based variant of TextTiling (Hearst, 1997) — among seam
    candidates within the min_words~max_words window, picks the one with the
    lowest similarity as the chunk boundary. A scene_break paragraph is
    always a hard boundary regardless of similarity.

    On embed_fn failure, either raises or falls back to
    _paginate_by_word_count_only depending on config["on_error"] (the
    default).
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
            detail = f" Cause: {type(reason).__name__}: {reason}" if reason else ""
            raise EmbeddingAPIError(
                f"The embedding API call still failed after retries — cannot proceed with semantic pagination.{detail}"
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
            cut_at = end  # the last chunk absorbs all remaining units
        else:
            candidates = [k for k in range(i, end + 1) if prefix[k + 1] - prefix[i] >= min_words]
            cut_at = min(candidates, key=lambda k: seam_scores[k]) if candidates else end

        chunks.append(_emit_semantic_chunk(units[i : cut_at + 1], len(chunks)))
        i = cut_at + 1

    return chunks


def chunk_containing_position(chunks: list[Chunk], char_position: int) -> Chunk | None:
    """Given a source char offset, returns the chunk that contains that position."""
    for chunk in chunks:
        if chunk.char_start <= char_position < chunk.char_end:
            return chunk
    return None
