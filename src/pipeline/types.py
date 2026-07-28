"""Shared Paragraph/Chunk dataclasses — used by both chuncking.py and gisting.py.

Split out to avoid a circular import: chuncking.paginate_semantic needs to
reference embeddings' embed function as a default parameter, and if
Chunk/Paragraph lived in chuncking.py that would create
chuncking -> embeddings -> chuncking.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Paragraph:
    text: str
    index: int
    char_offset: int
    is_scene_break: bool = False
    is_quote: bool = False


@dataclass
class Chunk:
    text: str
    index: int
    paragraph_indices: list[int] = field(default_factory=list)
    char_start: int = 0
    char_end: int = 0
    original_text: str = ""  # preserves the source text when gisting replaces text with a compressed version