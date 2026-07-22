"""Paragraph/Chunk 공유 데이터 클래스 — chuncking.py와 gisting.py가 함께 쓴다.

순환 임포트 방지용으로 분리했다: chuncking.paginate_semantic이 embeddings의
임베딩 함수를 기본 파라미터로 참조해야 하는데, Chunk/Paragraph가 chuncking.py에
있으면 chuncking → embeddings → chuncking 순환 임포트가 생긴다.
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
    original_text: str = ""  # gisting이 text를 압축본으로 교체할 때 원문 보존용