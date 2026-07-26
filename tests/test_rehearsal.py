"""rehearsal(유지형 되뇌기 A) 단위 테스트 — 전부 fake model/tokenizer/embed_fn으로
동작, 실제 모델 다운로드·API 키·네트워크 불필요.
"""

import inspect
import os

import pytest
import torch
from transformers import BatchEncoding

from src.pipeline.embeddings import EmbeddingAPIError, load_config
from src.pipeline.rehearsal import _snap_to_verbatim, novel_ngram_ratio, rehearse_maintenance
from src.pipeline.types import Chunk


@pytest.fixture
def config():
    cfg = load_config("configs/importance_filter.yaml")
    os.environ[cfg["api_key_env"]] = "dummy-key-for-test"
    return cfg


class _EchoTokenizer:
    """chunk.text 전체를 정수 ID 하나로 등록/조회하는 가짜 토크나이저 — 실제
    서브워드 분할 없이 rehearse_maintenance의 토큰화->생성->디코드 제어 흐름만
    검증한다.
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
    """입력 토큰을 그대로 돌려주는 가짜 모델 — decode(generate(encode(t))) == t가
    성립하게 만들어, "생성문이 원문과 완전히 같을 때" 경로(verbatim 스냅이
    필요 없는 경우)를 검증한다.
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
    raise AssertionError("embed_fn이 호출되면 안 되는 상황에서 호출됨")


@pytest.fixture
def sample_chunk():
    return Chunk(
        text="고양이가 창밖을 바라본다. 주가 지수가 급락했다.",
        index=2,
        paragraph_indices=[5, 6],
        char_start=100,
        char_end=150,
    )


def test_rehearse_maintenance_is_query_agnostic():
    """가이드 §2 하드 제약: question/query류 파라미터가 있으면 안 됨."""
    params = set(inspect.signature(rehearse_maintenance).parameters)
    forbidden = {p for p in params if "question" in p.lower() or "query" in p.lower()}
    assert not forbidden, f"query-agnostic 위반 — 금지된 파라미터 발견: {forbidden}"


def test_rehearse_maintenance_preserves_position_metadata(sample_chunk, config):
    # echo 모델이라 생성문 == 원문 -> 모든 문장이 이미 verbatim이라 embed_fn은 호출되면 안 됨
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
        Chunk(text="첫 번째 청크 문장입니다.", index=0),
        Chunk(text="두 번째 청크 문장입니다.", index=1),
    ]
    result = rehearse_maintenance(
        chunks, model=_EchoModel(), tokenizer=_EchoTokenizer(), embed_cfg=config, embed_fn=_poison_embed_fn
    )
    assert [c.original_text for c in result] == [c.text for c in chunks]
    assert [c.index for c in result] == [0, 1]


def _fake_embed_fixed_vectors(vectors_by_text: dict):
    def _embed(texts, input_type, model, truncate, api_key, timeout=30.0, dimensions=None):
        return [vectors_by_text[t] for t in texts]

    return _embed


def test_snap_to_verbatim_keeps_exact_matches_without_calling_embed_fn():
    generated = ["고양이가 창밖을 바라본다.", "주가 지수가 급락했다."]
    source = ["고양이가 창밖을 바라본다.", "주가 지수가 급락했다.", "부동산 시장이 얼어붙었다."]

    result = _snap_to_verbatim(generated, source, embed_cfg={}, embed_fn=_poison_embed_fn)
    assert result == generated


def test_snap_to_verbatim_replaces_non_matching_sentence_with_closest_by_cosine(config):
    generated = ["고양이가 밖을 응시한다."]  # 원문에 없는 표현(패러프레이즈)
    source = ["고양이가 창밖을 바라본다.", "부동산 시장이 얼어붙었다."]

    vectors = {
        "고양이가 밖을 응시한다.": [1.0, 0.0],
        "고양이가 창밖을 바라본다.": [0.9, 0.1],  # 코사인 유사도 더 높음
        "부동산 시장이 얼어붙었다.": [0.0, 1.0],
    }
    result = _snap_to_verbatim(generated, source, embed_cfg=config, embed_fn=_fake_embed_fixed_vectors(vectors))
    assert result == ["고양이가 창밖을 바라본다."]


def test_snap_to_verbatim_falls_back_to_generated_on_embedding_failure(config):
    def always_fails(*args, **kwargs):
        raise EmbeddingAPIError("simulated embedding outage")

    config = dict(config)
    config["max_retries"] = 0
    generated = ["원문에 없는 새 표현."]
    source = ["원문 문장 하나."]

    result = _snap_to_verbatim(generated, source, embed_cfg=config, embed_fn=always_fails)
    assert result == generated  # 죽지 않고 생성문 그대로 반환


def test_novel_ngram_ratio_is_zero_for_pure_verbatim_text():
    source = "고양이가 창밖을 바라본다 주가 지수가 급락했다 고양이는 낮잠을 잔다"
    generated = "고양이가 창밖을 바라본다"
    assert novel_ngram_ratio(generated, source, n=3) == 0.0


def test_novel_ngram_ratio_is_positive_for_novel_text():
    source = "고양이가 창밖을 바라본다 주가 지수가 급락했다"
    generated = "완전히 다른 단어들로 구성된 새로운 문장 생성"
    assert novel_ngram_ratio(generated, source, n=3) == 1.0


def test_novel_ngram_ratio_short_generation_returns_zero():
    # n-gram을 만들기엔 너무 짧은 생성문 -> 0 (예외 대신 안전한 기본값)
    assert novel_ngram_ratio("한 단어", "아무 원문", n=3) == 0.0
