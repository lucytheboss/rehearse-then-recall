"""저수준 임베딩 API 공용 유틸 — NIM 임베딩 API(llama-nemotron-embed-1b-v2) 호출 + 코사인 유사도.

chuncking.py의 paginate_semantic과 gisting.py가 함께 쓴다. 순수 임베딩 API
호출 + 벡터 연산만 수행. (LLM prompt 기반 생성형 로직 사용하지 않음)
"""

from __future__ import annotations

import os
import sys

import numpy as np
import requests
import yaml
from dotenv import find_dotenv, load_dotenv

# .env에서 NVIDIA_NIM_API_KEY 등 환경변수 로드. usecwd=True: 노트북 커널에서도
# 안정적으로 .env를 찾기 위함. 이미 export된 환경변수가 있으면 그쪽 우선(override=False).
load_dotenv(find_dotenv(usecwd=True))
_API_URL = "https://integrate.api.nvidia.com/v1/embeddings"


class EmbeddingAPIError(RuntimeError):
    pass


def load_config(path: str = "configs/importance_filter.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def embed_texts(
    texts: list[str],
    input_type: str,
    model: str,
    truncate: str,
    api_key: str,
    timeout: float = 30.0,
    dimensions: int | None = None,
) -> list[list[float]]:
    """NIM 임베딩 API 원시 호출 1회. 테스트에서는 이 함수를 mock으로 교체한다.
    dimensions는 임베딩 차원 축소 옵션(모델이 지원하는 경우) — None이면 모델 기본 차원.
    """
    if input_type not in ("passage", "query"):
        raise ValueError(f"input_type은 'passage' 또는 'query'여야 함: {input_type!r}")

    body = {"input": texts, "model": model, "input_type": input_type, "truncate": truncate}
    if dimensions is not None:
        body["dimensions"] = dimensions

    response = requests.post(
        _API_URL,
        headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
        json=body,
        timeout=timeout,
    )
    if response.status_code != 200:
        raise EmbeddingAPIError(f"NIM 임베딩 API 실패 ({response.status_code}): {response.text}")

    data = response.json()["data"]
    data.sort(key=lambda item: item["index"])  # 응답 순서가 입력 순서와 다를 수 있어 정렬
    return [item["embedding"] for item in data]


def _batched(items: list, batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def _resolve_api_key(config: dict) -> str:
    api_key = os.environ.get(config["api_key_env"])
    if not api_key:
        raise EmbeddingAPIError(f"환경변수 {config['api_key_env']}가 설정되어 있지 않습니다.")
    return api_key


# 가장 최근 embed_with_retry 실패 원인 — last_embedding_error()로 조회
_last_embedding_error: Exception | None = None


def last_embedding_error() -> Exception | None:
    """가장 최근 embed_with_retry 호출이 실패한 원인 예외. 마지막 호출이 성공했으면 None."""
    return _last_embedding_error


def embed_with_retry(texts: list[str], input_type: str, config: dict, embed_fn) -> list[list[float]] | None:
    """embed_fn을 config["max_retries"]만큼 재시도. 재시도를 다 쓰고도 실패하면
    stderr에 원인을 남기고 None을 반환한다. 원인은 last_embedding_error()로도 조회 가능.
    """
    global _last_embedding_error
    api_key = _resolve_api_key(config)
    last_error: Exception | None = None
    for attempt in range(config["max_retries"] + 1):
        try:
            result = embed_fn(
                texts,
                input_type=input_type,
                model=config["model"],
                truncate=config["truncate"],
                api_key=api_key,
                timeout=config["timeout"],
                dimensions=config.get("dimensions"),
            )
            _last_embedding_error = None
            return result
        except (EmbeddingAPIError, requests.RequestException) as e:
            last_error = e
            if attempt == config["max_retries"]:
                _last_embedding_error = last_error
                print(
                    f"[embeddings] 임베딩 API 호출 실패 (재시도 {config['max_retries']}회 소진, "
                    f"텍스트 {len(texts)}개, input_type={input_type!r}): "
                    f"{type(last_error).__name__}: {last_error}",
                    file=sys.stderr,
                )
                return None
    return None


def cosine_similarity(a: list[float], b: list[float]) -> float:
    a_arr, b_arr = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    denom = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
    if denom == 0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / denom)
