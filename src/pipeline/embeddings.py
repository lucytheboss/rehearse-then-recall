"""Low-level embedding API shared utilities — calls the NIM embedding API
(llama-nemotron-embed-1b-v2) + cosine similarity.

Used by both chuncking.py's paginate_semantic and gisting.py. Pure
embedding API calls + vector math only (no LLM-prompt-based generative
logic).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import requests
import yaml
from dotenv import find_dotenv, load_dotenv

# Load env vars (NVIDIA_NIM_API_KEY etc.) from .env. usecwd=True so this
# reliably finds .env from notebook kernels too. Vars already exported in
# the environment take priority (override=False).
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
    """One raw NIM embedding API call. Tests substitute a mock for this function.
    dimensions is the embedding-dimension-reduction option (if the model
    supports it) — None uses the model's default dimensionality.
    """
    if input_type not in ("passage", "query"):
        raise ValueError(f"input_type must be 'passage' or 'query': {input_type!r}")

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
        raise EmbeddingAPIError(f"NIM embedding API failed ({response.status_code}): {response.text}")

    data = response.json()["data"]
    data.sort(key=lambda item: item["index"])  # response order can differ from input order
    return [item["embedding"] for item in data]


def _batched(items: list, batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def _resolve_api_key(config: dict) -> str:
    api_key = os.environ.get(config["api_key_env"])
    if not api_key:
        raise EmbeddingAPIError(f"Environment variable {config['api_key_env']} is not set.")
    return api_key


# Reason for the most recent embed_with_retry failure — queryable via last_embedding_error()
_last_embedding_error: Exception | None = None


def last_embedding_error() -> Exception | None:
    """The exception behind the most recent embed_with_retry failure, or None
    if the last call succeeded."""
    return _last_embedding_error


def embed_with_retry(
    texts: list[str], input_type: str, config: dict, embed_fn, rate_limiter=None,
) -> list[list[float]] | None:
    """Retries embed_fn up to config["max_retries"] times. If retries are
    exhausted, logs the cause to stderr and returns None. The cause can also
    be queried via last_embedding_error().

    `rate_limiter`, if given, is a `rate_limit.RateLimiter` whose `.acquire()`
    is called before every attempt (including retries) — see that module for
    why this is a separate knob from how many callers run concurrently.
    """
    global _last_embedding_error
    api_key = _resolve_api_key(config)
    last_error: Exception | None = None
    for attempt in range(config["max_retries"] + 1):
        if rate_limiter is not None:
            rate_limiter.acquire()
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
                    f"[embeddings] Embedding API call failed (retries exhausted: {config['max_retries']}, "
                    f"{len(texts)} texts, input_type={input_type!r}): "
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
