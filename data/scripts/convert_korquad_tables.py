"""KorQuAD 2.0 문서 하나를 골라 표(<table>)를 NIM 채팅 API로 자연어 문장으로
풀어쓰고, 문단(<p>)과 원래 문서 순서 그대로 합쳐 plain text로 저장한다.

일회성 데이터 준비 스크립트다 — chuncking.py 등 핵심 파이프라인은 안 건드리고,
결과물(plain text + QA 위치 재계산용 audit JSON)만 남긴다.

표 하나당 NIM 호출 1회로 변환한다(행 단위 템플릿이 아니라 LLM에게 표 전체를
보여줘서 컬럼 의미를 스스로 파악하게 함 — 표마다 컬럼 구성이 달라서 범용적으로
동작하려면 이 방식이 낫다). navbox/전거통제 각주 표는 실제 정보가 아니라
탐색용 UI라 변환 대상에서 제외한다.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

root = Path(__file__).resolve().parent
while not (root / "src").exists() and root != root.parent:
    root = root.parent
sys.path.insert(0, str(root))

import requests
from bs4 import BeautifulSoup
from datasets import load_dataset
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))

TITLE = "20세기_폭스"
QA_MODEL = "meta/llama-3.1-8b-instruct"
_CHAT_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
_TABLE_PROMPT = (
    "다음은 위키백과 표(HTML)입니다. 각 행을 하나의 완결된 한국어 문장으로 바꿔서, "
    "행 순서 그대로 줄바꿈으로 구분해 나열하세요. 숫자·고유명사·날짜는 원문 그대로 "
    "유지하고, 표에 없는 새 정보를 추가하지 마세요. 표 제목이나 헤더 행 자체는 "
    "문장으로 만들지 말고, 데이터 행만 문장으로 바꾸세요. 다른 설명 없이 문장들만 출력하세요."
)


class TableConversionError(RuntimeError):
    pass


def call_chat(messages: list[dict], model: str, api_key: str, timeout: float = 90.0, max_retries: int = 3) -> str:
    """NIM 채팅 API 1회 호출(+재시도). nb02의 call_chat과 같은 패턴, 이 스크립트 전용 축소판."""
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            response = requests.post(
                _CHAT_API_URL,
                headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
                json={"model": model, "messages": messages, "temperature": 0.0},
                timeout=timeout,
            )
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            last_error = e
            if attempt < max_retries:
                time.sleep(3.0)
                continue
            raise TableConversionError(f"NIM 채팅 API 호출 실패(재시도 소진): {e}") from e

        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        if response.status_code in (429, 500, 502, 503, 504) and attempt < max_retries:
            time.sleep(3.0)
            continue
        raise TableConversionError(f"NIM 채팅 API 실패 ({response.status_code}): {response.text}")

    raise TableConversionError(f"NIM 채팅 API 호출 실패(재시도 소진): {last_error}")


_META_NOTICE_PATTERNS = (
    "이 문단은 비어있습니다",
    "이 문서의 내용은 출처",
    "이 부분은",
    "위키백과의 정책",
    "출처가 필요합니다",
)


def is_navbox_or_footer(table) -> bool:
    """실제 표 데이터가 아니라 위키백과 UI/편집 안내문이라 변환 대상에서
    제외해야 하는 표들: 탐색용 navbox("v d e h" 템플릿), 전거 통제 각주,
    "이 문단은 비어있습니다" 같은 편집 안내 placeholder."""
    text = table.get_text(separator=" ", strip=True)
    if text.startswith("v d e h") or " v d e h " in text[:20]:
        return True
    if "전거 통제" in text[:20] or "WorldCat" in text[:50]:
        return True
    if any(pattern in text[:40] for pattern in _META_NOTICE_PATTERNS):
        return True
    return False


def convert_table_to_sentences(table, api_key: str) -> str:
    messages = [
        {"role": "system", "content": "당신은 표를 자연스러운 한국어 문장으로 바꾸는 도우미입니다."},
        {"role": "user", "content": f"{_TABLE_PROMPT}\n\n{table.decode()}"},
    ]
    return call_chat(messages, model=QA_MODEL, api_key=api_key)


def main() -> None:
    api_key = os.environ.get("NVIDIA_NIM_API_KEY")
    if not api_key:
        raise SystemExit("NVIDIA_NIM_API_KEY가 설정되어 있지 않습니다.")

    print(f"KorQuAD/squad_kor_v2 로드 중... (title={TITLE!r})")
    ds = load_dataset("KorQuAD/squad_kor_v2", split="train", revision="refs/convert/parquet")
    rows = [r for r in ds if r["title"] == TITLE]
    if not rows:
        raise SystemExit(f"title={TITLE!r}인 문서를 찾지 못했습니다.")
    print(f"문항 수: {len(rows)}")

    soup = BeautifulSoup(rows[0]["context"], "lxml")
    blocks = soup.find_all(["p", "table"])
    print(f"문서 순서상 p/table 블록 수: {len(blocks)}")

    audit: list[dict] = []
    text_blocks: list[str] = []
    n_converted = 0
    n_skipped_navbox = 0

    for i, block in enumerate(blocks):
        if block.name == "p":
            text = block.get_text(separator=" ", strip=True)
            if text:
                text_blocks.append(text)
            continue

        # block.name == "table"
        if is_navbox_or_footer(block):
            n_skipped_navbox += 1
            continue

        print(f"  [{i + 1}/{len(blocks)}] 표 변환 중... ({len(block.find_all('tr'))}행)")
        try:
            converted = convert_table_to_sentences(block, api_key)
        except TableConversionError as e:
            print(f"    실패, 이 표는 건너뜀: {e}")
            audit.append({"block_index": i, "type": "table", "status": "failed", "error": str(e)})
            continue

        text_blocks.append(converted)
        audit.append(
            {
                "block_index": i,
                "type": "table",
                "status": "converted",
                "original_html": block.decode(),
                "converted_text": converted,
            }
        )
        n_converted += 1
        time.sleep(0.5)

    final_text = "\n\n".join(text_blocks)
    print(f"\n변환된 표: {n_converted}개, 제외(navbox/footer): {n_skipped_navbox}개")
    print(f"최종 문서: {len(final_text)}자, {len(final_text.split())}단어")

    out_dir = root / "data" / "processed" / "korquad2"
    out_dir.mkdir(parents=True, exist_ok=True)

    text_path = out_dir / f"{TITLE}_tables_converted.txt"
    text_path.write_text(final_text, encoding="utf-8")

    audit_path = out_dir / f"{TITLE}_conversion_audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    questions_path = out_dir / f"{TITLE}_questions_raw.json"
    questions = [
        {"question": r["question"], "answer": r["answer"], "url": r["url"]}
        for r in rows
    ]
    questions_path.write_text(json.dumps(questions, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n저장 완료:\n  {text_path}\n  {audit_path}\n  {questions_path}")


if __name__ == "__main__":
    main()
