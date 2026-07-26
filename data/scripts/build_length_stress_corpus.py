"""KorQuAD 2.0에서 "문항당 단어수" 밀도가 높은 문서들을 그리디하게 골라
이어붙여, 길이별 스트레스 테스트(notebooks/04_length_stress_test.ipynb)용
단일 원문 + 문항 CSV를 만든다.

표 변환 없이 문단(<p>)만 쓴다 — 문단만으로 문항의 상당수가 이미 답변
가능하다는 게 확인됐고(전체 후보 풀 45% answerable), 표 변환은 정답 위치가
극도로 모호해지는 문제(예: "0"이 3,534회 등장)가 있어서 이번 코퍼스에서는
안 쓴다(단, 20세기_폭스처럼 이미 표 변환해둔 문서가 있다면 별도로 병합 가능
— 이 스크립트는 그 통합까지는 하지 않음, 필요하면 후속 스크립트에서).

핵심 아이디어: 문서 하나가 이미 몇천~2만 단어라 "길이"는 문서 1~2개로도
넘치지만, KorQuAD 2.0은 문서당 네이티브 문항이 몇 개 안 돼서(스포츠/목록형
제외 시 보통 3~4개) 정작 "answerable 문항 수"가 부족하다. 그래서 문서를
단순히 길이 순이 아니라 **"문항당 단어수"가 낮은(=효율 좋은) 순**으로
그리디하게 골라 이어붙인다 — 같은 예산(1만~1.2만 단어) 안에 최대한 많은
answerable 문항을 담기 위함.
"""

from __future__ import annotations

import ast
import json
from collections import Counter, defaultdict
from pathlib import Path

from bs4 import BeautifulSoup
from datasets import load_dataset

root = Path(__file__).resolve().parent
while not (root / "src").exists() and root != root.parent:
    root = root.parent

MIN_QUESTION_COUNT = 3  # 문서당 최소 문항 수(너무 적으면 통계적으로 의미 없음)
QC_MAX_OCCURRENCE = 3  # 답이 본문에 이 횟수 이하로 등장해야 "위치 신뢰 가능"으로 채택
TARGET_MIN_WORDS = 10_000
TARGET_MAX_WORDS = 12_000
SCENE_BREAK_SEPARATOR = "\n\n***\n\n"  # chuncking._is_scene_break_text가 하드 경계로 인식

_SPORTS_KEYWORDS = [
    "FC", "축구", "월드컵", "올림픽", "리그", "시즌", "선수권", "유나이티드",
    "시티", "야구", "농구", "배구", "K리그", "챔피언스", "토너먼트", "선거",
    "지방선거", "국회의원", "총선", "목록",
]
# 카테고리 필터로 걸렸지만 이미 문단만으로 71%+ answerable로 검증된 예외
_MANUAL_ALLOWLIST = {"2010년_동계_올림픽"}


def _looks_excluded(title: str) -> bool:
    if title in _MANUAL_ALLOWLIST:
        return False
    return any(kw in title for kw in _SPORTS_KEYWORDS)


def _clean_answer(raw) -> str:
    if isinstance(raw, str):
        raw = ast.literal_eval(raw)
    return BeautifulSoup(raw["text"], "lxml").get_text(strip=True)


def _extract_p_text(context_html: str) -> str:
    soup = BeautifulSoup(context_html, "lxml")
    return " ".join(p.get_text(separator=" ", strip=True) for p in soup.find_all("p"))


def _word_index(text: str, char_pos: int) -> int:
    """char_pos 이전까지의 어절 수 + 1 = 1-인덱스 어절 위치(eval_sample01.csv 관례와 동일)."""
    return len(text[:char_pos].split()) + 1


def collect_candidates(ds) -> list[dict]:
    titles = Counter(ds["title"])
    candidate_titles = {t for t, c in titles.items() if c >= MIN_QUESTION_COUNT}

    by_title: dict[str, list] = defaultdict(list)
    for row in ds:
        if row["title"] in candidate_titles:
            by_title[row["title"]].append(row)

    candidates = []
    for title, rows in by_title.items():
        if _looks_excluded(title):
            continue

        p_text = _extract_p_text(rows[0]["context"])
        word_count = len(p_text.split())
        if word_count < 300:  # 사실상 빈 문서 제외
            continue

        answerable = []
        for r in rows:
            clean = _clean_answer(r["answer"])
            if len(clean) < 2:
                continue
            count = p_text.count(clean)
            if 1 <= count <= QC_MAX_OCCURRENCE:
                local_char_pos = p_text.find(clean)
                answerable.append(
                    {
                        "question": r["question"],
                        "answer_text": clean,
                        "local_char_pos": local_char_pos,
                    }
                )

        if not answerable:
            continue

        candidates.append(
            {
                "title": title,
                "p_text": p_text,
                "word_count": word_count,
                "answerable": answerable,
                "density": word_count / len(answerable),  # 낮을수록 효율적
            }
        )

    return candidates


def greedy_select(candidates: list[dict]) -> list[dict]:
    candidates_sorted = sorted(candidates, key=lambda c: c["density"])
    selected = []
    cumulative_words = 0
    for cand in candidates_sorted:
        if cumulative_words >= TARGET_MIN_WORDS:
            break
        if cumulative_words > 0 and cumulative_words + cand["word_count"] > TARGET_MAX_WORDS:
            continue  # 예산 초과하는 문서는 스킵하고 더 작은 다음 후보 시도
        selected.append(cand)
        cumulative_words += cand["word_count"]
    return selected


def build_corpus(selected: list[dict]) -> tuple[str, list[dict]]:
    text_parts: list[str] = []
    questions: list[dict] = []
    cumulative_chars = 0
    q_index = 1

    for doc_index, doc in enumerate(selected):
        text_parts.append(doc["p_text"])
        doc_start_char = cumulative_chars

        for item in doc["answerable"]:
            global_char_pos = doc_start_char + item["local_char_pos"]
            questions.append(
                {
                    "문항번호": q_index,
                    "원본_문서": doc["title"],
                    "문제": item["question"],
                    "정답": item["answer_text"],
                    "근거_문자위치": global_char_pos,
                    "유형": "KorQuAD_추출형",
                }
            )
            q_index += 1

        cumulative_chars += len(doc["p_text"])
        if doc_index < len(selected) - 1:
            cumulative_chars += len(SCENE_BREAK_SEPARATOR)

    full_text = SCENE_BREAK_SEPARATOR.join(text_parts)

    # 근거_어절위치는 최종 이어붙인 텍스트 기준으로 한 번에 재계산(구분자 포함 오프셋 정합성 보장)
    for q in questions:
        q["근거_어절위치"] = _word_index(full_text, q["근거_문자위치"])
        # 위치가 실제로 그 답을 가리키는지 최종 검증(디버깅 안전망)
        assert full_text[q["근거_문자위치"] : q["근거_문자위치"] + len(q["정답"])] == q["정답"], (
            f"위치 불일치: {q['문항번호']}번 문항"
        )

    return full_text, questions


def main() -> None:
    print("KorQuAD/squad_kor_v2 로드 중...")
    ds = load_dataset("KorQuAD/squad_kor_v2", split="train", revision="refs/convert/parquet")

    print("후보 수집 중...")
    candidates = collect_candidates(ds)
    print(f"후보 문서: {len(candidates)}개")

    selected = greedy_select(candidates)
    total_words = sum(c["word_count"] for c in selected)
    total_answerable = sum(len(c["answerable"]) for c in selected)
    print(f"\n선정된 문서: {len(selected)}개, 총 {total_words}단어, answerable 문항 {total_answerable}개")
    for c in selected:
        print(f"  - {c['title']}: {c['word_count']}단어, {len(c['answerable'])}문항, 밀도(단어/문항)={c['density']:.0f}")

    full_text, questions = build_corpus(selected)
    print(f"\n최종 코퍼스: {len(full_text)}자, {len(full_text.split())}단어, 문항 {len(questions)}개")

    out_dir = root / "data" / "processed" / "korquad2"
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_dir = root / "data" / "sample"
    sample_dir.mkdir(parents=True, exist_ok=True)

    text_path = out_dir / "length_stress_corpus.txt"
    text_path.write_text(full_text, encoding="utf-8")

    import pandas as pd

    questions_df = pd.DataFrame(questions)
    csv_path = sample_dir / "korquad2_length_stress.csv"
    questions_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    manifest_path = out_dir / "length_stress_corpus_manifest.json"
    manifest_path.write_text(
        json.dumps(
            [{"title": c["title"], "word_count": c["word_count"], "n_answerable": len(c["answerable"])} for c in selected],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\n저장 완료:\n  {text_path}\n  {csv_path}\n  {manifest_path}")


if __name__ == "__main__":
    main()
