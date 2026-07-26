"""유지형 되뇌기(A) — 청크 내부 salience 압축.

`11_되뇌기 구현 가이드.md` §3 구현. 각 Chunk를 독립적으로(청크 간 맥락 없이,
롤링 없음 — 이게 정교화 되뇌기 B와의 핵심 차이) seq2seq 모델에 넣어 압축본을
생성한 뒤, 생성된 문장이 원문에 실제로 존재하는지 검증하고(verbatim), 없으면
원문 내 코사인 유사도가 가장 높은 실제 문장으로 스냅(snap)한다 — 아키텍처를
바꾸지 않고 디코딩 후처리만으로 verbatim을 강제하는 장치(가이드 §3.2).

하드 제약(가이드 §2, A/B 공통):
- query-agnostic — 이 모듈의 어떤 함수도 question/query류 인자를 받지 않는다.
- 청크 위치 메타데이터(index/paragraph_indices/char_start/char_end) 보존 —
  text만 압축본으로 교체, original_text에 압축 전 원문 저장.
- 텍스트 레벨 개입 — 모델 내부 hidden state가 아니라 순수 텍스트 in/out.
"""

from __future__ import annotations

from dataclasses import replace

import torch

from src.pipeline.embeddings import cosine_similarity, embed_texts, embed_with_retry
from src.pipeline.gisting import split_into_sentences
from src.pipeline.types import Chunk


def _snap_to_verbatim(
    generated_sentences: list[str],
    source_sentences: list[str],
    embed_cfg: dict,
    embed_fn=embed_texts,
) -> list[str]:
    """생성된 각 문장이 source_sentences(원문 문장)에 정확히 존재하면 그대로,
    없으면 코사인 유사도가 가장 높은 원문 문장으로 교체한다.

    임베딩 API가 재시도 후에도 실패하면(embed_with_retry가 None 반환) verbatim
    보장은 포기하고 생성 문장을 그대로 반환한다 — 파이프라인이 죽지는 않는다
    (paginate_semantic의 on_error="pass_through"와 같은 철학).
    """
    if not generated_sentences or not source_sentences:
        return generated_sentences

    source_set = set(source_sentences)
    needs_snap = [s for s in generated_sentences if s not in source_set]
    if not needs_snap:
        return generated_sentences

    source_embeddings = embed_with_retry(source_sentences, "passage", embed_cfg, embed_fn)
    if source_embeddings is None:
        return generated_sentences

    needed_embeddings = embed_with_retry(needs_snap, "passage", embed_cfg, embed_fn)
    if needed_embeddings is None:
        return generated_sentences

    snap_map: dict[str, str] = {}
    for gen_sentence, gen_embedding in zip(needs_snap, needed_embeddings):
        scores = [cosine_similarity(gen_embedding, src_embedding) for src_embedding in source_embeddings]
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        snap_map[gen_sentence] = source_sentences[best_idx]

    return [snap_map.get(s, s) for s in generated_sentences]


def rehearse_maintenance(
    chunks: list[Chunk],
    model,
    tokenizer,
    embed_cfg: dict,
    embed_fn=embed_texts,
    max_input_length: int = 1300,
    max_new_tokens: int = 400,
) -> list[Chunk]:
    """각 Chunk를 독립적으로(청크 간 맥락 없이) 처리해 압축본으로 교체한다.

    반환 Chunk: text=verbatim 강제 후처리 적용 완료된 압축본, original_text=압축
    전 원문, index/paragraph_indices/char_start/char_end는 입력과 동일하게
    유지된다. question 등 질문 관련 인자를 받지 않는다(query-agnostic).

    max_input_length는 pko-t5 공식 학습 설정(input_ids max_length=1300)을
    기본값으로 쓴다 — 09_청킹 구현 가이드 §5에서 추정치였던 어절→토큰 환산
    비율을 실측한 결과 어절당 약 2.79토큰으로(추정치 1.5~2보다 높음),
    max_words=350 청크가 실제로 약 975토큰을 쓰는 걸 확인했다.
    """
    model.eval()
    device = next(model.parameters()).device

    result: list[Chunk] = []
    for chunk in chunks:
        inputs = tokenizer(
            chunk.text,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_length,
        ).to(device)

        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        generated_sentences = [s for s in split_into_sentences(generated_text) if s.strip()]
        source_sentences = split_into_sentences(chunk.text)
        verbatim_sentences = _snap_to_verbatim(generated_sentences, source_sentences, embed_cfg, embed_fn)

        compressed_text = " ".join(verbatim_sentences)
        result.append(replace(chunk, text=compressed_text, original_text=chunk.text))

    return result


def novel_ngram_ratio(generated_text: str, source_text: str, n: int = 3) -> float:
    """생성 텍스트의 n-gram 중 원문에 없는("novel") 비율 — 가이드 §6 조작 점검.

    A(유지형)는 이 값이 0에 가까워야 정상(거의 원문 그대로 뽑아 붙인 것이므로).
    0보다 유의미하게 크면 verbatim 후처리(_snap_to_verbatim)가 실제로는
    새로운 표현을 걸러내지 못하고 있다는 신호. 어절(공백 분리) 단위 n-gram을
    쓴다 — 형태소 분석기를 새로 끌어오지 않고 간단하게 유지.
    """
    gen_words = generated_text.split()
    src_words = source_text.split()
    if len(gen_words) < n:
        return 0.0

    gen_ngrams = {tuple(gen_words[i : i + n]) for i in range(len(gen_words) - n + 1)}
    if not gen_ngrams:
        return 0.0
    src_ngrams = {tuple(src_words[i : i + n]) for i in range(len(src_words) - n + 1)}

    novel = gen_ngrams - src_ngrams
    return len(novel) / len(gen_ngrams)
