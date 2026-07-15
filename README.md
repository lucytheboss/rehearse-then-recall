# rehearse-then-recall

## 프로젝트 개요


## 논문 / 발표 링크
- 논문: TBD
- 발표 자료: TBD

## 구조

```
rehearse-then-recall/
├── README.md
├── LICENSE
├── .gitignore
├── pyproject.toml / requirements.txt
│
├── configs/                       # 실험별 하이퍼파라미터 (YAML)
│   └── chunking.yaml              # 청킹 min/max words, 니들 폐기 이후 위치 사후 산출 설정
│
├── data/
│   ├── raw/                       # 원본 데이터 (gitignore 대상)
│   ├── processed/                 # 전처리된 학습용 데이터
│   ├── eval_texts/                # 평가용 텍스트 (주간 문학동네 단편 등)
│   └── scripts/                   # 데이터 전처리 스크립트
│
├── src/
│   ├── models/                    # 유지형/정교화 되뇌기, QG 모델 정의
│   ├── pipeline/                  # 청킹, 중요도 필터, 파이프라인 오케스트레이션
│   ├── train/                     # 모델별 학습 스크립트
│   └── eval/                      # 평가 메트릭 (EM/F1 등)
│
├── experiments/
│   ├── ablation_extraction/       # 추출 단계 유무 ablation 결과
│   └── logs/
│
├── notebooks/                     # Colab 학습/실험용 노트북
│
└── docs/
    └── paper/                     # 발표/논문용 자료
```

> `.gitkeep`이 들어있는 폴더는 아직 실제 파일이 없는 자리(구조만 확보). 채워지는 대로 이 트리도 갱신 필요.
