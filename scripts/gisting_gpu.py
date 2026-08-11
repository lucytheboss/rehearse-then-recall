"""Standalone gisting step (10's §1-2 + §9's full_gists_by_genre) for a
rented CUDA GPU -- the bottleneck in `10`: ~1,200 chunks across the 4 eval
corpora need a real `rehearse_elaborative` generate() call each, which is
the same GPU-bound cost the retention check had, just ~5x more of it.

Everything downstream of gisting in `10` (the QA conditions themselves) is
NIM chat API calls, not compute -- cheap in wall-clock time and better run
locally where the notebook's condition-specific logic (lookup window,
retrieval targeting, adaptive-k) already lives, rather than re-implementing
all of it here. This script only produces `full_gists_by_genre` and saves it
to disk; `10` loads it back in instead of recomputing it.

Needs, copied over from the local repo:
    src/                                                    (whole package)
    configs/                                                (chunking.yaml, curation.yaml, importance_filter.yaml)
    .env                                                    (NVIDIA_NIM_API_KEY)
    experiments/rehearsal_elaborative_stage2_flant5base_ratio0/
    data/processed/wiki_eval/wiki_eval_corpus.txt
    data/processed/news_eval/news_eval_corpus.txt
    data/processed/narrativeqa_length_stress/novel_eval_corpus.txt
    data/processed/caselaw_eval/caselaw_eval_corpus.txt

Usage on the remote instance:
    pip install torch transformers requests pyyaml python-dotenv pandas numpy tqdm nltk
    python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
    python scripts/gisting_gpu.py

Then copy results/full_gists_by_genre.json back to this repo's same path --
`10`'s gisting cell loads it from there if present instead of recomputing.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

import torch
import yaml
from safetensors.torch import load_file
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.pipeline.chuncking import paginate_semantic, plain_text_to_paragraphs
from src.pipeline.embeddings import embed_texts, load_config as load_embed_config
from src.pipeline.rehearsal import rehearse_elaborative
from src.pipeline.types import Chunk

STAGE2_DIR = Path("experiments/rehearsal_elaborative_stage2_flant5base_ratio0")

GENRES = {
    "wiki": {"corpus_path": "data/processed/wiki_eval/wiki_eval_corpus.txt"},
    "news": {"corpus_path": "data/processed/news_eval/news_eval_corpus.txt"},
    "novel": {"corpus_path": "data/processed/narrativeqa_length_stress/novel_eval_corpus.txt"},
    "caselaw": {"corpus_path": "data/processed/caselaw_eval/caselaw_eval_corpus.txt"},
}


def load_stage2(path: Path, device: str):
    """Same lm_head fix as retention_check_gpu.py / 07b -- see those for why
    config.tie_word_embeddings=False on from_pretrained() is too blunt."""
    model = AutoModelForSeq2SeqLM.from_pretrained(path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(path)
    state = load_file(str(path / "model.safetensors"))
    if "lm_head.weight" in state:
        model.lm_head.weight = torch.nn.Parameter(
            state["lm_head.weight"].to(device=device, dtype=model.lm_head.weight.dtype)
        )
        model.config.tie_word_embeddings = False
    assert model.lm_head.weight is not model.shared.weight, "lm_head got re-tied -- fix didn't take"
    return model, tokenizer


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    if device == "cpu":
        print("WARNING: no CUDA GPU detected -- this defeats the point of running this remotely.")

    if not STAGE2_DIR.exists():
        raise SystemExit(f"missing: {STAGE2_DIR} -- copy it over first.")

    chunk_cfg = yaml.safe_load(open("configs/chunking.yaml", encoding="utf-8"))
    chunk_embed_cfg = load_embed_config("configs/importance_filter.yaml")

    corpora: dict[str, str] = {}
    for genre, cfg in GENRES.items():
        with open(cfg["corpus_path"], encoding="utf-8") as f:
            corpora[genre] = f.read()

    chunks_by_genre: dict[str, list[Chunk]] = {}
    for genre, text in corpora.items():
        paragraphs = plain_text_to_paragraphs(text)
        chunks_by_genre[genre] = paginate_semantic(
            paragraphs,
            min_words=chunk_cfg["min_words"],
            max_words=chunk_cfg["max_words"],
            granularity="paragraph",
            config=chunk_embed_cfg,
            embed_fn=embed_texts,
        )
        print(f"{genre:10} chunks: {len(chunks_by_genre[genre])}")

    model, tokenizer = load_stage2(STAGE2_DIR, device)
    embed_cfg = load_embed_config()

    full_gists_by_genre: dict[str, list[Chunk]] = {}
    for genre, chunk_list in tqdm(chunks_by_genre.items(), desc="genres"):
        full_chunks = [Chunk(text=c.text, index=i) for i, c in enumerate(chunk_list)]
        rehearsed, _ = rehearse_elaborative(
            full_chunks, model, tokenizer, embed_cfg, embed_texts,
        )
        full_gists_by_genre[genre] = rehearsed
        raw_words = sum(len(c.original_text.split()) for c in rehearsed)
        gist_words = sum(len(c.text.split()) for c in rehearsed)
        print(f"{genre:10} {len(rehearsed):4} chunks | raw {raw_words:6} words -> "
              f"gist {gist_words:6} words ({gist_words / raw_words:.1%})")

    Path("results").mkdir(exist_ok=True)
    out_path = "results/full_gists_by_genre.json"
    serializable = {genre: [asdict(c) for c in chunks] for genre, chunks in full_gists_by_genre.items()}
    json.dump(serializable, open(out_path, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"saved to {out_path}")


if __name__ == "__main__":
    main()
