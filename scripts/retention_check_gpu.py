"""Standalone stage-2 retention/collapse check (07b's §5+§6) for a rented CUDA
GPU — the actual result that decides whether stage 2 worked. Needs the full
src/pipeline package (unlike train_stage{1,2}_gpu.py, which were kept
dependency-free) since this reuses ThreadMemory / rehearse_elaborative /
retention_probes as-is rather than reimplementing them.

BUG (2026-08-09): AutoModelForSeq2SeqLM.from_pretrained() re-ties
lm_head.weight to shared.weight on load because config.tie_word_embeddings=True
in these checkpoints' config, silently discarding the actually-trained (and
genuinely different -- flan-t5-base is not really tied despite the config
flag) lm_head.weight. Produces fluent-looking garbage generation with no
error/warning. Fixed below by forcing tie_word_embeddings=False before
from_pretrained on every load of these checkpoints.

Needs, copied over from the local repo:
    src/                                                    (whole package)
    configs/                                                (curation.yaml, importance_filter.yaml)
    .env                                                    (NVIDIA_NIM_API_KEY)
    experiments/rehearsal_elaborative_flant5base/           (stage-1 checkpoint)
    experiments/rehearsal_elaborative_stage2_flant5base_ratio0/  (stage-2 checkpoint)
    data/processed/rehearsal_elaborative_stage2/curation_cache.jsonl
    data/processed/rehearsal_elaborative_stage2_ratio0/test_pairs_raw.csv

Usage on the remote instance:
    pip install torch transformers datasets requests pyyaml python-dotenv pandas numpy tqdm
    python retention_check_gpu.py

Then copy results/elaborative_stage2_retention_by_age_flant5base_ratio0.csv
back to this repo's same path.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.pipeline.curation import probe_accuracy_by_position, retention_probes
from src.pipeline.embeddings import embed_texts, load_config as load_embed_config
from src.pipeline.rehearsal import rehearse_elaborative
from src.pipeline.teacher import load_curation_config
from src.pipeline.types import Chunk

STAGE1_DIR = Path("experiments/rehearsal_elaborative_flant5base")
OUTPUT_DIR = Path("experiments/rehearsal_elaborative_stage2_flant5base_ratio0")
CACHE_DIR = Path("data/processed/rehearsal_elaborative_stage2")
DATA_DIR = Path("data/processed/rehearsal_elaborative_stage2_ratio0")


def load_untied(path: Path, device: str):
    """Loads normally (so encoder/decoder embed_tokens stay correctly tied to
    shared.weight, which is structural to T5 and fine) then surgically
    replaces just lm_head.weight with the checkpoint's actual saved value.

    Passing config.tie_word_embeddings=False to from_pretrained() is too
    blunt: on this transformers version it also untangles encoder/decoder
    embed_tokens from shared.weight, and since those are never saved as
    separate keys in a T5 checkpoint, they come back MISSING and get
    randomly reinitialized (confirmed via the remote LOAD REPORT). Loading
    normally first, then only detaching+overwriting lm_head.weight,
    leaves everything else's tie intact.
    """
    from safetensors.torch import load_file

    model = AutoModelForSeq2SeqLM.from_pretrained(path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(path)
    state = load_file(str(Path(path) / "model.safetensors"))
    if "lm_head.weight" in state:
        model.lm_head.weight = torch.nn.Parameter(
            state["lm_head.weight"].to(device=device, dtype=model.lm_head.weight.dtype)
        )
        model.config.tie_word_embeddings = False
    assert model.lm_head.weight is not model.shared.weight, f"lm_head got re-tied for {path}"
    return model, tokenizer


def memory_states_for(m, tok, chunk_texts: list[str], embed_cfg: dict) -> list[str]:
    chunks = [Chunk(text=t, index=i) for i, t in enumerate(chunk_texts)]
    _, recs = rehearse_elaborative(
        chunks, m, tok, embed_cfg, embed_texts,
        max_new_tokens=256, record_memory_state=True,
    )
    return [r.memory_state for r in recs]


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    if device == "cpu":
        print("WARNING: no CUDA GPU detected -- this defeats the point of running this remotely.")

    for p in (STAGE1_DIR, OUTPUT_DIR, CACHE_DIR / "curation_cache.jsonl", DATA_DIR / "test_pairs_raw.csv"):
        if not Path(p).exists():
            raise SystemExit(f"missing: {p} -- copy it over first.")

    cfg = load_curation_config()
    embed_cfg = load_embed_config()

    records = {}
    for line in (CACHE_DIR / "curation_cache.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            records[record["key"]] = record

    test_keys = set(pd.read_csv(DATA_DIR / "test_pairs_raw.csv")["key"])
    test_records = [records[k] for k in test_keys if k in records]
    print(f"{len(test_records)} held-out documents")

    stage1_model, stage1_tokenizer = load_untied(STAGE1_DIR, device)
    stage2_model, stage2_tokenizer = load_untied(OUTPUT_DIR, device)
    print(f"stage1 params: {sum(p.numel() for p in stage1_model.parameters()):,}")
    print(f"stage2 params: {sum(p.numel() for p in stage2_model.parameters()):,}")

    curves = {}
    for name, (m, tok) in {
        "stage 1 (one-shot, driven incrementally)": (stage1_model, stage1_tokenizer),
        "stage 2 (rolling, threaded)": (stage2_model, stage2_tokenizer),
        "teacher (upper bound)": (None, None),
    }.items():
        probes = []
        for record in tqdm(test_records, desc=name):
            answers = {int(k): v for k, v in record["answers_by_chunk"].items()}
            states = record["memory_states"] if m is None else memory_states_for(m, tok, record["chunk_texts"], embed_cfg)
            probes += retention_probes(states, answers, cfg["probe_f1_threshold"])
        report = probe_accuracy_by_position(probes, collapse_after=3)
        curves[name] = report
        print(f"{name:45} early {report['early_accuracy']:.3f}  late {report['late_accuracy']:.3f}  drop {report['drop']:+.3f}")

    comparison = pd.DataFrame({name: report["by_position"] for name, report in curves.items()})
    comparison.index.name = "age (compressions since the fact was read)"
    print(comparison.round(3).to_string())

    Path("results").mkdir(exist_ok=True)
    out_path = "results/elaborative_stage2_retention_by_age_flant5base_ratio0.csv"
    comparison.to_csv(out_path)
    print(f"saved to {out_path}")


if __name__ == "__main__":
    main()
