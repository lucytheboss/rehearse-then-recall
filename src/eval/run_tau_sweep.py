"""CLI for Ablation A4's tau sweep.

    python -m src.eval.run_tau_sweep --dry-run           # synthetic, no API, no checkpoint
    python -m src.eval.run_tau_sweep --genre news        # real corpus + checkpoint + embeddings

Writes `results/tau_sweep{_suffix}.csv` and `.md`. The markdown table is the
one that goes into `03_실험 설계` Ablation A4 and the slide.

⚠ Which checkpoint you sweep with matters for how the numbers read. Driving
this with the stage-1 (`07`) model is the one-shot-compressor-in-a-rolling-loop
mismatch C-DIC Table 1 measures, so absolute retention will sit low. The sweep
is still valid for *choosing* tau — the failure modes at both ends are
properties of the threshold, not of the checkpoint — but re-run it after `07b`
before quoting absolute EM/F1 anywhere.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import asdict
from pathlib import Path

from src.eval.tau_sweep import (
    DEFAULT_TAUS,
    degeneration_checks,
    recommend_tau,
    sweep_tau,
    to_markdown,
)


def _load_real_inputs(genre: str, corpus: str, doc_index: int, checkpoint: Path):
    """One fixed document from the train corpora, plus the checkpoint.

    Reuses `06b`'s document shaping so the sweep scores against the same chunk
    boundaries and the same answer spans the stage-2 work uses — the sweep must
    not introduce a second, incompatible definition of "a document".
    """
    import pandas as pd
    import yaml
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    from src.pipeline.embeddings import embed_texts, load_config
    from src.pipeline.teacher import load_curation_config
    from src.pipeline.types import Chunk

    curation_cfg = load_curation_config()
    max_words = yaml.safe_load(open("configs/chunking.yaml", encoding="utf-8"))["max_words"]
    doc_chunks = curation_cfg["doc_chunks"]

    corpus_dir = Path("data/processed") / f"{genre}_train" / corpus
    if not corpus_dir.exists():
        raise SystemExit(f"no corpus at {corpus_dir}")

    text = (corpus_dir / "corpus.txt").read_text(encoding="utf-8")
    starts, cursor = [], 0
    for word in text.split():
        i = text.find(word, cursor)
        starts.append(i)
        cursor = i + len(word)
    windows = [
        (starts[k], starts[k + max_words] if k + max_words < len(starts) else len(text))
        for k in range(0, len(starts), max_words)
    ]
    window = windows[doc_index * doc_chunks : (doc_index + 1) * doc_chunks]
    if len(window) < doc_chunks:
        raise SystemExit(f"corpus too short for doc_index={doc_index}")

    questions = pd.read_csv(corpus_dir / "questions.csv").dropna(
        subset=["answer", "evidence_char_pos"]
    )

    chunks, answers_by_chunk = [], {}
    for position, (start, end) in enumerate(window):
        chunks.append(Chunk(text=text[start:end].strip(), index=position, char_start=start, char_end=end))
        hits = questions[
            (questions["evidence_char_pos"] >= start) & (questions["evidence_char_pos"] < end)
        ]
        answers_by_chunk[position] = [str(a) for a in hits["answer"].tolist()]

    if not checkpoint.exists():
        raise SystemExit(f"no checkpoint at {checkpoint} — run 07 (or 07b) first")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

    return chunks, answers_by_chunk, model, tokenizer, load_config(), embed_texts, curation_cfg


def _load_synthetic_inputs(n_chunks: int = 24):
    """A deterministic stand-in so the sweep runs with no API key and no
    checkpoint.

    Each topic owns its own 2D rotation plane inside an 8D space, and a chunk
    sits at angle `index * DRIFT` within its topic's plane. So:

        same topic, `gap` chunks apart  ->  cosine = cos(gap * DRIFT)
        different topics                ->  cosine = 0 (orthogonal planes)

    Three properties are needed and none is decorative:

    - **A gradient, not 0/1.** With one-hot topics every tau in (0, 1) behaves
      identically and the table would be flat by construction, so a threshold
      sweep over it would prove nothing.
    - **DRIFT > 0.** Without it a topic's later occurrence is *identical* to
      its earlier one (cosine exactly 1.0), which clears any tau below 1.0, so
      revise would keep firing no matter how strict the threshold and the sweep
      could never reach the append-only failure mode. Real gists drift; the
      fixture has to as well.
    - **Irregular topic recurrence.** A chunk's best similarity is set by the
      *gap* back to the nearest earlier chunk of its topic, so a regular
      `i % n_topics` sequence gives every chunk the same gap and the table
      collapses to a step function. `TOPIC_SEQUENCE` was searched for a spread
      of gaps; it yields insert_share ~0.13 at tau=0.3 rising to ~0.91 at
      tau=0.9, bracketing both failure modes.

    An earlier version put every chunk on one shared circle. With 24 points on
    a 2π circumference the nearest neighbour is always close, so best-similarity
    could never fall far enough for the append-only mode to appear at any tau —
    the per-topic planes exist to give the geometry room.

    Use it to check the sweep mechanics and the shape of the output. The
    numbers are synthetic and must never be quoted as results.
    """
    import math as _math

    import torch
    from transformers import BatchEncoding

    from src.pipeline.types import Chunk

    DRIFT = 0.18  # radians a topic rotates per chunk of document distance
    TOPIC_SEQUENCE = [3, 0, 2, 1, 0, 3, 3, 2, 1, 3, 0, 1,
                      2, 2, 3, 1, 2, 0, 1, 3, 2, 1, 0, 2]
    n_topics = 4

    class _Tokenizer:
        def __init__(self):
            self._id_of, self._text_of = {}, {}

        def __call__(self, text, return_tensors=None, truncation=True, max_length=None,
                     add_special_tokens=True):
            if return_tensors == "pt":
                if text not in self._id_of:
                    new_id = len(self._id_of)
                    self._id_of[text] = new_id
                    self._text_of[new_id] = text
                return BatchEncoding({
                    "input_ids": torch.tensor([[self._id_of[text]]]),
                    "attention_mask": torch.tensor([[1]]),
                })
            return BatchEncoding({"input_ids": list(range(len(text.split())))})

        def decode(self, ids, skip_special_tokens=True):
            # Echo the current passage only, so the gist stays on its own topic
            # and the pool does not homogenise into one blob.
            text = self._text_of[int(ids[0])]
            body = text.split("current passage:", 1)[-1]
            return body.split("related context:", 1)[0].strip()

    class _Model:
        def __init__(self):
            self._param = torch.nn.Parameter(torch.zeros(1))

        def eval(self):
            return self

        def parameters(self):
            yield self._param

        def generate(self, input_ids, attention_mask=None, max_new_tokens=None):
            return input_ids

    def embed_fn(texts, input_type, model, truncate, api_key, timeout=30.0, dimensions=None):
        vectors = []
        for text in texts:
            marker = text.split("topic", 1)[-1]
            topic = int(marker[0]) if marker and marker[0].isdigit() else 0
            index = 0
            if " chunk " in text:
                digits = text.split(" chunk ", 1)[1].split(" ", 1)[0]
                index = int(digits) if digits.isdigit() else 0

            angle = index * DRIFT
            vector = [0.0] * (2 * n_topics)
            vector[2 * topic] = _math.cos(angle)
            vector[2 * topic + 1] = _math.sin(angle)
            vectors.append(vector)
        return vectors

    chunks, answers_by_chunk = [], {}
    for i in range(n_chunks):
        topic = TOPIC_SEQUENCE[i % len(TOPIC_SEQUENCE)]
        answer = f"fact{i:02d}"
        chunks.append(Chunk(text=f"topic{topic} chunk {i} states {answer} plainly.", index=i))
        answers_by_chunk[i] = [answer]

    embed_cfg = {"model": "synthetic", "truncate": "NONE", "timeout": 1.0,
                 "max_retries": 0, "api_key_env": "SYNTHETIC_KEY", "dimensions": None}
    import os

    os.environ["SYNTHETIC_KEY"] = "not-a-real-key"
    return chunks, answers_by_chunk, _Model(), _Tokenizer(), embed_cfg, embed_fn


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Ablation A4 — tau sweep")
    parser.add_argument("--dry-run", action="store_true",
                        help="synthetic inputs; no API key, no checkpoint, no budget")
    parser.add_argument("--genre", default="news")
    parser.add_argument("--corpus", default="corpus_00")
    parser.add_argument("--doc-index", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("experiments/rehearsal_elaborative_small"))
    parser.add_argument("--taus", type=float, nargs="+", default=list(DEFAULT_TAUS))
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--out", type=Path, default=Path("results/tau_sweep"))
    args = parser.parse_args(argv)

    if args.dry_run:
        chunks, answers, model, tokenizer, embed_cfg, embed_fn = _load_synthetic_inputs()
        probe_threshold = 0.6
        out_base = args.out.with_name(args.out.name + "_dryrun")
        print("*** DRY RUN — synthetic inputs. Numbers are not results. ***\n")
    else:
        (chunks, answers, model, tokenizer, embed_cfg, embed_fn,
         curation_cfg) = _load_real_inputs(args.genre, args.corpus, args.doc_index, args.checkpoint)
        probe_threshold = curation_cfg["probe_f1_threshold"]
        out_base = args.out.with_name(f"{args.out.name}_{args.genre}_{args.corpus}")
        print(f"corpus     : {args.genre}/{args.corpus} doc {args.doc_index}")
        print(f"checkpoint : {args.checkpoint}")

    n_answers = sum(len(a) for a in answers.values())
    print(f"chunks     : {len(chunks)}   answer spans: {n_answers}")
    print(f"taus       : {args.taus}\n")

    rows, cached = sweep_tau(
        chunks, answers, model, tokenizer, embed_cfg, embed_fn,
        taus=args.taus, top_k=args.top_k, probe_f1_threshold=probe_threshold,
    )

    print(to_markdown(rows))

    naive = len(chunks) * 2 * len(args.taus)
    print(f"\nembedding cache: {cached.hit_rate:.1%} hit rate, "
          f"{cached.calls} batched calls issued (naive re-run would be ~{naive} texts)")

    report = degeneration_checks(rows)
    print("\nfailure-mode checks:")
    print(report)

    tau, why = recommend_tau(rows)
    print(f"\nrecommendation: {why}" if not math.isnan(tau) else f"\nrecommendation: {why}")

    out_base.parent.mkdir(parents=True, exist_ok=True)
    csv_path = out_base.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0])))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    md_path = out_base.with_suffix(".md")
    md_path.write_text(
        f"# Ablation A4 — tau sweep\n\n"
        f"{'**DRY RUN — synthetic inputs, not results.**' if args.dry_run else ''}\n\n"
        f"{to_markdown(rows)}\n\n## Failure-mode checks\n\n```\n{report}\n```\n\n"
        f"## Recommendation\n\n{why}\n",
        encoding="utf-8",
    )
    print(f"\nwrote {csv_path}\nwrote {md_path}")

    # A failed check means the sweep range never reached a boundary, so a tau
    # chosen from it was chosen from the flat middle of a curve whose edges were
    # never located. Non-zero exit makes that impossible to miss in a shell.
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
