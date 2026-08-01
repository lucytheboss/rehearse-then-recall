"""Rehearsal inference pipelines — maintenance (A) and elaborative (B).

A (`rehearse_maintenance_extractive`) implements `11_되뇌기 구현 가이드.md` §3
(Obsidian). Each Chunk is processed independently — no cross-chunk context, no
rolling state, which is the key difference from elaborative rehearsal B — by
scoring its sentences with the RoBERTa selector in `extractive.py` and keeping
the top `ceil(R * n)` in source order. Verbatim retention is structural: a
kept sentence *is* a source sentence.

`rehearse_maintenance` is the earlier seq2seq path, kept for reference. It
generated text and then snapped each sentence back onto the nearest source
sentence by cosine similarity (guide §3.2) — the same guarantee, but enforced
by post-processing and dependent on the embedding API.

B (`rehearse_elaborative`) is the mirror: a rolling pass that rewrites each
chunk while conditioned on previously rehearsed material, and deliberately
does *not* snap to verbatim. See `retrieve_aligned_context` for how that
prior material is selected.

Hard constraints (guide §2, shared by A/B):
- query-agnostic — no function in this module takes a question/query-like
  argument.
- Preserves chunk position metadata (index/paragraph_indices/char_start/
  char_end) — only `text` is replaced with the compressed version;
  `original_text` stores the pre-compression source.
- Text-level intervention — plain text in/out, not the model's internal
  hidden states.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import math

import torch

from src.pipeline.embeddings import cosine_similarity, embed_texts, embed_with_retry
from src.pipeline.extractive import score_sentences, select_sentences
from src.pipeline.gisting import split_into_sentences
from src.pipeline.types import Chunk


def _snap_to_verbatim(
    generated_sentences: list[str],
    source_sentences: list[str],
    embed_cfg: dict,
    embed_fn=embed_texts,
) -> list[str]:
    """Keeps each generated sentence as-is if it exists verbatim in
    source_sentences; otherwise replaces it with the source sentence with the
    highest cosine similarity.

    If the embedding API still fails after retries (embed_with_retry returns
    None), gives up on the verbatim guarantee and returns the generated
    sentences unchanged — the pipeline doesn't die (same philosophy as
    paginate_semantic's on_error="pass_through").
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


def dependent_ratio(chunks: list[Chunk], tokenizer, target_context_tokens: int) -> float:
    """Compression ratio R = K / |D|, computed once for the whole document.

    From the multi-step summarization pipeline of Sie, Beek, Bots, Brinkkemper
    & Gatt (NLLP 2024, arXiv:2408.09777) §3.2, which compares three ways of
    setting the ratio:

      - fixed    R = 0.4 always, N = ceil(log(K/|D|) / log(R)) extraction passes
      - dependent R = K/|D| per document, always N = 1
      - hybrid   fixed for passes 1..N-1, dependent for the last

    Dependent is what we use, and it is the reason compression here is a
    *single* pass: the ratio is sized so that one extraction already fits the
    downstream budget, so there is nothing left for a second pass to do. The
    paper reports RoBERTa with a single dependent-ratio pass as the best
    extractive configuration in its grid.

    K = `target_context_tokens`, the budget the compressed document has to fit
    into downstream. |D| = the document's own token count. Ratios above 1.0
    are clamped — a document already inside the budget needs no compression.
    """
    doc_tokens = sum(len(tokenizer(chunk.text, add_special_tokens=False).input_ids) for chunk in chunks)
    if doc_tokens == 0:
        return 1.0
    return min(1.0, target_context_tokens / doc_tokens)


def _apply_ratio(
    kept_sentences: list[str],
    source_sentences: list[str],
    ratio: float | None,
) -> list[str]:
    """Trims to `ceil(ratio * len(source_sentences))` sentences, in source order.

    The ratio is a document-level quantity (see `dependent_ratio`) applied
    per chunk, so a long chunk yields proportionally more sentences than a
    short one and the concatenation lands near the intended budget.
    """
    if not kept_sentences:
        return kept_sentences

    # Always restore source order first. _snap_to_verbatim returns sentences in
    # the order the model emitted them, which need not be the order they appear
    # in the chunk — and a maintenance-rehearsal output that reshuffles the
    # source is no longer a faithful verbatim retention of it.
    order = {sentence: i for i, sentence in enumerate(source_sentences)}
    ordered = sorted(kept_sentences, key=lambda s: order.get(s, len(order)))

    if ratio is None:
        return ordered
    budget = max(1, math.ceil(ratio * len(source_sentences)))
    return ordered[:budget]


def rehearse_maintenance_extractive(
    chunks: list[Chunk],
    model,
    tokenizer,
    target_context_tokens: int | None = None,
    compression_ratio: float | None = None,
    score_fn=score_sentences,
) -> list[Chunk]:
    """Maintenance rehearsal (A), extractive path — **the canonical one**.

    Each chunk is split into sentences, scored by the `roberta-base`
    `SentenceScorer` from `extractive.py`, and the top `ceil(R * n)` are kept
    in source order. No cross-chunk context, exactly as in the seq2seq version.

    Why this replaces `rehearse_maintenance`
    ----------------------------------------
    Maintenance rehearsal is surface repetition, so a *selector* is the natural
    model and verbatim retention becomes **structural** — a kept sentence is
    the source sentence, by construction. The seq2seq path had to generate text
    and then repair it back onto the source with `_snap_to_verbatim`, which
    needed the embedding API and could silently give up on the guarantee when
    that API failed. This path calls no API at all, so there is no degraded
    mode and no `embed_cfg`/`embed_fn` argument.

    Measured on the held-out test set (`05`, 2026-08-01): recall@k 0.571 at
    R=0.3 and 0.741 at R=0.5, MAP 0.493 against a ~0.19 chance level. Note
    what the recall figure means downstream — at R=0.3, 43% of the sentences
    the oracle marked important do not survive, which is the quantitative form
    of Ablation A3's concern. Since `dependent_ratio` derives R from K/|D|,
    longer documents sit at the lossier end.

    Ratio selection follows the same precedence as the seq2seq version:
      - `target_context_tokens` set  -> R = K/|D| via `dependent_ratio`
      - `compression_ratio` set      -> that fixed R
      - neither                      -> R = 1.0, i.e. keep every sentence

    Preserves index/paragraph_indices/char_start/char_end; only `text` is
    replaced, with `original_text` holding the source. Takes no
    question-related argument (query-agnostic).

    `score_fn` is injectable for testing, mirroring `embed_fn` elsewhere in
    this module.

    To verify the output with `seam_report`, re-split it with
    `split_into_sentences` — the same splitter that produced it. Splitting on
    `". "` instead strips the terminal periods, and `seam_report` then reports
    every sentence as altered, which looks exactly like a real defect.
    """
    model.eval()

    if target_context_tokens is not None:
        ratio = dependent_ratio(chunks, tokenizer, target_context_tokens)
        print(f"[rehearse_maintenance_extractive] dependent ratio R = K/|D| = {ratio:.3f}")
    elif compression_ratio is not None:
        ratio = compression_ratio
    else:
        ratio = 1.0

    result: list[Chunk] = []
    for chunk in chunks:
        sentences = [s for s in split_into_sentences(chunk.text) if s.strip()]
        if not sentences:
            # Nothing to select from. Keep the chunk as-is rather than emitting
            # an empty one, which would break the position bookkeeping later
            # stages read.
            result.append(replace(chunk, original_text=chunk.text))
            continue

        scores = score_fn(sentences, model, tokenizer)
        kept = select_sentences(sentences, scores, ratio)
        result.append(replace(chunk, text=" ".join(kept), original_text=chunk.text))

    return result


def rehearse_maintenance(
    chunks: list[Chunk],
    model,
    tokenizer,
    embed_cfg: dict,
    embed_fn=embed_texts,
    max_input_length: int = 512,
    max_new_tokens: int = 400,
    target_context_tokens: int | None = None,
    compression_ratio: float | None = None,
) -> list[Chunk]:
    """Legacy seq2seq path — superseded by `rehearse_maintenance_extractive`.

    Kept because its tests still pass and it documents what the earlier
    results were produced with. New pipeline code should use the extractive
    function; this one generates text and repairs it back onto the source,
    which makes the verbatim guarantee contingent on an embedding API call.

    Processes each Chunk independently (no cross-chunk context) and replaces
    it with a compressed version.

    Returned Chunks: text=the compressed version with verbatim post-processing
    already applied, original_text=the pre-compression source;
    index/paragraph_indices/char_start/char_end stay identical to the input.
    Takes no question-related argument (query-agnostic).

    Compression is a **single pass** — N = 1 in the terms of Sie et al. (NLLP
    2024) §3.2, which is what the dependent-ratio strategy implies: the ratio
    is sized so one pass already fits the downstream budget.

    Ratio selection, in precedence order:
      - `target_context_tokens` set  -> R = K/|D| via `dependent_ratio`
      - `compression_ratio` set      -> that fixed R (the paper's "fixed ratio")
      - neither                      -> no trimming, model output kept as-is

    max_input_length=512 is a common convention for t5-base-family
    summarization fine-tuning (the old pko-t5 backbone's 1300 was tuned to
    that model's own training setup and doesn't carry over). Re-measure
    against `configs/chunking.yaml`'s max_words and the actual English
    tokenizer ratio and adjust as needed — English generally runs fewer
    tokens per word than Korean (roughly 1.2-1.5 tokens/word with
    SentencePiece), so there's room to raise the chunk max_words a bit.
    """
    model.eval()
    device = next(model.parameters()).device

    if target_context_tokens is not None:
        ratio = dependent_ratio(chunks, tokenizer, target_context_tokens)
        print(f"[rehearse_maintenance] dependent ratio R = K/|D| = {ratio:.3f}")
    else:
        ratio = compression_ratio

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
        verbatim_sentences = _apply_ratio(verbatim_sentences, source_sentences, ratio)

        compressed_text = " ".join(verbatim_sentences)
        result.append(replace(chunk, text=compressed_text, original_text=chunk.text))

    return result


# --------------------------------------------------------------------------
# Elaborative rehearsal (B) — rolling compression with position-independent
# retrieval over the already-rehearsed history.
# --------------------------------------------------------------------------

# Field markers for B's model input. Defined here, once, because the stage-2
# prep notebook has to build training pairs in the *identical* shape that
# `rehearse_elaborative` feeds at inference — a template that drifts between
# the two is a silent train/inference mismatch, and t5 will absorb it as
# degraded output rather than an error.
#
# The current passage comes first on purpose. The input is right-truncated at
# `max_input_length`, so whatever sits last is what gets cut; putting the
# retrieved context there means overflow costs us the least-similar retrieved
# material instead of the chunk actually being rehearsed.
PASSAGE_FIELD = "current passage:"
CONTEXT_FIELD = "related context:"


def format_elaborative_input(current_text: str, context_texts: list[str]) -> str:
    """Assembles B's model input from the current chunk plus retrieved context.

    Shared by stage-2 data prep and `rehearse_elaborative` so both sides use
    one template.
    """
    if not context_texts:
        return f"{PASSAGE_FIELD} {current_text}"
    return f"{PASSAGE_FIELD} {current_text} {CONTEXT_FIELD} " + " ".join(context_texts)


@dataclass
class RetrievalRecord:
    """One chunk's retrieval decision, kept so the notebooks can audit the
    mechanism instead of trusting it.

    `retrieved_indices` are source chunk indices, so `chunk_index - j` is how
    far back in the document each hit reached. That distance is the whole
    point of the mechanism (see `retrieval_span_report`).

    `write_back_action` is C-DIC Eq. 6 / Algorithm 1 read off the retrieval
    that already happened. The paper branches on `delta_t = max_i S(q_t, Z_i)`:
    below `tau` the turn opens a new slot (topic shift), at or above it the
    best-matching slot is replaced. Since `retrieve_aligned_context` filters at
    exactly that `tau`, an empty result already *means* `delta_t < tau` — so
    the branch needs no second similarity pass, and `similarities[0]` is
    `delta_t` whenever the action is "revise".

    This is a signal for the teacher and for auditing, not a control: the
    curation prompt in `curation.py` asks the teacher to make the same
    insert/revise call from the text itself. Two independent estimates of the
    same decision are more useful than one, since disagreement localises
    whether a bad summary came from a mis-set `tau` or from the teacher
    ignoring the instruction.
    """

    chunk_index: int
    retrieved_indices: list[int] = field(default_factory=list)
    similarities: list[float] = field(default_factory=list)
    pool_size: int = 0
    used_recency_fallback: bool = False
    # This chunk's gist could not be embedded, so it never entered the pool and
    # is invisible to every later chunk. Tracked separately from
    # `used_recency_fallback` because the damage lands on *other* chunks: under
    # a sustained outage the pool stays empty, every chunk truthfully reports
    # "no prior history", and the run would otherwise look like a document with
    # nothing worth retrieving rather than a broken embedding endpoint.
    pool_write_failed: bool = False
    write_back_action: str = "insert"  # "insert" (topic shift) | "revise" (on-topic)


def retrieve_aligned_context(
    current_text: str,
    pool_texts: list[str],
    pool_indices: list[int],
    pool_embeddings: list[list[float]],
    embed_cfg: dict,
    embed_fn=embed_texts,
    top_k: int = 3,
    min_similarity: float = 0.35,
    fallback_top1: bool = False,
) -> list[tuple[int, str, float]] | None:
    """Selects prior rehearsed material semantically aligned with `current_text`,
    **regardless of where it sits in the history**.

    This is C-DIC's thread-aware retrieval (Jung, Kim, Jung, Wang, Zhang,
    Cheung, See & Chen, ICML 2026, arXiv:2606.12411, Eq. 3), transposed from
    dialogue turns to document chunks. The contrast it is meant to break is
    recency: a rolling state `(S_{i-1}, C_i) -> S_i` can only ever elaborate
    against the immediately preceding chunk, so a callback to something 30
    chunks earlier is unreachable no matter how relevant it is. Ranking the
    whole prior history by cosine similarity makes position irrelevant and
    leaves relevance as the only selection criterion.

    `current_text` is the **current chunk** — never the downstream question.
    Keying this on a question would make B question-aware and break both the
    module's query-agnostic constraint and the A/B contrast the experiment
    rests on.

    That substitution is what makes importing Eq. 3 legitimate at all. C-DIC's
    score is keyed on `q_t`, and `03_실험 설계` initially rejected the whole
    mechanism as query-aware on that basis. But `q_t` there is the current
    dialogue *turn* — the new content being written into memory — not a
    question posed about the memory afterwards. Its counterpart in this
    pipeline is `C_i`, the chunk being rehearsed, which is available without
    knowing anything about the final QA. The eval question has no counterpart
    in C-DIC's loop at all.

    Eq. 3 also applies a recency decay `exp(-alpha * dt)` to the similarity,
    which is deliberately dropped: dialogue turns arrive in time and older
    threads genuinely go stale, whereas a document's chunk 0 is not staler than
    its chunk 40 — the whole text is present at once. Keeping the decay would
    reintroduce the recency bias this function exists to remove.

    Embedded asymmetrically (`"query"` for the current chunk, `"passage"` for
    the pool) because the NIM embedding model is trained for that split.

    `min_similarity` is C-DIC's threshold `tau`, and it is a relevance floor
    rather than a formality: with top-k alone, a chunk that genuinely relates
    to nothing earlier still drags in the three least-unrelated candidates, and
    unrelated context in the prompt is what makes a small seq2seq invent
    connections. Below the floor, B correctly falls back to elaborating the
    chunk on its own.

    The same `tau` decides the write-back branch in Eq. 6 (see
    `RetrievalRecord.write_back_action`), so one threshold governs both — as in
    the paper, where the write-back reuses the retrieval score. C-DIC App. K
    reports the failure mode for setting it too strictly: at `tau = 0.9`
    nothing ever clears the bar, so nothing is ever revised and slots grow
    without bound. The text-level analogue is a running summary that only ever
    gains sentences.

    `fallback_top1` implements the other half of Eq. 3 — "if no slot exceeds
    tau, fall back to the single best match". Note what this separates: the
    *generation* context falls back to the best match, while the *write-back*
    still branches at tau and inserts. A topic-shift chunk is therefore
    elaborated against its nearest neighbour rather than against nothing, but
    is still recorded as opening a new thread. The two decisions share a score
    and are not the same decision.

    Returns (chunk index, text, similarity) sorted by similarity descending,
    or None if the embedding API failed after retries — the caller decides
    what to degrade to, matching `_snap_to_verbatim`'s "don't kill the
    pipeline" philosophy. With `fallback_top1`, a returned hit may sit below
    `min_similarity`; read its similarity to tell the cases apart.
    """
    if not pool_texts:
        return []

    query_embedding = embed_with_retry([current_text], "query", embed_cfg, embed_fn)
    if query_embedding is None:
        return None

    scored = [
        (index, text, cosine_similarity(query_embedding[0], embedding))
        for index, text, embedding in zip(pool_indices, pool_texts, pool_embeddings)
    ]
    scored.sort(key=lambda item: item[2], reverse=True)

    cleared = [item for item in scored if item[2] >= min_similarity][:top_k]
    if cleared or not fallback_top1:
        return cleared
    return scored[:1]


def _fit_context_to_budget(
    current_text: str,
    context_texts: list[str],
    tokenizer,
    max_input_length: int,
) -> list[str]:
    """Drops the lowest-ranked retrieved passages until the assembled input
    fits `max_input_length`.

    Right-truncation would already protect the current passage (it is placed
    first), but it would cut a retrieved passage mid-sentence and hand the
    model a dangling fragment. Dropping whole passages instead keeps every
    piece of context that survives a complete one.
    """
    kept = list(context_texts)
    while kept:
        assembled = format_elaborative_input(current_text, kept)
        if len(tokenizer(assembled, add_special_tokens=True).input_ids) <= max_input_length:
            break
        kept.pop()
    return kept


def rehearse_elaborative(
    chunks: list[Chunk],
    model,
    tokenizer,
    embed_cfg: dict,
    embed_fn=embed_texts,
    max_input_length: int = 512,
    max_new_tokens: int = 64,
    top_k: int = 3,
    min_similarity: float = 0.35,
    carry_previous: bool = False,
) -> tuple[list[Chunk], list[RetrievalRecord]]:
    """Rewrites each Chunk conditioned on semantically aligned prior material.

    Runs left to right, and the retrieval pool holds every chunk already
    rehearsed — so chunk i can be elaborated against chunk 0 just as easily as
    against chunk i-1. The pool stores the *compressed* texts, not the
    originals: that is what keeps the accumulated history bounded as the
    document grows, which is the "incremental compression" half of the idea.

    The topic is re-derived from `C_i` at **every** chunk — there is no
    document-level topic fixed once up front, and never the eval question. So
    what counts as related context can change completely from one chunk to the
    next, which is the point: that is what lets chunk 40 elaborate against
    chunk 0 and chunk 41 against chunk 39.

    `carry_previous` defaults to False for that reason. Unconditionally
    appending `S_{i-1}` would mean every chunk is elaborated against the
    previous one *whether or not it is on topic*, which is a second, competing
    definition of relevance layered on top of the per-chunk one — and on an
    insert (topic shift) it forces together exactly the two unrelated topics
    the curation prompt in `curation.py` forbids blending. Cross-chunk
    continuity is preserved without it: `retrieve_aligned_context`'s
    `fallback_top1` guarantees non-empty context whenever anything has been
    rehearsed, so B never silently degenerates into A's independent per-chunk
    processing. Set `carry_previous=True` to restore the pure recency baseline
    as an ablation.

    Unlike A there is **no verbatim snapping** — snapping would undo exactly
    the rewriting B exists to perform.

    Returns the rehearsed chunks plus one `RetrievalRecord` per chunk. A
    returns bare chunks because it has no mechanism to audit; B does, and
    `retrieval_span_report` needs the trace to show the mechanism is actually
    reaching past recency.

    Costs two embedding calls per chunk (one query, one to add the fresh gist
    to the pool). If the API fails, that chunk degrades to recency-only rather
    than aborting the run, and the record is flagged `used_recency_fallback`.

    ⚠ Requires a stage-2 checkpoint. Driving this loop with the stage-1
    (`07`) model — trained on XSum one-shot `document -> summary` — is the
    exact mismatch C-DIC Table 1 measures: ICAE trained one-shot and applied
    incrementally scores PPL 513.774 against 27.656 for the same model used
    one-shot, and App. F attributes the collapse to the objective never having
    trained the compressor to consume its own output. `curation.py` builds the
    rolling-shaped targets that close that gap; until it has run, treat output
    from this function as unvalidated.
    """
    model.eval()
    device = next(model.parameters()).device

    pool_texts: list[str] = []
    pool_indices: list[int] = []
    pool_embeddings: list[list[float]] = []

    result: list[Chunk] = []
    records: list[RetrievalRecord] = []

    for position, chunk in enumerate(chunks):
        record = RetrievalRecord(chunk_index=chunk.index, pool_size=len(pool_texts))

        retrieved = retrieve_aligned_context(
            chunk.text, pool_texts, pool_indices, pool_embeddings,
            embed_cfg, embed_fn, top_k=top_k, min_similarity=min_similarity,
            fallback_top1=True,
        )
        if retrieved is None:
            record.used_recency_fallback = True
            retrieved = []
        else:
            record.retrieved_indices = [index for index, _, _ in retrieved]
            record.similarities = [round(score, 4) for _, _, score in retrieved]
            # Eq. 6 branches on delta_t = the best score, which is retrieved[0]
            # because the list is sorted. Checking the score rather than
            # emptiness matters now that fallback_top1 can return a below-tau
            # hit for the model to condition on: that chunk still opens a new
            # thread, it just is not elaborated against nothing.
            record.write_back_action = (
                "revise" if retrieved and retrieved[0][2] >= min_similarity else "insert"
            )

        context_texts = [text for _, text, _ in retrieved]
        if carry_previous and position > 0:
            previous_text = result[-1].text
            if previous_text not in context_texts:
                context_texts.append(previous_text)

        context_texts = _fit_context_to_budget(chunk.text, context_texts, tokenizer, max_input_length)
        model_input = format_elaborative_input(chunk.text, context_texts)

        inputs = tokenizer(
            model_input,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_length,
        ).to(device)

        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        elaborated = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

        # An empty generation would poison the pool for every later chunk, so
        # keep the source text instead of appending nothing.
        if not elaborated:
            elaborated = chunk.text

        result.append(replace(chunk, text=elaborated, original_text=chunk.text))
        records.append(record)

        fresh_embedding = embed_with_retry([elaborated], "passage", embed_cfg, embed_fn)
        if fresh_embedding is None:
            record.pool_write_failed = True
        else:
            pool_texts.append(elaborated)
            pool_indices.append(chunk.index)
            pool_embeddings.append(fresh_embedding[0])

    return result, records


def retrieval_span_report(records: list[RetrievalRecord]) -> dict:
    """Manipulation check for the retrieval mechanism itself.

    `mean_span` is the average `chunk_index - retrieved_index`. It answers the
    only question that decides whether this feature earns its place: **is
    retrieval reaching past the previous chunk?** If `mean_span` sits near 1.0
    and `recency_share` near 1.0, similarity ranking has simply rediscovered
    recency, B is doing what the rolling baseline already did, and the added
    embedding cost bought nothing. A high `mean_span` with a low `fire_rate`
    is the other failure — the mechanism triggers too rarely to matter.

    Read `pool_write_failures` before reading anything else: a nonzero value
    means chunks are missing from the pool they should have been retrievable
    from, so a low `eligible` or `fire_rate` is an infrastructure result, not
    a finding about the document.

    This is the same discipline `novel_ngram_ratio` applies to B's output:
    state what the mechanism must show, then measure it rather than assume it.
    """
    # "Fired" means cleared tau, i.e. a genuine on-topic match — not merely
    # "returned something", since fallback_top1 also returns a below-tau
    # neighbour on a topic shift. Spans are measured over everything actually
    # fed to the model, fallbacks included, because the question there is how
    # far back the context came from regardless of why it was chosen.
    eligible = [r for r in records if r.pool_size > 0]
    fired = [r for r in eligible if r.write_back_action == "revise"]
    used = [r for r in eligible if r.retrieved_indices]
    spans = [r.chunk_index - index for r in used for index in r.retrieved_indices]

    return {
        "chunks": len(records),
        "eligible": len(eligible),
        "fire_rate": len(fired) / len(eligible) if eligible else 0.0,
        "mean_hits": sum(len(r.retrieved_indices) for r in fired) / len(fired) if fired else 0.0,
        "mean_span": sum(spans) / len(spans) if spans else 0.0,
        "max_span": max(spans) if spans else 0,
        "recency_share": sum(s == 1 for s in spans) / len(spans) if spans else 0.0,
        "mean_similarity": (
            sum(s for r in fired for s in r.similarities) / sum(len(r.similarities) for r in fired)
            if fired else 0.0
        ),
        "api_fallbacks": sum(r.used_recency_fallback for r in records),
        "pool_write_failures": sum(r.pool_write_failed for r in records),
        # Eq. 6's branch, aggregated — the complement of fire_rate, kept under
        # its own name because it is read for a different question: near 1.0 on
        # a coherent document means tau is too strict, so nothing is ever
        # revised and the summary only grows (C-DIC App. K).
        "insert_share": (
            sum(r.write_back_action == "insert" for r in eligible) / len(eligible)
            if eligible else 0.0
        ),
    }


def novel_ngram_ratio(generated_text: str, source_text: str, n: int = 3) -> float:
    """Fraction of the generated text's n-grams that are "novel" (not present
    in the source) — the manipulation check from guide §6.

    For A (maintenance), this should be close to 0 — the output should be
    almost entirely lifted verbatim from the source. A value meaningfully
    above 0 signals that verbatim post-processing (_snap_to_verbatim) isn't
    actually filtering out new phrasing. Uses word-level (whitespace-split)
    n-grams — kept simple rather than pulling in a morphological analyzer.
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
