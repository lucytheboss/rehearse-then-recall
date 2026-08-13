"""Teacher LLM client for stage-2 curation — NIM chat completions.

Offline only. This is the one place in the project that issues *generative*
prompts, and it runs while building training data, never at evaluation time.
`03_실험 설계` §핵심 원칙 forbids generative prompts before the final QA step in
the *pipeline*; a data-prep script is not the pipeline. At inference B is still
a single seq2seq forward pass with no teacher anywhere in the loop.

Mirrors `embeddings.py` deliberately: same config shape, same retry helper,
same "return None rather than kill the run" contract. A curation pass is
thousands of sequential calls and will lose some of them; a single raised
exception at call 2,400 would discard the whole run.
"""

from __future__ import annotations

import os
import random
import sys
import time
from dataclasses import dataclass, field

import requests
import yaml
from dotenv import find_dotenv, load_dotenv

from src.pipeline.curation import (
    answer_supported,
    build_teacher_messages,
    build_teacher_messages_threaded,
    build_teacher_messages_threaded_corrective,
)
from src.pipeline.embeddings import embed_texts, embed_with_retry
from src.pipeline.thread_memory import ThreadMemory

load_dotenv(find_dotenv(usecwd=True))
_CHAT_URL = "https://integrate.api.nvidia.com/v1/chat/completions"


class TeacherAPIError(RuntimeError):
    pass


def load_curation_config(path: str = "configs/curation.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def chat_completion(
    messages: list[dict],
    model: str,
    api_key: str,
    temperature: float = 0.0,
    max_tokens: int = 512,
    timeout: float = 60.0,
) -> str:
    """One raw NIM chat-completions call. Tests substitute a fake for this.

    `temperature=0.0` by default: the teacher is producing *labels*, and a
    label that changes between runs makes the collapse diagnostic unreadable —
    a drop at position 6 could then be sampling noise rather than degradation.
    """
    response = requests.post(
        _CHAT_URL,
        headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
        json={
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        timeout=timeout,
    )
    if response.status_code != 200:
        raise TeacherAPIError(f"NIM chat API failed ({response.status_code}): {response.text}")
    return response.json()["choices"][0]["message"]["content"]


def complete_with_retry(
    messages: list[dict], config: dict, chat_fn=chat_completion, rate_limiter=None,
) -> str | None:
    """Retries `chat_fn` up to `config["max_retries"]` times, then gives up and
    returns None after logging the cause to stderr.

    `rate_limiter`, if given, is a `rate_limit.RateLimiter` whose `.acquire()`
    is called before every attempt (including retries) — see that module;
    this is what actually stops concurrent document curation from tripping
    NIM's 429 rate limit, which backoff alone cannot (backoff only slows down
    a caller *after* it has already been rejected).

    Backs off between attempts (2026-08-04). A pilot run against the real NIM
    endpoint measured **42 of 72 calls failing** (58%) — `ReadTimeout` at the
    60s ceiling and `503 ResourceExhausted: Worker local total request limit
    reached (31/16)` — over 3h15m for 6 documents. Retrying immediately after
    a 503 that says the worker is already over its concurrent-request limit
    only adds another request to the same overloaded worker; without a delay,
    a retry is not meaningfully different from the call that just failed.
    Exponential backoff (`base_delay * 2**attempt`, capped, plus jitter so a
    whole document's retries don't re-synchronize on the same clock tick) is
    the minimum fix — it does not address the endpoint being oversubscribed
    (that may need a different model, a quieter time window, or raising
    `timeout` further), but it stops the retry loop itself from being part of
    the problem.
    """
    api_key = os.environ.get(config["api_key_env"])
    if not api_key:
        raise TeacherAPIError(f"Environment variable {config['api_key_env']} is not set.")

    base_delay = config.get("retry_base_delay", 2.0)
    max_delay = config.get("retry_max_delay", 30.0)

    last_error: Exception | None = None
    for attempt in range(config["max_retries"] + 1):
        if rate_limiter is not None:
            rate_limiter.acquire()
        try:
            return chat_fn(
                messages,
                model=config["model"],
                api_key=api_key,
                temperature=config["temperature"],
                max_tokens=config["max_tokens"],
                timeout=config["timeout"],
            )
        except (TeacherAPIError, requests.RequestException, KeyError, IndexError) as e:
            last_error = e
            if attempt == config["max_retries"]:
                print(
                    f"[teacher] chat call failed (retries exhausted: {config['max_retries']}): "
                    f"{type(last_error).__name__}: {last_error}",
                    file=sys.stderr,
                )
                return None
            delay = min(max_delay, base_delay * (2**attempt)) + random.uniform(0, base_delay)
            time.sleep(delay)
    return None


@dataclass
class CurationResult:
    """One document's curated running summaries.

    `running_summaries[i]` is the summary **after** chunk i, which is the
    alignment `build_stage2_pairs` expects: chunk i conditions on
    `running_summaries[i-1]` and targets `running_summaries[i]`.

    `failed_positions` are chunks whose teacher call gave up. The summary is
    carried forward unchanged at those positions, so the document stays usable
    and correctly aligned — but a pair built across a failure has a target
    identical to its input's context, which teaches the model to do nothing.
    `drop_failed_pairs` in the prep notebook removes them; this list is what it
    reads.
    """

    running_summaries: list[str] = field(default_factory=list)
    failed_positions: list[int] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failed_positions


def curate_document(chunk_texts: list[str], config: dict, chat_fn=chat_completion) -> CurationResult:
    """Runs the teacher's rolling `(S_{i-1}, C_i) -> S_i` loop over one document.

    This is the loop that produces stage 2's targets. Stage 1 trains B on XSum
    `(document -> one-sentence summary)`, a *one-shot* objective, and C-DIC
    Table 1 measures what happens when a one-shot compressor is then driven
    incrementally: ICAE scores PPL 513.774 against 27.656 for the same model
    used one-shot as intended. Appendix F attributes it to the structural
    cause — an objective that never trained the compressor to consume its own
    output — so the fix has to be targets that are themselves rolling.

    Sequential by construction: chunk i's prompt contains chunk i-1's answer,
    so this cannot be batched or parallelised within a document. Documents are
    independent of each other and can be.
    """
    result = CurationResult()
    summary = ""

    for position, chunk_text in enumerate(chunk_texts):
        messages = build_teacher_messages(summary, chunk_text)
        completion = complete_with_retry(messages, config, chat_fn)

        if completion is None:
            # Carry the previous summary forward so positions stay aligned with
            # chunks. Truncating the document instead would silently shorten it
            # and bias the position axis the collapse diagnostic is plotted on.
            result.failed_positions.append(position)
        else:
            summary = completion.strip()

        result.running_summaries.append(summary)

    return result


# --------------------------------------------------------------------------
# Threaded curation — the teacher standing in for B inside C-DIC's actual
# retrieve / generate / write-back loop (see `curation.py`'s module comment
# above `TEACHER_SYSTEM_PROMPT_THREADED` for why the linear loop above trains
# a different, easier task than the one B is asked to perform at inference).
# --------------------------------------------------------------------------


@dataclass
class ThreadedCurationResult:
    """One document's curated multi-slot memory trace — C-DIC's actual shape,
    where `CurationResult` above is a single linear running text.

    Every field is one entry per chunk position, aligned like
    `CurationResult`'s:

      - `gists[i]` — the teacher's shortened passage for chunk i's thread,
        conditioned on `context_texts[i]`. This is `target_text` for the pair
        at position i.
      - `context_texts[i]` — the related slot text(s) retrieved *before*
        generating `gists[i]`, in the exact list form `format_elaborative_input`
        takes, so `input_text = format_elaborative_input(chunk_texts[i],
        context_texts[i])` reproduces the inference-time input exactly — no
        separate lookup step needed, unlike the linear version's
        `gold_summaries[i-1]`.
      - `write_back_actions[i]` — "insert" (new thread) or "revise" (replaced
        the matched slot) — C-DIC Eq. 6, actually executed via
        `ThreadMemory.write_back`, not merely labelled.
      - `memory_states[i]` — every slot's text concatenated, immediately after
        chunk i's write-back. This is what a downstream reader would see at
        that point, and the correct target for `retention_probes` — a fact
        can live in a *different* slot than the one this chunk touched, so
        `gists[i]` alone would miss it.

    `failed_positions` are chunks whose teacher call gave up; `gists[i]` at
    those positions falls back to the raw chunk text so the document stays
    usable and aligned, mirroring `CurationResult`'s "carry state forward,
    don't truncate" contract. `retrieval_failed_positions` /
    `store_failed_positions` are the two ways an embedding call can fail —
    kept separate because the damage lands differently: a failed *query*
    embedding means this chunk found nothing to retrieve (degrades to
    opening a new thread, same as a real topic shift would), while a failed
    *store* embedding means this chunk's gist never entered memory and is
    invisible to every later chunk's retrieval — the same asymmetry
    `RetrievalRecord.used_recency_fallback` / `.pool_write_failed` track in
    `rehearsal.py`.
    """

    gists: list[str] = field(default_factory=list)
    context_texts: list[list[str]] = field(default_factory=list)
    write_back_actions: list[str] = field(default_factory=list)
    memory_states: list[str] = field(default_factory=list)
    failed_positions: list[int] = field(default_factory=list)
    retrieval_failed_positions: list[int] = field(default_factory=list)
    store_failed_positions: list[int] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not (self.failed_positions or self.retrieval_failed_positions or self.store_failed_positions)


def curate_document_threaded(
    chunk_texts: list[str],
    embed_cfg: dict,
    config: dict,
    embed_fn=embed_texts,
    chat_fn=chat_completion,
    top_k: int = 3,
    tau_read: float = 0.35,
    tau_write: float | None = None,
    memory_capacity: int | None = None,
    on_overflow: str = "grow",
    on_chunk_done=None,
    rate_limiter=None,
) -> ThreadedCurationResult:
    """Runs C-DIC's retrieve / generate / write-back loop over one document,
    with the teacher chat model standing in for B at the generation step —
    the actual shape `rehearse_elaborative` uses at inference, not
    `curate_document`'s single linear running text.

    Why this exists
    ----------------
    `rehearse_elaborative` keeps a `ThreadMemory` of separate topic threads.
    Each chunk retrieves the threads related to it (Eq. 3), generates a new
    gist for *that chunk's thread* conditioned on them, and the result
    *replaces* the matched slot (Eq. 6) rather than being appended to a
    document-wide summary. `curate_document`'s linear loop only ever has one
    state to condition on — it never has to choose among several unrelated
    retrieved threads and decide which one, if any, this chunk continues.
    Training B on that shape and then asking it, at inference, to pick one of
    up to `top_k` retrieved slots and replace just that one is a second,
    shape-level version of the train/inference mismatch stage 2 exists to
    close in the first place (C-DIC Table 1) — this time in what the rolling
    task looks like, not merely in whether it is rolling. This function drives
    the identical `ThreadMemory` class `rehearse_elaborative` uses, so a pair
    built from its output is the literal shape of one inference step.

    ⚠ tau is uncalibrated here. `tau_read`/`tau_write` govern retrieval and
    write-back exactly as in `rehearse_elaborative`, but over the *teacher's*
    gist embeddings — llama-3.3-70b-instruct through llama-nemotron-embed, not
    the stage-1 t5-small's. Ablation A4's real-corpus sweep
    (`src/eval/tau_sweep.py`) was measured against the stage-1 checkpoint's
    embeddings, and for the same reason those numbers don't transfer from
    C-DIC App. K's latent-slot band, they don't transfer to this function's
    output either — different representation, different similarity scale.
    Treat 0.35 as a placeholder pending a sweep against this function's own
    embeddings, not a calibrated value.

    Cost: one query embedding, one teacher call, one store embedding — per
    chunk, the same shape as `rehearse_elaborative`. `on_overflow="grow"` by
    default, unlike `rehearse_elaborative`'s `"merge"`: curation is building
    the very targets stage 2 will learn from, so lossily merging them should
    be a deliberate, separately justified choice rather than a silent default.

    Sequential within a document — chunk i's retrieval depends on chunk
    i-1's write-back, so this cannot be batched or parallelised within a
    document. Documents are independent and can be.

    `on_chunk_done(position, total)`, if given, fires after every chunk
    (2026-08-04). Without it there is no signal at all between one document
    finishing and the next — three real network calls per chunk, all of them
    silent on success, so a document that is merely slow under NIM load and a
    document that has hung look identical from the outside: nothing prints
    until the whole document's `failed_positions` are known. A caller driving
    a `tqdm` bar per document (as `06b`'s §4 did) genuinely cannot tell "12
    chunks in, still working" apart from "stuck" without this.

    `rate_limiter`, if given, is shared across every concurrently-curated
    document (2026-08-04) — see `rate_limit.RateLimiter`. Document-level
    parallelism (multiple calls to this function at once, e.g. from a thread
    pool) is what makes a shared limiter necessary in the first place: each
    call's chat and embedding requests need to draw from one combined budget,
    not one budget per document, or NIM's actual per-key rate limit gets
    exceeded regardless of how conservative any single call looks alone.
    """
    memory = ThreadMemory(capacity=memory_capacity, on_overflow=on_overflow)
    write_threshold = tau_read if tau_write is None else tau_write

    def _embed_slot(text: str) -> list[float] | None:
        embedded = embed_with_retry([text], "passage", embed_cfg, embed_fn, rate_limiter=rate_limiter)
        return None if embedded is None else embedded[0]

    result = ThreadedCurationResult()

    for position, chunk_text in enumerate(chunk_texts):
        query_embedding = embed_with_retry([chunk_text], "query", embed_cfg, embed_fn, rate_limiter=rate_limiter)
        if query_embedding is None:
            result.retrieval_failed_positions.append(position)
            hits, best = [], None
        else:
            hits, best = memory.retrieve(
                query_embedding[0], position, top_k=top_k, tau=tau_read, fallback_top1=True,
            )

        context_texts = [hit.text for hit in hits]
        messages = build_teacher_messages_threaded(context_texts, chunk_text)
        completion = complete_with_retry(messages, config, chat_fn, rate_limiter=rate_limiter)

        if completion is None:
            result.failed_positions.append(position)
            # Carry the chunk itself forward so there is still something to
            # embed and write back — same "don't truncate the document"
            # contract as `curate_document`.
            gist = chunk_text
        else:
            gist = completion.strip() or chunk_text

        result.gists.append(gist)
        result.context_texts.append(context_texts)

        fresh_embedding = _embed_slot(gist)
        if fresh_embedding is None:
            result.store_failed_positions.append(position)
            result.write_back_actions.append("insert")  # never actually entered memory
        else:
            written = memory.write_back(
                gist, fresh_embedding, position, position,
                best=best, tau=write_threshold, embed_slot_fn=_embed_slot,
            )
            result.write_back_actions.append(written["action"])

        result.memory_states.append(" ".join(slot.text for slot in memory.slots))

        if on_chunk_done is not None:
            on_chunk_done(position, len(chunk_texts))

    return result


# --------------------------------------------------------------------------
# Testing effect (C) — retrieval practice folded into curation itself, not
# just measured after the fact.
# --------------------------------------------------------------------------
#
# `curate_document_threaded` above already gets probed post-hoc (06b §5, via
# `retention_probes`) to see whether its output holds up — but that check
# only ever *reports* collapse, it never acts on it. The function below is
# what `ProbeResult`'s own docstring in curation.py already called "the
# curation loop's §2-3 step" and this project's earlier design notes named
# "self-generated probes, a corrective retry, and selective augmentation
# against a verbatim fallback" (06's Summary) — specified before, not
# implemented until now.
#
# Psychology: Roediger & Karpicke (2006), "Test-Enhanced Learning: Taking
# Memory Tests Improves Long-Term Retention" (Psychological Science) — the
# testing effect is tied to the act of *retrieval*, not to who authored the
# question, so this reuses the project's own evidence-linked questions
# (already built for 06b's post-hoc check, `answers_by_chunk`) as the in-loop
# test material rather than synthesizing new ones with the QG model (`09`).
# Karpicke & Roediger (2008), "The Critical Importance of Retrieval for
# Learning" (Science), is the companion finding this leans on for the
# corrective-retry design specifically: *repeated* successful retrieval, not
# repeated exposure, is what predicts retention — so a probe that already
# passes is left alone (no retry spent re-testing something already fine),
# and a probe that fails gets another retrieval *attempt* (the corrective
# regeneration), not just another shortening pass.
#
# ML framing: mechanically this is a generate -> verify -> corrective-
# regenerate loop over natural-language feedback, the same shape as
# Self-Refine (Madaan et al., 2023, arXiv:2303.17651) and Reflexion (Shinn et
# al., 2023, arXiv:2303.11366). The "feedback" here is narrow by design —
# which answer span(s) are no longer recoverable, not a free-form critique —
# because the failure mode being corrected (a fact silently dropped during
# shortening) is narrow too.
@dataclass
class TestingCurationResult(ThreadedCurationResult):
    """Extends `ThreadedCurationResult` with the in-loop self-test outcome
    per position.

    `probe_passed[i]` — `None` if chunk i had no evidence-linked questions to
    test (most chunks); `True`/`False` otherwise, read *after* whatever
    retries happened (so `False` means retries were exhausted, not that the
    first attempt failed).
    `retry_counts[i]` — how many corrective regenerations position i needed
    (0 if it passed first try or had nothing to test).
    `fell_back_to_verbatim[i]` — `True` if retries were exhausted and the raw
    chunk text was written back instead of any teacher-generated gist.
    """

    probe_passed: list[bool | None] = field(default_factory=list)
    retry_counts: list[int] = field(default_factory=list)
    fell_back_to_verbatim: list[bool] = field(default_factory=list)


def curate_document_threaded_with_testing(
    chunk_texts: list[str],
    answers_by_chunk: dict[int, list[str]],
    embed_cfg: dict,
    config: dict,
    embed_fn=embed_texts,
    chat_fn=chat_completion,
    top_k: int = 3,
    tau_read: float = 0.35,
    tau_write: float | None = None,
    memory_capacity: int | None = None,
    on_overflow: str = "grow",
    max_retries: int = 2,
    probe_f1_threshold: float = 0.6,
    on_chunk_done=None,
    rate_limiter=None,
) -> TestingCurationResult:
    """`curate_document_threaded`, with a self-test after every write-back.

    Same retrieve/generate/write-back loop, same cost per chunk with no
    linked questions (one query embedding, one teacher call, one store
    embedding — nothing extra when there is nothing to test, which is most
    chunks: `answers_by_chunk` only has entries for chunks whose evidence
    backs a real question, same sparsity `retention_probes` already works
    with). When a chunk *does* have linked answers, after write-back this
    checks whether the memory state as it now stands still supports each one
    (`answer_supported`, the identical check `retention_probes` uses
    post-hoc) — up to `max_retries` corrective regenerations
    (`build_teacher_messages_threaded_corrective`), each one told exactly
    which answer span(s) are missing, not a generic "try again". Exhausting
    retries falls back to the raw chunk text for that slot — a fact that
    cannot be preserved in shortened form is better kept whole than lost.

    The corrective retry directly mutates the slot `write_back` already
    wrote to (via the `"slot"` index its return dict carries), rather than
    calling `write_back` again with the same `best` — a second `write_back`
    call would be safe on the "revise" branch (same slot again) but would
    silently open a *second* new slot on the "insert" branch, leaving the
    failed first attempt behind as an orphaned duplicate.

    Cost, when a chunk has linked answers: +1 corrective teacher call per
    retry actually needed (0 to `max_retries`), no extra embedding calls —
    the corrective regeneration overwrites the same slot's embedding, which
    still needs computing once per accepted gist, exactly as before.
    """
    memory = ThreadMemory(capacity=memory_capacity, on_overflow=on_overflow)
    write_threshold = tau_read if tau_write is None else tau_write

    def _embed_slot(text: str) -> list[float] | None:
        embedded = embed_with_retry([text], "passage", embed_cfg, embed_fn, rate_limiter=rate_limiter)
        return None if embedded is None else embedded[0]

    def _memory_state() -> str:
        return " ".join(slot.text for slot in memory.slots)

    result = TestingCurationResult()

    for position, chunk_text in enumerate(chunk_texts):
        query_embedding = embed_with_retry([chunk_text], "query", embed_cfg, embed_fn, rate_limiter=rate_limiter)
        if query_embedding is None:
            result.retrieval_failed_positions.append(position)
            hits, best = [], None
        else:
            hits, best = memory.retrieve(
                query_embedding[0], position, top_k=top_k, tau=tau_read, fallback_top1=True,
            )

        context_texts = [hit.text for hit in hits]
        messages = build_teacher_messages_threaded(context_texts, chunk_text)
        completion = complete_with_retry(messages, config, chat_fn, rate_limiter=rate_limiter)

        if completion is None:
            result.failed_positions.append(position)
            gist = chunk_text
        else:
            gist = completion.strip() or chunk_text

        result.context_texts.append(context_texts)

        fresh_embedding = _embed_slot(gist)
        if fresh_embedding is None:
            result.store_failed_positions.append(position)
            result.write_back_actions.append("insert")  # never actually entered memory
            written_slot = None
        else:
            written = memory.write_back(
                gist, fresh_embedding, position, position,
                best=best, tau=write_threshold, embed_slot_fn=_embed_slot,
            )
            result.write_back_actions.append(written["action"])
            written_slot = written["slot"]

        # -- self-test: does the memory, as it now actually stands, still
        # support every answer whose evidence is in this chunk? --
        answers = answers_by_chunk.get(position, [])
        probe_passed: bool | None = None
        retries = 0
        fell_back = False

        if answers and written_slot is not None and completion is not None:
            failed = [a for a in answers if not answer_supported(a, _memory_state(), probe_f1_threshold)]
            while failed and retries < max_retries:
                corrective_messages = build_teacher_messages_threaded_corrective(
                    context_texts, chunk_text, gist, failed,
                )
                corrective_completion = complete_with_retry(
                    corrective_messages, config, chat_fn, rate_limiter=rate_limiter,
                )
                retries += 1
                if corrective_completion is None:
                    break
                gist = corrective_completion.strip() or gist
                corrective_embedding = _embed_slot(gist)
                if corrective_embedding is None:
                    break
                slot = memory.slots[written_slot]
                slot.text, slot.embedding = gist, corrective_embedding
                failed = [a for a in answers if not answer_supported(a, _memory_state(), probe_f1_threshold)]

            if failed:
                fell_back = True
                memory.slots[written_slot].text = chunk_text
                gist = chunk_text
                # Embedding left as the last attempt's — the *text* is what
                # downstream reads (`memory_states`), and re-embedding a
                # slot that is about to be superseded by the next retrieval
                # round anyway is a cost this project already treats as
                # avoidable (see `rehearsal.py`'s analogous fallback path).
            probe_passed = not failed

        result.gists.append(gist)
        result.memory_states.append(_memory_state())
        result.probe_passed.append(probe_passed)
        result.retry_counts.append(retries)
        result.fell_back_to_verbatim.append(fell_back)

        if on_chunk_done is not None:
            on_chunk_done(position, len(chunk_texts))

    return result
