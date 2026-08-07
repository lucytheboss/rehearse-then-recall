"""Bounded, revisable thread memory for elaborative rehearsal (B).

This is C-DIC's memory M (Jung, Kim, Jung, Wang, Zhang, Cheung, See & Chen,
ICML 2026, arXiv:2606.12411) transposed from latent slots to text slots.

What this module fixes
----------------------
`rehearse_elaborative` previously kept its history in three parallel lists
(`pool_texts` / `pool_indices` / `pool_embeddings`) and **appended to them
unconditionally**. It computed `write_back_action` from the retrieval score
and then ignored it. So Eq. 6 was being *reported* but never *executed*:

  - the revise branch `M_{<t+1} = (M_{<t} \\ {Z_j}) u {Z_t}` never ran, so a
    thread's stale early gist stayed in the pool competing for retrieval
    against its own updated version;
  - memory grew one slot per chunk, i.e. O(N) — the exact cost C-DIC's
    single compact memory exists to avoid.

`ThreadMemory` executes the branch.

Where it departs from the paper
-------------------------------
1. **Slots are text, not latents.** C-DIC's `Z_i` are n=128 learned
   compression-token embeddings produced by an ICAE-initialised compressor.
   Ours are the rehearsed gist strings themselves. This is forced by the
   pipeline being a text-level intervention, but it also answers the paper's
   own Limitation 2 ("relies on pretrained latent-compression components, so
   results may vary with different compressor initialisations") — a text slot
   has no compressor state to be sensitive to.

2. **Capacity is bounded.** The paper's Limitation 1 is that "the total number
   of stored memory slots can grow when dialogue repeatedly shifts topics",
   and its proposed remedy — bounded *retrieval* — caps what generation reads,
   not what storage holds. `capacity` + `on_overflow` caps storage too. See
   `_make_room`.

3. **Provenance is tracked and deletable.** The paper's Limitation 3 is that
   "persistent latent memories can retain sensitive information ... latent
   compression should therefore not be viewed as a privacy or deletion
   mechanism." A latent slot cannot say which turns went into it. A text slot
   can: `MemorySlot.origin_indices` records every chunk that has written to
   the slot, `provenance()` exposes the whole map, and `forget()` removes a
   source chunk's contribution. That is a real consequence of choosing text
   over latents, not a cosmetic one.

4. **Recency decay is off by default, not deleted.** Eq. 3 multiplies
   similarity by `exp(-alpha * dt)`. Dialogue turns arrive in time and old
   threads go stale; a document's chunk 0 is not staler than its chunk 40,
   the whole text is present at once. Keeping the decay would reintroduce
   precisely the recency bias thread-aware retrieval exists to remove — so
   `recency_alpha` defaults to 0.0.

   It stays a *parameter* rather than a deletion because the paper's evidence
   does not obviously favour us. App. K Table 17 sweeps alpha in {0, 0.05, 0.1}
   and finds alpha = 0.1 gives both the *highest* LongMemEval accuracy (0.120
   vs 0.098 at alpha = 0) and the smallest memory (5.0 vs 23.9 slots) — decay
   is doing real work there. Our argument that documents are not
   time-ordered is a claim about our setting, not a refutation of theirs, so it
   has to be run as an ablation instead of asserted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

from src.pipeline.embeddings import cosine_similarity
from src.pipeline.gisting import split_into_sentences

# Overflow policies for a memory at capacity.
#   "merge" — fold the two most similar slots into one, keeping both
#             provenances. Lossy in wording, lossless in traceability.
#   "evict" — drop the least useful slot outright (least recently retrieved,
#             tie-broken by fewest revisions). Lossy in both.
#   "grow"  — ignore the capacity. This is the paper's behaviour, kept so the
#             unbounded baseline is runnable as an ablation rather than only
#             describable.
OVERFLOW_POLICIES = ("merge", "evict", "grow")


@dataclass
class MemorySlot:
    """One contextual thread.

    `origin_indices` is the slot's provenance in document order: the chunk that
    opened the thread, then every chunk that has since revised or been merged
    into it. Two indices matter downstream and they are *not* the same number:

      - `anchor_index` (first) — how far back the thread itself reaches. This
        is what "chunk 40 is still elaborating chunk 0's thread" means.
      - `head_index` (last) — which chunk the slot's current *text* actually
        came from. Retrieval span must be measured against this one, because
        after a revision the wording is the reviser's, not the anchor's.

    Reporting span against the anchor would inflate it every time a thread is
    revised, which would make the mechanism look like it reaches further the
    more it works. `retrieval_span_report` therefore tracks both separately.
    """

    text: str
    embedding: list[float]
    origin_indices: list[int] = field(default_factory=list)
    created_at: int = 0
    # Loop position of the last retrieval *or* revision. Drives both the Eq. 3
    # decay term and the eviction ordering, so a slot that keeps getting read
    # stays alive even if it is never rewritten.
    last_touched_at: int = 0
    revisions: int = 0
    merges: int = 0

    @property
    def anchor_index(self) -> int:
        return self.origin_indices[0] if self.origin_indices else -1

    @property
    def head_index(self) -> int:
        return self.origin_indices[-1] if self.origin_indices else -1


@dataclass
class Hit:
    """One retrieved slot, as handed to the generator."""

    slot: int  # position in ThreadMemory.slots — valid only until the next write
    text: str
    score: float
    anchor_index: int
    head_index: int


class ThreadMemory:
    """The retrieve / revise / write-back loop of C-DIC §3, over text slots.

    Usage is two calls per chunk, sharing one score:

        hits, best = memory.retrieve(query_embedding, position, tau=tau_read)
        ...generate using [h.text for h in hits]...
        memory.write_back(gist, gist_embedding, chunk.index, position,
                          best=best, tau=tau_write, embed_slot_fn=...)

    `best` is `(slot_index, delta_t)` where `delta_t = max_i S(q_t, Z_i)` —
    Eq. 6's branch variable. Passing it through means write-back reuses the
    retrieval score exactly as the paper does, with no second similarity pass.
    """

    def __init__(
        self,
        capacity: int | None = None,
        on_overflow: str = "merge",
        recency_alpha: float = 0.0,
        max_slot_sentences: int = 4,
    ):
        if on_overflow not in OVERFLOW_POLICIES:
            raise ValueError(f"on_overflow must be one of {OVERFLOW_POLICIES}: {on_overflow!r}")
        if capacity is not None and capacity < 1:
            raise ValueError(f"capacity must be >= 1 or None: {capacity!r}")

        self.capacity = capacity
        self.on_overflow = on_overflow
        self.recency_alpha = recency_alpha
        self.max_slot_sentences = max_slot_sentences
        self.slots: list[MemorySlot] = []
        # Every merge/eviction, so a run can report what storage pressure cost
        # it. An overflow event is a deliberate information loss and should
        # never be silent.
        self.overflow_events: list[dict] = []

    def __len__(self) -> int:
        return len(self.slots)

    def adopt(self, slot: MemorySlot) -> None:
        """Appends a pre-built slot with no capacity check.

        For reconstructing a memory from an existing pool (see
        `retrieve_aligned_context`), not for the write path — `write_back` is
        the only entry point that honours Eq. 6 and the capacity bound.
        """
        self.slots.append(slot)

    # -- retrieval (Eq. 3) --------------------------------------------------

    def score_all(self, query_embedding: list[float], position: int) -> list[tuple[int, float]]:
        """`S(q_t, Z_i)` for every slot, highest first.

        The decay factor `exp(-alpha * dt)` is applied only when
        `recency_alpha` is nonzero, so the default path is pure cosine and
        costs nothing.
        """
        scored: list[tuple[int, float]] = []
        for i, slot in enumerate(self.slots):
            score = cosine_similarity(query_embedding, slot.embedding)
            if self.recency_alpha:
                score *= math.exp(-self.recency_alpha * (position - slot.last_touched_at))
            scored.append((i, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored

    def retrieve(
        self,
        query_embedding: list[float],
        position: int,
        top_k: int = 3,
        tau: float = 0.35,
        fallback_top1: bool = True,
        touch: bool = True,
    ) -> tuple[list[Hit], tuple[int, float] | None]:
        """Slots aligned with the current chunk, plus Eq. 6's branch variable.

        Returns `(hits, best)`. `best` is `(slot_index, delta_t)` over the
        *unfiltered* ranking, or None on an empty memory — write-back needs the
        maximum whether or not it cleared `tau`.

        `tau` is a relevance floor, not a formality: with top-k alone a chunk
        that genuinely relates to nothing earlier still drags in the three
        least-unrelated slots, and unrelated context in the prompt is what
        makes a small seq2seq invent connections.

        `fallback_top1` is Eq. 3's other half — below `tau`, condition on the
        single nearest slot rather than on nothing. Note what this separates:
        *generation* falls back, while *write-back* still branches at `tau` and
        inserts. A topic-shift chunk is elaborated against its nearest
        neighbour but still recorded as opening a new thread. Read a hit's
        `score` to tell a real match from a fallback.

        `touch` updates `last_touched_at` on the returned slots, which is what
        makes "turns since last retrieval" mean what Eq. 3 says it means. Pass
        False for a read-only probe.
        """
        scored = self.score_all(query_embedding, position)
        if not scored:
            return [], None

        best = scored[0]
        cleared = [item for item in scored if item[1] >= tau][:top_k]
        chosen = cleared if (cleared or not fallback_top1) else scored[:1]

        hits = [
            Hit(
                slot=index,
                text=self.slots[index].text,
                score=score,
                anchor_index=self.slots[index].anchor_index,
                head_index=self.slots[index].head_index,
            )
            for index, score in chosen
        ]
        if touch:
            for hit in hits:
                self.slots[hit.slot].last_touched_at = position
        return hits, best

    # -- write-back (Eq. 6) -------------------------------------------------

    def write_back(
        self,
        text: str,
        embedding: list[float],
        chunk_index: int,
        position: int,
        best: tuple[int, float] | None,
        tau: float = 0.35,
        embed_slot_fn: Callable[[str], list[float] | None] | None = None,
    ) -> dict:
        """Eq. 6 / Algorithm 1, executed rather than merely labelled.

            delta_t < tau   -> insert: open a new slot (topic shift)
            delta_t >= tau  -> revise: replace the matched slot (on-topic)

        The revise branch replaces the slot's text outright, which is the
        faithful transposition of `M \\ {Z_j} u {Z_t}`: `text` was generated
        while conditioned on the retrieved slots, so it *is* the updated thread
        state — no extra model call is needed to produce one.

        `tau` here is `tau_write` and need not equal the `tau` passed to
        `retrieve`. The paper shares one threshold because retrieval and
        write-back share one score, but the two decisions have asymmetric
        costs: reading a marginal slot adds slightly noisy context to one
        prompt, while overwriting on a marginal match destroys a thread that
        was the only record of its content. `tau_write >= tau_read` buys
        caution on the destructive side for free.

        Returns a record of what happened, including any overflow event, for
        `RetrievalRecord` to carry into the report.
        """
        if best is not None and best[1] >= tau and best[0] < len(self.slots):
            index = best[0]
            slot = self.slots[index]
            slot.text = text
            slot.embedding = embedding
            slot.origin_indices.append(chunk_index)
            slot.last_touched_at = position
            slot.revisions += 1
            return {
                "action": "revise",
                "slot": index,
                "origin_indices": list(slot.origin_indices),
                "overflow": None,
            }

        overflow = self._make_room(embed_slot_fn, position)
        self.slots.append(
            MemorySlot(
                text=text,
                embedding=embedding,
                origin_indices=[chunk_index],
                created_at=position,
                last_touched_at=position,
            )
        )
        return {
            "action": "insert",
            "slot": len(self.slots) - 1,
            "origin_indices": [chunk_index],
            "overflow": overflow,
        }

    # -- capacity (paper Limitation 1) --------------------------------------

    def _make_room(self, embed_slot_fn, position: int) -> dict | None:
        """Frees a slot if inserting would exceed `capacity`.

        The paper's Limitation 1 concedes unbounded slot growth under repeated
        topic shifts and offers bounded *retrieval* as the mitigation — but
        that caps what generation reads, not what is stored, so the memory
        still grows without limit and every retrieval pays a longer scan.
        Capping storage forces the question the paper defers: when memory is
        full, what is lost?

        "merge" answers it least destructively. Folding the two most similar
        slots keeps both provenances, so nothing becomes untraceable, and the
        two most similar slots are by construction the pair whose separation
        carries the least information. "evict" is the honest crude baseline.
        """
        if self.capacity is None or self.on_overflow == "grow":
            return None
        if len(self.slots) < self.capacity:
            return None

        if self.on_overflow == "merge":
            event = self._merge_closest_pair(embed_slot_fn, position)
            if event is not None:
                self.overflow_events.append(event)
                return event
            # Merging needs an embedding for the merged text. If that call
            # failed, degrade to eviction rather than either aborting the run
            # or silently exceeding the bound — same philosophy as
            # `_snap_to_verbatim` returning unsnapped text on an API outage.

        event = self._evict_least_useful(position)
        self.overflow_events.append(event)
        return event

    def _trim_slot_text(self, text: str) -> str:
        """Caps a merged slot at `max_slot_sentences` sentences.

        Keeps the *last* n. Merge concatenates older-then-newer, and for a
        running thread summary the later sentences supersede the earlier ones —
        so trimming from the front drops the material most likely to already be
        restated. Without this cap, repeated merges would grow a slot without
        bound and reintroduce the O(N) memory the capacity exists to prevent.
        """
        sentences = [s for s in split_into_sentences(text) if s.strip()]
        if len(sentences) <= self.max_slot_sentences:
            return text
        return " ".join(sentences[-self.max_slot_sentences :])

    def _merge_closest_pair(self, embed_slot_fn, position: int) -> dict | None:
        """Folds the two most mutually similar slots into one.

        Similarity here is slot-to-slot and deliberately *undecayed* — the
        decay in Eq. 3 models how stale a slot is relative to the current
        query, which says nothing about whether two slots are redundant with
        each other.

        Returns None when the merge cannot be completed (fewer than two slots,
        or no/failed embedding function), leaving the caller to fall back.
        """
        if embed_slot_fn is None or len(self.slots) < 2:
            return None

        best: tuple[float, int, int] | None = None
        for i in range(len(self.slots)):
            for j in range(i + 1, len(self.slots)):
                score = cosine_similarity(self.slots[i].embedding, self.slots[j].embedding)
                if best is None or score > best[0]:
                    best = (score, i, j)

        score, i, j = best
        first, second = self.slots[i], self.slots[j]
        older, newer = (first, second) if first.created_at <= second.created_at else (second, first)

        merged_text = self._trim_slot_text(f"{older.text} {newer.text}".strip())
        merged_embedding = embed_slot_fn(merged_text)
        if merged_embedding is None:
            return None

        merged = MemorySlot(
            text=merged_text,
            embedding=merged_embedding,
            origin_indices=sorted(set(older.origin_indices) | set(newer.origin_indices)),
            created_at=min(older.created_at, newer.created_at),
            last_touched_at=max(older.last_touched_at, newer.last_touched_at),
            revisions=older.revisions + newer.revisions,
            merges=older.merges + newer.merges + 1,
        )
        for index in sorted((i, j), reverse=True):
            self.slots.pop(index)
        self.slots.append(merged)

        return {
            "kind": "merge",
            "at": position,
            "similarity": round(score, 4),
            "origin_indices": list(merged.origin_indices),
        }

    def _evict_least_useful(self, position: int) -> dict:
        """Drops the least recently retrieved slot, tie-broken by fewest
        revisions — i.e. the thread nothing has come back to and nothing has
        updated."""
        victim = min(
            range(len(self.slots)),
            key=lambda i: (self.slots[i].last_touched_at, self.slots[i].revisions),
        )
        dropped = self.slots.pop(victim)
        return {
            "kind": "evict",
            "at": position,
            "similarity": None,
            "origin_indices": list(dropped.origin_indices),
        }

    # -- audit (paper Limitation 3) -----------------------------------------

    def provenance(self) -> dict[int, list[int]]:
        """`{slot position: contributing chunk indices}`.

        The paper states latent memory "should not be viewed as a privacy or
        deletion mechanism" because a latent slot cannot report what went into
        it. A text slot can, and this is that report — the basis for both
        `forget` and for auditing which source material a given answer could
        possibly have drawn on.
        """
        return {i: list(slot.origin_indices) for i, slot in enumerate(self.slots)}

    def forget(self, chunk_index: int) -> int:
        """Removes every slot a given source chunk contributed to; returns how
        many were removed.

        Coarse on purpose. Once a chunk has been folded into a thread its
        wording is no longer separable from the rest of the slot, so removing
        the *slot* is the only removal that can actually be guaranteed —
        anything finer would claim a deletion it cannot deliver, which is the
        failure mode the paper's limitation is warning about.
        """
        before = len(self.slots)
        self.slots = [slot for slot in self.slots if chunk_index not in slot.origin_indices]
        return before - len(self.slots)

    def stats(self) -> dict:
        return {
            "slots": len(self.slots),
            "capacity": self.capacity,
            "on_overflow": self.on_overflow,
            "merges": sum(e["kind"] == "merge" for e in self.overflow_events),
            "evictions": sum(e["kind"] == "evict" for e in self.overflow_events),
            "total_revisions": sum(slot.revisions for slot in self.slots),
            "mean_slot_origins": (
                sum(len(slot.origin_indices) for slot in self.slots) / len(self.slots)
                if self.slots
                else 0.0
            ),
        }
