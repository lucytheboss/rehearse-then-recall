"""No-merge thread clustering.

`ThreadMemory` (thread_memory.py) collapses because its "revise" branch
replaces a slot's text with a model's regeneration of "retrieved thread +
current chunk" -- and that regeneration reproduces the retrieved thread
near-verbatim, ignoring the current chunk (§4.3.3, ANALYSIS_REPORT.md).
The regeneration is what collapses. The *decision* of which thread a chunk
belongs to (C-DIC Eq. 4-5, pure cosine similarity) is not -- it never calls a
model at all.

This module keeps the decision and discards the regeneration: chunks are
grouped into threads by embedding similarity, but each thread's content is
just its members' independently-produced texts concatenated, never merged or
rewritten. Content is expected to come from something that already can't
collapse on its own -- `rehearse_elaborative_independent` (B, run with no
retrieved context, so there is nothing to collapse toward) or maintenance
rehearsal's verbatim extraction (A) -- so grouping adds cross-chunk structure
without reintroducing the failure mode that made grouping dangerous in the
first place.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.pipeline.embeddings import cosine_similarity


@dataclass
class GistThread:
    """A group of chunks whose content is topically related, per embedding
    similarity, with member texts kept side by side rather than merged.

    `anchor_embedding` is fixed at thread creation (the embedding of whichever
    chunk opened it) and never updated as members are added. A running
    average would let a thread's notion of "on topic" drift with every
    addition, so chunk 40 being grouped with chunk 0 would depend on
    everything added in between rather than on chunk 0 itself -- fixing the
    anchor keeps "belongs in this thread" meaning the same thing throughout.
    """

    member_indices: list[int] = field(default_factory=list)
    member_texts: list[str] = field(default_factory=list)
    anchor_embedding: list[float] = field(default_factory=list)

    @property
    def text(self) -> str:
        """All member texts, concatenated in the order they were added.
        Never merged, rewritten, or summarized -- this is the whole point."""
        return "\n".join(self.member_texts)


def cluster_into_threads(
    chunk_texts: list[str],
    chunk_embeddings: list[list[float]],
    tau: float = 0.35,
) -> list[GistThread]:
    """Groups chunks into threads by embedding similarity, in source order.

    For each chunk: join the most similar existing thread if that similarity
    is >= `tau` (C-DIC's "revise" condition, Eq. 4-5), else open a new thread
    (C-DIC's "insert"). Similarity is always measured against a thread's
    `anchor_embedding`, not its most recent member.

    No model call happens here -- this is pure embedding arithmetic. Pass
    whichever embeddings already exist for chunk selection elsewhere in the
    pipeline (e.g. `chunk_embeddings_by_genre`) rather than re-embedding the
    generated content; clustering on raw-chunk similarity keeps this
    consistent with how every other condition in this project selects chunks,
    and avoids extra embedding calls.

    `chunk_texts[i]` should already be independently produced (context-free
    B, or extractive A) -- this function only decides grouping, and never
    inspects or alters the texts themselves.
    """
    threads: list[GistThread] = []
    for chunk_index, (text, embedding) in enumerate(zip(chunk_texts, chunk_embeddings)):
        best_index, best_score = None, -1.0
        for i, thread in enumerate(threads):
            score = cosine_similarity(embedding, thread.anchor_embedding)
            if score > best_score:
                best_index, best_score = i, score

        if best_index is not None and best_score >= tau:
            chosen = threads[best_index]
        else:
            chosen = GistThread(anchor_embedding=embedding)
            threads.append(chosen)

        chosen.member_indices.append(chunk_index)
        chosen.member_texts.append(text)

    return threads
