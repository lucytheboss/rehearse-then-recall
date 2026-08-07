"""Unit tests for ThreadMemory — C-DIC Eq. 6 write-back, the capacity bound
that answers the paper's Limitation 1, and the provenance that answers its
Limitation 3. No model, API key, or network required.
"""

import pytest

from src.pipeline.thread_memory import MemorySlot, ThreadMemory


def _onehot(index: int, dim: int = 8) -> list[float]:
    vector = [0.0] * dim
    vector[index] = 1.0
    return vector


def _memory_with_topics(topics, **kwargs) -> ThreadMemory:
    """A memory holding one slot per topic id, opened at consecutive positions.

    One-hot embeddings make similarity exactly topic identity, so every
    retrieval and write-back decision below is deterministic.
    """
    memory = ThreadMemory(**kwargs)
    for position, topic in enumerate(topics):
        memory.adopt(
            MemorySlot(
                text=f"topic{topic} slot",
                embedding=_onehot(topic),
                origin_indices=[position],
                created_at=position,
                last_touched_at=position,
            )
        )
    return memory


# -- Eq. 6: the branch is executed, not merely labelled ---------------------


def test_revise_replaces_the_matched_slot_instead_of_appending():
    """The bug this class exists to fix: the old loop appended unconditionally,
    so the revise arm of Eq. 6 never ran and memory grew one slot per chunk."""
    memory = _memory_with_topics([0, 1, 2])
    _, best = memory.retrieve(_onehot(0), position=3, tau=0.35)

    written = memory.write_back(
        "topic0 updated", _onehot(0), chunk_index=3, position=3, best=best, tau=0.35
    )

    assert written["action"] == "revise"
    assert len(memory) == 3, "revise must not grow memory"
    assert memory.slots[0].text == "topic0 updated"
    assert memory.slots[0].revisions == 1


def test_insert_opens_a_new_slot_on_a_topic_shift():
    memory = _memory_with_topics([0, 1])
    _, best = memory.retrieve(_onehot(5), position=2, tau=0.35)

    written = memory.write_back(
        "topic5 new", _onehot(5), chunk_index=2, position=2, best=best, tau=0.35
    )

    assert written["action"] == "insert"
    assert len(memory) == 3


def test_revise_accumulates_provenance_in_document_order():
    memory = _memory_with_topics([0])
    for chunk_index in (4, 9):
        _, best = memory.retrieve(_onehot(0), position=chunk_index, tau=0.35)
        memory.write_back(
            f"topic0 v{chunk_index}", _onehot(0), chunk_index, chunk_index, best=best, tau=0.35
        )

    slot = memory.slots[0]
    assert slot.origin_indices == [0, 4, 9]
    assert slot.anchor_index == 0, "the thread still starts at chunk 0"
    assert slot.head_index == 9, "but its current text came from chunk 9"


def test_tau_write_can_be_stricter_than_tau_read():
    """Read generously, overwrite only on a confident match. A hit good enough
    to condition on must not automatically be good enough to destroy a slot."""
    memory = ThreadMemory()
    memory.adopt(MemorySlot(text="prior", embedding=[1.0, 0.0], origin_indices=[0]))

    hits, best = memory.retrieve([0.8, 0.6], position=1, tau=0.5)
    assert hits, "the slot cleared tau_read"

    written = memory.write_back(
        "new", [0.8, 0.6], chunk_index=1, position=1, best=best, tau=0.95
    )
    assert written["action"] == "insert"
    assert len(memory) == 2


# -- Eq. 3: retrieval ------------------------------------------------------


def test_retrieval_reaches_past_the_most_recent_slot():
    memory = _memory_with_topics([0, 1, 2, 3])
    hits, _ = memory.retrieve(_onehot(0), position=4, tau=0.35)
    assert [hit.head_index for hit in hits] == [0]


def test_retrieval_respects_the_relevance_floor():
    memory = _memory_with_topics([0, 1])
    hits, best = memory.retrieve(_onehot(5), position=2, tau=0.35, fallback_top1=False)
    assert hits == []
    assert best is not None, "write-back still needs delta_t even when nothing cleared tau"


def test_fallback_top1_returns_a_below_tau_neighbour():
    memory = _memory_with_topics([0, 1])
    hits, _ = memory.retrieve(_onehot(5), position=2, tau=0.35, fallback_top1=True)
    assert len(hits) == 1
    assert hits[0].score < 0.35


def test_recency_decay_is_off_by_default_and_reorders_when_enabled():
    """A document's chunk 0 is not staler than its chunk 40, so alpha=0 is the
    default — but the claim has to be runnable as an ablation, not asserted."""
    plain = _memory_with_topics([0, 1])
    plain.slots[0].embedding = [1.0, 0.0]
    plain.slots[1].embedding = [0.95, 0.31]
    plain.slots[0].last_touched_at = 0
    plain.slots[1].last_touched_at = 40

    assert plain.score_all([1.0, 0.0], position=41)[0][0] == 0, "no decay: the older slot wins"

    decayed = _memory_with_topics([0, 1], recency_alpha=0.1)
    decayed.slots[0].embedding = [1.0, 0.0]
    decayed.slots[1].embedding = [0.95, 0.31]
    decayed.slots[0].last_touched_at = 0
    decayed.slots[1].last_touched_at = 40

    assert decayed.score_all([1.0, 0.0], position=41)[0][0] == 1, "decay: the recent slot wins"


def test_retrieval_touches_slots_so_decay_measures_time_since_last_read():
    memory = _memory_with_topics([0, 1])
    memory.retrieve(_onehot(0), position=7, tau=0.35)
    assert memory.slots[0].last_touched_at == 7
    assert memory.slots[1].last_touched_at == 1, "an unread slot must keep ageing"


def test_touch_false_leaves_the_memory_unmodified():
    memory = _memory_with_topics([0, 1])
    memory.retrieve(_onehot(0), position=7, tau=0.35, touch=False)
    assert memory.slots[0].last_touched_at == 0


# -- capacity: the paper's Limitation 1 ------------------------------------


def test_grow_policy_reproduces_the_papers_unbounded_memory():
    """The baseline has to be runnable, not just describable."""
    memory = ThreadMemory(capacity=2, on_overflow="grow")
    for i in range(5):
        _, best = memory.retrieve(_onehot(i), position=i, tau=0.35, fallback_top1=False)
        memory.write_back(f"topic{i}", _onehot(i), i, i, best=best, tau=0.35)
    assert len(memory) == 5


def test_capacity_holds_under_repeated_topic_shifts():
    """The paper concedes slots 'can grow when dialogue repeatedly shifts
    topics'. Every chunk here shifts topic, which is the worst case."""
    memory = ThreadMemory(capacity=3, on_overflow="evict")
    for i in range(8):
        _, best = memory.retrieve(_onehot(i % 8), position=i, tau=0.35, fallback_top1=False)
        memory.write_back(f"topic{i}", _onehot(i % 8), i, i, best=best, tau=0.35)
    assert len(memory) == 3
    assert memory.stats()["evictions"] == 5


def test_eviction_drops_the_least_recently_retrieved_slot():
    memory = _memory_with_topics([0, 1, 2], capacity=3, on_overflow="evict")
    memory.retrieve(_onehot(0), position=3, tau=0.35)  # slot 0 stays useful

    _, best = memory.retrieve(_onehot(6), position=4, tau=0.35, fallback_top1=False)
    memory.write_back("topic6", _onehot(6), 4, 4, best=best, tau=0.35)

    kept = {slot.anchor_index for slot in memory.slots}
    assert 1 not in kept, "the least recently touched slot should have gone"
    assert 0 in kept, "a slot that was just read must survive"


def test_merge_folds_the_closest_pair_and_keeps_both_provenances():
    """Merge is the least destructive overflow answer: the two most similar
    slots are the pair whose separation carries the least information, and
    nothing becomes untraceable."""
    memory = ThreadMemory(capacity=3, on_overflow="merge")
    memory.adopt(MemorySlot(text="alpha.", embedding=[1.0, 0.0], origin_indices=[0], created_at=0))
    memory.adopt(MemorySlot(text="beta.", embedding=[0.99, 0.14], origin_indices=[1], created_at=1))
    memory.adopt(MemorySlot(text="gamma.", embedding=[0.0, 1.0], origin_indices=[2], created_at=2))

    _, best = memory.retrieve([0.0, -1.0], position=3, tau=0.35, fallback_top1=False)
    written = memory.write_back(
        "delta.", [0.0, -1.0], chunk_index=3, position=3, best=best, tau=0.35,
        embed_slot_fn=lambda text: [0.5, 0.5],
    )

    assert len(memory) == 3
    assert written["overflow"]["kind"] == "merge"
    assert written["overflow"]["origin_indices"] == [0, 1]

    merged = [slot for slot in memory.slots if slot.merges][0]
    assert merged.origin_indices == [0, 1]
    assert "alpha." in merged.text and "beta." in merged.text


def test_merge_falls_back_to_eviction_when_the_embedding_fails():
    """An embedding outage must not silently break the capacity bound — same
    'don't kill the pipeline' philosophy as _snap_to_verbatim."""
    memory = ThreadMemory(capacity=2, on_overflow="merge")
    memory.adopt(MemorySlot(text="alpha", embedding=[1.0, 0.0], origin_indices=[0]))
    memory.adopt(MemorySlot(text="beta", embedding=[0.9, 0.1], origin_indices=[1]))

    _, best = memory.retrieve([0.0, 1.0], position=2, tau=0.35, fallback_top1=False)
    written = memory.write_back(
        "gamma", [0.0, 1.0], chunk_index=2, position=2, best=best, tau=0.35,
        embed_slot_fn=lambda text: None,
    )

    assert len(memory) == 2
    assert written["overflow"]["kind"] == "evict"


def test_merged_slot_text_is_capped_so_merges_cannot_grow_unbounded():
    memory = ThreadMemory(capacity=2, on_overflow="merge", max_slot_sentences=2)
    memory.adopt(MemorySlot(text="One. Two. Three.", embedding=[1.0, 0.0], origin_indices=[0]))
    memory.adopt(MemorySlot(text="Four. Five.", embedding=[0.99, 0.1], origin_indices=[1], created_at=1))

    _, best = memory.retrieve([0.0, 1.0], position=2, tau=0.35, fallback_top1=False)
    memory.write_back(
        "Six.", [0.0, 1.0], chunk_index=2, position=2, best=best, tau=0.35,
        embed_slot_fn=lambda text: [0.5, 0.5],
    )

    merged = [slot for slot in memory.slots if slot.merges][0]
    # Keeps the *last* two — for a running thread summary the later sentences
    # supersede the earlier ones.
    assert merged.text == "Four. Five."


def test_overflow_events_are_recorded_rather_than_silent():
    memory = ThreadMemory(capacity=1, on_overflow="evict")
    for i in range(3):
        _, best = memory.retrieve(_onehot(i), position=i, tau=0.35, fallback_top1=False)
        memory.write_back(f"topic{i}", _onehot(i), i, i, best=best, tau=0.35)
    assert len(memory.overflow_events) == 2
    assert all(event["kind"] == "evict" for event in memory.overflow_events)


def test_rejects_an_unknown_overflow_policy():
    with pytest.raises(ValueError):
        ThreadMemory(capacity=2, on_overflow="discard-everything")


# -- provenance: the paper's Limitation 3 ----------------------------------


def test_provenance_maps_every_slot_to_its_source_chunks():
    """A latent slot cannot report what went into it, which is why the paper
    says latent memory is not a deletion mechanism. A text slot can."""
    memory = _memory_with_topics([0, 1])
    _, best = memory.retrieve(_onehot(0), position=5, tau=0.35)
    memory.write_back("topic0 v2", _onehot(0), chunk_index=5, position=5, best=best, tau=0.35)

    assert memory.provenance() == {0: [0, 5], 1: [1]}


def test_forget_removes_every_slot_a_chunk_contributed_to():
    memory = _memory_with_topics([0, 1, 2])
    _, best = memory.retrieve(_onehot(1), position=7, tau=0.35)
    memory.write_back("topic1 v2", _onehot(1), chunk_index=7, position=7, best=best, tau=0.35)

    assert memory.forget(7) == 1
    assert all(7 not in slot.origin_indices for slot in memory.slots)
    assert len(memory) == 2


def test_forget_is_a_no_op_for_an_unknown_chunk():
    memory = _memory_with_topics([0, 1])
    assert memory.forget(99) == 0
    assert len(memory) == 2
