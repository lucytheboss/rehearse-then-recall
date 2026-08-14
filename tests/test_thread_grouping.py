"""Unit tests for no-merge thread clustering (`cluster_into_threads`)."""

from src.pipeline.thread_grouping import GistThread, cluster_into_threads

# Orthogonal-ish 2D embeddings so similarity is exact and easy to reason about.
A1, A2 = [1.0, 0.0], [0.9, 0.1]  # similar to each other
B1 = [0.0, 1.0]  # dissimilar to A1/A2


def test_single_chunk_opens_one_thread():
    threads = cluster_into_threads(["gist 0"], [A1], tau=0.35)
    assert len(threads) == 1
    assert threads[0].member_indices == [0]
    assert threads[0].member_texts == ["gist 0"]


def test_similar_chunks_join_the_same_thread():
    threads = cluster_into_threads(["gist 0", "gist 1"], [A1, A2], tau=0.5)
    assert len(threads) == 1
    assert threads[0].member_indices == [0, 1]
    assert threads[0].member_texts == ["gist 0", "gist 1"]


def test_dissimilar_chunks_open_separate_threads():
    threads = cluster_into_threads(["gist 0", "gist 1"], [A1, B1], tau=0.5)
    assert len(threads) == 2
    assert [t.member_indices for t in threads] == [[0], [1]]


def test_third_chunk_compares_against_anchor_not_most_recent_member():
    # gist 1 (B1) opens a second thread; gist 2 is similar to gist 0's thread
    # (A1) specifically, not to whatever thread 1 currently looks like.
    threads = cluster_into_threads(
        ["gist 0", "gist 1", "gist 2"], [A1, B1, A2], tau=0.5
    )
    assert len(threads) == 2
    assert threads[0].member_indices == [0, 2]
    assert threads[1].member_indices == [1]


def test_text_property_concatenates_members_in_order_without_rewriting():
    threads = cluster_into_threads(["first", "second"], [A1, A2], tau=0.5)
    assert threads[0].text == "first\nsecond"


def test_no_content_is_ever_merged_or_altered():
    """The whole point: member_texts holds exactly what was passed in."""
    inputs = ["alpha", "beta", "gamma"]
    threads = cluster_into_threads(inputs, [A1, A2, A1], tau=0.5)
    all_kept_texts = [t for thread in threads for t in thread.member_texts]
    assert sorted(all_kept_texts) == sorted(inputs)


def test_empty_input_returns_no_threads():
    assert cluster_into_threads([], [], tau=0.35) == []


def test_anchor_embedding_is_fixed_at_thread_creation():
    threads = cluster_into_threads(["a", "b"], [A1, A2], tau=0.5)
    assert threads[0].anchor_embedding == A1


def test_gist_thread_default_construction_is_empty():
    thread = GistThread()
    assert thread.member_indices == []
    assert thread.member_texts == []
    assert thread.text == ""
