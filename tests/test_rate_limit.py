"""Unit tests for the shared token-bucket rate limiter — no network required."""

import threading
import time

from src.pipeline.rate_limit import RateLimiter


def test_rate_limiter_spaces_out_sequential_acquires():
    """A single caller issuing calls back-to-back must not exceed the
    configured rate — this is what stops one document's chunk loop alone
    from tripping NIM's rate limit."""
    limiter = RateLimiter(rate_per_second=20.0)  # 1 call per 0.05s
    start = time.monotonic()
    for _ in range(5):
        limiter.acquire()
    elapsed = time.monotonic() - start
    assert elapsed >= 4 * 0.05 - 0.01  # 5 acquires span 4 intervals


def test_rate_limiter_first_acquire_does_not_block():
    limiter = RateLimiter(rate_per_second=1.0)
    start = time.monotonic()
    limiter.acquire()
    assert time.monotonic() - start < 0.05


def test_rate_limiter_shared_across_threads_caps_combined_rate():
    """The whole point: one limiter shared by N concurrent documents must cap
    the *combined* issue rate, not allow each thread the full configured rate
    independently — that would be a fleet-wide rate of rate*n_threads, the
    exact bug that let MAX_WORKERS concurrency trip NIM's 429 limit."""
    limiter = RateLimiter(rate_per_second=20.0)
    n_acquires_per_thread = 5
    n_threads = 4

    def worker():
        for _ in range(n_acquires_per_thread):
            limiter.acquire()

    start = time.monotonic()
    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.monotonic() - start

    total_acquires = n_threads * n_acquires_per_thread
    min_expected = (total_acquires - 1) * (1.0 / 20.0)
    assert elapsed >= min_expected - 0.02
