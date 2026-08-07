"""Thread-safe token-bucket rate limiter for the NIM API.

2026-08-04: `06b`'s document-parallel curation (`MAX_WORKERS=2`) measured NIM
chat calls returning 429 "Too Many Requests" — a genuine per-key request-rate
limit, distinct from the 503 "Worker local total request limit reached"
congestion seen earlier against a different model deployment (that one was a
specific deployment being oversubscribed; this one showed up on embedding
calls too, so it reads as a limit on the key, not one model). Concurrency
(how many requests are in flight at once, which is what lets network wait
overlap) and issue rate (how fast new requests start) are different knobs —
`MAX_WORKERS` controls the former, `RateLimiter` controls the latter. Raising
`MAX_WORKERS` without this just launches requests faster into the same limit.
"""

from __future__ import annotations

import threading
import time


class RateLimiter:
    """Blocks each caller until at least `1/rate_per_second` has elapsed
    since the previously granted acquire, across every thread sharing this
    instance. One instance must be shared by all concurrent callers — a
    limiter per thread would each allow the configured rate independently,
    which is a fleet-wide rate of `rate_per_second * n_threads`, not the cap
    this exists to enforce.
    """

    def __init__(self, rate_per_second: float):
        self._interval = 1.0 / rate_per_second
        self._lock = threading.Lock()
        self._next_allowed = time.monotonic()

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = self._next_allowed - now
            self._next_allowed = max(now, self._next_allowed) + self._interval
        if wait > 0:
            time.sleep(wait)
