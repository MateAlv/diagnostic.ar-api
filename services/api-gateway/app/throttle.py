"""Per-IP rate limiting and a global concurrency cap for /extract-hpo.

These are independent of authentication and worth having either way: auth
decides who may call, these decide how often and how many at once. An
authenticated user can still run up a bill.

The concurrency cap is the one that bounds spend. A per-IP limit is evaded by
having more addresses; a semaphore holds regardless of who is calling.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict

from fastapi import HTTPException, Request, status

logger = logging.getLogger(__name__)

FORWARDED_FOR_HEADER = "x-forwarded-for"
UNKNOWN_CLIENT = "unknown"


def client_identifier(request: Request, trusted_proxies: list[str]) -> str:
    """The address to attribute a request to.

    `request.client.host` is the peer, which behind Traefik is the proxy on
    every request -- keying a limiter on it throttles everyone at once or no
    one. The real address is in X-Forwarded-For, but that header is caller
    supplied: believing it unconditionally lets anyone pick their own bucket
    and never hit the limit. So it is read only when the peer is a proxy we
    named.
    """
    peer = request.client.host if request.client else None

    if peer and peer in trusted_proxies:
        forwarded = request.headers.get(FORWARDED_FOR_HEADER, "")
        # Leftmost entry is the original client; the rest are intermediaries.
        original = forwarded.split(",")[0].strip()
        if original:
            return original

    return peer or UNKNOWN_CLIENT


class PerIpRateLimiter:
    """A fixed-window request budget per client address."""

    def __init__(self, requests_per_minute: int, max_tracked: int = 8192):
        self.requests_per_minute = requests_per_minute
        self.max_tracked = max_tracked
        self._windows: OrderedDict[str, tuple[float, int]] = OrderedDict()
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self.requests_per_minute > 0

    def check(self, identifier: str) -> None:
        """Raise HTTPException(429) when the caller is over budget."""
        if not self.enabled:
            return

        now = time.monotonic()
        window_start = now - (now % 60.0)

        with self._lock:
            started_at, count = self._windows.get(identifier, (window_start, 0))

            if started_at < window_start:
                started_at, count = window_start, 0

            if count >= self.requests_per_minute:
                self._windows[identifier] = (started_at, count)
                self._windows.move_to_end(identifier)
                retry_after = max(1, int(started_at + 60.0 - now))
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded",
                    headers={"Retry-After": str(retry_after)},
                )

            self._windows[identifier] = (started_at, count + 1)
            self._windows.move_to_end(identifier)

            # Bounded, so a flood of distinct addresses cannot grow this
            # without limit. Evicting the least recently seen can forgive an
            # idle caller's earlier requests, which is the right trade.
            while len(self._windows) > self.max_tracked:
                self._windows.popitem(last=False)


class ConcurrencyLimiter:
    """Caps extractions in flight, and sheds load past a queue depth.

    Extraction calls a paid model, so the bound that matters for cost is how
    many run at once rather than how many any one caller makes. Queueing a
    little smooths a burst; queueing without limit turns a slow model into a
    slow service for everyone.
    """

    def __init__(self, max_concurrent: int, max_queue: int = 8):
        self.max_concurrent = max_concurrent
        self.max_queue = max_queue
        self._semaphore = threading.BoundedSemaphore(max_concurrent) if max_concurrent > 0 else None
        self._waiting = 0
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self._semaphore is not None

    def __enter__(self):
        if not self.enabled:
            return self

        with self._lock:
            if self._waiting >= self.max_queue:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Service is busy, try again shortly",
                    headers={"Retry-After": "30"},
                )
            self._waiting += 1

        try:
            self._semaphore.acquire()
        finally:
            with self._lock:
                self._waiting -= 1

        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.enabled:
            self._semaphore.release()
