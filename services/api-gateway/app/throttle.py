"""Rate limits, a global extraction budget and a concurrency cap for /extract-hpo.

These are independent of authentication and worth having either way: auth
decides who may call, these decide how often and how many at once. An
authenticated user can still run up a bill.

The global budget is the one that bounds spend: per-IP and per-user limits
are evaded with more addresses and more accounts, and the concurrency cap
bounds the rate, not the total. See docs/EXTRACTION_LIMITS.md.
"""
from __future__ import annotations

import ipaddress
import json
import logging
import os
import threading
import time
from collections import OrderedDict

from fastapi import HTTPException, Request, status

logger = logging.getLogger(__name__)

FORWARDED_FOR_HEADER = "x-forwarded-for"
UNKNOWN_CLIENT = "unknown"


def _is_trusted(address: str, trusted_proxies: list[str]) -> bool:
    """Whether `address` is one of the named proxies, as an IP or a CIDR.

    A CIDR matters on Docker: Traefik's address on the shared network changes
    whenever its container is recreated, and an exact IP that has gone stale
    fails silently -- every request then keys on the proxy again.
    """
    try:
        ip = ipaddress.ip_address(address)
    except ValueError:
        return False
    for entry in trusted_proxies:
        try:
            if ip in ipaddress.ip_network(entry, strict=False):
                return True
        except ValueError:
            continue
    return False


def client_identifier(request: Request, trusted_proxies: list[str]) -> str:
    """The address to attribute a request to.

    `request.client.host` is the peer, which behind Traefik is the proxy on
    every request -- keying a limiter on it throttles everyone at once or no
    one. The real address is in X-Forwarded-For, but that header is caller
    supplied: believing it unconditionally lets anyone pick their own bucket
    and never hit the limit. So it is read only when the peer is a proxy we
    named, and from the right: each proxy appends the address it saw, so the
    first entry that is not one of ours is the last one a caller could not
    have written. The leftmost entry is whatever the caller sent.
    """
    peer = request.client.host if request.client else None

    if peer and _is_trusted(peer, trusted_proxies):
        forwarded = request.headers.get(FORWARDED_FOR_HEADER, "")
        hops = [hop.strip() for hop in forwarded.split(",") if hop.strip()]
        for hop in reversed(hops):
            if not _is_trusted(hop, trusted_proxies):
                return hop

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


def _too_many(error: str, retry_after: float, message: str) -> HTTPException:
    """A 429 the frontend can tell apart: `error` says which limit it was."""
    seconds = max(1, int(retry_after + 0.999))
    return HTTPException(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        detail={"error": error, "retry_after": seconds, "message": message},
        headers={"Retry-After": str(seconds)},
    )


class SlidingWindowLimiter:
    """At most N events per window, for several windows at once, per key.

    Sliding rather than fixed: a fixed hour lets a caller spend the whole
    budget at 10:59 and again at 11:00. Each key keeps the timestamps of its
    last events, at most the largest limit of them, so memory is bounded by
    keys x limit.
    """

    def __init__(self, limits: list[tuple[int, float]], max_tracked: int = 8192):
        # (count, seconds) pairs; a count of 0 turns that window off.
        self.limits = [(count, seconds) for count, seconds in limits if count > 0]
        self.max_tracked = max_tracked
        self._events: OrderedDict[str, list[float]] = OrderedDict()
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return bool(self.limits)

    def _retry_after(self, events: list[float], now: float) -> float:
        """Seconds until every window has room again; 0 when there is room now."""
        wait = 0.0
        for count, seconds in self.limits:
            recent = [t for t in events if t > now - seconds]
            if len(recent) >= count:
                # The oldest event that keeps this window full has to age out.
                wait = max(wait, recent[-count] + seconds - now)
        return wait

    def acquire(self, key: str, now: float | None = None) -> float:
        """Record an event for `key` and return 0, or return the seconds to
        wait without recording anything."""
        if not self.enabled:
            return 0.0
        now = time.time() if now is None else now
        longest = max(seconds for _, seconds in self.limits)
        keep = max(count for count, _ in self.limits)
        with self._lock:
            events = [t for t in self._events.get(key, []) if t > now - longest]
            wait = self._retry_after(events, now)
            if wait > 0:
                self._events[key] = events
                self._events.move_to_end(key)
                return wait
            events.append(now)
            self._events[key] = events[-keep:]
            self._events.move_to_end(key)
            while len(self._events) > self.max_tracked:
                self._events.popitem(last=False)
            return 0.0

    def count(self, key: str, seconds: float, now: float | None = None) -> int:
        now = time.time() if now is None else now
        with self._lock:
            return sum(1 for t in self._events.get(key, []) if t > now - seconds)


class PerUserRateLimiter:
    """A budget per signed-in account, keyed on the token's subject.

    Per-IP limits miss a user who changes networks and punish a hospital
    behind one address; the account is the unit that should be limited. It is
    not the unit that bounds spend, since sign-up is open and accounts are
    free -- that is GlobalExtractionBudget's job.
    """

    def __init__(self, per_hour: int, per_day: int):
        self._limiter = SlidingWindowLimiter([(per_hour, 3600.0), (per_day, 86400.0)])

    @property
    def enabled(self) -> bool:
        return self._limiter.enabled

    def check(self, user_id: str | None) -> None:
        if user_id is None or not self.enabled:
            return
        wait = self._limiter.acquire(user_id)
        if wait > 0:
            raise _too_many(
                "user-rate-limit",
                wait,
                "Alcanzaste el límite de análisis de texto por ahora.",
            )


class GlobalExtractionBudget:
    """A ceiling on extractions for the whole service, whoever asks.

    This is the limit that bounds spend: per-user and per-IP limits are
    evaded with more accounts and more addresses. Its windows are chosen to
    match what is being protected -- on staging, a Claude subscription, which
    meters a rolling 5-hour session and a week.

    Counted when an extraction is about to call the model, not when a request
    arrives, so a caller turned away by another limit spends nothing here.
    With `state_path` the recent extraction times survive a restart; without
    it a redeploy would reset the week.
    """

    KEY = "global"

    def __init__(self, per_5h: int, per_week: int, state_path: str = ""):
        self._limiter = SlidingWindowLimiter(
            [(per_5h, 5 * 3600.0), (per_week, 7 * 86400.0)], max_tracked=1
        )
        self.per_5h = per_5h
        self.per_week = per_week
        self.state_path = state_path
        self._load()

    @property
    def enabled(self) -> bool:
        return self._limiter.enabled

    def _load(self) -> None:
        if not (self.state_path and self.enabled):
            return
        try:
            with open(self.state_path, encoding="utf-8") as handle:
                events = [float(t) for t in json.load(handle)]
        except FileNotFoundError:
            return
        except (OSError, ValueError, TypeError) as exc:
            # A budget that cannot read its history must not start from zero
            # silently: that is how a week's spend gets repeated.
            logger.error("extraction budget state unreadable at %s: %s", self.state_path, exc)
            raise
        cutoff = time.time() - 7 * 86400.0
        self._limiter._events[self.KEY] = sorted(t for t in events if t > cutoff)

    def _save(self) -> None:
        if not self.state_path:
            return
        events = self._limiter._events.get(self.KEY, [])
        tmp = f"{self.state_path}.tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(events, handle)
        os.replace(tmp, self.state_path)

    def consume(self) -> None:
        """Count one extraction, or raise 429 when the budget is spent."""
        if not self.enabled:
            return
        wait = self._limiter.acquire(self.KEY)
        if wait > 0:
            logger.warning("extraction budget exhausted; next slot in %.0fs", wait)
            raise _too_many(
                "global-budget",
                wait,
                "El análisis de texto libre llegó a su límite por ahora.",
            )
        try:
            self._save()
        except OSError as exc:
            logger.error("could not persist the extraction budget to %s: %s", self.state_path, exc)

    def status(self) -> dict:
        if not self.enabled:
            return {"enabled": False}
        return {
            "enabled": True,
            "used_5h": self._limiter.count(self.KEY, 5 * 3600.0),
            "limit_5h": self.per_5h or None,
            "used_week": self._limiter.count(self.KEY, 7 * 86400.0),
            "limit_week": self.per_week or None,
            "persisted": bool(self.state_path),
        }
