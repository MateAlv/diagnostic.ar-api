"""Tests for per-IP rate limiting and the concurrency cap."""
from __future__ import annotations

import threading
import time

import pytest
from fastapi import HTTPException

from app.throttle import (
    ConcurrencyLimiter,
    PerIpRateLimiter,
    UNKNOWN_CLIENT,
    client_identifier,
)


class FakeRequest:
    def __init__(self, peer: str | None, headers: dict | None = None):
        self.client = type("C", (), {"host": peer})() if peer else None
        self.headers = {k.lower(): v for k, v in (headers or {}).items()}


class TestClientIdentifier:
    def test_peer_is_used_when_no_proxy_is_trusted(self):
        request = FakeRequest("203.0.113.9")
        assert client_identifier(request, []) == "203.0.113.9"

    def test_forwarded_header_is_used_behind_a_trusted_proxy(self):
        request = FakeRequest("10.0.0.1", {"X-Forwarded-For": "203.0.113.9"})
        assert client_identifier(request, ["10.0.0.1"]) == "203.0.113.9"

    def test_forwarded_header_is_ignored_from_an_untrusted_peer(self):
        # Otherwise any caller sets the header and picks their own bucket.
        request = FakeRequest("203.0.113.9", {"X-Forwarded-For": "1.2.3.4"})
        assert client_identifier(request, ["10.0.0.1"]) == "203.0.113.9"

    def test_leftmost_entry_is_the_original_client(self):
        request = FakeRequest("10.0.0.1", {"X-Forwarded-For": "203.0.113.9, 10.0.0.5"})
        assert client_identifier(request, ["10.0.0.1"]) == "203.0.113.9"

    def test_empty_forwarded_header_falls_back_to_the_peer(self):
        request = FakeRequest("10.0.0.1", {"X-Forwarded-For": "  "})
        assert client_identifier(request, ["10.0.0.1"]) == "10.0.0.1"

    def test_missing_client_is_not_an_error(self):
        assert client_identifier(FakeRequest(None), []) == UNKNOWN_CLIENT


class TestPerIpRateLimiter:
    def test_disabled_by_default_value(self):
        limiter = PerIpRateLimiter(0)
        assert not limiter.enabled
        for _ in range(100):
            limiter.check("1.1.1.1")

    def test_requests_within_budget_pass(self):
        limiter = PerIpRateLimiter(3)
        for _ in range(3):
            limiter.check("1.1.1.1")

    def test_exceeding_the_budget_is_429(self):
        limiter = PerIpRateLimiter(2)
        limiter.check("1.1.1.1")
        limiter.check("1.1.1.1")
        with pytest.raises(HTTPException) as excinfo:
            limiter.check("1.1.1.1")
        assert excinfo.value.status_code == 429
        assert "Retry-After" in excinfo.value.headers

    def test_budgets_are_per_address(self):
        limiter = PerIpRateLimiter(1)
        limiter.check("1.1.1.1")
        limiter.check("2.2.2.2")  # unaffected by the first caller

    def test_tracking_is_bounded(self):
        limiter = PerIpRateLimiter(5, max_tracked=10)
        for i in range(50):
            limiter.check(f"10.0.0.{i}")
        assert len(limiter._windows) <= 10

    def test_concurrent_callers_do_not_exceed_the_budget(self):
        limiter = PerIpRateLimiter(10)
        allowed = []
        lock = threading.Lock()

        def call():
            try:
                limiter.check("1.1.1.1")
                with lock:
                    allowed.append(1)
            except HTTPException:
                pass

        threads = [threading.Thread(target=call) for _ in range(40)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(allowed) == 10


class TestConcurrencyLimiter:
    def test_disabled_by_default_value(self):
        limiter = ConcurrencyLimiter(0)
        assert not limiter.enabled
        with limiter:
            pass

    def test_permits_are_released_on_exit(self):
        limiter = ConcurrencyLimiter(1)
        for _ in range(5):
            with limiter:
                pass

    def test_permit_is_released_even_when_the_body_raises(self):
        limiter = ConcurrencyLimiter(1)
        with pytest.raises(ValueError):
            with limiter:
                raise ValueError("extraction failed")
        with limiter:  # would block forever if the permit leaked
            pass

    def test_never_more_than_max_concurrent_in_flight(self):
        limiter = ConcurrencyLimiter(3, max_queue=100)
        peak = 0
        current = 0
        lock = threading.Lock()

        def work():
            nonlocal peak, current
            with limiter:
                with lock:
                    current += 1
                    peak = max(peak, current)
                time.sleep(0.02)
                with lock:
                    current -= 1

        threads = [threading.Thread(target=work) for _ in range(12)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert peak <= 3

    def test_queue_depth_sheds_load_with_503(self):
        limiter = ConcurrencyLimiter(1, max_queue=1)
        rejected = []

        def work():
            try:
                with limiter:
                    time.sleep(0.05)
            except HTTPException as exc:
                rejected.append(exc.status_code)

        threads = [threading.Thread(target=work) for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert rejected, "expected some callers to be shed"
        assert set(rejected) == {503}
