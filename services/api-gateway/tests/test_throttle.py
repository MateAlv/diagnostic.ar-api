"""Tests for the rate limits, the global extraction budget and the concurrency cap."""
from __future__ import annotations

import json
import threading
import time

import pytest
from fastapi import HTTPException

from app.throttle import (
    ConcurrencyLimiter,
    GlobalExtractionBudget,
    PerIpRateLimiter,
    PerUserRateLimiter,
    SlidingWindowLimiter,
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

    def test_trusted_hops_are_skipped_from_the_right(self):
        request = FakeRequest("10.0.0.1", {"X-Forwarded-For": "203.0.113.9, 10.0.0.5"})
        assert client_identifier(request, ["10.0.0.1", "10.0.0.5"]) == "203.0.113.9"

    def test_a_caller_cannot_pick_their_bucket_by_prepending(self):
        # The proxy appends what it saw; anything left of that is the caller's.
        request = FakeRequest("10.0.0.1", {"X-Forwarded-For": "1.2.3.4, 203.0.113.9"})
        assert client_identifier(request, ["10.0.0.1"]) == "203.0.113.9"

    def test_proxies_can_be_named_by_cidr(self):
        request = FakeRequest("172.18.0.7", {"X-Forwarded-For": "203.0.113.9"})
        assert client_identifier(request, ["172.18.0.0/16"]) == "203.0.113.9"

    def test_a_malformed_entry_is_ignored(self):
        request = FakeRequest("10.0.0.1", {"X-Forwarded-For": "203.0.113.9"})
        assert client_identifier(request, ["not-an-ip", "10.0.0.1"]) == "203.0.113.9"

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


class TestSlidingWindowLimiter:
    def test_allows_up_to_the_limit_then_reports_the_wait(self):
        limiter = SlidingWindowLimiter([(2, 60.0)])
        assert limiter.acquire("a", now=100.0) == 0
        assert limiter.acquire("a", now=110.0) == 0
        # Full until the first event, at 100, is older than 60 s.
        assert limiter.acquire("a", now=120.0) == pytest.approx(40.0)

    def test_the_window_slides_rather_than_resetting(self):
        # A fixed window would allow 2 at 59 s and 2 more at 61 s.
        limiter = SlidingWindowLimiter([(2, 60.0)])
        limiter.acquire("a", now=58.0)
        limiter.acquire("a", now=59.0)
        assert limiter.acquire("a", now=61.0) > 0
        assert limiter.acquire("a", now=118.5) == 0

    def test_a_refused_request_is_not_counted(self):
        limiter = SlidingWindowLimiter([(1, 60.0)])
        limiter.acquire("a", now=0.0)
        for t in (10.0, 20.0, 30.0):
            assert limiter.acquire("a", now=t) > 0
        assert limiter.acquire("a", now=60.5) == 0

    def test_every_window_must_have_room(self):
        limiter = SlidingWindowLimiter([(10, 60.0), (3, 3600.0)])
        for t in (0.0, 1.0, 2.0):
            assert limiter.acquire("a", now=t) == 0
        # The minute has room, the hour does not.
        assert limiter.acquire("a", now=120.0) == pytest.approx(3480.0)

    def test_keys_are_independent(self):
        limiter = SlidingWindowLimiter([(1, 60.0)])
        assert limiter.acquire("a", now=0.0) == 0
        assert limiter.acquire("b", now=0.0) == 0

    def test_zero_turns_a_window_off(self):
        limiter = SlidingWindowLimiter([(0, 60.0), (0, 3600.0)])
        assert not limiter.enabled
        assert all(limiter.acquire("a") == 0 for _ in range(100))


class TestPerUserRateLimiter:
    def test_limits_one_account_without_touching_another(self):
        limiter = PerUserRateLimiter(per_hour=2, per_day=0)
        limiter.check("user-1")
        limiter.check("user-1")
        with pytest.raises(HTTPException) as refused:
            limiter.check("user-1")
        assert refused.value.status_code == 429
        assert refused.value.detail["error"] == "user-rate-limit"
        assert int(refused.value.headers["Retry-After"]) > 0
        limiter.check("user-2")

    def test_an_anonymous_caller_is_not_its_business(self):
        # Unauthenticated calls are refused by auth before they cost anything.
        limiter = PerUserRateLimiter(per_hour=1, per_day=1)
        for _ in range(5):
            limiter.check(None)


class TestGlobalExtractionBudget:
    def test_refuses_everyone_once_spent(self):
        budget = GlobalExtractionBudget(per_5h=2, per_week=0)
        budget.consume()
        budget.consume()
        with pytest.raises(HTTPException) as refused:
            budget.consume()
        assert refused.value.status_code == 429
        assert refused.value.detail["error"] == "global-budget"
        assert budget.status()["used_5h"] == 2

    def test_the_week_caps_what_the_5_hours_would_allow(self):
        budget = GlobalExtractionBudget(per_5h=10, per_week=3)
        for _ in range(3):
            budget.consume()
        with pytest.raises(HTTPException):
            budget.consume()

    def test_survives_a_restart(self, tmp_path):
        state = tmp_path / "budget.json"
        first = GlobalExtractionBudget(per_5h=0, per_week=2, state_path=str(state))
        first.consume()
        first.consume()
        # A redeploy must not hand out a second week's budget.
        second = GlobalExtractionBudget(per_5h=0, per_week=2, state_path=str(state))
        with pytest.raises(HTTPException):
            second.consume()
        assert second.status()["used_week"] == 2

    def test_forgets_what_is_older_than_a_week(self, tmp_path):
        state = tmp_path / "budget.json"
        state.write_text(json.dumps([time.time() - 8 * 86400]))
        budget = GlobalExtractionBudget(per_5h=0, per_week=1, state_path=str(state))
        budget.consume()

    def test_an_unreadable_history_stops_the_service(self, tmp_path):
        # Starting from zero instead would repeat a week's spend.
        state = tmp_path / "budget.json"
        state.write_text("not json")
        with pytest.raises(ValueError):
            GlobalExtractionBudget(per_5h=0, per_week=5, state_path=str(state))

    def test_disabled_by_default(self):
        budget = GlobalExtractionBudget(per_5h=0, per_week=0)
        assert budget.status() == {"enabled": False}
        for _ in range(100):
            budget.consume()
