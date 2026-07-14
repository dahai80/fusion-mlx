# SPDX-License-Identifier: Apache-2.0
"""Regression tests for RateLimiter (#57 deque O(1) rewrite).

Locks in:
- RPM enforcement: allowed up to requests_per_minute, then denied.
- retry_after correctness (oldest + window - now + 1, min 1).
- deque popleft eviction: after window_size passes, requests allowed again.
- per-client isolation: one client hitting limit does not block another.
- disabled limiter always allows.
"""

import threading

import pytest

from fusion_mlx.middleware import auth as auth_mod
from fusion_mlx.middleware.auth import RateLimiter


@pytest.fixture
def controlled_time(monkeypatch):
    """Drive auth.time.time so we can advance the window without sleeping."""
    state = {"t": 1000.0}

    def fake_time():
        return state["t"]

    monkeypatch.setattr(auth_mod.time, "time", fake_time)
    return state


def _advance(state, seconds):
    state["t"] += seconds


def test_disabled_limiter_always_allows(controlled_time):
    rl = RateLimiter(requests_per_minute=2, enabled=False)
    for _ in range(10):
        allowed, retry = rl.is_allowed("c1")
        assert allowed is True
        assert retry == 0


def test_allows_until_rpm_then_denies(controlled_time):
    rl = RateLimiter(requests_per_minute=3, enabled=True)
    for i in range(3):
        allowed, retry = rl.is_allowed("c1")
        assert allowed is True, f"request {i} should be allowed"
        assert retry == 0
    allowed, retry = rl.is_allowed("c1")
    assert allowed is False
    # window_size=60, oldest at t=1000, now=1000 -> 1000+60-1000+1 = 61
    assert retry == 61


def test_retry_after_decreases_as_window_advances(controlled_time):
    rl = RateLimiter(requests_per_minute=1, enabled=True)
    assert rl.is_allowed("c1") == (True, 0)
    allowed, retry = rl.is_allowed("c1")
    assert allowed is False
    assert retry == 61
    _advance(controlled_time, 10)
    allowed, retry = rl.is_allowed("c1")
    assert allowed is False
    assert retry == 51


def test_window_eviction_allows_again(controlled_time):
    # #57 core: deque popleft drops expired entries so a client is
    # re-admitted once the oldest timestamp leaves the window.
    rl = RateLimiter(requests_per_minute=2, enabled=True)
    assert rl.is_allowed("c1")[0] is True
    assert rl.is_allowed("c1")[0] is True
    assert rl.is_allowed("c1")[0] is False  # limit hit
    _advance(controlled_time, 61)  # past full window
    allowed, retry = rl.is_allowed("c1")
    assert allowed is True
    assert retry == 0
    # second hit still allowed (deque evicted the stale pair)
    assert rl.is_allowed("c1")[0] is True
    assert rl.is_allowed("c1")[0] is False  # back at limit


def test_per_client_isolation(controlled_time):
    rl = RateLimiter(requests_per_minute=2, enabled=True)
    assert rl.is_allowed("A")[0] is True
    assert rl.is_allowed("A")[0] is True
    assert rl.is_allowed("A")[0] is False  # A saturated
    # B is independent and unaffected
    assert rl.is_allowed("B")[0] is True
    assert rl.is_allowed("B")[0] is True
    assert rl.is_allowed("B")[0] is False


def test_retry_after_minimum_is_one(controlled_time):
    # Edge: if (oldest + window - now) rounds to 0, retry_after must
    # still be >= 1 so clients don't get a zero Retry-After.
    rl = RateLimiter(requests_per_minute=1, enabled=True)
    assert rl.is_allowed("c1")[0] is True
    _advance(controlled_time, 59)  # 1s before window expiry
    allowed, retry = rl.is_allowed("c1")
    assert allowed is False
    assert retry >= 1


def test_is_allowed_is_thread_safe(controlled_time):
    # _lock guards the deque; hammer from many threads and assert no
    # exception and total admits never exceed rpm over a frozen window.
    rl = RateLimiter(requests_per_minute=50, enabled=True)
    errors = []
    admits = {"n": 0}
    admits_lock = threading.Lock()

    def worker():
        try:
            for _ in range(20):
                allowed, _ = rl.is_allowed("shared")
                if allowed:
                    with admits_lock:
                        admits["n"] += 1
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    # frozen window -> admits bounded by rpm regardless of thread count
    assert admits["n"] <= 50
