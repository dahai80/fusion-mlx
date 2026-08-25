# Issue #635: ``--rate-limit 0`` must disable the limiter. The module-level
# RateLimiter defaults to enabled=True @ 60rpm; cli_serve previously skipped
# configure_rate_limiter when the flag was 0, leaving the limiter active.
from __future__ import annotations

import pytest

from fusion_mlx.middleware.auth import configure_rate_limiter, rate_limiter


@pytest.fixture
def _restore_limiter():
    # Snapshot so a test failure can't leave the module limiter enabled for
    # the rest of the suite (the module global is shared).
    prev_rpm = rate_limiter.requests_per_minute
    prev_enabled = rate_limiter.enabled
    yield
    rate_limiter.requests_per_minute = prev_rpm
    rate_limiter.enabled = prev_enabled


class TestRateLimitZeroDisables:
    def test_zero_disables_limiter(self, _restore_limiter):
        # configure_rate_limiter(0, enabled=False) mirrors what cli_serve does
        # for --rate-limit 0. is_allowed must always return True.
        configure_rate_limiter(0, enabled=False)
        assert rate_limiter.enabled is False
        # Even after many requests, the disabled limiter never throttles.
        for i in range(100):
            allowed, _ = rate_limiter.is_allowed(f"client-{i}")
            assert allowed

    def test_positive_enables_limiter(self, _restore_limiter):
        configure_rate_limiter(5, enabled=True)
        assert rate_limiter.enabled is True
        assert rate_limiter.requests_per_minute == 5
        # 6th request from one client in the window is throttled.
        for _ in range(5):
            assert rate_limiter.is_allowed("burst-client")[0]
        allowed, retry = rate_limiter.is_allowed("burst-client")
        assert allowed is False
        assert retry >= 1

    def test_zero_flag_maps_to_disabled(self, _restore_limiter):
        # Mirrors cli_serve: enabled=args.rate_limit > 0.
        enabled = 0 > 0
        configure_rate_limiter(0, enabled=enabled)
        assert rate_limiter.enabled is False
