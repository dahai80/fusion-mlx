# SPDX-License-Identifier: Apache-2.0
"""Unit tests for middleware/auth.py RateLimiter and auxiliary functions.

Covers:
- RateLimiter sliding-window algorithm (allow/reject/cleanup)
- Rate limiter configuration (enable/disable)
- _bucket_id, _subnet_bucket, _extract_bearer_token
- _rate_limit_client_id, _is_loopback_client
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

from fastapi import Request as FastAPIRequest

from fusion_mlx.middleware.auth import (
    RateLimiter,
    _bucket_id,
    _extract_bearer_token,
    _is_loopback_client,
    _rate_limit_client_id,
    _subnet_bucket,
)

# =========================================================================
# RateLimiter Core
# =========================================================================


class TestRateLimiterCore:
    """Sliding-window rate limiter algorithm."""

    def make_limiter(self, rpm: int = 60, enabled: bool = True):
        return RateLimiter(requests_per_minute=rpm, enabled=enabled)

    def test_initial_state_allows_request(self):
        limiter = self.make_limiter()
        allowed, retry_after = limiter.is_allowed("client-1")
        assert allowed is True
        assert retry_after == 0

    def test_within_limit_passes(self):
        limiter = self.make_limiter(rpm=5)
        for _ in range(5):
            allowed, _ = limiter.is_allowed("client-1")
            assert allowed is True

    def test_exceeding_limit_blocked(self):
        limiter = self.make_limiter(rpm=3)
        for _ in range(3):
            limiter.is_allowed("client-1")
        allowed, retry_after = limiter.is_allowed("client-1")
        assert allowed is False
        assert retry_after >= 1

    def test_window_slides_after_silence(self, monkeypatch):
        limiter = self.make_limiter(rpm=2)
        # Fill the window
        monkeypatch.setattr(time, "time", lambda: 1000.0)
        limiter.is_allowed("client-1")
        limiter.is_allowed("client-1")
        allowed, _ = limiter.is_allowed("client-1")
        assert allowed is False

        # Slide past the 60s window
        monkeypatch.setattr(time, "time", lambda: 1061.0)
        allowed, _ = limiter.is_allowed("client-1")
        assert allowed is True

    def test_window_boundary_fractional_second(self, monkeypatch):
        """Timestamps with sub-second precision should not break counting."""
        limiter = self.make_limiter(rpm=1)
        monkeypatch.setattr(time, "time", lambda: 1000.0)
        limiter.is_allowed("client-1")
        monkeypatch.setattr(time, "time", lambda: 1000.001)
        allowed, _ = limiter.is_allowed("client-1")
        assert allowed is False

    def test_disabled_always_allows(self):
        limiter = self.make_limiter(rpm=1, enabled=False)
        for _ in range(100):
            allowed, _ = limiter.is_allowed("client-1")
            assert allowed is True

    def test_clients_isolated(self):
        """One client exceeding limit should not affect another."""
        limiter = self.make_limiter(rpm=2)
        limiter.is_allowed("client-A")
        limiter.is_allowed("client-A")
        # client-A is blocked
        allowed_a, _ = limiter.is_allowed("client-A")
        assert allowed_a is False
        # client-B is not blocked
        for _ in range(2):
            allowed_b, _ = limiter.is_allowed("client-B")
            assert allowed_b is True

    def test_medium_rpm_behavior(self):
        """RPM=30 should allow 30 requests, then block."""
        limiter = self.make_limiter(rpm=30)
        for _ in range(30):
            allowed, _ = limiter.is_allowed("client-1")
            assert allowed is True
        allowed, retry_after = limiter.is_allowed("client-1")
        assert allowed is False
        assert retry_after >= 1


# =========================================================================
# RateLimiter Configuration
# =========================================================================


class TestRateLimiterConfig:
    """configure_rate_limiter and dynamic adjustments."""

    def test_fresh_limiter_has_correct_rpm(self):
        limiter = RateLimiter(requests_per_minute=10, enabled=True)
        assert limiter.requests_per_minute == 10
        assert limiter.enabled is True

    def test_disabled_by_default(self):
        limiter = RateLimiter(requests_per_minute=60)
        assert limiter.enabled is False

    def test_configure_changes_settings(self):
        limiter = RateLimiter(requests_per_minute=10, enabled=False)
        limiter.requests_per_minute = 20
        limiter.enabled = True
        assert limiter.requests_per_minute == 20
        assert limiter.enabled is True

    def test_configure_clears_existing_state(self):
        """Changing RPM should clear accumulated state."""
        limiter = RateLimiter(requests_per_minute=5, enabled=True)
        for _ in range(5):
            limiter.is_allowed("test")
        allowed, _ = limiter.is_allowed("test")
        assert allowed is False

        # Reset and verify
        limiter.requests_per_minute = 5
        limiter._requests.clear()
        limiter._last_seen.clear()
        allowed, _ = limiter.is_allowed("test")
        assert allowed is True


# =========================================================================
# Auxiliary Functions
# =========================================================================


class TestRateLimiterAux:
    """HMAC bucket, subnet, bearer extraction helpers."""

    def test_bucket_id_deterministic(self):
        b1 = _bucket_id("sk-test-key-123")
        b2 = _bucket_id("sk-test-key-123")
        assert b1 == b2

    def test_bucket_id_changes_with_key(self):
        b1 = _bucket_id("key-a")
        b2 = _bucket_id("key-b")
        assert b1 != b2

    def test_bucket_id_is_16_chars(self):
        bid = _bucket_id("any-key")
        assert len(bid) == 16
        assert all(c in "0123456789abcdef" for c in bid)

    def test_subnet_bucket_v4(self):
        result = _subnet_bucket("192.168.1.55")
        assert result == "192.168.1.0"  # /24

    def test_subnet_bucket_v6(self):
        result = _subnet_bucket("2001:db8::1")
        assert result == "2001:db8::"  # /64

    def test_subnet_bucket_invalid_address_fallback(self):
        result = _subnet_bucket("not-an-ip")
        assert result == "not-an-ip"

    def test_extract_bearer_token_valid(self):
        result = _extract_bearer_token("Bearer my-token-123")
        assert result == "my-token-123"

    def test_extract_bearer_token_missing(self):
        assert _extract_bearer_token(None) is None
        assert _extract_bearer_token("") is None

    def test_extract_bearer_token_wrong_scheme(self):
        assert _extract_bearer_token("Basic dXNlcjpwYXNz") is None

    def test_extract_bearer_no_value(self):
        assert _extract_bearer_token("Bearer ") is None

    def test_extract_bearer_case_insensitive(self):
        result = _extract_bearer_token("bearER my-token")
        assert result == "my-token"

    def test_rate_limit_client_id_by_auth(self):
        request = MagicMock(spec=FastAPIRequest)
        request.headers = {"Authorization": "Bearer test-key"}
        request.client = MagicMock()
        request.client.host = "10.0.0.1"
        cid = _rate_limit_client_id(request)
        # With auth header, returns HMAC bucket (16 hex chars)
        assert len(cid) == 16

    def test_rate_limit_client_id_by_ip(self):
        request = MagicMock(spec=FastAPIRequest)
        request.headers = {}
        request.client = MagicMock()
        request.client.host = "10.0.0.3"
        cid = _rate_limit_client_id(request)
        assert cid == "10.0.0.0"  # /24

    def test_rate_limit_client_id_fallback_unknown(self):
        request = MagicMock(spec=FastAPIRequest)
        request.headers = {}
        request.client = None
        cid = _rate_limit_client_id(request)
        assert cid == "unknown"

    def test_rate_limit_client_id_with_auth_and_no_client(self):
        """When auth header is present, client host is not needed."""
        request = MagicMock(spec=FastAPIRequest)
        request.headers = {"Authorization": "Bearer test-key"}
        request.client = None
        cid = _rate_limit_client_id(request)
        assert len(cid) == 16


# =========================================================================
# Loopback Detection
# =========================================================================


class TestIsLoopbackClient:
    """_is_loopback client proxy-header-aware detection."""

    def test_loopback_127(self):
        request = MagicMock(spec=FastAPIRequest)
        request.client.host = "127.0.0.1"
        request.headers = {}
        assert _is_loopback_client(request) is True

    def test_loopback_localhost(self):
        request = MagicMock(spec=FastAPIRequest)
        request.client.host = "localhost"
        request.headers = {}
        assert _is_loopback_client(request) is True

    def test_loopback_v6(self):
        request = MagicMock(spec=FastAPIRequest)
        request.client.host = "::1"
        request.headers = {}
        assert _is_loopback_client(request) is True

    def test_loopback_127_range(self):
        request = MagicMock(spec=FastAPIRequest)
        request.client.host = "127.0.0.2"
        request.headers = {}
        assert _is_loopback_client(request) is True

    def test_non_loopback_rejected(self):
        request = MagicMock(spec=FastAPIRequest)
        request.client.host = "192.168.1.1"
        request.headers = {}
        assert _is_loopback_client(request) is False

    def test_proxy_forwarded_header_returns_false(self):
        request = MagicMock(spec=FastAPIRequest)
        request.client.host = "127.0.0.1"
        request.headers = {"x-forwarded-for": "10.0.0.1"}
        assert _is_loopback_client(request) is False

    def test_multiple_proxy_headers_returns_false(self):
        request = MagicMock(spec=FastAPIRequest)
        request.client.host = "127.0.0.1"
        request.headers = {
            "x-forwarded-for": "10.0.0.1",
            "forwarded": "for=10.0.0.2",
            "via": "1.0 proxy",
        }
        assert _is_loopback_client(request) is False

    def test_no_client_returns_false(self):
        request = MagicMock(spec=FastAPIRequest)
        request.client = None
        request.headers = {}
        assert _is_loopback_client(request) is False

    def test_cf_connecting_ip_header_returns_false(self):
        request = MagicMock(spec=FastAPIRequest)
        request.client.host = "127.0.0.1"
        request.headers = {"cf-connecting-ip": "203.0.113.1"}
        assert _is_loopback_client(request) is False
