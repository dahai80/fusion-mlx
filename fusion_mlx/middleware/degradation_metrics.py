# SPDX-License-Identifier: Apache-2.0
"""Degradation-point counters for fusion-mlx.

OP-2 (#0907 audit): critical degradation events (SSRF rejection, enforcer
lock timeout, cloud fallback, rate-limit 429) were log-only. An operator had
no metric to alert on, so production degradation surfaced only by a human
grepping logs. Each degradation point now increments a module-level counter
exposed via /metrics so a dashboard can alert on a non-zero rate.

Mirrors the response_format_metrics pattern: module-level ints guarded by a
lock, a ``snapshot()`` for the /metrics render path, and a
``reset_for_tests()`` hook. Prometheus counters are contractually monotonic
for the process lifetime — production code MUST NOT call reset_for_tests.
"""

import logging
import threading

logger = logging.getLogger(__name__)

_lock = threading.Lock()

# OP-2 counters. Each ticks once per degradation event so
# ``rate(fusion_mlx_degradation_total[5m]) > 0`` is the alert condition.
_ssrf_rejected_total: int = 0
_enforcer_timeout_total: int = 0
_cloud_fallback_total: int = 0
_rate_limit_rejected_total: int = 0
_route_guard_rejected_total: int = 0

# Reason sub-breakdowns (dict[str, int]) so the dashboard can split a
# counter by cause without a new metric per reason.
_ssrf_rejected_by_reason: dict[str, int] = {}
_cloud_fallback_by_reason: dict[str, int] = {}
_route_guard_rejected_by_reason: dict[str, int] = {}


def record_ssrf_rejection(reason: str = "private_ip") -> None:
    """Tick the SSRF-rejection counter.

    Called from ``fusion_mlx.api._url_safety`` whenever a fetch is denied
    because the resolved address is a private/loopback/reserved range, a
    redirect hops to one, or DNS rebinding to a private IP is detected.
    """
    global _ssrf_rejected_total
    with _lock:
        _ssrf_rejected_total += 1
        _ssrf_rejected_by_reason[reason] = _ssrf_rejected_by_reason.get(reason, 0) + 1


def record_enforcer_timeout() -> None:
    """Tick the memory-enforcer pool-lock-timeout counter.

    Called from ``fusion_mlx.pool.memory_enforcer`` when the 2s pool-lock
    acquire times out — the relief valve was blocked by contention, so
    memory pressure is not being relieved.
    """
    global _enforcer_timeout_total
    with _lock:
        _enforcer_timeout_total += 1


def record_cloud_fallback(reason: str = "large_context") -> None:
    """Tick the cloud-fallback counter.

    Called from ``fusion_mlx.dispatch`` when a request is routed to a
    third-party cloud provider instead of the local engine (large uncached
    context, consent-gated).
    """
    global _cloud_fallback_total
    with _lock:
        _cloud_fallback_total += 1
        _cloud_fallback_by_reason[reason] = _cloud_fallback_by_reason.get(reason, 0) + 1


def record_rate_limit_rejection() -> None:
    """Tick the rate-limit (429) counter.

    Called from ``fusion_mlx.middleware.auth`` rate-limit enforcement.
    """
    global _rate_limit_rejected_total
    with _lock:
        _rate_limit_rejected_total += 1


def record_route_guard_rejection(reason: str = "missing_token") -> None:
    """Tick the route-guard rejection counter.

    OP-16 (#0907 audit): route-guard 403s were log-only with no metric, so an
    operator could not tell how often a misconfigured gateway token or a
    direct-port probe was being rejected. Called from
    ``fusion_mlx.middleware.route_guard`` at every reject branch.
    """
    global _route_guard_rejected_total
    with _lock:
        _route_guard_rejected_total += 1
        _route_guard_rejected_by_reason[reason] = (
            _route_guard_rejected_by_reason.get(reason, 0) + 1
        )


def snapshot() -> dict[str, int | dict[str, int]]:
    """Return a consistent snapshot of all degradation counters for /metrics."""
    with _lock:
        return {
            "ssrf_rejected_total": _ssrf_rejected_total,
            "ssrf_rejected_by_reason": dict(_ssrf_rejected_by_reason),
            "enforcer_timeout_total": _enforcer_timeout_total,
            "cloud_fallback_total": _cloud_fallback_total,
            "cloud_fallback_by_reason": dict(_cloud_fallback_by_reason),
            "rate_limit_rejected_total": _rate_limit_rejected_total,
            "route_guard_rejected_total": _route_guard_rejected_total,
            "route_guard_rejected_by_reason": dict(_route_guard_rejected_by_reason),
        }


def reset_for_tests() -> None:
    """Test-only hook: zero the counters between cases.

    Production code MUST NOT call this — Prometheus counters are
    contractually monotonic for the process lifetime.
    """
    global _ssrf_rejected_total, _enforcer_timeout_total
    global _cloud_fallback_total, _rate_limit_rejected_total
    global _route_guard_rejected_total
    with _lock:
        _ssrf_rejected_total = 0
        _enforcer_timeout_total = 0
        _cloud_fallback_total = 0
        _rate_limit_rejected_total = 0
        _route_guard_rejected_total = 0
        _ssrf_rejected_by_reason.clear()
        _cloud_fallback_by_reason.clear()
        _route_guard_rejected_by_reason.clear()
