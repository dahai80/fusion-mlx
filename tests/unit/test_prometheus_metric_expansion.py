# SPDX-License-Identifier: Apache-2.0
import logging

import pytest

from fusion_mlx.routes_internal.metrics import render_prometheus_metrics

logger = logging.getLogger(__name__)


def _render() -> str:
    body = render_prometheus_metrics()
    logger.info("prometheus body length=%d", len(body))
    return body


def test_no_rapid_mlx_prefix():
    body = _render()
    assert "rapid_mlx_" not in body, "stale rapid_mlx_ prefix still present"
    assert "fusion_mlx_response_format_strict_total" in body


def test_queue_gauges_render():
    body = _render()
    assert "fusion_mlx_requests_running" in body
    assert "fusion_mlx_requests_waiting" in body
    assert "# TYPE fusion_mlx_requests_running gauge" in body
    assert "# TYPE fusion_mlx_requests_waiting gauge" in body


def test_uptime_gauge_renders():
    body = _render()
    assert "fusion_mlx_uptime_seconds" in body
    assert "# TYPE fusion_mlx_uptime_seconds gauge" in body


def test_metal_gauges_render_or_absent():
    body = _render()
    mlx_available = False
    try:
        import mlx.core as mx

        mlx_available = bool(mx.metal.is_available())
    except Exception:
        mlx_available = False
    logger.info("mlx metal available=%s", mlx_available)
    if mlx_available:
        assert "fusion_mlx_metal_active_bytes" in body
        assert "fusion_mlx_metal_cache_bytes" in body
        assert "fusion_mlx_metal_peak_bytes" in body
    else:
        assert "fusion_mlx_metal_active_bytes" not in body


def test_response_format_metrics_fusion_prefix():
    body = _render()
    expected = [
        "fusion_mlx_response_format_strict_total",
        "fusion_mlx_response_format_strict_violations_total",
        "fusion_mlx_response_format_strict_repairs_attempted_total",
        "fusion_mlx_response_format_strict_repairs_succeeded_total",
        "fusion_mlx_response_format_strict_repairs_skipped_context_overflow_total",
    ]
    for name in expected:
        assert name in body, f"missing metric: {name}"


def test_queue_gauges_zero_without_engine_pool():
    body = _render()
    for line in body.splitlines():
        if line.startswith("fusion_mlx_requests_running "):
            value = float(line.rsplit(" ", 1)[-1])
            assert value == 0.0
            logger.info("queue running value=0.0 (no engine pool) OK")
            return
    pytest.fail("fusion_mlx_requests_running line not found")
