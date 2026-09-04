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


def test_kv_checkpoint_metrics_render():
    import fusion_mlx.runtime.disk_kv_checkpoint as dkc

    dkc.reset_stats_for_tests()
    logger.info("dkc stats after reset: %s", dkc.get_stats())
    body = _render()
    expected_names = [
        "fusion_mlx_kv_checkpoint_writes_total",
        "fusion_mlx_kv_checkpoint_loads_total",
        "fusion_mlx_kv_checkpoint_bytes_total",
        "fusion_mlx_kv_checkpoint_evictions_total",
    ]
    for name in expected_names:
        assert name in body, f"missing metric: {name}"
    assert "# TYPE fusion_mlx_kv_checkpoint_writes_total counter" in body
    assert "fusion_mlx_kv_checkpoint_writes_total 0" in body
    logger.info("kv checkpoint metrics rendered with 4 families")


def test_ubc_metrics_render():
    import fusion_mlx.runtime.ubc_evict as ubc_mod

    ubc_mod.reset_for_tests()
    logger.info("ubc snapshot after reset: %s", ubc_mod.snapshot())
    body = _render()
    assert "fusion_mlx_ubc_evicted_bytes_total" in body
    assert "# TYPE fusion_mlx_ubc_evicted_bytes_total counter" in body
    assert "fusion_mlx_ubc_evict_calls_total" in body
    assert "fusion_mlx_ubc_evict_failed_total" in body
    logger.info("ubc metrics rendered (evicted_bytes + calls + failed)")


def test_radix_cache_metrics_empty_when_no_cache():
    from fusion_mlx.cache.radix_diffusion_cache import all_cache_stats

    caches = all_cache_stats()
    logger.info("live radix caches before render: %d", len(caches))
    body = _render()
    assert "fusion_mlx_radix_cache_hits_total" not in body
    logger.info("radix cache metrics absent with no live cache (no fabrication)")


def test_radix_cache_metrics_render_with_live_cache():
    from fusion_mlx.cache.radix_diffusion_cache import _REGISTRY

    class _FakeCache:
        name = "test-cache"

        def stats(self):
            return {
                "hits": 2,
                "misses": 1,
                "evictions": 0,
                "insertions": 1,
                "leaf_count": 1,
                "total_bytes": 128,
            }

    fake = _FakeCache()
    _REGISTRY.add(fake)
    try:
        body = _render()
        assert 'fusion_mlx_radix_cache_hits_total{cache="test-cache"} 2' in body
        assert 'fusion_mlx_radix_cache_misses_total{cache="test-cache"} 1' in body
        assert 'fusion_mlx_radix_cache_leaf_count{cache="test-cache"} 1' in body
        assert 'fusion_mlx_radix_cache_bytes{cache="test-cache"} 128' in body
        logger.info("radix cache metrics rendered with live fake cache")
    finally:
        _REGISTRY.discard(fake)
        logger.info("cleaned up fake cache from _REGISTRY")
