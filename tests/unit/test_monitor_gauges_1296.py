# SPDX-License-Identifier: Apache-2.0
# #1296: swap rate / model residency / p95 latency monitor additions.

import logging
import time
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

logger = logging.getLogger(__name__)


@pytest.fixture(autouse=True)
def _reset_swap_state():
    from fusion_mlx.utils import swap_stats

    swap_stats._reset_for_tests()
    yield
    swap_stats._reset_for_tests()


def test_sample_swap_rate_from_sin_sout_deltas(monkeypatch):
    from fusion_mlx.utils import swap_stats

    times = iter([100.0, 102.0])
    counters = iter(
        [
            {
                "swap_total_bytes": 0,
                "swap_used_bytes": 0,
                "swap_free_bytes": 0,
                "swap_percent": 0.0,
                "_sin": 1000,
                "_sout": 500,
            },
            {
                "swap_total_bytes": 0,
                "swap_used_bytes": 0,
                "swap_free_bytes": 0,
                "swap_percent": 0.0,
                "_sin": 3000,
                "_sout": 1500,
            },
        ]
    )
    monkeypatch.setattr(swap_stats, "_monotonic", lambda: next(times))
    monkeypatch.setattr(swap_stats, "_read_swap_counters", lambda: next(counters))

    first = swap_stats.sample_swap()
    assert first["swap_rate_bytes_per_second"] is None
    second = swap_stats.sample_swap()
    # Signed net rate: page-in delta 2000 minus page-out delta 1000 over 2s.
    assert second["swap_rate_bytes_per_second"] == pytest.approx(500.0)
    logger.info("swap rate computed from sin/sout deltas: %s", second)


def test_sample_swap_rate_goes_negative_when_paging_out(monkeypatch):
    from fusion_mlx.utils import swap_stats

    times = iter([200.0, 203.0])
    counters = iter(
        [
            {
                "swap_total_bytes": 0,
                "swap_used_bytes": 0,
                "swap_free_bytes": 0,
                "swap_percent": 0.0,
                "_sin": 5000,
                "_sout": 1000,
            },
            {
                "swap_total_bytes": 0,
                "swap_used_bytes": 0,
                "swap_free_bytes": 0,
                "swap_percent": 0.0,
                "_sin": 5000,
                "_sout": 4000,
            },
        ]
    )
    monkeypatch.setattr(swap_stats, "_monotonic", lambda: next(times))
    monkeypatch.setattr(swap_stats, "_read_swap_counters", lambda: next(counters))

    assert swap_stats.sample_swap()["swap_rate_bytes_per_second"] is None
    paging_out = swap_stats.sample_swap()
    assert paging_out["swap_rate_bytes_per_second"] == pytest.approx(-1000.0)
    logger.info("signed swap rate goes negative on page-out only: %s", paging_out)


def test_sample_swap_rate_rebaselines_on_counter_reset(monkeypatch):
    from fusion_mlx.utils import swap_stats

    times = iter([300.0, 305.0])
    counters = iter(
        [
            {
                "swap_total_bytes": 0,
                "swap_used_bytes": 0,
                "swap_free_bytes": 0,
                "swap_percent": 0.0,
                "_sin": 9000,
                "_sout": 9000,
            },
            {
                "swap_total_bytes": 0,
                "swap_used_bytes": 0,
                "swap_free_bytes": 0,
                "swap_percent": 0.0,
                "_sin": 10,
                "_sout": 10,
            },
        ]
    )
    monkeypatch.setattr(swap_stats, "_monotonic", lambda: next(times))
    monkeypatch.setattr(swap_stats, "_read_swap_counters", lambda: next(counters))

    assert swap_stats.sample_swap()["swap_rate_bytes_per_second"] is None
    after_reset = swap_stats.sample_swap()
    # Wrapped counters must not emit a garbage negative rate.
    assert after_reset["swap_rate_bytes_per_second"] is None
    logger.info("swap rate rebaselined after counter reset: %s", after_reset)


def test_swap_failure_warns_once(caplog):
    from fusion_mlx.utils import swap_stats

    def boom():
        raise OSError("psutil unavailable")

    def sample():
        swap_stats._last_at = 0.0
        return swap_stats.sample_swap()

    original = swap_stats._read_swap_counters
    swap_stats._read_swap_counters = boom
    try:
        with caplog.at_level(logging.WARNING, logger="fusion_mlx.utils.swap_stats"):
            sample()
            sample()
    finally:
        swap_stats._read_swap_counters = original
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    logger.info("swap read failure warned %d time(s) across two samples", len(warnings))


def test_sample_swap_rate_holds_baseline_inside_min_interval(monkeypatch):
    from fusion_mlx.utils import swap_stats

    counters = iter(
        [
            {
                "swap_total_bytes": 0,
                "swap_used_bytes": 0,
                "swap_free_bytes": 0,
                "swap_percent": 0.0,
                "_sin": 0,
                "_sout": 0,
            },
            {
                "swap_total_bytes": 0,
                "swap_used_bytes": 0,
                "swap_free_bytes": 0,
                "swap_percent": 0.0,
                "_sin": 100,
                "_sout": 0,
            },
            {
                "swap_total_bytes": 0,
                "swap_used_bytes": 0,
                "swap_free_bytes": 0,
                "swap_percent": 0.0,
                "_sin": 300,
                "_sout": 0,
            },
        ]
    )
    times = iter([100.0, 100.2, 101.5])
    monkeypatch.setattr(swap_stats, "_monotonic", lambda: next(times))
    monkeypatch.setattr(swap_stats, "_read_swap_counters", lambda: next(counters))

    assert swap_stats.sample_swap()["swap_rate_bytes_per_second"] is None
    held = swap_stats.sample_swap()
    assert held["swap_rate_bytes_per_second"] is None
    computed = swap_stats.sample_swap()
    # Baseline stayed at t=100 (sin=0), so the 1.5s window covers sin 0 -> 300.
    assert computed["swap_rate_bytes_per_second"] == pytest.approx(200.0)
    logger.info("swap rate held inside min interval, then computed: %s", computed)


def test_sample_swap_unavailable_returns_nulls(monkeypatch):
    from fusion_mlx.utils import swap_stats

    def _boom():
        raise RuntimeError("psutil exploded")

    monkeypatch.setattr(swap_stats, "_read_swap_counters", _boom)
    snap = swap_stats.sample_swap()
    assert set(snap) == {
        "swap_total_bytes",
        "swap_used_bytes",
        "swap_free_bytes",
        "swap_percent",
        "swap_rate_bytes_per_second",
    }
    assert all(value is None for value in snap.values())
    logger.info("swap unavailable -> all fields null")


def test_render_swap_metrics_with_sample(monkeypatch):
    from fusion_mlx.routes_internal import metrics
    from fusion_mlx.utils import swap_stats

    monkeypatch.setattr(
        swap_stats,
        "sample_swap",
        lambda: {
            "swap_total_bytes": 2048,
            "swap_used_bytes": 42,
            "swap_free_bytes": 2006,
            "swap_percent": 2.05,
            "swap_rate_bytes_per_second": 1234.5,
        },
    )
    body = "\n".join(metrics._render_swap_metrics())
    assert "fusion_mlx_swap_total_bytes 2048.0" in body
    assert "fusion_mlx_swap_used_bytes 42.0" in body
    assert "fusion_mlx_swap_free_bytes 2006.0" in body
    assert "fusion_mlx_swap_percent 2.05" in body
    assert "fusion_mlx_swap_rate_bytes_per_second 1234.5" in body
    logger.info("swap metric family rendered from stub")


def test_render_swap_metrics_omits_null_rate(monkeypatch):
    from fusion_mlx.routes_internal import metrics
    from fusion_mlx.utils import swap_stats

    monkeypatch.setattr(
        swap_stats,
        "sample_swap",
        lambda: {
            "swap_total_bytes": None,
            "swap_used_bytes": None,
            "swap_free_bytes": None,
            "swap_percent": None,
            "swap_rate_bytes_per_second": None,
        },
    )
    body = "\n".join(metrics._render_swap_metrics())
    assert "fusion_mlx_swap_rate_bytes_per_second" not in body
    assert body == ""
    logger.info("null swap sample -> family omitted (no fabricated zero)")


def test_histogram_quantile_bucket_search():
    from fusion_mlx.routes_internal.metrics import _histogram_quantile
    from fusion_mlx.server_metrics import _TTFT_BUCKETS, Histogram

    assert _histogram_quantile(None, 0.95) is None
    assert _histogram_quantile(Histogram(_TTFT_BUCKETS), 0.95) is None

    hist = Histogram(_TTFT_BUCKETS)
    for _ in range(100):
        hist.observe(0.2)
    assert _histogram_quantile(hist, 0.95) == pytest.approx(0.25)

    overflow = Histogram(_TTFT_BUCKETS)
    overflow.observe(100.0)
    assert _histogram_quantile(overflow, 0.95) == pytest.approx(100.0)
    logger.info("histogram quantile nearest-rank search verified")


def test_merged_histogram_quantile_across_models():
    from fusion_mlx.routes_internal.metrics import _merged_histogram_quantile
    from fusion_mlx.server_metrics import _TTFT_BUCKETS, Histogram

    fast = Histogram(_TTFT_BUCKETS)
    for _ in range(90):
        fast.observe(0.2)
    slow = Histogram(_TTFT_BUCKETS)
    for _ in range(10):
        slow.observe(0.8)
    merged = _merged_histogram_quantile({"fast": fast, "slow": slow}, 0.95)
    assert merged == pytest.approx(1.0)
    assert _merged_histogram_quantile({}, 0.95) is None

    mismatched = Histogram([0.1, float("inf")])
    mismatched.observe(0.05)
    assert _merged_histogram_quantile({"a": fast, "b": mismatched}, 0.95) is None
    logger.info("merged histogram quantile across models verified")


def test_p95_gauges_render_from_ttft_histogram():
    from fusion_mlx.routes_internal.metrics import render_prometheus_metrics
    from fusion_mlx.server_metrics import _TTFT_BUCKETS, Histogram, get_server_metrics

    sm = get_server_metrics()
    sm.clear_metrics()
    try:
        hist = Histogram(_TTFT_BUCKETS)
        for _ in range(100):
            hist.observe(0.2)
        sm._ttft_hist["m1"] = hist
        body = render_prometheus_metrics()
        assert 'fusion_mlx_ttft_p95_seconds{model="m1"} 0.25' in body
        assert "fusion_mlx_ttft_p95_seconds_all_models 0.25" in body
        assert "fusion_mlx_request_latency_p95_seconds" not in body
        logger.info("p95 gauges rendered from stubbed ttft histogram")
    finally:
        sm.clear_metrics()


def test_model_residency_metric_renders(monkeypatch):
    import fusion_mlx.server as srv
    from fusion_mlx.routes_internal.metrics import render_prometheus_metrics

    entry = SimpleNamespace(engine=object(), loaded_at=time.time() - 120.0)
    unknown = SimpleNamespace(engine=object(), loaded_at=0.0)
    pool = SimpleNamespace(_entries={"m1": entry, "m2": unknown})
    monkeypatch.setitem(srv._server_state, "engine_pool", pool)

    body = render_prometheus_metrics()
    line = next(
        (
            raw
            for raw in body.splitlines()
            if raw.startswith('fusion_mlx_model_resident_seconds{model="m1"}')
        ),
        None,
    )
    assert line is not None
    value = float(line.rsplit(" ", 1)[-1])
    assert 118.0 <= value <= 123.0
    assert 'fusion_mlx_model_resident_seconds{model="m2"}' not in body
    logger.info("model residency gauge rendered: %s", line)


def test_engine_pool_status_exposes_residency():
    from fusion_mlx.pool.engine_pool import EngineEntry, EnginePool

    entry = EngineEntry(
        model_id="m1",
        model_path="/tmp/m1",
        model_type="llm",
        engine_type="batched",
        estimated_size=10,
        engine=object(),
        loaded_at=time.time() - 30.0,
    )
    pool = SimpleNamespace(
        _entries={"m1": entry},
        _current_model_memory=10,
        _load_seconds_per_gb_ema=0.0,
        _load_time_observations=0,
        _suppress_ttl=False,
        _current_ceiling=lambda: 0,
        _global_idle_timeout_seconds=lambda: 0,
        _resolve_effective_ttl=lambda mid: None,
        _model_memory_plan_dict=lambda e: {},
    )
    row = EnginePool.get_status(pool)["models"][0]
    assert row["loaded_at"] is not None
    assert 29.0 <= row["resident_seconds"] <= 31.0
    logger.info("pool status residency fields: %s", row)


def test_engine_pool_register_entry_sets_loaded_at():
    from fusion_mlx.pool.engine_pool import EnginePool

    pool = SimpleNamespace(_entries={}, _current_model_memory=0)
    before = time.time()
    EnginePool._register_entry(
        pool, "m1", "/tmp/m1", "llm", "batched", engine=object(), estimated_size=10
    )
    entry = pool._entries["m1"]
    assert before <= entry.loaded_at <= time.time()
    logger.info("register_entry set loaded_at=%s", entry.loaded_at)


@pytest.fixture
def health_client(monkeypatch):
    from fusion_mlx.config import reset_config
    from fusion_mlx.routes_internal import health as health_route

    cfg = reset_config()
    cfg.model_name = "qwen3.5-4b"
    cfg.api_key = "test-secret"

    app = FastAPI()
    app.include_router(health_route.router)
    monkeypatch.setattr(
        health_route,
        "_mlx_memory_stats",
        lambda: {"active": 0, "cache": 0, "peak": 0},
    )
    yield SimpleNamespace(client=TestClient(app), monkeypatch=monkeypatch)
    reset_config()


def _set_health_pool(monkeypatch, models):
    import fusion_mlx.server as srv

    pool = SimpleNamespace(
        get_status=lambda: {"models": models},
        get_loaded_model_ids=lambda: [m["id"] for m in models if m.get("loaded")],
    )
    monkeypatch.setattr(srv, "_server_state", {"engine_pool": pool})


def test_v1_health_includes_swap_and_residency(health_client):
    from fusion_mlx.utils import swap_stats

    health_client.monkeypatch.setattr(
        swap_stats,
        "sample_swap",
        lambda: {
            "swap_total_bytes": 4096,
            "swap_used_bytes": 1024,
            "swap_free_bytes": 3072,
            "swap_percent": 25.0,
            "swap_rate_bytes_per_second": 64.0,
        },
    )
    _set_health_pool(
        health_client.monkeypatch,
        models=[
            {
                "id": "qwen3.5-4b",
                "loaded": True,
                "estimated_size": 2048,
                "loaded_at": 1_700_000_000.0,
                "resident_seconds": 42.5,
            }
        ],
    )
    body = health_client.client.get("/v1/health").json()
    mem = body["memory"]
    assert mem["swap_total_bytes"] == 4096
    assert mem["swap_used_bytes"] == 1024
    assert mem["swap_free_bytes"] == 3072
    assert mem["swap_percent"] == 25.0
    assert mem["swap_rate_bytes_per_second"] == 64.0
    assert mem["free_bytes"] > 0
    row = mem["per_model"][0]
    assert row["name"] == "qwen3.5-4b"
    assert row["bytes"] == 2048
    assert row["loaded_at"] == 1_700_000_000.0
    assert row["resident_seconds"] == 42.5
    logger.info("v1/health memory additions: %s", mem)


def test_v1_health_swap_and_residency_null_when_unknown(health_client):
    from fusion_mlx.utils import swap_stats

    health_client.monkeypatch.setattr(
        swap_stats,
        "sample_swap",
        lambda: {
            "swap_total_bytes": None,
            "swap_used_bytes": None,
            "swap_free_bytes": None,
            "swap_percent": None,
            "swap_rate_bytes_per_second": None,
        },
    )
    _set_health_pool(
        health_client.monkeypatch,
        models=[{"id": "m1", "loaded": True, "estimated_size": 10}],
    )
    mem = health_client.client.get("/v1/health").json()["memory"]
    assert mem["swap_used_bytes"] is None
    assert mem["swap_rate_bytes_per_second"] is None
    row = mem["per_model"][0]
    assert row["loaded_at"] is None
    assert row["resident_seconds"] is None
    logger.info("unknown swap/residency surfaces as null, not zero")
