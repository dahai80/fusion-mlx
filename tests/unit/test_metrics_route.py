# SPDX-License-Identifier: Apache-2.0
"""Wire-level tests for the Prometheus ``/metrics`` endpoint.

The route renders from process-level state — the ServerMetrics singleton,
the EnginePool stashed in ``server._server_state``, and module counter
snapshots — not from a per-request engine handle. These tests inject a
fake pool into ``_server_state`` and drive ServerMetrics directly, keeping
the suite at unit-test speed with no model download.

Pins the ``fusion_mlx_*`` exposition contract from the #0907 ops-audit
rewrite: the old ``rapid_mlx_*`` engine-stats series (requests_processed,
steps_executed, engine-stats metal bytes, prefix evictions/tokens_saved)
were replaced by server-level counters, pool gauges, and per-engine
scheduler-cache series.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture
def metrics_client():
    """FastAPI TestClient mounting only the metrics router.

    Using a per-test app instance keeps the global ``ServerConfig``
    singleton in a known shape and avoids interfering with other tests
    that share the same process. Also resets the module-level sticky
    counter accumulator and clears the ServerMetrics singleton so each
    test sees fresh counters.
    """
    from fusion_mlx.config import reset_config
    from fusion_mlx.routes_internal.metrics import _reset_accumulator_for_tests, router
    from fusion_mlx.server_metrics import get_server_metrics

    cfg = reset_config()
    cfg.model_name = "qwen3.5-4b"
    cfg.api_key = "test-secret"  # #344: /metrics now gated; functional tests pass via FUSION_ALLOW_ANONYMOUS (conftest).
    _reset_accumulator_for_tests()
    sm = get_server_metrics()
    sm.clear_metrics()

    app = FastAPI()
    app.include_router(router)
    yield SimpleNamespace(client=TestClient(app), cfg=cfg, sm=sm)
    reset_config()
    _reset_accumulator_for_tests()
    sm.clear_metrics()


def _fake_pool(entries: dict[str, SimpleNamespace] | None = None):
    # Minimal EnginePool stand-in: the renderers read _entries,
    # model_count, loaded_model_count, current_model_memory and
    # process_memory_enforcer — nothing else.
    return SimpleNamespace(
        _entries=entries or {},
        model_count=len(entries or {}),
        loaded_model_count=len(entries or {}),
        current_model_memory=0,
        process_memory_enforcer=None,
    )


def _stub_engine(
    get_stats: Any = None, scheduler: SimpleNamespace | None = None
) -> SimpleNamespace:
    return SimpleNamespace(
        get_stats=get_stats or (lambda: {}),
        scheduler=scheduler,
    )


def _entry(engine: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(engine=engine)


def _set_pool(monkeypatch, pool) -> None:
    import fusion_mlx.server as srv

    monkeypatch.setitem(srv._server_state, "engine_pool", pool)


# ---------------------------------------------------------------------------
# Basic protocol surface
# ---------------------------------------------------------------------------


def test_metrics_returns_200_and_text_content_type(metrics_client):
    """200 OK with Prometheus text exposition content-type."""
    resp = metrics_client.client.get("/metrics")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/plain")
    # version=0.0.4 is what every Prometheus 2.x scraper expects.
    assert "version=0.0.4" in resp.headers["content-type"]


def test_metrics_no_engine_pool_still_returns_200(metrics_client, monkeypatch):
    # No pool at all (server mid-boot) → 200 with build_info, never 500.
    # Prometheus drops a scrape target after a single non-2xx; if /metrics
    # 500'd between restarts the dashboard would lose continuity.
    _set_pool(monkeypatch, None)
    resp = metrics_client.client.get("/metrics")
    assert resp.status_code == 200
    body = resp.text
    assert "fusion_mlx_build_info" in body
    # Queue gauges are server-level and always present — 0 with no pool,
    # not absent (a missing series reads as "no data" on a dashboard).
    assert "fusion_mlx_requests_running 0" in body
    assert "fusion_mlx_requests_waiting 0" in body


def test_metrics_engine_get_stats_raises_falls_back_to_build_info(
    metrics_client, monkeypatch
):
    # A half-initialized engine whose get_stats() raises must not take the
    # scrape down: the queue renderer logs+skips, the rest still renders.
    def _explode() -> dict[str, Any]:
        raise RuntimeError("engine half-initialized")

    _set_pool(monkeypatch, _fake_pool({"m1": _entry(_stub_engine(get_stats=_explode))}))
    resp = metrics_client.client.get("/metrics")
    assert resp.status_code == 200
    assert "fusion_mlx_build_info" in resp.text
    # The broken engine aborts the queue render mid-loop (logged+skipped),
    # so the queue series are absent for that pool — not fake zeros.
    assert "fusion_mlx_requests_running" not in resp.text


# #344: /metrics is gated behind verify_management_access. With an API key
# configured, a request carrying no credentials and no X-Fusion-Route header
# must be rejected (401). Prometheus scrapers reach /metrics through the
# gateway, which injects X-Fusion-Route.
def test_metrics_rejects_unauthenticated_when_api_key_set(metrics_client, monkeypatch):
    monkeypatch.delenv("FUSION_ALLOW_ANONYMOUS", raising=False)
    assert metrics_client.cfg.api_key == "test-secret"
    resp = metrics_client.client.get("/metrics")  # no Authorization header
    assert resp.status_code == 401


# #344: X-Fusion-Route is provenance, not auth. With an api_key configured
# (test-secret), a caller that omits the key but sets the header must be 401.
def test_metrics_rejects_x_fusion_route_without_key(metrics_client, monkeypatch):
    monkeypatch.delenv("FUSION_ALLOW_ANONYMOUS", raising=False)
    resp = metrics_client.client.get("/metrics", headers={"X-Fusion-Route": "gateway"})
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Exposition contents
# ---------------------------------------------------------------------------

_QUEUE_STATS = {
    "num_running": 3,
    "num_waiting": 2,
    "num_prefilling": 1,
}


def test_metrics_exposes_all_expected_series(metrics_client, monkeypatch):
    # Every always-present series carries HELP + TYPE banners. Pool-
    # dependent gauges are pinned with a fake pool attached; optional-
    # import renderers (spec-decode extras, moe cache) are deliberately
    # not pinned — their presence depends on installed extras.
    _set_pool(monkeypatch, _fake_pool({"m1": _entry(_stub_engine())}))
    body = metrics_client.client.get("/metrics").text

    expected_names = [
        "fusion_mlx_build_info",
        "fusion_mlx_requests_total",
        "fusion_mlx_requests_failed_total",
        "fusion_mlx_requests_cancelled_total",
        "fusion_mlx_prompt_tokens_total",
        "fusion_mlx_completion_tokens_total",
        "fusion_mlx_requests_running",
        "fusion_mlx_requests_waiting",
        "fusion_mlx_requests_prefilling",
        "fusion_mlx_uptime_seconds",
        "fusion_mlx_kv_cache_dtype",
        "fusion_mlx_response_format_strict_total",
        "fusion_mlx_response_format_strict_violations_total",
        "fusion_mlx_models_discovered",
        "fusion_mlx_models_loaded",
        "fusion_mlx_model_memory_bytes",
        "fusion_mlx_ane_resident_bytes_total",
        "fusion_mlx_metal_wired_bytes",
        "fusion_mlx_iosurface_shared_bytes",
    ]
    for name in expected_names:
        assert f"# HELP {name}" in body, f"missing HELP for {name}"
        assert f"# TYPE {name}" in body, f"missing TYPE for {name}"


def test_metrics_values_match_server_metrics(metrics_client, monkeypatch):
    # Counters come from the ServerMetrics singleton; queue gauges from
    # per-engine get_stats(). Values render verbatim — never rescaled.
    sm = metrics_client.sm
    sm.total_requests = 17
    sm.total_tokens_prompt = 1234
    sm.total_tokens_generated = 5678
    sm.failed_requests = 2
    _set_pool(
        monkeypatch,
        _fake_pool({"m1": _entry(_stub_engine(get_stats=lambda: dict(_QUEUE_STATS)))}),
    )
    body = metrics_client.client.get("/metrics").text

    assert "fusion_mlx_requests_total 17" in body
    assert "fusion_mlx_requests_failed_total 2" in body
    assert "fusion_mlx_prompt_tokens_total 1234" in body
    assert "fusion_mlx_completion_tokens_total 5678" in body
    assert "fusion_mlx_requests_running 3" in body
    assert "fusion_mlx_requests_waiting 2" in body
    assert "fusion_mlx_requests_prefilling 1" in body


def test_metrics_build_info_labels_carry_version(metrics_client):
    # build_info exposes the version label; the sample value is 1.
    metrics_client.cfg.engine = None
    body = metrics_client.client.get("/metrics").text

    sample_line = next(
        line for line in body.splitlines() if line.startswith("fusion_mlx_build_info{")
    )
    assert 'version="' in sample_line
    assert sample_line.endswith(" 1")


def test_metrics_tolerates_engine_stat_none_fields(metrics_client, monkeypatch):
    # Engine stats carrying None in fields the renderer no longer reads
    # (legacy metal_*_gb keys) must not break the render; the num_* keys
    # the queue gauges do read still pass through verbatim.
    stats = {
        "num_running": 3,
        "num_waiting": 2,
        "metal_active_memory_gb": None,
        "metal_peak_memory_gb": None,
        "metal_cache_memory_gb": None,
    }
    _set_pool(
        monkeypatch, _fake_pool({"m1": _entry(_stub_engine(get_stats=lambda: stats))})
    )

    body = metrics_client.client.get("/metrics").text
    assert "fusion_mlx_requests_running 3" in body
    assert "fusion_mlx_requests_waiting 2" in body


def test_metrics_prefix_cache_series_from_ssd_cache_stats(metrics_client, monkeypatch):
    # OP-19: per-engine prefix-cache series render from
    # scheduler.get_ssd_cache_stats()["prefix_cache"], labeled by engine.
    sched = SimpleNamespace(
        get_ssd_cache_stats=lambda: {
            "prefix_cache": {"hits": 11, "misses": 7, "hit_rate": 0.6111}
        }
    )
    _set_pool(monkeypatch, _fake_pool({"m1": _entry(_stub_engine(scheduler=sched))}))
    body = metrics_client.client.get("/metrics").text
    assert 'fusion_mlx_prefix_cache_hits_total{engine="m1"} 11' in body
    assert 'fusion_mlx_prefix_cache_misses_total{engine="m1"} 7' in body
    assert 'fusion_mlx_prefix_cache_hit_rate{engine="m1"} 0.6111' in body


def test_metrics_queue_gauges_prefer_lockfree_snapshot(metrics_client, monkeypatch):
    """/metrics must read the engine's lock-free stats snapshot instead of
    calling get_stats(): the latter walks scheduler.get_stats() → paged-cache
    block/free-queue locks, which the executor thread holds for the whole of
    a long prefill/decode step — rendering blocked >10s and 503'd whenever a
    big generation was active."""
    snapshot = {"num_running": 5, "num_waiting": 1, "num_prefilling": 2}
    get_stats_calls = []

    def _blocking_get_stats():
        get_stats_calls.append(1)
        return {"num_running": 99}

    engine = SimpleNamespace(
        get_stats_snapshot=lambda: snapshot,
        get_stats=_blocking_get_stats,
    )
    _set_pool(monkeypatch, _fake_pool({"m1": _entry(engine)}))

    body = metrics_client.client.get("/metrics").text
    assert "fusion_mlx_requests_running 5" in body
    assert "fusion_mlx_requests_waiting 1" in body
    assert "fusion_mlx_requests_prefilling 2" in body
    assert get_stats_calls == [], (
        "renderer must not call get_stats() when a lock-free "
        "get_stats_snapshot() is available"
    )


def test_metrics_queue_gauges_fall_back_to_get_stats(metrics_client, monkeypatch):
    """Engines without get_stats_snapshot (other modality engines) still
    render via the legacy get_stats() path."""
    engine = SimpleNamespace(get_stats=lambda: {"num_running": 4})
    _set_pool(monkeypatch, _fake_pool({"m1": _entry(engine)}))

    body = metrics_client.client.get("/metrics").text
    assert "fusion_mlx_requests_running 4" in body


def test_metrics_omits_prefix_cache_series_when_scheduler_lacks_stats(
    metrics_client, monkeypatch
):
    # A scheduler without get_ssd_cache_stats (cache feature off) → the
    # prefix-cache series are silently omitted, not fake zeros implying a
    # working-but-ineffective cache. Server-level series still render.
    _set_pool(monkeypatch, _fake_pool({"m1": _entry(_stub_engine())}))
    body = metrics_client.client.get("/metrics").text

    assert "fusion_mlx_prefix_cache_hits_total" not in body
    assert "fusion_mlx_requests_total 0" in body


def test_metrics_paged_kv_gauges_from_block_aware_cache(metrics_client, monkeypatch):
    # R7-M1 successor: cache capacity/usage gauges now come from the
    # paged KV cache (scheduler.block_aware_cache.paged_cache.get_stats),
    # rendered as block counts + utilization instead of the old
    # prefix_cache cap/current byte gauges.
    pcs = SimpleNamespace(
        total_blocks=100,
        allocated_blocks=40,
        free_blocks=60,
        shared_blocks=5,
        total_tokens_cached=1234,
        cow_copies=7,
    )
    sched = SimpleNamespace(
        block_aware_cache=SimpleNamespace(
            paged_cache=SimpleNamespace(get_stats=lambda: pcs)
        )
    )
    _set_pool(monkeypatch, _fake_pool({"m1": _entry(_stub_engine(scheduler=sched))}))
    body = metrics_client.client.get("/metrics").text

    assert "# TYPE fusion_mlx_paged_kv_total_blocks gauge" in body
    assert "fusion_mlx_paged_kv_total_blocks 100" in body
    assert "fusion_mlx_paged_kv_allocated_blocks 40" in body
    assert "fusion_mlx_paged_kv_free_blocks 60" in body
    assert "fusion_mlx_paged_kv_shared_blocks 5" in body
    assert "fusion_mlx_paged_kv_tokens_cached_total 1234" in body
    assert "fusion_mlx_paged_kv_cow_copies_total 7" in body
    assert "fusion_mlx_paged_kv_utilization 0.4" in body


def test_metrics_paged_kv_gauges_default_zero_when_fields_missing(
    metrics_client, monkeypatch
):
    # A paged-cache stats object lacking the attr fields renders explicit
    # zeros rather than dropping the series — dashboards prefer a known-0
    # gauge to a missing one for a pool that is present but idle.
    sched = SimpleNamespace(
        block_aware_cache=SimpleNamespace(
            paged_cache=SimpleNamespace(get_stats=lambda: SimpleNamespace())
        )
    )
    _set_pool(monkeypatch, _fake_pool({"m1": _entry(_stub_engine(scheduler=sched))}))
    body = metrics_client.client.get("/metrics").text

    assert "fusion_mlx_paged_kv_total_blocks 0" in body
    assert "fusion_mlx_paged_kv_allocated_blocks 0" in body
    assert "fusion_mlx_paged_kv_utilization 0.0" in body


def test_metrics_escapes_quotes_in_engine_label(metrics_client, monkeypatch):
    # Label values containing ``"`` / ``\`` are escaped per the
    # exposition spec — a model-id with a quote would break the scraper's
    # parser otherwise. The escape function is the only sensitive part of
    # the renderer and deserves a regression test.
    sched = SimpleNamespace(
        get_ssd_cache_stats=lambda: {
            "prefix_cache": {"hits": 1, "misses": 0, "hit_rate": 1.0}
        }
    )
    _set_pool(
        monkeypatch, _fake_pool({'foo"bar\\baz': _entry(_stub_engine(scheduler=sched))})
    )
    body = metrics_client.client.get("/metrics").text
    sample = next(
        line
        for line in body.splitlines()
        if line.startswith("fusion_mlx_prefix_cache_hits_total{")
    )
    assert 'engine="foo\\"bar\\\\baz"' in sample


def test_metrics_body_ends_with_newline(metrics_client):
    """Prometheus text exposition requires a trailing newline."""
    metrics_client.cfg.engine = _stub_engine()
    body = metrics_client.client.get("/metrics").text
    assert body.endswith("\n")


# ---------------------------------------------------------------------------
# Sticky-counter accumulator (retained unit contract)
# ---------------------------------------------------------------------------


def test_sticky_accumulator_unit_behavior():
    """Direct unit test of the accumulator — covers branches the route test misses.

    The accumulator is currently unused by the render path (the #0907
    rewrite dropped the engine-stats counters it served), but the class
    is exported for any future counter that needs reset-tolerant
    monotonicity, so its contract stays pinned.
    """
    from fusion_mlx.routes_internal.metrics import _StickyCounterAccumulator

    acc = _StickyCounterAccumulator()

    # First advance establishes baseline=0, last_raw=raw.
    assert acc.advance("k1", 5) == 5
    assert acc.advance("k2", 100) == 100

    # Same value twice in a row — no-op.
    assert acc.advance("k1", 5) == 5

    # Monotonic increase.
    assert acc.advance("k1", 8) == 8

    # Reset — raw drops; exposed must NOT drop.
    assert acc.advance("k1", 0) == 8  # 8 (baseline) + 0 (raw)
    assert acc.advance("k1", 2) == 10  # 8 (baseline) + 2 (raw)

    # The other key is untouched by k1's reset.
    assert acc.advance("k2", 105) == 105

    # Defensive: negative raw is floored to 0.
    assert acc.advance("k1", -1) == 10  # 10 (baseline so far) + 0
    # After previous step state is (last_raw=0, baseline=10); next reset
    # check needs raw < 0 (impossible after flooring) so baseline stays.
    assert acc.advance("k1", 4) == 14


def test_metrics_prefix_cache_hits_render_verbatim_across_scrapes(
    metrics_client, monkeypatch
):
    # Successive scrapes reflect the live cache stats verbatim — the
    # route is a pure pass-through of scheduler.get_ssd_cache_stats().
    pc_stats = {"hits": 10, "misses": 5, "hit_rate": 0.6667}
    sched = SimpleNamespace(get_ssd_cache_stats=lambda: {"prefix_cache": pc_stats})
    _set_pool(monkeypatch, _fake_pool({"m1": _entry(_stub_engine(scheduler=sched))}))

    def _hits_value(body: str) -> int:
        for line in body.splitlines():
            if line.startswith(
                "fusion_mlx_prefix_cache_hits_total{"
            ) and not line.startswith("#"):
                return int(line.rsplit(" ", 1)[1])
        raise AssertionError("hits_total line not found")

    body = metrics_client.client.get("/metrics").text
    assert _hits_value(body) == 10

    pc_stats["hits"] = 15
    body = metrics_client.client.get("/metrics").text
    assert _hits_value(body) == 15


# ---------------------------------------------------------------------------
# Format-compliance smoke test (codex r1 LOW: tests should parse the body)
# ---------------------------------------------------------------------------


def test_metrics_output_parses_cleanly_via_prometheus_client(
    metrics_client, monkeypatch
):
    """The emitted body parses without errors via the official Prometheus parser.

    Catches whole classes of regression that the hand-picked substring
    assertions don't: malformed escapes, locale-dependent float formatting,
    bad HELP/TYPE ordering, etc. The dependency is dev-only (see
    ``pyproject.toml`` ``dev`` extras); the runtime route still hand-rolls
    the format.
    """
    from prometheus_client.parser import text_string_to_metric_families

    sm = metrics_client.sm
    sm.total_requests = 17
    _set_pool(
        monkeypatch,
        _fake_pool({"m1": _entry(_stub_engine(get_stats=lambda: dict(_QUEUE_STATS)))}),
    )
    body = metrics_client.client.get("/metrics").text

    families = list(text_string_to_metric_families(body))
    assert len(families) > 0, "parser found zero metric families"

    # Every emitted family should have at least one sample.
    for fam in families:
        assert len(fam.samples) > 0, f"family {fam.name} has no samples"

    # Verify a few specific families round-trip the values we set.
    by_name = {fam.name: fam for fam in families}
    assert "fusion_mlx_requests" in by_name  # parser strips _total
    assert by_name["fusion_mlx_requests"].samples[0].value == 17

    # build_info gauge round-trips labels.
    assert "fusion_mlx_build_info" in by_name
    build_sample = by_name["fusion_mlx_build_info"].samples[0]
    assert build_sample.labels["version"]
    assert build_sample.value == 1.0


def test_metrics_output_parses_cleanly_with_escaped_label(metrics_client, monkeypatch):
    """Output with escaped quote/backslash in a label still parses cleanly.

    Regression guard for ``_escape_label_value`` — a broken escape would
    let the body pass substring assertions but choke the real parser.
    """
    from prometheus_client.parser import text_string_to_metric_families

    sched = SimpleNamespace(
        get_ssd_cache_stats=lambda: {
            "prefix_cache": {"hits": 1, "misses": 0, "hit_rate": 1.0}
        }
    )
    _set_pool(
        monkeypatch, _fake_pool({'foo"bar\\baz': _entry(_stub_engine(scheduler=sched))})
    )
    body = metrics_client.client.get("/metrics").text

    families = list(text_string_to_metric_families(body))
    by_name = {fam.name: fam for fam in families}
    assert "fusion_mlx_prefix_cache_hits" in by_name
    # The parser should reverse the escapes to recover the original string.
    assert (
        by_name["fusion_mlx_prefix_cache_hits"].samples[0].labels["engine"]
        == 'foo"bar\\baz'
    )


def test_metrics_output_parses_cleanly_when_engine_missing(metrics_client, monkeypatch):
    """Even the no-pool fallback path parses cleanly.

    The fallback emits build_info plus the H-06 response_format counters
    (which live in process-local module state, not on the engine) so
    dashboards never flip to "no data" between restarts. Strict-mode
    counters always present even at zero.
    """
    from prometheus_client.parser import text_string_to_metric_families

    _set_pool(monkeypatch, None)
    body = metrics_client.client.get("/metrics").text

    families = list(text_string_to_metric_families(body))
    by_name = {fam.name: fam for fam in families}
    # build_info + H-06 response_format counters.
    assert "fusion_mlx_build_info" in by_name
    assert "fusion_mlx_response_format_strict" in by_name
    assert "fusion_mlx_response_format_strict_violations" in by_name
