# SPDX-License-Identifier: Apache-2.0
"""Wire-level tests for GET /v1/health read-only memory/OOM endpoint (#564).

Mounts only the health router (management-gated) with a stubbed engine
pool + patched MLX memory stats, so the suite stays at unit-test speed
(no model load). Pins the contract fusion-code /doctor + MLX OOM
auto-recovery depend on.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture
def health_client(monkeypatch):
    from fusion_mlx.config import reset_config
    from fusion_mlx.routes_internal import health as health_route

    cfg = reset_config()
    cfg.model_name = "qwen3.5-4b"
    cfg.api_key = "test-secret"

    app = FastAPI()
    app.include_router(health_route.router)
    client = TestClient(app)
    ns = SimpleNamespace(client=client, cfg=cfg, monkeypatch=monkeypatch)
    yield ns
    reset_config()


def _set_mlx(monkeypatch, *, active, cache, peak):
    from fusion_mlx.routes_internal import health as health_route

    monkeypatch.setattr(
        health_route,
        "_mlx_memory_stats",
        lambda: {"active": active, "cache": cache, "peak": peak},
    )


def _set_pool(monkeypatch, *, models):
    from fusion_mlx import server as srv

    fake_pool = SimpleNamespace(
        get_status=lambda: {"models": models},
        get_loaded_model_ids=lambda: [m["id"] for m in models if m.get("loaded")],
    )
    monkeypatch.setattr(srv, "_server_state", {"engine_pool": fake_pool})


# ---------------------------------------------------------------------------
# Basic shape
# ---------------------------------------------------------------------------


def test_v1_health_returns_200_and_full_shape(health_client):
    _set_mlx(health_client.monkeypatch, active=512 * 1024 * 1024, cache=0, peak=0)
    _set_pool(
        health_client.monkeypatch,
        models=[{"id": "qwen3.5-4b", "loaded": True, "estimated_size": 2 * 1024**3}],
    )
    r = health_client.client.get("/v1/health")
    assert r.status_code == 200, r.text
    body = r.json()

    assert body["version"]
    assert isinstance(body["uptime_seconds"], int)
    assert body["active_models"] == ["qwen3.5-4b"]
    assert body["oom_risk"] in {"none", "low", "high", "imminent"}

    mem = body["memory"]
    assert mem["mlx_active_bytes"] == 512 * 1024 * 1024
    assert mem["mlx_peak_bytes"] == 0
    assert mem["total_bytes"] > 0
    assert mem["rss_bytes"] > 0
    assert mem["per_model"] == [{"name": "qwen3.5-4b", "bytes": 2 * 1024**3}]


def test_v1_health_no_pool_no_models(health_client):
    _set_mlx(health_client.monkeypatch, active=None, cache=None, peak=None)
    from fusion_mlx import server as srv

    health_client.monkeypatch.setattr(srv, "_server_state", {})
    r = health_client.client.get("/v1/health")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["active_models"] == []
    assert body["memory"]["per_model"] == []
    # MLX absent -> None fields, not 0 (distinct from "measured zero").
    assert body["memory"]["mlx_active_bytes"] is None
    assert body["memory"]["mlx_peak_bytes"] is None


# ---------------------------------------------------------------------------
# OOM risk classifier (deterministic, Rule 5)
# ---------------------------------------------------------------------------


def test_oom_risk_none_when_plenty_free():
    from fusion_mlx.routes_internal.health import _oom_risk

    assert _oom_risk(0.80, 0.10) == "none"


def test_oom_risk_low_band():
    from fusion_mlx.routes_internal.health import _oom_risk

    assert _oom_risk(0.25, 0.20) == "low"
    assert _oom_risk(0.50, 0.55) == "low"


def test_oom_risk_high_when_free_low():
    from fusion_mlx.routes_internal.health import _oom_risk

    assert _oom_risk(0.10, 0.20) == "high"
    assert _oom_risk(0.50, 0.80) == "high"


def test_oom_risk_imminent_below_5pct_free():
    from fusion_mlx.routes_internal.health import _oom_risk

    assert _oom_risk(0.03, 0.10) == "imminent"


def test_oom_risk_imminent_mlx_peak_over_90pct():
    from fusion_mlx.routes_internal.health import _oom_risk

    assert _oom_risk(0.80, 0.95) == "imminent"


def test_oom_risk_mlx_peak_none_falls_back_to_free_only():
    from fusion_mlx.routes_internal.health import _oom_risk

    # peak unknown -> only available_ratio drives it.
    assert _oom_risk(0.80, None) == "none"
    assert _oom_risk(0.10, None) == "high"
    assert _oom_risk(0.03, None) == "imminent"


def test_v1_health_status_maps_from_risk(health_client):
    _set_mlx(health_client.monkeypatch, active=0, cache=0, peak=0)
    _set_pool(health_client.monkeypatch, models=[])
    # free ratio high -> risk none -> status ok
    r = health_client.client.get("/v1/health")
    assert r.json()["status"] == "ok"


# ---------------------------------------------------------------------------
# Auth (management-gated, #344 parity)
# ---------------------------------------------------------------------------


def test_v1_health_rejects_unauthenticated_when_api_key_set(health_client, monkeypatch):
    monkeypatch.delenv("FUSION_ALLOW_ANONYMOUS", raising=False)
    assert health_client.cfg.api_key == "test-secret"
    r = health_client.client.get("/v1/health")
    assert r.status_code == 401


if __name__ == "__main__":  # pragma: no cover — convenience only
    pytest.main([__file__, "-v"])
