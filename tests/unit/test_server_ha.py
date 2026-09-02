# SPDX-License-Identifier: Apache-2.0
"""Wire + unit coverage for the server-side HA slice (#754).

Four parts, all bounded:

* ``POST/DELETE /v1/drain`` admin routes toggle ``cfg.draining`` so a
  gateway/CLI doing health-driven failover can drain an instance (new
  requests refused via /healthz 503) and restore it after maintenance.
* ``get_instance_id()`` honors ``FUSION_INSTANCE_ID`` (operator-set) and
  falls back to a stable ``<hostname>:<pid>`` derivation so a gateway can
  distinguish replicas even without an explicit id.
* ``/health`` exposes ``version`` + ``instance_id`` + ``draining`` + a
  ``status`` that flips to ``"draining"`` so a gateway's routing rules
  have a single rich field to branch on.
* ``SessionTracker`` JSON snapshot persistence (``FUSION_SESSION_STATE_DIR``)
  survives a failover/restart so cumulative per-session token usage is
  rehydrated rather than lost.

The drain routes do NOT touch the engine pool — they only flip a runtime
flag — so no mock engine is needed for the wire tests (unlike the
cancel/cache-clear routes in test_internal_route_auth.py).
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fusion_mlx.config import get_config
from fusion_mlx.instance import get_instance_id

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ha_client(monkeypatch):
    """Mount the admin + probe routers on a bare FastAPI app.

    Drain routes mutate ``cfg.draining`` and read ``get_instance_id()``;
    no engine pool is required. ``FUSION_ALLOW_ANONYMOUS=true`` is set
    by the autouse conftest fixture, so no credential is needed (mirrors
    test_internal_route_auth.py's no-credential path).
    """
    from fusion_mlx.routes_internal.health import admin_router, probe_router

    cfg = get_config()
    prev_draining = cfg.draining
    cfg.draining = False

    app = FastAPI()
    app.include_router(probe_router)
    app.include_router(admin_router)
    client = TestClient(app)
    try:
        yield client, cfg
    finally:
        cfg.draining = prev_draining


# ---------------------------------------------------------------------------
# (A) drain toggle
# ---------------------------------------------------------------------------


class TestDrainToggle:
    def test_post_drain_sets_flag_and_returns_draining(self, ha_client):
        client, cfg = ha_client
        assert cfg.draining is False
        r = client.post("/v1/drain")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["status"] == "draining"
        assert body["instance_id"] == get_instance_id()
        assert cfg.draining is True

    def test_delete_drain_clears_flag_and_returns_healthy(self, ha_client):
        client, cfg = ha_client
        cfg.draining = True
        r = client.delete("/v1/drain")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["status"] == "healthy"
        assert body["instance_id"] == get_instance_id()
        assert cfg.draining is False

    def test_drain_is_idempotent(self, ha_client):
        client, _ = ha_client
        r1 = client.post("/v1/drain")
        assert r1.status_code == 200
        r2 = client.post("/v1/drain")
        assert r2.status_code == 200
        assert r2.json()["status"] == "draining"

    def test_undrain_is_idempotent(self, ha_client):
        client, _ = ha_client
        r1 = client.delete("/v1/drain")
        assert r1.status_code == 200
        r2 = client.delete("/v1/drain")
        assert r2.status_code == 200
        assert r2.json()["status"] == "healthy"

    def test_healthz_reflects_503_when_draining(self, ha_client):
        # /healthz route handler (fall-through, not the ASGI fast-path) must
        # return 503 draining once the flag flips. The fast-path parity is
        # pinned in test_probe_fastpath.py; this covers the route handler.
        client, cfg = ha_client
        assert client.get("/healthz").status_code == 200
        client.post("/v1/drain")
        assert cfg.draining is True
        r = client.get("/healthz")
        assert r.status_code == 503, r.text
        body = r.json()
        assert body["status"] == "draining"
        assert body["ready"] is False


# ---------------------------------------------------------------------------
# (B) instance identity
# ---------------------------------------------------------------------------


class TestInstanceId:
    def test_env_var_honored(self, monkeypatch):
        monkeypatch.setenv("FUSION_INSTANCE_ID", "mlx-node-1")
        assert get_instance_id() == "mlx-node-1"

    def test_env_var_stripped(self, monkeypatch):
        monkeypatch.setenv("FUSION_INSTANCE_ID", "  mlx-node-2  ")
        assert get_instance_id() == "mlx-node-2"

    def test_derived_fallback_when_unset(self, monkeypatch):
        monkeypatch.delenv("FUSION_INSTANCE_ID", raising=False)
        iid = get_instance_id()
        assert iid and ":" in iid
        host, _, pid = iid.rpartition(":")
        assert host
        assert pid.isdigit()

    def test_derived_fallback_stable_across_calls(self, monkeypatch):
        monkeypatch.delenv("FUSION_INSTANCE_ID", raising=False)
        assert get_instance_id() == get_instance_id()

    def test_empty_env_falls_back_to_derived(self, monkeypatch):
        monkeypatch.setenv("FUSION_INSTANCE_ID", "   ")
        iid = get_instance_id()
        assert ":" in iid  # derived shape, not the empty string


# ---------------------------------------------------------------------------
# (C) /health richness
# ---------------------------------------------------------------------------


class TestHealthRichness:
    def test_health_has_version_instance_draining(self, ha_client):
        client, _ = ha_client
        r = client.get("/health")
        assert r.status_code == 200, r.text
        body = r.json()
        assert isinstance(body.get("version"), str)
        assert body.get("instance_id") == get_instance_id()
        assert body["draining"] is False
        assert body["status"] == "healthy"

    def test_health_status_flips_to_draining(self, ha_client):
        client, _ = ha_client
        client.post("/v1/drain")
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "draining"
        assert body["draining"] is True
        assert body["ready"] is False


# ---------------------------------------------------------------------------
# (D) session persistence
# ---------------------------------------------------------------------------


class TestSessionPersistence:
    def test_record_without_state_dir_no_snapshot(self, tmp_path):
        from fusion_mlx.sessions.tracker import SessionTracker

        tracker = SessionTracker(state_dir=None)
        tracker.record("sess-1", prompt_tokens=10, completion_tokens=5)
        assert not (tmp_path / "sessions.json").exists()

    def test_snapshot_round_trip_rehydrates_stats(self, tmp_path):
        from fusion_mlx.sessions.tracker import SessionTracker

        # First tracker: record, force a snapshot, then drop it.
        t1 = SessionTracker(state_dir=str(tmp_path))
        t1.record("sess-1", prompt_tokens=100, completion_tokens=40)
        t1.set_max_context("sess-1", 8192)
        t1._maybe_snapshot()
        # Bypass the debounce so the file is written now.
        t1._last_snapshot = 0.0
        t1._maybe_snapshot()
        assert (tmp_path / "sessions.json").exists()

        # Second tracker on the same state dir rehydrates.
        t2 = SessionTracker(state_dir=str(tmp_path))
        got = t2.get("sess-1")
        assert got is not None
        assert got.prompt_tokens == 100
        assert got.completion_tokens == 40
        assert got.total_tokens == 140
        assert got.request_count == 1
        assert got.max_context_tokens == 8192

    def test_rehydrate_skips_empty_session_id(self, tmp_path):
        import json

        from fusion_mlx.sessions.tracker import SessionTracker

        path = tmp_path / "sessions.json"
        path.write_text(
            json.dumps(
                [
                    {"principal": "default", "session_id": "", "prompt_tokens": 5},
                    {"principal": "default", "session_id": "real", "prompt_tokens": 7},
                ]
            )
        )
        tracker = SessionTracker(state_dir=str(tmp_path))
        assert tracker.get("real") is not None
        assert tracker.get("real").prompt_tokens == 7
        assert tracker.get("") is None

    def test_rehydrate_corrupt_file_is_fail_visible(self, tmp_path, caplog):
        from fusion_mlx.sessions.tracker import SessionTracker

        (tmp_path / "sessions.json").write_text("{not valid json")
        with caplog.at_level("WARNING"):
            tracker = SessionTracker(state_dir=str(tmp_path))
        # Corrupt file must not crash construction; tracker is usable.
        tracker.record("fresh", prompt_tokens=1)
        assert tracker.get("fresh") is not None
        assert any("rehydrate failed" in rec.message for rec in caplog.records)

    def test_snapshot_is_atomic_via_tmp_replace(self, tmp_path):
        from fusion_mlx.sessions.tracker import SessionTracker

        t1 = SessionTracker(state_dir=str(tmp_path))
        t1.record("s", prompt_tokens=3)
        t1._last_snapshot = 0.0
        t1._maybe_snapshot()
        assert (tmp_path / "sessions.json").exists()
        # The .tmp sidecar must not linger after a successful snapshot.
        assert not (tmp_path / "sessions.json.tmp").exists()
