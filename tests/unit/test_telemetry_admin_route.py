# SPDX-License-Identifier: Apache-2.0
"""Tests for the telemetry admin route (#5) + endpoint-allowlist fix.

Covers:
- /api/telemetry/status, /queue, /activations, /alerts return the
  expected counters without raising (even when telemetry is disabled and
  no queue exists).
- the endpoint allowlist now maps the real image/video/audio routes
  instead of collapsing them to "other".
- alerts surface flush failures and drops; an all-healthy queue yields
  a single "ok" alert.
"""

from __future__ import annotations

import logging

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fusion_mlx.admin.telemetry_route import router as telemetry_router
from fusion_mlx.telemetry import emit

logger = logging.getLogger(__name__)


@pytest.fixture
def app(monkeypatch):
    application = FastAPI()
    application.include_router(telemetry_router)
    # Bypass admin auth via dependency override. Depends(require_admin)
    # captured the require_admin ref at route-def time, so we override
    # that exact callable rather than monkeypatching a module attr.
    from fusion_mlx.admin.auth import require_admin

    application.dependency_overrides[require_admin] = lambda: True
    return application


@pytest.fixture(autouse=True)
def _reset():
    emit._reset_for_tests()
    yield
    emit._reset_for_tests()


@pytest.fixture
def client(app):
    return TestClient(app)


def test_status_disabled_no_queue(client, monkeypatch):
    monkeypatch.delenv("FUSION_MLX_TELEMETRY", raising=False)
    resp = client.get("/api/telemetry/status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["enabled"] is False
    assert body["queue"]["active"] is False
    assert body["queue"]["pending"] == 0
    assert isinstance(body["activations"], dict)


def test_queue_snapshot_no_queue(client):
    resp = client.get("/api/telemetry/queue")
    assert resp.status_code == 200
    body = resp.json()
    assert body["active"] is False
    assert body["enqueued_total"] == 0


def test_activations_returns_all_kinds(client):
    resp = client.get("/api/telemetry/activations")
    assert resp.status_code == 200
    body = resp.json()
    from fusion_mlx.telemetry.activation_spec import ACTIVATION_KINDS

    assert set(body.keys()) == set(ACTIVATION_KINDS)
    assert all(isinstance(v, bool) for v in body.values())


def test_alerts_ok_when_healthy(client):
    resp = client.get("/api/telemetry/alerts")
    assert resp.status_code == 200
    alerts = resp.json()["alerts"]
    assert len(alerts) == 1
    assert alerts[0]["level"] == "ok"


def test_alerts_surface_flush_failures(client, monkeypatch):
    # Inject a fake queue with a failed flush to drive the alert.
    class _FakeQueue:
        def snapshot(self):
            return {
                "pending": 5,
                "enqueued_total": 100,
                "dropped_total": 3,
                "flushes_ok": 8,
                "flushes_failed": 2,
            }

    monkeypatch.setattr(emit, "_queue", _FakeQueue(), raising=False)
    resp = client.get("/api/telemetry/alerts")
    assert resp.status_code == 200
    codes = {a["code"] for a in resp.json()["alerts"]}
    assert "flush_failures" in codes
    assert "events_dropped" in codes


# ---------------------------------------------------------------- endpoint allowlist


def test_endpoint_allowlist_has_real_routes():
    from fusion_mlx.telemetry.emit import _ALLOWED_ENDPOINTS

    # The image route is /v1/images/generate (prefix + "/generate"), NOT
    # the OpenAI /v1/images/generations. The old allowlist had the wrong
    # path and silently collapsed every image request to "other".
    assert "/v1/images/generate" in _ALLOWED_ENDPOINTS
    assert "/v1/videos/generate" in _ALLOWED_ENDPOINTS
    assert "/v1/audio/transcriptions" in _ALLOWED_ENDPOINTS
    assert "/v1/audio/speech" in _ALLOWED_ENDPOINTS
    assert "/v1/audio/process" in _ALLOWED_ENDPOINTS
    assert "/v1/images/generations" not in _ALLOWED_ENDPOINTS


def test_normalize_endpoint_matches_real_route():
    from fusion_mlx.telemetry.emit import _normalize_endpoint

    assert _normalize_endpoint("/v1/images/generate") == "/v1/images/generate"
    assert _normalize_endpoint("/v1/videos/generate") == "/v1/videos/generate"
    assert _normalize_endpoint("/v1/audio/speech") == "/v1/audio/speech"
