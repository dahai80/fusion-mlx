# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fusion_mlx.admin import settings as admin_settings
from fusion_mlx.admin.auth import require_admin


@pytest.fixture
def client(monkeypatch):
    async def _fake_require_admin():
        return True

    app = FastAPI()
    app.include_router(admin_settings.router)
    app.dependency_overrides[require_admin] = _fake_require_admin
    return TestClient(app)


class TestRestartServerRoute:
    def test_returns_503_when_unsupervised(self, client, monkeypatch):
        monkeypatch.delenv("OMLX_SUPERVISED", raising=False)

        r = client.post("/api/server/restart")
        assert r.status_code == 503
        body = r.json()
        assert "detail" in body
        assert "supervisor" in body["detail"].lower()

    def test_returns_202_when_supervised(self, client, monkeypatch):
        monkeypatch.setenv("OMLX_SUPERVISED", "menubar")

        with patch("fusion_mlx.admin.settings._schedule_self_terminate") as spy:
            r = client.post("/api/server/restart")

        assert r.status_code == 202, r.text
        body = r.json()
        assert body["status"] == "restarting"
        assert body["supervisor"] == "menubar"
        assert body["expected_downtime_seconds"] > 0
        spy.assert_called_once()
        (delay,), _kwargs = spy.call_args
        assert delay > 0

    def test_supervisor_label_round_trips(self, client, monkeypatch):
        monkeypatch.setenv("OMLX_SUPERVISED", "launchd")

        with patch("fusion_mlx.admin.settings._schedule_self_terminate"):
            r = client.post("/api/server/restart")

        assert r.status_code == 202
        assert r.json()["supervisor"] == "launchd"

    def test_unsupervised_does_not_schedule_termination(self, client, monkeypatch):
        monkeypatch.delenv("OMLX_SUPERVISED", raising=False)

        with patch("fusion_mlx.admin.settings._schedule_self_terminate") as spy:
            r = client.post("/api/server/restart")

        assert r.status_code == 503
        spy.assert_not_called()
