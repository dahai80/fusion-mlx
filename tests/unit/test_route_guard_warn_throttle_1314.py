# SPDX-License-Identifier: Apache-2.0
# #1314: warn-only route-guard log must be throttled per (host, path).

import logging

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fusion_mlx.middleware import route_guard

logger = logging.getLogger(__name__)


@pytest.fixture(autouse=True)
def _reset_warn_throttle(monkeypatch):
    monkeypatch.setattr(route_guard, "_warn_last_at", {})
    monkeypatch.delenv("FUSION_ROUTE_ENFORCE", raising=False)
    monkeypatch.delenv("FUSION_ROUTE_TOKEN", raising=False)
    monkeypatch.delenv("FUSION_TENANT_ISOLATION", raising=False)
    monkeypatch.setenv("FUSION_ROUTE_WARN_ONLY", "true")
    yield


class _Clock:
    def __init__(self, now):
        self.now = now

    def monotonic(self):
        return self.now


def _guard_app() -> FastAPI:
    app = FastAPI()

    @app.api_route("/{p:path}", methods=["GET", "POST"])
    async def _catch_all(p: str):
        return {"ok": True}

    route_guard.install_route_guard_middleware(app)
    return app


def _warnings(caplog):
    return [
        record
        for record in caplog.records
        if record.levelno == logging.WARNING
        and "missing X-Fusion-Route (warn-only)" in record.getMessage()
    ]


def test_repeated_scrapes_log_once(monkeypatch, caplog):
    client = TestClient(_guard_app())
    with caplog.at_level(logging.WARNING, logger="fusion_mlx.middleware.route_guard"):
        for _ in range(5):
            assert client.get("/metrics").status_code == 200
    assert len(_warnings(caplog)) == 1
    logger.info("5 scrapes of /metrics produced 1 route-guard warning")


def test_different_paths_warn_independently(caplog):
    client = TestClient(_guard_app())
    with caplog.at_level(logging.WARNING, logger="fusion_mlx.middleware.route_guard"):
        client.get("/metrics")
        client.get("/v1/models")
    assert len(_warnings(caplog)) == 2
    logger.info("distinct paths each keep their own warning")


def test_warning_repeats_after_interval(monkeypatch):
    clock = _Clock(1000.0)
    monkeypatch.setattr(route_guard, "time", clock)
    assert route_guard._warn_due("127.0.0.1", "/metrics") is True
    assert route_guard._warn_due("127.0.0.1", "/metrics") is False
    clock.now += route_guard._WARN_THROTTLE_S
    assert route_guard._warn_due("127.0.0.1", "/metrics") is True
    logger.info("throttle window reopens after the interval")


def test_throttle_map_is_bounded(monkeypatch):
    clock = _Clock(500.0)
    monkeypatch.setattr(route_guard, "time", clock)
    monkeypatch.setattr(route_guard, "_WARN_THROTTLE_MAX_KEYS", 4)
    for i in range(4):
        assert route_guard._warn_due("h", f"/p{i}") is True
    clock.now += route_guard._WARN_THROTTLE_S
    assert route_guard._warn_due("h", "/fresh") is True
    assert len(route_guard._warn_last_at) <= 4
    logger.info("throttle map stayed bounded: %d keys", len(route_guard._warn_last_at))
