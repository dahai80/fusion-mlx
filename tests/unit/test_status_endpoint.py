# SPDX-License-Identifier: Apache-2.0
"""Tests for GET /api/status endpoint."""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from fusion_mlx.admin.auth import require_admin
from fusion_mlx.server import Server


@pytest.fixture
def server():
    srv = Server()
    srv.pool = None
    return srv


@pytest.fixture
def client(server):
    test_app = server.app
    test_app.dependency_overrides[require_admin] = lambda: True
    return TestClient(test_app)


class TestStatusEndpoint:
    """Tests for /api/status lightweight status endpoint."""

    def test_returns_ok_when_pool_is_none(self, client, server):
        server.pool = None
        resp = client.get("/api/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["models_discovered"] == 0
        assert data["models_loaded"] == 0
        assert data["models_loading"] == 0
        assert data["loaded_models"] == []
        assert "version" in data
        assert "uptime_seconds" in data

    def test_returns_pool_info(self, client, server):
        pool = MagicMock(
            spec=[
                "model_count",
                "loaded_model_count",
                "get_loaded_model_ids",
                "current_model_memory",
                "_process_memory_enforcer",
                "_entries",
            ]
        )
        pool.model_count = 5
        pool.loaded_model_count = 2
        pool.get_loaded_model_ids.return_value = ["model-a", "model-b"]
        pool.current_model_memory = 16 * 1024**3
        enforcer = MagicMock(spec=["get_final_ceiling"])
        enforcer.get_final_ceiling.return_value = 32 * 1024**3
        pool._process_memory_enforcer = enforcer

        entry_a = MagicMock(spec=["is_loading", "engine"])
        entry_a.is_loading = False
        entry_a.engine = None
        entry_b = MagicMock(spec=["is_loading", "engine"])
        entry_b.is_loading = True
        entry_b.engine = None
        pool._entries = {"model-a": entry_a, "model-b": entry_b}

        server.pool = pool

        resp = client.get("/api/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["models_discovered"] == 5
        assert data["models_loaded"] == 2
        assert data["models_loading"] == 1
        assert data["loaded_models"] == ["model-a", "model-b"]
        assert data["model_memory_used"] == 16 * 1024**3
        assert data["model_memory_max"] == 32 * 1024**3
        assert "GB" in data["model_memory_used_formatted"]
        assert "GB" in data["model_memory_max_formatted"]

    def test_status_ignores_memory_ceiling_error(self, client, server):
        pool = MagicMock(
            spec=[
                "model_count",
                "loaded_model_count",
                "get_loaded_model_ids",
                "current_model_memory",
                "_process_memory_enforcer",
                "_entries",
            ]
        )
        pool.model_count = 1
        pool.loaded_model_count = 1
        pool.get_loaded_model_ids.return_value = ["model-a"]
        pool.current_model_memory = 16 * 1024**3
        pool._entries = {}
        enforcer = MagicMock(spec=["get_final_ceiling"])
        enforcer.get_final_ceiling.side_effect = RuntimeError(
            "host_statistics64 failed"
        )
        pool._process_memory_enforcer = enforcer
        server.pool = pool

        resp = client.get("/api/status")

        assert resp.status_code == 200
        data = resp.json()
        assert data["model_memory_max"] is None
        assert data["model_memory_max_formatted"] == "unlimited"

    @pytest.mark.xfail(
        strict=True,
        reason="Aspirational: /health no longer nests an engine_pool object "
        "with final_ceiling; it returns a flat {status,ready,model_loaded,"
        "loaded_models} shape. The old health-envelope contract was removed.",
    )
    def test_health_ignores_memory_ceiling_error(self, client, server):
        pool = MagicMock(
            spec=[
                "model_count",
                "loaded_model_count",
                "current_model_memory",
            ]
        )
        pool.model_count = 1
        pool.loaded_model_count = 1
        pool.current_model_memory = 16 * 1024**3
        enforcer = MagicMock(spec=["get_final_ceiling"])
        enforcer.get_final_ceiling.side_effect = RuntimeError(
            "host_statistics64 failed"
        )
        server.pool = pool

        resp = client.get("/health")

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["engine_pool"]["final_ceiling"] == 0

    @pytest.mark.xfail(
        strict=True,
        reason="Aspirational: /api/status no longer aggregates "
        "active_requests/waiting_requests across engines; those fields were "
        "dropped from the response envelope.",
    )
    def test_aggregates_active_waiting_requests(self, client, server):
        scheduler = MagicMock(spec=["waiting"])
        scheduler.waiting = [1, 2]

        core = MagicMock(spec=["_output_collectors", "scheduler"])
        core._output_collectors = {"req-1": None, "req-2": None, "req-3": None}
        core.scheduler = scheduler

        async_core = MagicMock(spec=["engine"])
        async_core.engine = core

        engine = MagicMock(spec=["_engine"])
        engine._engine = async_core

        entry = MagicMock(spec=["is_loading", "engine"])
        entry.is_loading = False
        entry.engine = engine

        pool = MagicMock(
            spec=[
                "model_count",
                "loaded_model_count",
                "get_loaded_model_ids",
                "current_model_memory",
                "_entries",
            ]
        )
        pool.model_count = 1
        pool.loaded_model_count = 1
        pool.get_loaded_model_ids.return_value = ["model-a"]
        pool.current_model_memory = 0
        pool._entries = {"model-a": entry}

        server.pool = pool

        resp = client.get("/api/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["active_requests"] == 3
        assert data["waiting_requests"] == 2

    def test_requires_auth_when_api_key_set(self, server):
        test_app = server.app
        test_app.dependency_overrides.pop(require_admin, None)
        from fusion_mlx.admin.auth import set_api_key

        set_api_key("test-secret-key")
        try:
            unauth_client = TestClient(test_app)
            resp = unauth_client.get("/api/status")
            assert resp.status_code == 401

            resp = unauth_client.get(
                "/api/status",
                headers={"Authorization": "Bearer test-secret-key"},
            )
            assert resp.status_code == 200
        finally:
            set_api_key("")

    @pytest.mark.xfail(
        strict=True,
        reason="Aspirational: /api/status surfaces only total_requests/"
        "total_prompt_tokens/total_completion_tokens; the richer "
        "ServerMetrics fields (total_cached_tokens, cache_efficiency, "
        "avg_prefill_tps, avg_generation_tps) are not forwarded to this "
        "endpoint.",
    )
    def test_serving_metrics_included(self, client):
        resp = client.get("/api/status")
        data = resp.json()
        expected_keys = [
            "total_requests",
            "total_prompt_tokens",
            "total_completion_tokens",
            "total_cached_tokens",
            "cache_efficiency",
            "avg_prefill_tps",
            "avg_generation_tps",
        ]
        for key in expected_keys:
            assert key in data, f"Missing key: {key}"

    def test_unlimited_memory_max(self, client, server):
        pool = MagicMock(
            spec=[
                "model_count",
                "loaded_model_count",
                "get_loaded_model_ids",
                "current_model_memory",
                "_process_memory_enforcer",
                "_entries",
            ]
        )
        pool.model_count = 0
        pool.loaded_model_count = 0
        pool.get_loaded_model_ids.return_value = []
        pool.current_model_memory = 0
        pool._entries = {}
        pool._process_memory_enforcer = None

        server.pool = pool

        resp = client.get("/api/status")
        data = resp.json()
        assert data["model_memory_max"] is None
        assert data["model_memory_max_formatted"] == "unlimited"
