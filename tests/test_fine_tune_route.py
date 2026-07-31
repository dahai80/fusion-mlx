# SPDX-License-Identifier: Apache-2.0
"""Tests for fine_tune_route — HTTP API endpoints.

callers: pytest
API: /admin/api/fine-tune/* routes (FastAPI TestClient)
schemas: FineTuneConfig, FineTuneJob, FineTuneAdapter (via service)
instruction: "继续实现二期和三期"
"""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from fusion_mlx.admin.fine_tune_route import _router, set_fine_tune_context
from fusion_mlx.admin.helpers import _admin_getters
from fusion_mlx.training.service import (
    FineTuneConfig,
    FineTuneJob,
    FineTuneService,
    JobStatus,
)


@pytest.fixture
def tmp_adapter_dir(tmp_path):
    with patch("fusion_mlx.training.service.ADAPTER_BASE_DIR", tmp_path):
        yield tmp_path


@pytest.fixture
def mock_service(tmp_adapter_dir):
    svc = FineTuneService()
    return svc


@pytest.fixture
def mock_pool():
    pool = MagicMock()
    pool.get_entry = MagicMock(return_value=None)
    pool.get_engine = AsyncMock(return_value=MagicMock())
    pool.unload_engine_async = AsyncMock()
    pool._entries = {}
    return pool


@pytest.fixture
def client(mock_service, mock_pool):
    from fastapi import FastAPI

    from fusion_mlx.admin.auth import require_admin

    app = FastAPI()
    app.include_router(_router)
    app.dependency_overrides[require_admin] = lambda: True

    set_fine_tune_context(mock_pool, mock_service)
    _admin_getters["engine_pool"] = lambda: mock_pool

    with TestClient(app) as tc:
        yield tc

    _admin_getters["engine_pool"] = None


class TestJobEndpoints:
    def test_create_job(self, client, mock_service, mock_pool):
        mock_pool.get_entry.return_value = MagicMock(
            model_type="llm", model_path="/tmp/model"
        )
        resp = client.post("/api/fine-tune/jobs", json={
            "model_id": "qwen3",
            "dataset": "/tmp/data",
            "config": {"lora_rank": 16},
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["model_id"] == "qwen3"
        assert data["status"] == "running"

    def test_create_job_missing_model_id(self, client):
        resp = client.post("/api/fine-tune/jobs", json={
            "dataset": "/tmp/data",
        })
        assert resp.status_code == 400

    def test_create_job_missing_dataset(self, client):
        resp = client.post("/api/fine-tune/jobs", json={
            "model_id": "qwen3",
        })
        assert resp.status_code == 400

    def test_create_job_model_not_found(self, client, mock_pool):
        mock_pool.get_entry.return_value = None
        resp = client.post("/api/fine-tune/jobs", json={
            "model_id": "nonexistent",
            "dataset": "/tmp/data",
        })
        assert resp.status_code == 404

    def test_list_jobs(self, client, mock_service):
        mock_service.create_job(model_id="qwen3", dataset="/tmp/data")
        resp = client.get("/api/fine-tune/jobs")
        assert resp.status_code == 200
        assert len(resp.json()) == 1

    def test_get_job(self, client, mock_service):
        job = mock_service.create_job(model_id="qwen3", dataset="/tmp/data")
        resp = client.get(f"/api/fine-tune/jobs/{job.job_id}")
        assert resp.status_code == 200
        assert resp.json()["job_id"] == job.job_id

    def test_get_job_not_found(self, client):
        resp = client.get("/api/fine-tune/jobs/nonexistent")
        assert resp.status_code == 404

    def test_cancel_job(self, client, mock_service):
        job = mock_service.create_job(model_id="qwen3", dataset="/tmp/data")
        resp = client.post(f"/api/fine-tune/jobs/{job.job_id}/cancel")
        assert resp.status_code == 200
        assert resp.json()["status"] == "cancelled"

    def test_cancel_job_not_found(self, client):
        resp = client.post("/api/fine-tune/jobs/nonexistent/cancel")
        assert resp.status_code == 404

    def test_delete_job(self, client, mock_service):
        job = mock_service.create_job(model_id="qwen3", dataset="/tmp/data")
        mock_service.cancel_job(job.job_id)
        resp = client.delete(f"/api/fine-tune/jobs/{job.job_id}")
        assert resp.status_code == 200

    def test_delete_job_not_found(self, client):
        resp = client.delete("/api/fine-tune/jobs/nonexistent")
        assert resp.status_code == 404


class TestAdapterEndpoints:
    def test_list_adapters(self, client, mock_service, tmp_adapter_dir):
        adapter_dir = tmp_adapter_dir / "qwen3" / "my-lora"
        adapter_dir.mkdir(parents=True)
        resp = client.get("/api/fine-tune/adapters")
        assert resp.status_code == 200
        assert len(resp.json()) == 1

    def test_list_adapters_filter_model(self, client, mock_service, tmp_adapter_dir):
        for m in ["qwen3", "llama"]:
            d = tmp_adapter_dir / m / "a1"
            d.mkdir(parents=True)
        resp = client.get("/api/fine-tune/adapters?model_id=qwen3")
        assert resp.status_code == 200
        assert len(resp.json()) == 1

    def test_delete_adapter(self, client, mock_service, tmp_adapter_dir):
        adapter_dir = tmp_adapter_dir / "qwen3" / "my-lora"
        adapter_dir.mkdir(parents=True)
        resp = client.request(
            "DELETE", "/api/fine-tune/adapters",
            content=json.dumps({"model_id": "qwen3", "adapter_name": "my-lora"}),
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 200

    def test_delete_adapter_missing_fields(self, client):
        resp = client.request(
            "DELETE", "/api/fine-tune/adapters",
            content=json.dumps({"model_id": "qwen3"}),
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 400


class TestAdapterServeEndpoints:
    def test_serve_adapter(self, client, mock_service, mock_pool, tmp_adapter_dir):
        adapter_dir = tmp_adapter_dir / "qwen3" / "my-lora"
        adapter_dir.mkdir(parents=True)
        (adapter_dir / "adapters.safetensors").write_bytes(b"\x00")
        resp = client.post("/api/fine-tune/adapters/qwen3/my-lora/serve")
        assert resp.status_code == 200
        data = resp.json()
        assert "served_model_id" in data

    def test_serve_adapter_not_found(self, client, mock_service, mock_pool):
        resp = client.post("/api/fine-tune/adapters/nonexistent/nope/serve")
        assert resp.status_code == 404

    def test_unload_adapter(self, client, mock_service, mock_pool, tmp_adapter_dir):
        adapter_dir = tmp_adapter_dir / "qwen3" / "my-lora"
        adapter_dir.mkdir(parents=True)
        (adapter_dir / "adapters.safetensors").write_bytes(b"\x00")
        mock_entry = MagicMock()
        mock_entry.engine = MagicMock()
        mock_pool.get_entry.return_value = mock_entry
        resp = client.post("/api/fine-tune/adapters/qwen3/my-lora/unload")
        assert resp.status_code == 200

    def test_unload_adapter_not_loaded(self, client, mock_service, mock_pool, tmp_adapter_dir):
        adapter_dir = tmp_adapter_dir / "qwen3" / "my-lora"
        adapter_dir.mkdir(parents=True)
        mock_entry = MagicMock()
        mock_entry.engine = None
        mock_pool.get_entry.return_value = mock_entry
        resp = client.post("/api/fine-tune/adapters/qwen3/my-lora/unload")
        assert resp.status_code == 404


class TestModelEndpoint:
    def test_list_models_no_pool(self, client, mock_pool):
        _saved = _admin_getters["engine_pool"]
        _admin_getters["engine_pool"] = None
        try:
            resp = client.get("/api/fine-tune/models")
            assert resp.status_code == 503
        finally:
            _admin_getters["engine_pool"] = _saved

    def test_list_models_with_entries(self, client, mock_pool):
        entry = MagicMock(model_type="llm", model_path="/tmp/qwen3")
        entry.engine = None
        mock_pool._entries = {"qwen3": entry}
        resp = client.get("/api/fine-tune/models")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["model_id"] == "qwen3"

    def test_list_models_filters_non_text(self, client, mock_pool):
        entry = MagicMock(model_type="diffusion", model_path="/tmp/sd")
        mock_pool._entries = {"sd3": entry}
        resp = client.get("/api/fine-tune/models")
        assert resp.status_code == 200
        assert resp.json() == []
