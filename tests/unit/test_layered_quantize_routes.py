# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fusion_mlx.admin.auth import require_admin
from fusion_mlx.api import layered_quantize_routes
from fusion_mlx.api.convert_routes import router as convert_router


@pytest.fixture()
def app():
    a = FastAPI()

    async def _fake_require_admin():
        return True

    a.dependency_overrides[require_admin] = _fake_require_admin
    a.include_router(convert_router)
    a.include_router(layered_quantize_routes.router)
    return a


def test_layered_jobs_list_route_mounted(app):
    client = TestClient(app)
    resp = client.get("/v1/quantize/layered/jobs")
    assert resp.status_code == 200


def test_layered_job_detail_route_mounted(app):
    client = TestClient(app)
    resp = client.get("/v1/quantize/layered/jobs/nonexistent-job-id")
    assert resp.status_code == 404  # route exists, job doesn't


def _wait_layered(client, job_id, timeout=5.0):
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        r = client.get(f"/v1/quantize/layered/jobs/{job_id}")
        assert r.status_code == 200, r.text
        job = r.json()
        if job["status"] in ("completed", "failed"):
            return job
        time.sleep(0.02)
    raise AssertionError(f"layered job {job_id} did not finish within {timeout}s")


def test_layered_quantize_terminal_status_is_completed(app, monkeypatch):
    def _fake_ok(model, **kwargs):
        return kwargs["mlx_path"]

    monkeypatch.setattr("fusion_mlx.cli_convert._run_convert", _fake_ok)
    client = TestClient(app)
    resp = client.post(
        "/v1/quantize/layered",
        json={
            "model": "test/repo",
            "default_bits": 4,
            "layer_rules": [{"pattern": "lm_head", "bits": 8}],
        },
    )
    assert resp.status_code == 200, resp.text
    job_id = resp.json()["job_id"]
    job = _wait_layered(client, job_id)
    assert job["status"] == "completed"
    assert job["kind"] == "layered-quantize"
