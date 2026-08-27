# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

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
