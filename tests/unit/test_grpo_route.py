# Route tests for /admin/api/fine-tune/grpo/* endpoints (#363 Phase 2).
# Minimal FastAPI app with the admin router + require_admin override.
# GRPOService._execute_grpo is never reached (no start_processing trigger
# without a running event loop), so no real model loads in CI.

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from fusion_mlx.admin.auth import require_admin
from fusion_mlx.admin.fine_tune_route import set_grpo_context
from fusion_mlx.admin.routes import router as admin_router
from fusion_mlx.training.grpo_service import GRPOService


def _build_app():
    app = FastAPI()
    svc = GRPOService()
    # Neutralize queue processing so route handlers never spin up a real
    # model load in a background task during sync TestClient requests.
    svc.start_processing = lambda *a, **kw: None
    svc._process_queue = lambda *a, **kw: None
    set_grpo_context(None, svc)
    app.include_router(admin_router)
    app.dependency_overrides[require_admin] = lambda: True
    return app, svc


def test_grpo_create_job_returns_id():
    app, svc = _build_app()
    client = TestClient(app)
    resp = client.post(
        "/admin/api/fine-tune/grpo/jobs",
        json={
            "model_id": "m1",
            "prompts": ["Hello", "Goodbye"],
            "config": {"group_size": 2, "iters": 1},
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "job_id" in body
    assert body["model_id"] == "m1"
    assert body["prompts"] == ["Hello", "Goodbye"]
    assert body["config"]["group_size"] == 2
    assert body["status"] in ("queued", "running", "completed", "cancelled")


def test_grpo_create_missing_model_id():
    app, _ = _build_app()
    client = TestClient(app)
    resp = client.post(
        "/admin/api/fine-tune/grpo/jobs",
        json={"prompts": ["x"]},
    )
    assert resp.status_code == 400
    assert "model_id" in resp.json()["detail"]


def test_grpo_create_missing_prompts():
    app, _ = _build_app()
    client = TestClient(app)
    resp = client.post(
        "/admin/api/fine-tune/grpo/jobs",
        json={"model_id": "m1"},
    )
    assert resp.status_code == 400
    assert "prompts" in resp.json()["detail"]


def test_grpo_create_invalid_config():
    app, _ = _build_app()
    client = TestClient(app)
    resp = client.post(
        "/admin/api/fine-tune/grpo/jobs",
        json={"model_id": "m1", "prompts": ["x"], "config": {"unknown_field": 1}},
    )
    assert resp.status_code == 400
    assert "Invalid config" in resp.json()["detail"]


def test_grpo_list_jobs():
    app, svc = _build_app()
    client = TestClient(app)
    svc.create_job(model_id="m1", prompts=["p1"], adapter_name="a1")
    resp = client.get("/admin/api/fine-tune/grpo/jobs")
    assert resp.status_code == 200
    jobs = resp.json()
    assert any(j["adapter_name"] == "a1" for j in jobs)


def test_grpo_get_job_not_found():
    app, _ = _build_app()
    client = TestClient(app)
    resp = client.get("/admin/api/fine-tune/grpo/jobs/nonexistent")
    assert resp.status_code == 404


def test_grpo_cancel_queued_job():
    app, svc = _build_app()
    client = TestClient(app)
    job = svc.create_job(model_id="m1", prompts=["p1"], adapter_name="q1")
    resp = client.post(f"/admin/api/fine-tune/grpo/jobs/{job.job_id}/cancel")
    assert resp.status_code == 200
    assert resp.json()["status"] == "cancelled"


def test_grpo_delete_job():
    app, svc = _build_app()
    client = TestClient(app)
    job = svc.create_job(model_id="m1", prompts=["p1"], adapter_name="d1")
    resp = client.delete(f"/admin/api/fine-tune/grpo/jobs/{job.job_id}")
    assert resp.status_code == 200
    assert resp.json()["status"] == "deleted"


def test_grpo_cancel_not_found():
    app, _ = _build_app()
    client = TestClient(app)
    resp = client.post("/admin/api/fine-tune/grpo/jobs/nope/cancel")
    assert resp.status_code == 404
