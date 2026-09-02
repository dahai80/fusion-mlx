# Route tests for /admin/api/fine-tune/grpo/* endpoints (#363 Phase 2).
# Minimal FastAPI app with the admin router + require_admin override.
# GRPOService._execute_grpo is never reached (no start_processing trigger
# without a running event loop), so no real model loads in CI.

from __future__ import annotations

import pytest
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


def test_grpo_execute_grpo_rebinds_generation_stream():
    # Regression for #751: _execute_grpo runs in a worker thread, but
    # mlx_lm.generate's module-level generation_stream was captured on the main
    # thread at import. generate_step does `with mx.stream(generation_stream)`
    # explicitly, so the old default-stream-only fix (#430/PR#432) did not
    # cover it. _execute_grpo must rebind the module global to a fresh
    # worker-thread-local stream (and restore it after).
    try:
        import importlib

        import mlx.core as mx

        # `import mlx_lm.generate as x` binds the re-exported generate()
        # FUNCTION (mlx_lm/__init__ shadow), not the submodule -- resolve the
        # module via importlib so .generation_stream is reachable.
        mlx_gen = importlib.import_module("mlx_lm.generate")
    except Exception:
        pytest.skip("mlx not available in this CI environment")

    app, svc = _build_app()
    job = svc.create_job(model_id="m1", prompts=["p"], adapter_name="s1")

    original_stream = mlx_gen.generation_stream
    captured = {}

    def fake_run_grpo(j):
        # Inside the worker context the module global must be rebound to a
        # different (worker-thread-local) stream -- not the import-time one.
        captured["during"] = mlx_gen.generation_stream
        captured["default"] = mx.default_stream(mx.default_device())

    svc._run_grpo = fake_run_grpo
    svc._execute_grpo(job)

    # Rebound to a fresh stream distinct from the import-time one.
    assert captured["during"] is not None
    assert captured["during"] is not original_stream
    # Restored after the call (finally block), even though no exception.
    assert mlx_gen.generation_stream is original_stream


def test_grpo_execute_grpo_restores_stream_on_error():
    # The finally branch: if _run_grpo raises, the module global must still be
    # restored so a failed GRPO job does not corrupt generation_stream for
    # later normal serve-side generation.
    try:
        import importlib

        mlx_gen = importlib.import_module("mlx_lm.generate")
    except Exception:
        pytest.skip("mlx not available in this CI environment")

    app, svc = _build_app()
    job = svc.create_job(model_id="m1", prompts=["p"], adapter_name="s2")
    original_stream = mlx_gen.generation_stream

    def fake_run_grpo(j):
        raise RuntimeError("boom inside train loop")

    svc._run_grpo = fake_run_grpo
    with pytest.raises(RuntimeError, match="boom"):
        svc._execute_grpo(job)
    assert mlx_gen.generation_stream is original_stream
