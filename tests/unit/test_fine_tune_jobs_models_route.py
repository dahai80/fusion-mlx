# Route tests for /admin/api/fine-tune/jobs/models (#397).
# fusion-trainer enumerates trainable models via this path. The static
# route MUST be registered before /jobs/{job_id}, else the parameter
# route captures job_id=="models" and shadows it. These tests assert both
# the response shape and the routing precedence.
#
# Importers/callers: fusion-trainer calls GET /admin/api/fine-tune/jobs/models.
# Affected API: new GET route aliasing list_finetunable_models (no existing
# API changed). Engine pool injected via helpers._admin_getters["engine_pool"].
# Schema: JSON array of {model_id, model_type, model_path, loaded}.

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from fusion_mlx.admin import helpers
from fusion_mlx.admin.auth import require_admin
from fusion_mlx.admin.routes import router as admin_router


def _build_app(engine_pool=None):
    app = FastAPI()
    app.include_router(admin_router)
    app.dependency_overrides[require_admin] = lambda: True
    helpers._admin_getters["engine_pool"] = (
        (lambda: engine_pool) if engine_pool is not None else (lambda: None)
    )
    return app


class _Entry:
    def __init__(self, model_id, model_type="llm", path="/models/x", loaded=True):
        self.model_id = model_id
        self.model_type = model_type
        self.model_path = path
        self.engine = object() if loaded else None


class _FakePool:
    def __init__(self, entries):
        self._entries = entries


def test_jobs_models_route_returns_list():
    pool = _FakePool(
        {
            "m1": _Entry("m1", "llm", "/models/m1", True),
            "m2": _Entry("m2", "vlm", "/models/m2", False),
        }
    )
    app = _build_app(pool)
    client = TestClient(app)

    r = client.get("/admin/api/fine-tune/jobs/models")

    assert r.status_code == 200
    body = r.json()
    assert isinstance(body, list)
    assert len(body) == 2
    ids = {m["model_id"] for m in body}
    assert ids == {"m1", "m2"}
    m2 = next(m for m in body if m["model_id"] == "m2")
    assert m2["loaded"] is False
    assert m2["model_type"] == "vlm"


def test_jobs_models_route_not_shadowed_by_job_id():
    # The static /jobs/models path must win over /jobs/{job_id} even when
    # the parametric route exists. A 200 with a list body proves we hit
    # list_finetunable_models, not get_job(job_id="models").
    pool = _FakePool({"only": _Entry("only", "llm", "/models/only", True)})
    app = _build_app(pool)
    client = TestClient(app)

    r = client.get("/admin/api/fine-tune/jobs/models")

    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_jobs_models_route_aliases_legacy_models_path():
    # /api/fine-tune/models and /api/fine-tune/jobs/models must return the
    # same payload — the new path is an alias for the old one.
    pool = _FakePool({"a": _Entry("a", "llm", "/models/a", True)})
    app = _build_app(pool)
    client = TestClient(app)

    legacy = client.get("/admin/api/fine-tune/models").json()
    new = client.get("/admin/api/fine-tune/jobs/models").json()

    assert legacy == new


def test_jobs_models_route_503_without_pool():
    app = _build_app(None)
    client = TestClient(app)

    r = client.get("/admin/api/fine-tune/jobs/models")

    assert r.status_code == 503
