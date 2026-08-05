# Route tests for the /admin/api/fine-tune/logprob endpoint (#363 Phase 1).
# Minimal FastAPI app with the fine-tune router + dependency override for
# require_admin so requests reach the handler. score_text is monkey-patched
# so no real model is loaded in CI.

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from fusion_mlx.admin.auth import require_admin
from fusion_mlx.admin.fine_tune_route import set_fine_tune_context
from fusion_mlx.admin.routes import router as admin_router


class _FakeEntry:
    def __init__(self, model_path="/tmp/fake-model", model_type="llm"):
        self.model_path = model_path
        self.model_type = model_type
        self.engine = None


class _FakePool:
    def __init__(self, model_path="/tmp/fake-model"):
        self._entries = {"m1": _FakeEntry(model_path=model_path)}

    def get_entry(self, model_id):
        return self._entries.get(model_id)

    def unload_if_idle_unpinned(self, model_id):
        return False


class _FakeService:
    def __init__(self, model_path="/tmp/fake-model"):
        self._pool = _FakePool(model_path=model_path)

    def set_engine_pool(self, pool):
        self._pool = pool

    def _resolve_model_path(self, model_id):
        entry = self._pool.get_entry(model_id)
        if entry is not None and hasattr(entry, "model_path"):
            return entry.model_path
        return model_id


def _build_app(service=None):
    app = FastAPI()
    set_fine_tune_context(_FakePool(), service or _FakeService())
    app.include_router(admin_router)
    app.dependency_overrides[require_admin] = lambda: True
    return app


def test_logprob_happy_path(monkeypatch):
    captured = {}

    def fake_score_text(model_path, prompt, completion, adapter_path=None):
        captured.update(
            model_path=model_path,
            prompt=prompt,
            completion=completion,
            adapter_path=adapter_path,
        )
        from fusion_mlx.training.logprob import LogprobResult

        return LogprobResult(logprob=-2.5, token_count=3, per_token=[-1.0, -1.0, -0.5])

    import fusion_mlx.training.logprob as _lp

    monkeypatch.setattr(_lp, "score_text", fake_score_text)

    app = _build_app()
    client = TestClient(app)

    resp = client.post(
        "/admin/api/fine-tune/logprob",
        json={"model_id": "m1", "prompt": "Hi", "completion": " there"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["logprob"] == -2.5
    assert body["token_count"] == 3
    assert body["per_token"] == [-1.0, -1.0, -0.5]
    assert captured["prompt"] == "Hi"
    assert captured["completion"] == " there"
    assert captured["adapter_path"] is None


def test_logprob_with_adapter(monkeypatch, tmp_path):
    fake_adapter = tmp_path / "m1" / "myadapter"
    fake_adapter.mkdir(parents=True)

    def fake_score_text(model_path, prompt, completion, adapter_path=None):
        from fusion_mlx.training.logprob import LogprobResult

        return LogprobResult(logprob=-1.0, token_count=1, per_token=[-1.0])

    import fusion_mlx.training.logprob as _lp

    monkeypatch.setattr(_lp, "score_text", fake_score_text)
    monkeypatch.setattr("fusion_mlx.training.service.ADAPTER_BASE_DIR", tmp_path)

    app = _build_app()
    client = TestClient(app)

    resp = client.post(
        "/admin/api/fine-tune/logprob",
        json={
            "model_id": "m1",
            "prompt": "Hi",
            "completion": "x",
            "adapter_name": "myadapter",
        },
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["logprob"] == -1.0


def test_logprob_missing_model_id():
    app = _build_app()
    client = TestClient(app)
    resp = client.post(
        "/admin/api/fine-tune/logprob",
        json={"prompt": "Hi", "completion": "x"},
    )
    assert resp.status_code == 400
    assert "model_id" in resp.json()["detail"]


def test_logprob_missing_completion():
    app = _build_app()
    client = TestClient(app)
    resp = client.post(
        "/admin/api/fine-tune/logprob",
        json={"model_id": "m1", "prompt": "Hi"},
    )
    assert resp.status_code == 400
    assert "completion" in resp.json()["detail"]


def test_logprob_adapter_not_found(monkeypatch, tmp_path):
    monkeypatch.setattr("fusion_mlx.training.service.ADAPTER_BASE_DIR", tmp_path)
    app = _build_app()
    client = TestClient(app)
    resp = client.post(
        "/admin/api/fine-tune/logprob",
        json={
            "model_id": "m1",
            "prompt": "Hi",
            "completion": "x",
            "adapter_name": "nope",
        },
    )
    assert resp.status_code == 404
    assert "Adapter not found" in resp.json()["detail"]
