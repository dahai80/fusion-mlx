# Route tests for /admin/api/fine-tune/reward/score (#431).
# Minimal FastAPI app + dependency override for require_admin. The model
# load (mlx_utils.load) and reward scoring (reward.score_text) are both
# monkey-patched so no real model is loaded in CI.

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


def _patch_load_and_score(monkeypatch, rewards_out):
    # Stub mlx_utils.load (called inside the handler's _run) and
    # reward.score_text so no real model is touched.
    import fusion_mlx.training.reward as _reward

    captured = {}

    def fake_score_text(
        model, tokenizer, model_path, prompt, completions, adapter_path=None
    ):
        captured.update(
            prompt=prompt,
            completions=completions,
            adapter_path=adapter_path,
        )
        return list(rewards_out)

    monkeypatch.setattr(_reward, "score_text", fake_score_text)

    import mlx_lm.utils as _mlx_utils

    def fake_load(model_path, adapter_path=None):
        captured["load_model_path"] = model_path
        captured["load_adapter_path"] = adapter_path
        return ("fake-model", "fake-tokenizer")

    monkeypatch.setattr(_mlx_utils, "load", fake_load)
    return captured


def test_reward_score_happy_path(monkeypatch, tmp_path):
    fake_adapter = tmp_path / "m1" / "rm_adapter"
    fake_adapter.mkdir(parents=True)
    monkeypatch.setattr("fusion_mlx.training.service.ADAPTER_BASE_DIR", tmp_path)

    captured = _patch_load_and_score(monkeypatch, [6.7, -7.2])

    app = _build_app()
    client = TestClient(app)

    resp = client.post(
        "/admin/api/fine-tune/reward/score",
        json={
            "model_id": "m1",
            "adapter_name": "rm_adapter",
            "prompt": "What is 2+2?",
            "completions": ["4", "five"],
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["rewards"] == [6.7, -7.2]
    assert body["model_id"] == "m1"
    assert body["adapter_name"] == "rm_adapter"
    assert captured["prompt"] == "What is 2+2?"
    assert captured["completions"] == ["4", "five"]
    assert captured["load_adapter_path"] is not None


def test_reward_score_query_params(monkeypatch, tmp_path):
    # GRPO's callback protocol sends only {prompt, completions} in the body;
    # model_id/adapter_name must come from query params in the reward_endpoint URL.
    fake_adapter = tmp_path / "m1" / "rm_adapter"
    fake_adapter.mkdir(parents=True)
    monkeypatch.setattr("fusion_mlx.training.service.ADAPTER_BASE_DIR", tmp_path)

    captured = _patch_load_and_score(monkeypatch, [1.0])

    app = _build_app()
    client = TestClient(app)

    resp = client.post(
        "/admin/api/fine-tune/reward/score" "?model_id=m1&adapter_name=rm_adapter",
        json={"prompt": "Hi", "completions": ["x"]},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["rewards"] == [1.0]
    assert captured["prompt"] == "Hi"


def test_reward_score_missing_model_id():
    app = _build_app()
    client = TestClient(app)
    resp = client.post(
        "/admin/api/fine-tune/reward/score",
        json={"adapter_name": "a", "prompt": "Hi", "completions": ["x"]},
    )
    assert resp.status_code == 400
    assert "model_id" in resp.json()["detail"]


def test_reward_score_missing_adapter_name():
    app = _build_app()
    client = TestClient(app)
    resp = client.post(
        "/admin/api/fine-tune/reward/score",
        json={"model_id": "m1", "prompt": "Hi", "completions": ["x"]},
    )
    assert resp.status_code == 400
    assert "adapter_name" in resp.json()["detail"]


def test_reward_score_missing_completions():
    app = _build_app()
    client = TestClient(app)
    resp = client.post(
        "/admin/api/fine-tune/reward/score",
        json={"model_id": "m1", "adapter_name": "a", "prompt": "Hi"},
    )
    assert resp.status_code == 400
    assert "completions" in resp.json()["detail"]


def test_reward_score_adapter_not_found(monkeypatch, tmp_path):
    monkeypatch.setattr("fusion_mlx.training.service.ADAPTER_BASE_DIR", tmp_path)
    app = _build_app()
    client = TestClient(app)
    resp = client.post(
        "/admin/api/fine-tune/reward/score",
        json={
            "model_id": "m1",
            "adapter_name": "nope",
            "prompt": "Hi",
            "completions": ["x"],
        },
    )
    assert resp.status_code == 404
    assert "Adapter not found" in resp.json()["detail"]


def test_reward_score_not_a_reward_model(monkeypatch, tmp_path):
    # Adapter dir exists but score_text raises ValueError -> endpoint 400.
    fake_adapter = tmp_path / "m1" / "sft_only"
    fake_adapter.mkdir(parents=True)
    monkeypatch.setattr("fusion_mlx.training.service.ADAPTER_BASE_DIR", tmp_path)

    import fusion_mlx.training.reward as _reward

    def fake_score_text(
        model, tokenizer, model_path, prompt, completions, adapter_path=None
    ):
        raise ValueError("not a reward model")

    monkeypatch.setattr(_reward, "score_text", fake_score_text)

    import mlx_lm.utils as _mlx_utils

    monkeypatch.setattr(_mlx_utils, "load", lambda p, adapter_path=None: ("m", "t"))

    app = _build_app()
    client = TestClient(app)
    resp = client.post(
        "/admin/api/fine-tune/reward/score",
        json={
            "model_id": "m1",
            "adapter_name": "sft_only",
            "prompt": "Hi",
            "completions": ["x"],
        },
    )
    assert resp.status_code == 400
    assert "not a reward model" in resp.json()["detail"]
