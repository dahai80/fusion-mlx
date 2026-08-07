# Route + unit tests for DPO/ORPO preference training (#399).
# Importers/callers: pytest test runner (no production importers).
# Affected API: tests new /admin/api/fine-tune/dpo/jobs + /orpo/jobs routes
#   + DPOConfig / DPOTrainer DPO+ORPO loss math.
# Data schemas: DPOJob, DPOConfig (method dpo|orpo), DPOStepResult.
# User verbatim instruction: "启动3个功能issue的修复落地"
# Mirrors test_grpo_route.py: minimal FastAPI app with admin router +
# require_admin override; DPOService queue processing neutralized so no
# real model loads in CI. Loss-math unit tests use a stub model.

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fusion_mlx.admin.auth import require_admin
from fusion_mlx.admin.fine_tune_route import set_dpo_context
from fusion_mlx.admin.routes import router as admin_router
from fusion_mlx.training.dpo import DPOConfig, DPOTrainer
from fusion_mlx.training.dpo_service import DPOJob, DPOService


def _build_app():
    app = FastAPI()
    svc = DPOService()
    svc.start_processing = lambda *a, **kw: None
    svc._process_queue = lambda *a, **kw: None
    set_dpo_context(None, svc)
    app.include_router(admin_router)
    app.dependency_overrides[require_admin] = lambda: True
    return app, svc


_PAIRS = [
    {"prompt": "Q?", "chosen": "good", "rejected": "bad"},
    {"prompt": "Q2?", "chosen": "better", "rejected": "worse"},
]


# =============================================================================
# Route tests
# =============================================================================


def test_dpo_create_job_returns_id():
    app, svc = _build_app()
    client = TestClient(app)
    resp = client.post(
        "/admin/api/fine-tune/dpo/jobs",
        json={"model_id": "m1", "preference_pairs": _PAIRS, "config": {"iters": 1}},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "job_id" in body
    assert body["model_id"] == "m1"
    assert body["preference_pairs"] == _PAIRS
    assert body["config"]["method"] == "dpo"
    assert body["config"]["iters"] == 1


def test_orpo_create_job_forces_method():
    app, svc = _build_app()
    client = TestClient(app)
    resp = client.post(
        "/admin/api/fine-tune/orpo/jobs",
        json={
            "model_id": "m1",
            "preference_pairs": _PAIRS,
            "config": {"iters": 1, "lambda_odds": 0.5},
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["config"]["method"] == "orpo"
    assert body["config"]["lambda_odds"] == 0.5


def test_dpo_create_missing_model_id():
    app, _ = _build_app()
    client = TestClient(app)
    resp = client.post(
        "/admin/api/fine-tune/dpo/jobs",
        json={"preference_pairs": _PAIRS},
    )
    assert resp.status_code == 400
    assert "model_id" in resp.json()["detail"]


def test_dpo_create_missing_pairs():
    app, _ = _build_app()
    client = TestClient(app)
    resp = client.post(
        "/admin/api/fine-tune/dpo/jobs",
        json={"model_id": "m1"},
    )
    assert resp.status_code == 400
    assert "preference_pairs" in resp.json()["detail"]


def test_dpo_create_malformed_pair():
    app, _ = _build_app()
    client = TestClient(app)
    resp = client.post(
        "/admin/api/fine-tune/dpo/jobs",
        json={"model_id": "m1", "preference_pairs": [{"prompt": "x"}]},
    )
    assert resp.status_code == 400
    assert "preference_pairs[0]" in resp.json()["detail"]


def test_dpo_create_invalid_config():
    app, _ = _build_app()
    client = TestClient(app)
    resp = client.post(
        "/admin/api/fine-tune/dpo/jobs",
        json={
            "model_id": "m1",
            "preference_pairs": _PAIRS,
            "config": {"unknown_field": 1},
        },
    )
    assert resp.status_code == 400
    assert "Invalid config" in resp.json()["detail"]


def test_dpo_list_jobs():
    app, svc = _build_app()
    client = TestClient(app)
    svc.create_job(model_id="m1", preference_pairs=_PAIRS, adapter_name="a1")
    resp = client.get("/admin/api/fine-tune/dpo/jobs")
    assert resp.status_code == 200
    jobs = resp.json()
    assert any(j["adapter_name"] == "a1" for j in jobs)


def test_dpo_get_job_not_found():
    app, _ = _build_app()
    client = TestClient(app)
    resp = client.get("/admin/api/fine-tune/dpo/jobs/nonexistent")
    assert resp.status_code == 404


def test_dpo_cancel_queued_job():
    app, svc = _build_app()
    client = TestClient(app)
    job = svc.create_job(model_id="m1", preference_pairs=_PAIRS, adapter_name="q1")
    resp = client.post(f"/admin/api/fine-tune/dpo/jobs/{job.job_id}/cancel")
    assert resp.status_code == 200
    assert resp.json()["status"] == "cancelled"


def test_dpo_delete_job():
    app, svc = _build_app()
    client = TestClient(app)
    job = svc.create_job(model_id="m1", preference_pairs=_PAIRS, adapter_name="d1")
    resp = client.delete(f"/admin/api/fine-tune/dpo/jobs/{job.job_id}")
    assert resp.status_code == 200
    assert resp.json()["status"] == "deleted"


def test_dpo_cancel_not_found():
    app, _ = _build_app()
    client = TestClient(app)
    resp = client.post("/admin/api/fine-tune/dpo/jobs/nope/cancel")
    assert resp.status_code == 404


# =============================================================================
# Service / config unit tests
# =============================================================================


def test_dpo_config_defaults():
    cfg = DPOConfig()
    assert cfg.method == "dpo"
    assert cfg.iters == 50
    assert cfg.beta == 0.1
    assert cfg.lambda_odds == 1.0


def test_dpo_job_to_dict_roundtrip():
    cfg = DPOConfig(method="orpo", iters=3)
    job = DPOJob(
        job_id="abc",
        model_id="m1",
        preference_pairs=_PAIRS,
        config=cfg,
        adapter_name="orpo-abc",
        adapter_path="/tmp/orpo-abc",
    )
    d = job.to_dict()
    assert d["job_id"] == "abc"
    assert d["config"]["method"] == "orpo"
    assert d["config"]["iters"] == 3
    assert d["preference_pairs"] == _PAIRS
    assert d["status"] == "queued"


class _StubTokenizer:
    def encode(self, text):
        return mx.array([ord(c) for c in text][:4])


class _StubModel(nn.Module):
    # Minimal differentiable model: projects token ids to logits over vocab.
    def __init__(self, vocab=32):
        super().__init__()
        self.embed = nn.Embedding(vocab, 4)
        self.head = nn.Linear(4, vocab)
        self._vocab = vocab

    def __call__(self, ids):
        x = self.embed(ids)
        return self.head(x)


def test_dpo_loss_runs_and_returns_metrics():
    # DPO loss with a stub model + precomputed ref logprobs. Verifies the loss
    # graph executes, returns finite metrics, and chosen-acc is in [0, 1].
    model = _StubModel()
    cfg = DPOConfig(method="dpo", iters=1, beta=0.1, lora_layers=0)
    trainer = DPOTrainer(model, _StubTokenizer(), "/dev/null", cfg)
    batch = [
        {
            "prompt_ids": [1, 2],
            "chosen_ids": [3, 4],
            "rejected_ids": [5, 6],
            "ref_w": -1.0,
            "ref_l": -2.0,
        }
    ]
    loss, margins, accs = trainer._dpo_loss(model, batch)
    assert mx.isfinite(loss)
    assert 0.0 <= accs[0] <= 1.0
    assert len(margins) == 1


def test_orpo_loss_runs_without_ref():
    # ORPO loss needs no ref logprobs; batch omits ref_w/ref_l.
    model = _StubModel()
    cfg = DPOConfig(method="orpo", iters=1, lambda_odds=0.5, lora_layers=0)
    trainer = DPOTrainer(model, _StubTokenizer(), "/dev/null", cfg)
    batch = [
        {
            "prompt_ids": [1, 2],
            "chosen_ids": [3, 4],
            "rejected_ids": [5, 6],
        }
    ]
    loss, margins, accs = trainer._orpo_loss(model, batch)
    assert mx.isfinite(loss)
    assert 0.0 <= accs[0] <= 1.0
    assert len(margins) == 1
