# Route tests for /admin/api/fine-tune/reward/jobs endpoints (#424).
# Minimal FastAPI app with the admin router + require_admin override.
# RewardService queue processing is neutralized so route handlers never
# spin up a real model load during sync TestClient requests.

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fusion_mlx.admin.auth import require_admin
from fusion_mlx.admin.fine_tune_route import set_reward_context
from fusion_mlx.admin.routes import router as admin_router
from fusion_mlx.training.reward import RewardConfig, RewardTrainer
from fusion_mlx.training.reward_service import RewardJob, RewardService


def _build_app():
    app = FastAPI()
    svc = RewardService()
    svc.start_processing = lambda *a, **kw: None
    svc._process_queue = lambda *a, **kw: None
    set_reward_context(None, svc)
    app.include_router(admin_router)
    app.dependency_overrides[require_admin] = lambda: True
    return app, svc


def test_reward_create_job_returns_id():
    app, svc = _build_app()
    client = TestClient(app)
    resp = client.post(
        "/admin/api/fine-tune/reward/jobs",
        json={
            "model_id": "m1",
            "preference_pairs": [{"prompt": "p", "chosen": "good", "rejected": "bad"}],
            "config": {"iters": 1, "batch_size": 1},
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "job_id" in body
    assert body["model_id"] == "m1"
    assert body["preference_pairs"][0]["chosen"] == "good"
    assert body["config"]["iters"] == 1
    assert body["status"] in ("queued", "running", "completed", "cancelled")


def test_reward_create_missing_model_id():
    app, _ = _build_app()
    client = TestClient(app)
    resp = client.post(
        "/admin/api/fine-tune/reward/jobs",
        json={"preference_pairs": [{"prompt": "p", "chosen": "c", "rejected": "r"}]},
    )
    assert resp.status_code == 400
    assert "model_id" in resp.json()["detail"]


def test_reward_create_missing_pairs():
    app, _ = _build_app()
    client = TestClient(app)
    resp = client.post(
        "/admin/api/fine-tune/reward/jobs",
        json={"model_id": "m1"},
    )
    assert resp.status_code == 400
    assert "preference_pairs" in resp.json()["detail"]


def test_reward_create_malformed_pair():
    app, _ = _build_app()
    client = TestClient(app)
    resp = client.post(
        "/admin/api/fine-tune/reward/jobs",
        json={"model_id": "m1", "preference_pairs": [{"prompt": "p"}]},
    )
    assert resp.status_code == 400
    assert "prompt/chosen/rejected" in resp.json()["detail"]


def test_reward_create_invalid_config():
    app, _ = _build_app()
    client = TestClient(app)
    resp = client.post(
        "/admin/api/fine-tune/reward/jobs",
        json={
            "model_id": "m1",
            "preference_pairs": [{"prompt": "p", "chosen": "c", "rejected": "r"}],
            "config": {"unknown_field": 1},
        },
    )
    assert resp.status_code == 400
    assert "Invalid config" in resp.json()["detail"]


def test_reward_list_jobs():
    app, svc = _build_app()
    client = TestClient(app)
    svc.create_job(
        model_id="m1",
        preference_pairs=[{"prompt": "p", "chosen": "c", "rejected": "r"}],
        adapter_name="a1",
    )
    resp = client.get("/admin/api/fine-tune/reward/jobs")
    assert resp.status_code == 200
    jobs = resp.json()
    assert any(j["adapter_name"] == "a1" for j in jobs)


def test_reward_get_job_not_found():
    app, _ = _build_app()
    client = TestClient(app)
    resp = client.get("/admin/api/fine-tune/reward/jobs/nonexistent")
    assert resp.status_code == 404


def test_reward_cancel_queued_job():
    app, svc = _build_app()
    client = TestClient(app)
    job = svc.create_job(
        model_id="m1",
        preference_pairs=[{"prompt": "p", "chosen": "c", "rejected": "r"}],
        adapter_name="q1",
    )
    resp = client.post(f"/admin/api/fine-tune/reward/jobs/{job.job_id}/cancel")
    assert resp.status_code == 200
    assert resp.json()["status"] == "cancelled"


def test_reward_delete_job():
    app, svc = _build_app()
    client = TestClient(app)
    job = svc.create_job(
        model_id="m1",
        preference_pairs=[{"prompt": "p", "chosen": "c", "rejected": "r"}],
        adapter_name="d1",
    )
    resp = client.delete(f"/admin/api/fine-tune/reward/jobs/{job.job_id}")
    assert resp.status_code == 200
    assert resp.json()["status"] == "deleted"


def test_reward_cancel_not_found():
    app, _ = _build_app()
    client = TestClient(app)
    resp = client.post("/admin/api/fine-tune/reward/jobs/nope/cancel")
    assert resp.status_code == 404


# =============================================================================
# Service / config / trainer unit tests
# =============================================================================


_PAIRS = [
    {"prompt": "Q?", "chosen": "good", "rejected": "bad"},
    {"prompt": "Q2?", "chosen": "better", "rejected": "worse"},
]


def test_reward_config_defaults():
    cfg = RewardConfig()
    assert cfg.iters == 50
    assert cfg.learning_rate == 1e-5
    assert cfg.lora_rank == 8
    assert cfg.max_seq_length == 1024


def test_reward_job_to_dict_roundtrip():
    cfg = RewardConfig(iters=3, lora_rank=4)
    job = RewardJob(
        job_id="abc",
        model_id="m1",
        preference_pairs=_PAIRS,
        config=cfg,
        adapter_name="rm-abc",
        adapter_path="/tmp/rm-abc",
    )
    d = job.to_dict()
    assert d["job_id"] == "abc"
    assert d["config"]["iters"] == 3
    assert d["config"]["lora_rank"] == 4
    assert d["preference_pairs"] == _PAIRS
    assert d["status"] == "queued"


class _StubTokenizer:
    def encode(self, text):
        return [ord(c) for c in text][:4]


class _StubTrunk(nn.Module):
    # Returns (seq, hidden) hidden states so _score can take last-token.
    def __init__(self, hidden=4, vocab=32):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self._hidden = hidden

    def __call__(self, ids):
        # ids: (1, seq) -> (1, seq, hidden); _score indexes [0] -> (seq, hidden)
        return self.embed(ids)


class _StubModel(nn.Module):
    def __init__(self, hidden=4, vocab=32):
        super().__init__()
        self.transformer = _StubTrunk(hidden, vocab)
        self.hidden_size = hidden

    def __call__(self, ids):
        return self.transformer(ids)


def test_reward_train_step_runs_and_returns_metrics():
    # RewardTrainer.train_step with a stub backbone + registered value head.
    # Verifies the Bradley-Terry loss graph executes, returns finite metrics,
    # and the head is attached to the model.
    model = _StubModel()
    cfg = RewardConfig(iters=1, batch_size=1, lora_layers=0, learning_rate=1e-4)
    trainer = RewardTrainer(model, _StubTokenizer(), "/dev/null", cfg)
    result = trainer.train_step([_PAIRS[0]])
    assert mx.isfinite(mx.array(result.loss))
    assert -1.0 <= result.acc_chosen <= 1.0
    assert getattr(model, "value_head", None) is not None


def test_reward_loss_margin_sign():
    # With a fresh head, score_w - score_l should be ~0 (untrained), so the
    # loss is near log(2) (=-log sigmoid(0)) and acc is a bool in {0,1}.
    model = _StubModel()
    cfg = RewardConfig(iters=1, batch_size=1, lora_layers=0)
    trainer = RewardTrainer(model, _StubTokenizer(), "/dev/null", cfg)
    result = trainer.train_step([_PAIRS[0]])
    # -log sigmoid(0) = log(2) ~= 0.693
    assert abs(result.loss - 0.6931) < 0.05
    assert result.acc_chosen in (0.0, 1.0)
