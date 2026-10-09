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
    assert cfg.init_adapter_path == ""


def test_dpo_config_init_adapter_path_roundtrip():
    cfg = DPOConfig(method="dpo", init_adapter_path="/saves/sft-adapter")
    d = cfg.to_dict()
    assert d["init_adapter_path"] == "/saves/sft-adapter"
    cfg2 = DPOConfig(**d)
    assert cfg2.init_adapter_path == "/saves/sft-adapter"


def test_dpo_job_init_adapter_path_in_dict():
    cfg = DPOConfig(init_adapter_path="/saves/bnup-sft")
    job = DPOJob(
        job_id="chain1",
        model_id="m1",
        preference_pairs=_PAIRS,
        config=cfg,
        adapter_name="dpo-chain",
        adapter_path="/tmp/dpo-chain",
    )
    d = job.to_dict()
    assert d["config"]["init_adapter_path"] == "/saves/bnup-sft"


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


def test_dpo_execute_passes_init_adapter_path(monkeypatch, tmp_path):
    # #1149: _execute_dpo must pass init_adapter_path to mlx_utils.load
    # for the POLICY model. The reference model (dpo.py _ref_logprob) stays
    # adapter_path=None — tested by source inspection, not here.
    import fusion_mlx.training.dpo_service as svc_mod
    from fusion_mlx.training.dpo import DPOStepResult

    captured = {}

    class _FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(32, 4)
            self.head = nn.Linear(4, 32)

        def __call__(self, ids):
            return self.head(self.embed(ids))

        def freeze(self):
            pass

        def parameters(self):
            return []

    class _FakeTokenizer:
        def encode(self, text):
            return [ord(c) % 32 for c in text][:4]

        def decode(self, ids):
            return "x"

    def fake_load(path, adapter_path=None):
        captured["model_path"] = path
        captured["adapter_path"] = adapter_path
        return _FakeModel(), _FakeTokenizer()

    class _FakeTrainer:
        def __init__(self, model, tokenizer, model_path, config):
            pass

        def train_step(self, batch):
            return DPOStepResult(loss=0.1, reward_margin=0.2, acc_chosen=0.5)

        def save_adapter(self, path):
            pass

    monkeypatch.setattr("mlx_lm.utils.load", fake_load)
    monkeypatch.setattr(
        "mlx_lm.tuner.utils.linear_to_lora_layers", lambda *a, **k: None
    )
    monkeypatch.setattr(svc_mod, "DPOTrainer", _FakeTrainer)

    service = DPOService.__new__(DPOService)
    service._resolve_model_path = lambda mid: "/fake/model"
    service._push_event = lambda *a, **k: None
    service._persist_jobs = lambda: None

    cfg = DPOConfig(
        method="dpo",
        iters=1,
        lora_layers=0,
        init_adapter_path="/saves/bnup-sft",
    )
    job = DPOJob(
        job_id="chain-test",
        model_id="m1",
        preference_pairs=_PAIRS,
        config=cfg,
        adapter_name="dpo-chain",
        adapter_path=str(tmp_path / "dpo-chain"),
    )
    service._execute_dpo(job)
    assert (
        captured["adapter_path"] == "/saves/bnup-sft"
    ), f"policy load must pass init_adapter_path, got {captured['adapter_path']}"


def test_dpo_execute_no_init_adapter_backward_compat(monkeypatch, tmp_path):
    # #1149: when init_adapter_path is empty (default), load gets None —
    # identical to pre-fix behavior (backward compatible).
    import fusion_mlx.training.dpo_service as svc_mod
    from fusion_mlx.training.dpo import DPOStepResult

    captured = {}

    class _FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(32, 4)
            self.head = nn.Linear(4, 32)

        def __call__(self, ids):
            return self.head(self.embed(ids))

        def freeze(self):
            pass

        def parameters(self):
            return []

    class _FakeTokenizer:
        def encode(self, text):
            return [ord(c) % 32 for c in text][:4]

        def decode(self, ids):
            return "x"

    def fake_load(path, adapter_path=None):
        captured["adapter_path"] = adapter_path
        return _FakeModel(), _FakeTokenizer()

    class _FakeTrainer:
        def __init__(self, model, tokenizer, model_path, config):
            pass

        def train_step(self, batch):
            return DPOStepResult(loss=0.1, reward_margin=0.2, acc_chosen=0.5)

        def save_adapter(self, path):
            pass

    monkeypatch.setattr("mlx_lm.utils.load", fake_load)
    monkeypatch.setattr(
        "mlx_lm.tuner.utils.linear_to_lora_layers", lambda *a, **k: None
    )
    monkeypatch.setattr(svc_mod, "DPOTrainer", _FakeTrainer)

    service = DPOService.__new__(DPOService)
    service._resolve_model_path = lambda mid: "/fake/model"
    service._push_event = lambda *a, **k: None
    service._persist_jobs = lambda: None

    cfg = DPOConfig(method="dpo", iters=1, lora_layers=0)
    job = DPOJob(
        job_id="no-init",
        model_id="m1",
        preference_pairs=_PAIRS,
        config=cfg,
        adapter_name="dpo-basic",
        adapter_path=str(tmp_path / "dpo-no-init"),
    )
    service._execute_dpo(job)
    assert (
        captured["adapter_path"] is None
    ), f"empty init_adapter_path must yield None, got {captured['adapter_path']}"
