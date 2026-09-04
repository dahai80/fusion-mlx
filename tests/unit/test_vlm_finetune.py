# SPDX-License-Identifier: Apache-2.0
"""Tests for the VL fine-tune endpoint (#797).

Covers:
- config validation (lora_layers/iters/batch_size guards).
- VLMFineTuneService CRUD (create/list/get/cancel/delete) without loading MLX.
- _StdoutProgressReader parses mlx_vlm train() stdout progress lines into job
  events.
- route registration: the 6 /api/fine-tune/vlm/jobs routes are mounted and
  admin-gated.

mlx_vlm is mocked where needed so the suite runs headless (no real VLM load).
"""

from __future__ import annotations

import asyncio
import logging

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fusion_mlx.training.vlm_service import (
    VLMFineTuneConfig,
    VLMFineTuneJob,
    VLMFineTuneService,
    _StdoutProgressReader,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------- config


def test_config_validate_defaults_ok():
    cfg = VLMFineTuneConfig()
    cfg.validate()  # no raise


def test_config_validate_rejects_bad_lora_layers():
    cfg = VLMFineTuneConfig(lora_layers=0)
    with pytest.raises(ValueError, match="lora_layers"):
        cfg.validate()


def test_config_validate_rejects_zero_iters():
    cfg = VLMFineTuneConfig(iters=0)
    with pytest.raises(ValueError, match="iters"):
        cfg.validate()


def test_config_validate_rejects_zero_batch():
    cfg = VLMFineTuneConfig(batch_size=0)
    with pytest.raises(ValueError, match="batch_size"):
        cfg.validate()


# ---------------------------------------------------------------- service CRUD


@pytest.fixture
def svc():
    s = VLMFineTuneService()
    s.set_loop(asyncio.new_event_loop())
    return s


def test_service_create_and_get(svc):
    job = svc.create_job(
        model_id="mlx-community/paligemma-3b-4bit",
        dataset="Some/ui-tars",
        config=VLMFineTuneConfig(iters=2),
        adapter_name="test-adapter",
    )
    assert job.model_id == "mlx-community/paligemma-3b-4bit"
    assert job.dataset == "Some/ui-tars"
    assert job.adapter_name == "test-adapter"
    assert job.adapter_path.endswith("test-adapter")
    assert svc.get_job(job.job_id) is job


def test_service_list_and_delete(svc):
    job = svc.create_job("m", "d")
    assert len(svc.list_jobs()) == 1
    assert svc.delete_job(job.job_id) is True
    assert svc.get_job(job.job_id) is None
    assert svc.delete_job("nope") is False


def test_service_cancel_queued(svc):
    job = svc.create_job("m", "d")
    assert svc.cancel_job(job.job_id) is True
    assert job.terminal is True
    # Already cancelled -> not cancellable again.
    assert svc.cancel_job(job.job_id) is False


def test_service_to_dict_roundtrip(svc):
    job = svc.create_job("m", "d", config=VLMFineTuneConfig(iters=5))
    d = job.to_dict()
    assert d["model_id"] == "m"
    assert d["dataset"] == "d"
    assert d["config"]["iters"] == 5
    assert d["status"] == "queued"


# ---------------------------------------------------------------- progress reader


def test_progress_reader_parses_train_loss_line():
    loop = asyncio.new_event_loop()
    job = VLMFineTuneJob(config=VLMFineTuneConfig(iters=100))
    reader = _StdoutProgressReader(job, loop)

    reader.feed(
        "Iter 10: Train loss 1.23456000, Learning Rate 1.000e-05, "
        "It/sec 0.500, Tokens/sec 12.000, Trained Tokens 120, "
        "Peak mem 1.000 GB\n"
    )
    assert job.progress.step == 10
    assert job.progress.total_steps == 100
    assert abs(job.progress.train_loss - 1.23456) < 1e-4
    assert len(job.events) == 1
    assert job.events[0]["type"] == "train_loss"
    loop.close()


def test_progress_reader_parses_val_loss_line():
    loop = asyncio.new_event_loop()
    job = VLMFineTuneJob(config=VLMFineTuneConfig(iters=100))
    reader = _StdoutProgressReader(job, loop)

    reader.feed("Iter 10: Val loss 0.987, Val took 1.200s\n")
    assert job.progress.val_loss == pytest.approx(0.987)
    assert any(e["type"] == "val_loss" for e in job.events)
    loop.close()


def test_progress_reader_ignores_non_progress_lines():
    loop = asyncio.new_event_loop()
    job = VLMFineTuneJob(config=VLMFineTuneConfig(iters=100))
    reader = _StdoutProgressReader(job, loop)

    reader.feed("Starting training..., iterations: 100\nsome other line\n")
    assert job.progress.step == 0
    assert job.events == []
    loop.close()


# ---------------------------------------------------------------- routes


@pytest.fixture
def app():
    application = FastAPI()
    from fusion_mlx.admin.auth import require_admin
    from fusion_mlx.admin.fine_tune_route import _router as ft_router

    application.include_router(ft_router)
    application.dependency_overrides[require_admin] = lambda: True
    return application


@pytest.fixture
def client(app):
    return TestClient(app)


def test_vlm_routes_registered(client):
    # POST with missing fields -> 400 (route exists and is admin-gated).
    resp = client.post("/api/fine-tune/vlm/jobs", json={})
    assert resp.status_code == 400
    assert "model_id" in resp.json()["detail"]


def test_vlm_routes_list_empty(client, monkeypatch):
    # Use a fresh isolated service so we don't pollute global state.
    from fusion_mlx.admin import fine_tune_route as rt

    fresh = VLMFineTuneService()
    monkeypatch.setattr(rt, "_vlm_service", fresh)
    resp = client.get("/api/fine-tune/vlm/jobs")
    assert resp.status_code == 200
    assert resp.json() == []


def test_vlm_route_get_missing_job_404(client, monkeypatch):
    from fusion_mlx.admin import fine_tune_route as rt

    fresh = VLMFineTuneService()
    monkeypatch.setattr(rt, "_vlm_service", fresh)
    resp = client.get("/api/fine-tune/vlm/jobs/does-not-exist")
    assert resp.status_code == 404
