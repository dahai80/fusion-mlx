# SPDX-License-Identifier: Apache-2.0
"""Tests for FineTuneService — job CRUD, persistence, adapter management.

callers: pytest
API: FineTuneService job CRUD + adapter serve/unload + persistence
schemas: FineTuneConfig, FineTuneProgress, FineTuneJob, JobStatus
instruction: "继续实现二期和三期"
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fusion_mlx.training.service import (
    ADAPTER_BASE_DIR,
    FineTuneConfig,
    FineTuneJob,
    FineTuneProgress,
    FineTuneService,
    JobStatus,
    _ProgressCallback,
)


@pytest.fixture
def tmp_adapter_dir(tmp_path):
    with patch("fusion_mlx.training.service.ADAPTER_BASE_DIR", tmp_path):
        yield tmp_path


@pytest.fixture
def svc(tmp_adapter_dir):
    s = FineTuneService()
    return s


@pytest.fixture
def mock_pool():
    pool = MagicMock()
    pool.get_entry = MagicMock(return_value=None)
    pool.get_engine = AsyncMock(return_value=MagicMock())
    pool.unload_engine_async = AsyncMock()
    pool.unload_if_idle_unpinned = MagicMock(return_value=True)
    return pool


class TestFineTuneConfig:
    def test_defaults(self):
        cfg = FineTuneConfig()
        assert cfg.lora_layers == 16
        assert cfg.lora_rank == 8
        assert cfg.fine_tune_type == "lora"
        assert cfg.optimizer == "adamw"
        assert cfg.learning_rate == 1e-5
        assert cfg.batch_size == 4
        assert cfg.iters == 100

    def test_custom(self):
        cfg = FineTuneConfig(lora_rank=32, fine_tune_type="dora", iters=500)
        assert cfg.lora_rank == 32
        assert cfg.fine_tune_type == "dora"
        assert cfg.iters == 500

    def test_to_mlx_args(self):
        cfg = FineTuneConfig()
        args = cfg.to_mlx_args(
            adapter_path="/tmp/adapter",
            data_path="/tmp/data",
            model_path="/tmp/model",
        )
        assert args.model == "/tmp/model"
        assert args.adapter_path == "/tmp/adapter"
        assert args.data == "/tmp/data"
        assert args.batch_size == 4
        assert args.iters == 100


class TestFineTuneJob:
    def test_to_dict(self):
        job = FineTuneJob(model_id="test-model", dataset="/tmp/data")
        d = job.to_dict()
        assert d["model_id"] == "test-model"
        assert d["dataset"] == "/tmp/data"
        assert d["status"] == "queued"
        assert "config" in d
        assert "progress" in d

    def test_job_id_auto_generated(self):
        job1 = FineTuneJob()
        job2 = FineTuneJob()
        assert job1.job_id != job2.job_id
        assert len(job1.job_id) == 12


class TestJobCRUD:
    def test_create_job(self, svc, tmp_adapter_dir):
        job = svc.create_job(model_id="qwen3", dataset="/tmp/data")
        assert job.model_id == "qwen3"
        assert job.dataset == "/tmp/data"
        assert job.status == JobStatus.QUEUED
        assert job.adapter_path.startswith(str(tmp_adapter_dir))
        assert job.job_id in svc._jobs

    def test_create_job_custom_adapter_name(self, svc):
        job = svc.create_job(
            model_id="qwen3", dataset="/tmp/data", adapter_name="my-adapter"
        )
        assert job.adapter_name == "my-adapter"
        assert "my-adapter" in job.adapter_path

    def test_create_job_auto_adapter_name(self, svc):
        job = svc.create_job(model_id="qwen3", dataset="/tmp/data")
        assert job.adapter_name.startswith("lora-")

    def test_get_job(self, svc):
        job = svc.create_job(model_id="qwen3", dataset="/tmp/data")
        found = svc.get_job(job.job_id)
        assert found is job

    def test_get_job_not_found(self, svc):
        assert svc.get_job("nonexistent") is None

    def test_list_jobs(self, svc):
        svc.create_job(model_id="qwen3", dataset="/tmp/data1")
        svc.create_job(model_id="llama", dataset="/tmp/data2")
        jobs = svc.list_jobs()
        assert len(jobs) == 2

    def test_cancel_queued_job(self, svc):
        job = svc.create_job(model_id="qwen3", dataset="/tmp/data")
        assert svc.cancel_job(job.job_id) is True
        assert job.status == JobStatus.CANCELLED
        assert job.terminal is True
        assert job.job_id not in svc._queue

    def test_cancel_nonexistent_job(self, svc):
        assert svc.cancel_job("nonexistent") is False

    def test_cancel_completed_job(self, svc):
        job = svc.create_job(model_id="qwen3", dataset="/tmp/data")
        job.status = JobStatus.COMPLETED
        assert svc.cancel_job(job.job_id) is False

    def test_delete_job(self, svc):
        job = svc.create_job(model_id="qwen3", dataset="/tmp/data")
        assert svc.delete_job(job.job_id) is True
        assert svc.get_job(job.job_id) is None

    def test_delete_running_job_fails(self, svc):
        job = svc.create_job(model_id="qwen3", dataset="/tmp/data")
        job.status = JobStatus.RUNNING
        assert svc.delete_job(job.job_id) is False

    def test_delete_nonexistent_job(self, svc):
        assert svc.delete_job("nonexistent") is False


class TestJobPersistence:
    def test_persist_and_load(self, tmp_adapter_dir):
        svc1 = FineTuneService()
        job = svc1.create_job(model_id="qwen3", dataset="/tmp/data")
        svc1.cancel_job(job.job_id)

        svc2 = FineTuneService()
        loaded = svc2.get_job(job.job_id)
        assert loaded is not None
        assert loaded.model_id == "qwen3"

    def test_stale_running_jobs_cancelled_on_load(self, tmp_adapter_dir):
        jobs_file = tmp_adapter_dir / "jobs.json"
        jobs_file.parent.mkdir(parents=True, exist_ok=True)
        jobs_data = [
            {
                "job_id": "abc123",
                "model_id": "qwen3",
                "dataset": "/tmp/data",
                "config": {},
                "status": "running",
                "progress": {},
                "created_at": 0.0,
                "started_at": None,
                "finished_at": None,
                "adapter_path": "/tmp/adapter",
                "adapter_name": "lora-test",
                "error": "",
            }
        ]
        with open(jobs_file, "w") as f:
            json.dump(jobs_data, f)

        svc = FineTuneService()
        job = svc.get_job("abc123")
        assert job is not None
        assert job.status == JobStatus.CANCELLED
        assert job.terminal is True

    def test_stale_queued_jobs_cancelled_on_load(self, tmp_adapter_dir):
        jobs_file = tmp_adapter_dir / "jobs.json"
        jobs_file.parent.mkdir(parents=True, exist_ok=True)
        jobs_data = [
            {
                "job_id": "def456",
                "model_id": "qwen3",
                "dataset": "/tmp/data",
                "config": {},
                "status": "queued",
                "progress": {},
                "created_at": 0.0,
                "started_at": None,
                "finished_at": None,
                "adapter_path": "/tmp/adapter",
                "adapter_name": "lora-test",
                "error": "",
            }
        ]
        with open(jobs_file, "w") as f:
            json.dump(jobs_data, f)

        svc = FineTuneService()
        job = svc.get_job("def456")
        assert job is not None
        assert job.status == JobStatus.CANCELLED

    def test_completed_jobs_preserved_on_load(self, tmp_adapter_dir):
        jobs_file = tmp_adapter_dir / "jobs.json"
        jobs_file.parent.mkdir(parents=True, exist_ok=True)
        jobs_data = [
            {
                "job_id": "ghi789",
                "model_id": "qwen3",
                "dataset": "/tmp/data",
                "config": {},
                "status": "completed",
                "progress": {"step": 100},
                "created_at": 0.0,
                "started_at": None,
                "finished_at": None,
                "adapter_path": "/tmp/adapter",
                "adapter_name": "lora-test",
                "error": "",
            }
        ]
        with open(jobs_file, "w") as f:
            json.dump(jobs_data, f)

        svc = FineTuneService()
        job = svc.get_job("ghi789")
        assert job is not None
        assert job.status == JobStatus.COMPLETED
        assert job.progress.step == 100

    def test_load_missing_file_no_error(self, tmp_adapter_dir):
        svc = FineTuneService()
        assert len(svc._jobs) == 0


class TestAdapterManagement:
    def test_list_adapters_empty(self, svc, tmp_adapter_dir):
        assert svc.list_adapters() == []

    def test_list_adapters_with_adapter(self, svc, tmp_adapter_dir):
        adapter_dir = tmp_adapter_dir / "qwen3" / "my-lora"
        adapter_dir.mkdir(parents=True)
        (adapter_dir / "adapters.safetensors").write_bytes(b"\x00")
        (adapter_dir / "adapter_config.json").write_text(
            json.dumps({"num_layers": 16, "lora_parameters": {"rank": 8}, "fine_tune_type": "lora"})
        )

        adapters = svc.list_adapters()
        assert len(adapters) == 1
        assert adapters[0]["adapter_name"] == "my-lora"
        assert adapters[0]["model_id"] == "qwen3"
        assert adapters[0]["has_weights"] is True
        assert adapters[0]["has_config"] is True
        assert adapters[0]["lora_rank"] == 8

    def test_list_adapters_filter_model(self, svc, tmp_adapter_dir):
        for model in ["qwen3", "llama"]:
            d = tmp_adapter_dir / model / "adapter1"
            d.mkdir(parents=True)

        adapters = svc.list_adapters(model_id="qwen3")
        assert len(adapters) == 1
        assert adapters[0]["model_id"] == "qwen3"

    def test_delete_adapter(self, svc, tmp_adapter_dir):
        adapter_dir = tmp_adapter_dir / "qwen3" / "my-lora"
        adapter_dir.mkdir(parents=True)
        assert svc.delete_adapter("qwen3", "my-lora") is True
        assert not adapter_dir.exists()

    def test_delete_adapter_not_found(self, svc, tmp_adapter_dir):
        assert svc.delete_adapter("nonexistent", "nope") is False

    def test_get_adapter_path(self, svc, tmp_adapter_dir):
        adapter_dir = tmp_adapter_dir / "qwen3" / "my-lora"
        adapter_dir.mkdir(parents=True)
        path = svc.get_adapter_path("qwen3", "my-lora")
        assert path is not None
        assert "my-lora" in path

    def test_get_adapter_path_not_found(self, svc, tmp_adapter_dir):
        assert svc.get_adapter_path("nonexistent", "nope") is None


class TestAdapterServing:
    @pytest.mark.asyncio
    async def test_serve_adapter(self, svc, mock_pool, tmp_adapter_dir):
        svc.set_engine_pool(mock_pool)
        adapter_dir = tmp_adapter_dir / "qwen3" / "my-lora"
        adapter_dir.mkdir(parents=True)
        (adapter_dir / "adapters.safetensors").write_bytes(b"\x00")

        result = await svc.serve_adapter("qwen3", "my-lora")
        assert result["base_model_id"] == "qwen3"
        assert result["adapter_name"] == "my-lora"
        assert "served_model_id" in result
        mock_pool.get_engine.assert_called_once()

    @pytest.mark.asyncio
    async def test_serve_adapter_not_found(self, svc, mock_pool, tmp_adapter_dir):
        svc.set_engine_pool(mock_pool)
        with pytest.raises(ValueError, match="Adapter not found"):
            await svc.serve_adapter("nonexistent", "nope")

    @pytest.mark.asyncio
    async def test_serve_adapter_no_weights(self, svc, mock_pool, tmp_adapter_dir):
        svc.set_engine_pool(mock_pool)
        adapter_dir = tmp_adapter_dir / "qwen3" / "my-lora"
        adapter_dir.mkdir(parents=True)

        with pytest.raises(ValueError, match="weights missing"):
            await svc.serve_adapter("qwen3", "my-lora")

    @pytest.mark.asyncio
    async def test_serve_adapter_no_pool(self, svc, tmp_adapter_dir):
        adapter_dir = tmp_adapter_dir / "qwen3" / "my-lora"
        adapter_dir.mkdir(parents=True)
        (adapter_dir / "adapters.safetensors").write_bytes(b"\x00")

        with pytest.raises(RuntimeError, match="Engine pool"):
            await svc.serve_adapter("qwen3", "my-lora")

    @pytest.mark.asyncio
    async def test_unload_adapter_engine(self, svc, mock_pool, tmp_adapter_dir):
        svc.set_engine_pool(mock_pool)
        adapter_dir = tmp_adapter_dir / "qwen3" / "my-lora"
        adapter_dir.mkdir(parents=True)

        mock_entry = MagicMock()
        mock_entry.engine = MagicMock()
        mock_pool.get_entry.return_value = mock_entry

        result = await svc.unload_adapter_engine("qwen3", "my-lora")
        assert result is True
        mock_pool.unload_engine_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_unload_adapter_not_loaded(self, svc, mock_pool, tmp_adapter_dir):
        svc.set_engine_pool(mock_pool)
        adapter_dir = tmp_adapter_dir / "qwen3" / "my-lora"
        adapter_dir.mkdir(parents=True)

        mock_entry = MagicMock()
        mock_entry.engine = None
        mock_pool.get_entry.return_value = mock_entry

        result = await svc.unload_adapter_engine("qwen3", "my-lora")
        assert result is False


class TestProgressCallback:
    def test_on_train_loss_report(self):
        loop = MagicMock()
        job = FineTuneJob(config=FineTuneConfig(iters=100))
        cb = _ProgressCallback(job, loop)

        cb.on_train_loss_report({
            "iteration": 10,
            "train_loss": 2.5,
            "learning_rate": 1e-5,
            "tokens_per_second": 1000.0,
            "iterations_per_second": 5.0,
            "trained_tokens": 5000,
            "peak_memory": 12.5,
        })

        assert job.progress.step == 10
        assert job.progress.total_steps == 100
        assert job.progress.train_loss == 2.5
        assert job.progress.tokens_per_second == 1000.0
        assert len(job.events) == 1
        assert job.events[0]["type"] == "train_loss"

    def test_on_val_loss_report(self):
        loop = MagicMock()
        job = FineTuneJob(config=FineTuneConfig(iters=100))
        cb = _ProgressCallback(job, loop)

        cb.on_val_loss_report({
            "iteration": 20,
            "val_loss": 3.0,
            "val_time": 1.5,
        })

        assert job.progress.val_loss == 3.0
        assert len(job.events) == 1
        assert job.events[0]["type"] == "val_loss"
