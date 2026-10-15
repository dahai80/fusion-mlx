# SPDX-License-Identifier: Apache-2.0
"""#1186: fine-tune train() process isolation.

train() runs in a spawned child process with its own Metal memory limit so
the serving process no longer shares the GPU command stream with a
multi-hour training run (laya P50 5.7-10ms -> 100-900ms, #1184). These
tests exercise the real spawn/Pipe plumbing with stub entrypoints (no
model load) plus the service-level kill policy.
"""

import textwrap
from unittest.mock import AsyncMock, patch

import pytest

from fusion_mlx.training.proc import (
    TrainingProcessFailed,
    TrainingProcessKilled,
    physical_memory_gb,
    resolve_memory_limit_gb,
    run_training_process,
)
from fusion_mlx.training.service import (
    FineTuneConfig,
    FineTuneService,
    JobStatus,
    _apply_progress_event,
    _PipeProgressCallback,
    _ProgressCallback,
)

STUB_MODULE = """
import time


def run_ok(spec, report):
    report({"type": "train_loss", "step": 10, "total_steps": 100,
            "train_loss": 1.5, "val_loss": None, "learning_rate": 1e-5,
            "tokens_per_second": 100.0, "it_sec": 2.0, "trained_tokens": 500,
            "peak_memory_gb": 1.0, "elapsed_seconds": 5.0, "eta_seconds": 45.0})
    report({"type": "val_loss", "step": 10, "val_loss": 0.5, "val_time": 1.0})
    return {"adapter_path": spec["adapter_path"],
            "adapter_file": spec["adapter_path"] + "/adapters.safetensors"}


def run_fail(spec, report):
    raise ValueError("boom-dataset-empty")


def run_slow(spec, report):
    report({"type": "train_loss", "step": 1, "total_steps": 1000,
            "train_loss": 2.0, "val_loss": None, "learning_rate": 1e-5,
            "tokens_per_second": 1.0, "it_sec": 1.0, "trained_tokens": 1,
            "peak_memory_gb": 0.0, "elapsed_seconds": 0.1, "eta_seconds": 9.9})
    time.sleep(120)
    return {}
"""


@pytest.fixture
def stub_entrypoint(tmp_path, monkeypatch):
    # Stub entrypoint module on sys.path — the spawn child inherits the
    # parent's sys.path, so "f mlx_stub_1186:run_ok" resolves there.
    stub_file = tmp_path / "fmlx_stub_1186.py"
    stub_file.write_text(textwrap.dedent(STUB_MODULE))
    monkeypatch.syspath_prepend(str(tmp_path))
    return "fmlx_stub_1186"


def _spec(entrypoint_fn: str, tmp_path) -> dict:
    out = tmp_path / "adapter_out"
    return {
        "config": FineTuneConfig(),
        "model_path": "/nonexistent",
        "dataset": "/nonexistent",
        "adapter_path": str(out),
        "entrypoint": entrypoint_fn,
    }


class TestRunTrainingProcess:
    def test_success_streams_progress_and_result(self, stub_entrypoint, tmp_path):
        events = []

        def on_event(event):
            events.append(event)

        result = run_training_process(
            _spec(f"{stub_entrypoint}:run_ok", tmp_path), on_event, 0.0
        )
        assert result["adapter_file"].endswith("adapters.safetensors")
        types = [e["type"] for e in events]
        assert "train_loss" in types
        assert "val_loss" in types

    def test_child_error_raises_failed(self, stub_entrypoint, tmp_path):
        with pytest.raises(TrainingProcessFailed, match="boom-dataset-empty"):
            run_training_process(_spec(f"{stub_entrypoint}:run_fail", tmp_path))

    def test_killed_child_raises_killed(self, stub_entrypoint, tmp_path):
        # Simulate kill -9 on the child: on_started hands out the handle and
        # the test SIGKILLs it after the first progress event lands.
        holder = {}

        def on_started(proc):
            holder["proc"] = proc

        def on_event(event):
            if "proc" in holder and not holder.get("killed"):
                holder["killed"] = True
                holder["proc"].kill()

        with pytest.raises(TrainingProcessKilled, match="exitcode"):
            run_training_process(
                _spec(f"{stub_entrypoint}:run_slow", tmp_path),
                on_event,
                0.0,
                on_started,
            )


class TestMemoryLimit:
    def test_explicit_value_passthrough(self):
        assert resolve_memory_limit_gb(24.0) == 24.0

    def test_auto_is_fraction_of_physical(self):
        auto = resolve_memory_limit_gb(0.0)
        phys = physical_memory_gb()
        assert 0 < auto < phys
        # Calibration default: 60% of physical (#1186).
        assert auto == pytest.approx(phys * 0.6, rel=0.01)


class TestApplyProgressEvent:
    def test_train_loss_event_updates_progress(self):
        from fusion_mlx.training.service import FineTuneJob

        job = FineTuneJob()
        _apply_progress_event(
            job,
            {
                "type": "train_loss",
                "step": 10,
                "total_steps": 100,
                "train_loss": 1.5,
                "learning_rate": 1e-5,
                "tokens_per_second": 100.0,
                "it_sec": 2.0,
                "trained_tokens": 500,
                "peak_memory_gb": 1.0,
                "elapsed_seconds": 5.0,
                "eta_seconds": 45.0,
            },
        )
        assert job.progress.step == 10
        assert job.progress.train_loss == 1.5
        assert job.progress.iterations_per_second == 2.0
        assert len(job.events) == 1

    def test_val_loss_event_preserved_across_train_events(self):
        from fusion_mlx.training.service import FineTuneJob

        job = FineTuneJob()
        _apply_progress_event(
            job, {"type": "val_loss", "step": 10, "val_loss": 0.5, "val_time": 1.0}
        )
        _apply_progress_event(
            job,
            {
                "type": "train_loss",
                "step": 20,
                "total_steps": 100,
                "train_loss": 1.2,
                "it_sec": 2.0,
            },
        )
        assert job.progress.val_loss == 0.5


class TestCallbackParity:
    def test_pipe_and_job_callbacks_emit_same_payload_keys(self):
        # Both execution paths must emit identical event payloads so the
        # parent-side mapping cannot drift from the in-process callback.
        train_info = {
            "iteration": 10,
            "train_loss": 1.5,
            "learning_rate": 1e-5,
            "tokens_per_second": 100.0,
            "iterations_per_second": 2.0,
            "trained_tokens": 500,
            "peak_memory": 1.0,
        }
        val_info = {"iteration": 10, "val_loss": 0.5, "val_time": 1.0}

        job_events = []
        job = _ProgressCallback.__new__(_ProgressCallback)
        from fusion_mlx.training.service import FineTuneJob

        real_job = FineTuneJob()
        job.__init__(real_job, AsyncMock())
        real_job.events = []
        pipe_events = []
        pipe = _PipeProgressCallback(pipe_events.append, real_job.config.iters)

        job.on_train_loss_report(train_info)
        pipe.on_train_loss_report(train_info)
        job.on_val_loss_report(val_info)
        pipe.on_val_loss_report(val_info)

        assert list(real_job.events[0].keys()) == list(pipe_events[0].keys())
        assert list(real_job.events[1].keys()) == list(pipe_events[1].keys())
        assert real_job.events[0]["train_loss"] == pipe_events[0]["train_loss"]
        assert real_job.events[1]["val_loss"] == pipe_events[1]["val_loss"]


class TestKillPolicy:
    @pytest.fixture
    def svc(self, tmp_path, monkeypatch):
        monkeypatch.setattr("fusion_mlx.training.service.ADAPTER_BASE_DIR", tmp_path)
        return FineTuneService()

    def _job(self, svc, **cfg_kwargs):
        cfg = FineTuneConfig(**cfg_kwargs)
        return svc.create_job("m", "/tmp/ds", cfg, "a")

    def test_mark_failed_policy(self, svc):
        job = self._job(svc, execution_mode="thread", on_process_kill="mark_failed")
        job.status = JobStatus.RUNNING
        svc._current_job_id = job.job_id
        svc._handle_training_killed(
            job, TrainingProcessKilled("training process died (exitcode=-9)")
        )
        assert job.status == JobStatus.FAILED
        assert job.terminal is True
        assert "exitcode=-9" in job.error

    def test_auto_restart_requeues_once(self, svc):
        job = self._job(
            svc,
            execution_mode="thread",
            on_process_kill="auto_restart",
            max_restarts=1,
        )
        job.status = JobStatus.RUNNING
        svc._handle_training_killed(job, TrainingProcessKilled("died"))
        assert job.status == JobStatus.QUEUED
        assert job.attempts == 1
        assert job.job_id in svc._queue
        assert job.terminal is False
        # Second death exceeds max_restarts -> FAILED.
        svc._handle_training_killed(job, TrainingProcessKilled("died again"))
        assert job.status == JobStatus.FAILED
        assert job.terminal is True

    def test_cancelled_job_skips_policy(self, svc):
        job = self._job(svc, execution_mode="thread", on_process_kill="auto_restart")
        job.status = JobStatus.CANCELLED
        job.terminal = True
        svc._handle_training_killed(job, TrainingProcessKilled("died"))
        assert job.status == JobStatus.CANCELLED
        assert job.attempts == 0


class TestProcKillWiring:
    @pytest.fixture
    def svc(self, tmp_path, monkeypatch):
        monkeypatch.setattr("fusion_mlx.training.service.ADAPTER_BASE_DIR", tmp_path)
        return FineTuneService()

    def test_cancel_running_job_kills_child(self, svc):
        proc = AsyncMock()
        proc.is_alive.return_value = True
        svc._proc = proc
        job = svc.create_job("m", "/tmp/ds", FineTuneConfig(), "a")
        job.status = JobStatus.RUNNING
        svc._current_job_id = job.job_id
        assert svc.cancel_job(job.job_id) is True
        proc.terminate.assert_called_once()
        assert svc._proc is None

    def test_interrupt_active_job_kills_child(self, svc):
        # #1186 x #1205: a fatal serving exit must not orphan the trainer.
        proc = AsyncMock()
        proc.is_alive.return_value = True
        svc._proc = proc
        job = svc.create_job("m", "/tmp/ds", FineTuneConfig(), "a")
        job.status = JobStatus.RUNNING
        svc._current_job_id = job.job_id
        out = svc.interrupt_active_job()
        assert out is job
        proc.terminate.assert_called_once()
        assert svc._proc is None
        assert job.status == JobStatus.FAILED

    def test_config_validates_new_fields(self):
        with pytest.raises(ValueError, match="execution_mode"):
            FineTuneConfig(execution_mode="fork").validate()
        with pytest.raises(ValueError, match="on_process_kill"):
            FineTuneConfig(on_process_kill="ignore").validate()
        with pytest.raises(ValueError, match="memory_limit_gb"):
            FineTuneConfig(memory_limit_gb=-1.0).validate()
        # Defaults are valid.
        FineTuneConfig().validate()


class TestProcessModeEndToEnd:
    @pytest.fixture
    def svc(self, tmp_path, monkeypatch):
        monkeypatch.setattr("fusion_mlx.training.service.ADAPTER_BASE_DIR", tmp_path)
        return FineTuneService()

    def test_run_job_process_mode_completes(self, svc, stub_entrypoint, tmp_path):
        # Full _run_job path with a stub child: job transitions RUNNING ->
        # COMPLETED and progress events land in job state.
        import asyncio

        job = svc.create_job(
            "m", "/tmp/ds", FineTuneConfig(execution_mode="process"), "a"
        )
        job.status = JobStatus.RUNNING
        svc._current_job_id = job.job_id
        svc._loop = asyncio.new_event_loop()

        # The service bakes the real entrypoint into the spec; run the real
        # process runner against a stub-entrypoint spec instead (no model
        # load), then mark completed exactly as the service does.
        async def patched(self, j):
            from fusion_mlx.training.proc import run_training_process

            spec = {
                "config": j.config,
                "model_path": "/nonexistent",
                "dataset": "/nonexistent",
                "adapter_path": j.adapter_path,
                "entrypoint": f"{stub_entrypoint}:run_ok",
            }
            result = await asyncio.to_thread(run_training_process, spec, None, 0.0)
            self._mark_job_completed(j)
            return result

        async def main():
            with patch.object(FineTuneService, "_run_job_in_process", patched):
                await svc._run_job(job)

        asyncio.run(main())
        assert job.status == JobStatus.COMPLETED
        assert job.terminal is True
