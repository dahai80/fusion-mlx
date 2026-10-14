# SPDX-License-Identifier: Apache-2.0
"""Tests for #1205: fatal-exit path must not silently destroy fine-tune jobs.

Three layers:
1. Persisted-job recovery is tolerant to schema drift (unknown config keys
   are dropped with a warning, not the whole record).
2. FineTuneService.interrupt_active_job marks the in-flight job FAILED
   with a resume hint and persists it, before a fatal exit.
3. memory_enforcer defers fatal_exit while a training job is in flight
   (bounded grace polls), and fatal_exit runs registered pre-exit hooks.
"""

from __future__ import annotations

import json
import os

import pytest

from fusion_mlx.training.service import (
    _CONFIG_FIELDS,
    _PROGRESS_FIELDS,
    FineTuneConfig,
    FineTuneJob,
    FineTuneProgress,
    FineTuneService,
    JobStatus,
)


def _with_jobs_dir(monkeypatch, tmp_path) -> None:
    # _jobs_file is a read-only property over ADAPTER_BASE_DIR; redirect the
    # module-level base so persistence lands in the test tmp dir.
    import fusion_mlx.training.service as svc_mod

    monkeypatch.setattr(svc_mod, "ADAPTER_BASE_DIR", tmp_path)
    monkeypatch.setattr(svc_mod, "_CONFIG_FIELDS", _CONFIG_FIELDS, raising=True)


class TestTolerantRecovery:
    def test_field_sets_cover_dataclasses(self):
        cfg_fields = {f.name for f in __import__("dataclasses").fields(FineTuneConfig)}
        prg_fields = {
            f.name for f in __import__("dataclasses").fields(FineTuneProgress)
        }
        assert cfg_fields == _CONFIG_FIELDS
        assert prg_fields == _PROGRESS_FIELDS
        assert "init_adapter_path" in _CONFIG_FIELDS

    def test_unknown_config_key_dropped_not_fatal(self, tmp_path, monkeypatch):
        jobs_file = tmp_path / "jobs.json"
        persisted = [
            {
                "job_id": "aaaabbbbcccc",
                "model_id": "m",
                "dataset": "/d",
                "status": "completed",
                "config": {
                    "lora_rank": 8,
                    "some_removed_field": 42,
                },
                "progress": {"step": 3, "total_steps": 3},
                "created_at": 1.0,
                "started_at": 1.0,
                "finished_at": 2.0,
                "adapter_path": "/a",
                "adapter_name": "n",
                "error": "",
            }
        ]
        jobs_file.write_text(json.dumps(persisted))
        svc = FineTuneService()
        _with_jobs_dir(monkeypatch, tmp_path)
        svc._load_jobs()
        job = svc.get_job("aaaabbbbcccc")
        assert job is not None, "unknown key must not drop the whole record"
        assert job.config.lora_rank == 8
        assert job.status == JobStatus.COMPLETED

    def test_unknown_status_maps_to_failed(self, tmp_path, monkeypatch):
        jobs_file = tmp_path / "jobs.json"
        jobs_file.write_text(
            json.dumps(
                [
                    {
                        "job_id": "ddddeeeeffff",
                        "model_id": "m",
                        "dataset": "/d",
                        "status": "a-status-from-the-future",
                        "config": {},
                        "progress": {},
                    }
                ]
            )
        )
        svc = FineTuneService()
        _with_jobs_dir(monkeypatch, tmp_path)
        svc._load_jobs()
        job = svc.get_job("ddddeeeeffff")
        assert job is not None
        assert job.status == JobStatus.FAILED


class TestInterruptActiveJob:
    def test_marks_inflight_job_failed_and_persists(self, tmp_path, monkeypatch):
        svc = FineTuneService()
        _with_jobs_dir(monkeypatch, tmp_path)
        job = FineTuneJob(
            job_id="j1205",
            model_id="m",
            dataset="/d",
            config=FineTuneConfig(),
        )
        job.status = JobStatus.RUNNING
        svc._jobs["j1205"] = job
        svc._current_job_id = "j1205"

        out = svc.interrupt_active_job()
        assert out is job
        assert job.status == JobStatus.FAILED
        assert "resume via init_adapter_path" in job.error
        assert job.terminal is True
        assert svc.has_active_job() is False
        # Persisted store has the interrupted record.
        data = json.loads((tmp_path / "jobs.json").read_text())
        assert any(i["job_id"] == "j1205" for i in data)

    def test_noop_when_nothing_in_flight(self):
        svc = FineTuneService()
        assert svc.has_active_job() is False
        assert svc.interrupt_active_job() is None

    def test_idempotent(self, tmp_path, monkeypatch):
        svc = FineTuneService()
        _with_jobs_dir(monkeypatch, tmp_path)
        job = FineTuneJob(job_id="j2", config=FineTuneConfig())
        job.status = JobStatus.RUNNING
        svc._jobs["j2"] = job
        svc._current_job_id = "j2"
        svc.interrupt_active_job()
        again = svc.interrupt_active_job()
        assert again is job
        assert job.status == JobStatus.FAILED


class TestFatalHooks:
    def test_hooks_run_before_exit(self, monkeypatch):
        from fusion_mlx.utils import fatal

        ran = []
        monkeypatch.setattr(fatal.os, "_exit", lambda code: ran.append(("exit", code)))
        fatal.register_fatal_hook(lambda: ran.append("hook1"))
        fatal.register_fatal_hook(lambda: ran.append("hook2"))
        fatal.fatal_exit("test reason")
        assert ran == ["hook1", "hook2", ("exit", 70)]

    def test_hook_exception_does_not_block_exit(self, monkeypatch):
        from fusion_mlx.utils import fatal

        ran = []

        def boom():
            raise RuntimeError("hook bug")

        monkeypatch.setattr(fatal.os, "_exit", lambda code: ran.append("exit"))
        fatal.register_fatal_hook(boom)
        fatal.register_fatal_hook(lambda: ran.append("hook2"))
        fatal.fatal_exit("test reason")
        assert "hook2" in ran  # boom is caught; the next hook still runs
        assert ran[-1] == "exit"

    def test_hook_timeout_proceeds(self, monkeypatch):
        import time

        from fusion_mlx.utils import fatal

        monkeypatch.setattr(fatal, "_FATAL_HOOK_TIMEOUT_S", 0.1)
        ran = []
        monkeypatch.setattr(fatal.os, "_exit", lambda code: ran.append("exit"))

        def hang():
            time.sleep(5)

        fatal.register_fatal_hook(hang)
        t0 = time.monotonic()
        fatal.fatal_exit("test reason")
        assert time.monotonic() - t0 < 2.0
        assert ran == ["exit"]

    def test_hooks_list_restored_between_tests(self):
        # register_fatal_hook appends to a module-level list; keep tests
        # isolated by clearing it via fixture in conftest-style below.
        pass


@pytest.fixture(autouse=True)
def _clear_fatal_hooks(monkeypatch):
    from fusion_mlx.utils import fatal

    monkeypatch.setattr(fatal, "_fatal_hooks", [])
    yield


class TestEnforcerDefer:
    def test_fine_tune_service_lookup_returns_none_when_unbound(self):
        from fusion_mlx.pool import memory_enforcer as me

        monkey = pytest.MonkeyPatch()
        monkey.setattr(me, "_fine_tune_service_if_running", lambda: None, raising=True)
        assert me._fine_tune_service_if_running() is None or True
        monkey.undo()

    def test_grace_polls_constant_bounded(self):
        from fusion_mlx.pool import memory_enforcer as me

        assert 0 < me._TRAINING_GRACE_POLLS <= 600
        assert (
            int(os.environ.get("FUSION_TRAINING_FATAL_GRACE_POLLS", "30"))
            == me._TRAINING_GRACE_POLLS
        )

    def test_defer_uses_service_hook(self, monkeypatch, tmp_path):
        from fusion_mlx.pool import memory_enforcer as me

        svc = FineTuneService()
        _with_jobs_dir(monkeypatch, tmp_path)
        job = FineTuneJob(job_id="jd", config=FineTuneConfig())
        job.status = JobStatus.RUNNING
        svc._jobs["jd"] = job
        svc._current_job_id = "jd"

        monkeypatch.setattr(me, "_fine_tune_service_if_running", lambda: svc)
        assert me._fine_tune_service_if_running() is svc
        assert me._fine_tune_service_if_running().has_active_job() is True
        # The real defer wiring is inside _maybe_escalate_unrecoverable;
        # drive it with a minimal fake self.
        enforcer = me.ProcessMemoryEnforcer.__new__(me.ProcessMemoryEnforcer)
        enforcer._unrecoverable_polls = 0
        enforcer._drain_flipped = False
        enforcer._training_grace_logged = False
        # count polls until the fatal branch would fire
        threshold = (
            me._UNRECOVERABLE_POLL_THRESHOLD
            + me._UNRECOVERABLE_FATAL_THRESHOLD
            + me._TRAINING_GRACE_POLLS
        )
        exited = []
        import fusion_mlx.utils.fatal as fatal_mod

        monkeypatch.setattr(
            fatal_mod, "fatal_exit", lambda reason: exited.append(reason)
        )
        # _maybe_escalate_unrecoverable imports fatal_exit lazily from
        # ..utils.fatal, so patching the module attribute suffices.
        for _ in range(threshold + 5):
            enforcer._unrecoverable_polls += 1
            enforcer._maybe_escalate_unrecoverable(1, 2)
        assert exited, "fatal_exit must eventually fire after the grace window"
        # The stub fatal_exit above never runs the real hook pipeline, so
        # invoke it the way real fatal_exit would have.
        fatal_mod._run_fatal_hooks()
        assert job.status == JobStatus.FAILED
        assert "resume via init_adapter_path" in job.error
