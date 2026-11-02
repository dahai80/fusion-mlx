# SPDX-License-Identifier: Apache-2.0
# #1483 item 3: DPO/ORPO trainers run in a spawned child process by default
# (symmetric with #1186 for SFT). A training blow-up fails the job, not the
# serving process. The thread path stays reachable via
# DPOConfig(execution_mode="thread").

import asyncio

import pytest

import fusion_mlx.training.proc as proc_mod
from fusion_mlx.training.dpo import DPOConfig
from fusion_mlx.training.dpo_service import DPOJob, DPOService, execute_dpo_spec
from fusion_mlx.training.service import JobStatus

_PAIRS = [
    {"prompt": "Q?", "chosen": "good", "rejected": "bad"},
    {"prompt": "Q2?", "chosen": "better", "rejected": "worse"},
]


def _bare_service(job_id: str = "job-1"):
    svc = DPOService.__new__(DPOService)
    svc._jobs = {}
    svc._queue = []
    svc._current_job_id = job_id
    svc._running = False
    svc._proc = None
    svc._child_kill_note = None
    svc._resolve_model_path = lambda mid: "/fake/model"
    svc._push_event = lambda *a, **k: None
    svc._persist_jobs = lambda: None
    svc._notify_job = lambda *a, **k: None
    svc._process_queue = lambda *a, **k: None
    return svc


def _job(job_id: str = "job-1", **cfg_kwargs):
    cfg = DPOConfig(iters=1, **cfg_kwargs)
    return DPOJob(
        job_id=job_id,
        model_id="m1",
        preference_pairs=_PAIRS,
        config=cfg,
        adapter_name="dpo-x",
        adapter_path="/tmp/dpo-x",
    )


class TestDefaults:
    def test_default_execution_mode_is_process(self):
        assert DPOConfig().execution_mode == "process"

    def test_thread_mode_requested_explicitly(self):
        assert DPOConfig(execution_mode="thread").execution_mode == "thread"


class TestChildEntrypoint:
    def test_spec_maps_to_shared_core(self, monkeypatch):
        import fusion_mlx.training.dpo_service as d

        captured = {}

        def fake_core(**kwargs):
            captured.update(kwargs)
            return {"adapter_path": kwargs["adapter_path"]}

        monkeypatch.setattr(d, "_execute_dpo_core", fake_core)
        cfg = DPOConfig(iters=1)
        events = []
        out = execute_dpo_spec(
            {
                "job_id": "j1",
                "model_path": "/m",
                "preference_pairs": _PAIRS,
                "config": cfg,
                "adapter_path": "/a",
            },
            events.append,
        )
        assert out == {"adapter_path": "/a"}
        assert captured["job_id"] == "j1"
        assert captured["model_path"] == "/m"
        assert captured["pairs"] == _PAIRS
        assert captured["config"] is cfg
        assert captured["adapter_path"] == "/a"
        assert captured["emit"] == events.append


class TestProcessRunJob:
    def test_spawns_child_with_spec_and_streams_progress(self, monkeypatch):
        captured = {}

        def fake_run(spec, on_event, memory_limit_gb=0.0, on_started=None):
            captured["spec"] = spec
            captured["limit"] = memory_limit_gb
            on_event(
                {
                    "type": "dpo_step",
                    "iter": 0,
                    "total_iters": 1,
                    "loss": 0.5,
                    "reward_margin": 0.1,
                    "acc_chosen": 1.0,
                    "n_pairs": 2,
                }
            )
            return {"adapter_path": spec["adapter_path"]}

        monkeypatch.setattr(proc_mod, "run_training_process", fake_run)
        svc = _bare_service()
        job = _job(memory_limit_gb=42.0)
        asyncio.run(svc._run_job(job))

        spec = captured["spec"]
        assert spec["entrypoint"] == "fusion_mlx.training.dpo_service:execute_dpo_spec"
        assert spec["job_id"] == "job-1"
        assert spec["model_path"] == "/fake/model"
        assert spec["preference_pairs"] == _PAIRS
        assert spec["adapter_path"] == "/tmp/dpo-x"
        assert captured["limit"] == 42.0
        assert job.status == JobStatus.COMPLETED
        assert job.progress == {
            "iter": 1,
            "total_iters": 1,
            "loss": 0.5,
            "reward_margin": 0.1,
            "acc_chosen": 1.0,
        }

    def test_unresolvable_model_fails_loudly(self, monkeypatch):
        monkeypatch.setattr(
            proc_mod, "run_training_process", lambda *a, **k: pytest.fail("spawned")
        )
        svc = _bare_service()
        svc._resolve_model_path = lambda mid: None
        job = _job()
        asyncio.run(svc._run_job(job))
        assert job.status == JobStatus.FAILED
        assert "Cannot resolve model path" in job.error

    def test_child_kill_marks_failed(self, monkeypatch):
        def fake_run(*a, **k):
            raise proc_mod.TrainingProcessKilled(
                "training process died without reporting (pid=1, exitcode=-9)"
            )

        monkeypatch.setattr(proc_mod, "run_training_process", fake_run)
        svc = _bare_service()
        job = _job()
        asyncio.run(svc._run_job(job))
        assert job.status == JobStatus.FAILED
        assert "died without reporting" in job.error

    def test_rss_hard_stop_note_wins_over_generic_kill(self, monkeypatch):
        svc = _bare_service()

        def fake_run(*a, **k):
            svc._child_kill_note = "RSS hard stop (#1483): child RSS 20.0GB"
            raise proc_mod.TrainingProcessKilled("died without reporting")

        monkeypatch.setattr(proc_mod, "run_training_process", fake_run)
        job = _job()
        asyncio.run(svc._run_job(job))
        assert job.status == JobStatus.FAILED
        assert "RSS hard stop (#1483)" in job.error

    def test_child_death_after_cancel_stays_cancelled(self, monkeypatch):
        svc = _bare_service()
        job = _job()

        def fake_run(*a, **k):
            job.status = JobStatus.CANCELLED
            job.terminal = True
            raise proc_mod.TrainingProcessKilled("died after SIGTERM")

        monkeypatch.setattr(proc_mod, "run_training_process", fake_run)
        asyncio.run(svc._run_job(job))
        assert job.status == JobStatus.CANCELLED
        assert job.error == ""

    def test_thread_mode_skips_child(self, monkeypatch):
        monkeypatch.setattr(
            proc_mod, "run_training_process", lambda *a, **k: pytest.fail("spawned")
        )
        svc = _bare_service()
        ran = []
        svc._execute_dpo = lambda job: ran.append(job.job_id)
        job = _job(execution_mode="thread")
        asyncio.run(svc._run_job(job))
        assert ran == ["job-1"]
        assert job.status != JobStatus.FAILED


class TestCancelKillsChild:
    def test_cancel_running_job_terminates_child(self):
        svc = _bare_service()
        job = _job()
        job.status = JobStatus.RUNNING
        svc._jobs = {"job-1": job}
        killed = []
        svc._kill_training_process = lambda: killed.append(True) or True

        assert svc.cancel_job("job-1") is True
        assert killed == [True]
        assert job.status == JobStatus.CANCELLED
        assert svc._current_job_id is None

    def test_kill_training_process_without_child_is_noop(self):
        svc = _bare_service()
        assert svc._kill_training_process() is False
