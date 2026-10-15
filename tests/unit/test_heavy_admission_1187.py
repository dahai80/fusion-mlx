# SPDX-License-Identifier: Apache-2.0
"""#1187: unified heavy-workload admission gate.

Covers the gate itself (admit/defer/pause/resume/budget/env knobs), the
server_metrics counters + Prometheus render, source pins that all six
training services consult the gate in _process_queue, and behavior tests
for the fine-tune / VLM / GRPO reject + defer + admit paths.
"""

import asyncio
import importlib
import inspect
import json

import pytest

import fusion_mlx.admission as admission_mod
from fusion_mlx.admission import (
    ADMISSION_RECHECK_SECONDS,
    HeavyAdmissionDecision,
    HeavyAdmissionGate,
    schedule_recheck,
)
from fusion_mlx.server_metrics import ServerMetrics

GB = 2**30

TRAINING_SERVICE_MODULES = [
    ("fusion_mlx.training.service", "FineTuneService"),
    ("fusion_mlx.training.vlm_service", "VLMFineTuneService"),
    ("fusion_mlx.training.grpo_service", "GRPOService"),
    ("fusion_mlx.training.dpo_service", "DPOService"),
    ("fusion_mlx.training.rft_service", "RFTService"),
    ("fusion_mlx.training.reward_service", "RewardService"),
]


@pytest.fixture(autouse=True)
def _fresh_gate():
    admission_mod.reset_heavy_admission_gate()
    yield
    admission_mod.reset_heavy_admission_gate()


def _gate_with(physical: int, usage: int) -> HeavyAdmissionGate:
    gate = HeavyAdmissionGate()
    gate.physical_bytes = lambda: physical
    gate.current_usage_bytes = lambda: usage
    return gate


def _reject(reason: str = "projected 95.0GB exceeds safety line 85.0GB"):
    return HeavyAdmissionDecision(admitted=False, deferred=False, reason=reason)


def _deferred(reason: str = "training paused by serving window"):
    return HeavyAdmissionDecision(admitted=False, deferred=True, reason=reason)


class _StubGate:
    def __init__(self, decision):
        self._decision = decision
        self.budget_calls = []

    def training_budget_bytes(self, gb):
        self.budget_calls.append(gb)
        return 0

    def admit_training(self, budget):
        return self._decision


class _FakeLoop:
    def __init__(self):
        self.calls = []

    def is_closed(self):
        return False

    def call_later(self, delay, callback, *args):
        self.calls.append((delay, callback))


class _RecheckTarget:
    def __init__(self):
        self.calls = 0

    def _process_queue(self):
        self.calls += 1


class TestGate:
    def test_admit_under_line(self):
        gate = _gate_with(100 * GB, 10 * GB)
        d = gate.admit_training(5 * GB)
        assert d.admitted is True
        assert d.deferred is False

    def test_reject_over_line(self):
        gate = _gate_with(100 * GB, 80 * GB)
        d = gate.admit_training(10 * GB)
        assert d.admitted is False
        assert d.deferred is False
        assert "safety line" in d.reason
        assert d.detail["projected_bytes"] == 90 * GB

    def test_projected_exactly_at_line_admits(self):
        gate = _gate_with(100 * GB, 80 * GB)
        d = gate.admit_training(5 * GB)
        assert d.admitted is True

    def test_env_line_override_tightens(self, monkeypatch):
        monkeypatch.setenv("FUSION_MLX_HEAVY_ADMISSION_LINE", "0.6")
        gate = _gate_with(100 * GB, 55 * GB)
        # 55+10=65 <= 85 default line, but > 60 with the override.
        d = gate.admit_training(10 * GB)
        assert d.admitted is False

    def test_env_disable_admits_anything(self, monkeypatch):
        monkeypatch.setenv("FUSION_MLX_HEAVY_ADMISSION", "off")
        gate = _gate_with(100 * GB, 99 * GB)
        d = gate.admit_training(50 * GB)
        assert d.admitted is True
        assert d.reason == "gate disabled"

    def test_training_budget_bytes_default_fraction(self):
        gate = _gate_with(100 * GB, 0)
        assert gate.training_budget_bytes(0.0) == int(100 * GB * 0.60)

    def test_training_budget_bytes_explicit_passthrough(self):
        gate = _gate_with(100 * GB, 0)
        assert gate.training_budget_bytes(8.0) == 8 * GB

    def test_deferred_when_paused_then_admits_after_resume(self):
        gate = _gate_with(100 * GB, 10 * GB)
        gate.pause("training")
        d = gate.admit_training(5 * GB)
        assert d.admitted is False
        assert d.deferred is True
        assert "resume" in d.reason
        d2 = gate.admit_media("image", 5 * GB)
        # Pause scopes training only — media stays admissible.
        assert d2.admitted is True
        gate.resume()
        d3 = gate.admit_training(5 * GB)
        assert d3.admitted is True

    def test_pause_wrong_scope_raises(self):
        gate = HeavyAdmissionGate()
        with pytest.raises(ValueError):
            gate.pause("inference")

    def test_resume_invokes_recheck_targets(self):
        gate = HeavyAdmissionGate()
        target = _RecheckTarget()
        gate.add_recheck_target(target)
        gate.resume()
        assert target.calls == 1

    def test_resume_swallows_recheck_target_errors(self):
        gate = HeavyAdmissionGate()

        class _Boom:
            def _process_queue(self):
                raise RuntimeError("boom")

        gate.add_recheck_target(_Boom())
        gate.resume()

    def test_snapshot_fields(self):
        gate = _gate_with(100 * GB, 25 * GB)
        snap = gate.snapshot()
        assert snap["paused"] is False
        assert snap["safety_line_fraction"] == 0.85
        assert snap["physical_bytes"] == 100 * GB
        assert snap["safety_line_bytes"] == 85 * GB
        assert snap["usage_bytes"] == 25 * GB
        assert snap["training_budget_default_bytes"] == 60 * GB
        assert snap["recheck_seconds"] == ADMISSION_RECHECK_SECONDS


class TestRecheck:
    def test_schedule_recheck_guards_double_schedule(self):
        owner = SimpleOwner()
        loop = _FakeLoop()
        owner._loop = loop
        schedule_recheck(owner)
        schedule_recheck(owner)
        assert len(loop.calls) == 1
        assert loop.calls[0][0] == ADMISSION_RECHECK_SECONDS

    def test_schedule_recheck_no_loop_is_noop(self):
        owner = SimpleOwner()
        schedule_recheck(owner)
        assert not getattr(owner, "_admission_recheck_scheduled", False)

    def test_recheck_fire_calls_process_queue_once(self):
        owner = SimpleOwner()
        loop = _FakeLoop()
        owner._loop = loop
        schedule_recheck(owner)
        delay, callback = loop.calls[0]
        owner._process_queue = lambda: (_ for _ in ()).throw(RuntimeError("x"))
        callback()
        # Flag reset even when _process_queue raises.
        assert owner._admission_recheck_scheduled is False


class SimpleOwner:
    def __init__(self):
        self._loop = None
        self._admission_recheck_scheduled = False

    def _process_queue(self):
        pass


class TestMetrics:
    def test_record_admission_counters(self):
        sm = ServerMetrics()
        sm.record_admission("training", admitted=True)
        sm.record_admission("training", admitted=False)
        sm.record_admission("training", admitted=False, deferred=True)
        sm.record_admission("image", admitted=False)
        assert sm.admission_admitted_total == 1
        assert sm.admission_rejected_total == 2
        assert sm.admission_deferred_total == 1
        assert sm.admission_rejected_by_kind == {"training": 1, "image": 1}
        assert sm.admission_deferred_by_kind == {"training": 1}

    def test_render_admission_metrics(self, monkeypatch):
        sm = ServerMetrics()
        sm.record_admission("training", admitted=True)
        sm.record_admission("training", admitted=False)
        sm.record_admission("training", admitted=False, deferred=True)

        import fusion_mlx.routes_internal.metrics as metrics_mod

        monkeypatch.setattr(metrics_mod, "get_server_metrics", lambda: sm)
        text = "\n".join(metrics_mod._render_admission_metrics())
        assert "fusion_mlx_admission_admitted_total" in text
        assert 'kind="training"' in text
        assert "fusion_mlx_admission_rejected_total" in text
        assert "fusion_mlx_admission_deferred_total" in text


class TestServicePins:
    @pytest.mark.parametrize("mod_path,cls_name", TRAINING_SERVICE_MODULES)
    def test_process_queue_consults_gate(self, mod_path, cls_name):
        mod = importlib.import_module(mod_path)
        cls = getattr(mod, cls_name)
        src = inspect.getsource(cls._process_queue)
        assert "get_heavy_admission_gate" in src
        assert "schedule_recheck" in src
        assert "admit_training" in src


class _ServiceFixtureBase:
    @pytest.fixture
    def stub(self):
        return _StubGate(_reject())

    @pytest.fixture
    def patched_gate(self, monkeypatch, stub):
        monkeypatch.setattr(admission_mod, "get_heavy_admission_gate", lambda: stub)


class TestFineTuneService(_ServiceFixtureBase):
    @pytest.fixture
    def svc(self, tmp_path, monkeypatch):
        monkeypatch.setattr("fusion_mlx.training.service.ADAPTER_BASE_DIR", tmp_path)
        from fusion_mlx.training.service import FineTuneService

        return FineTuneService()

    def _job(self, svc):
        from fusion_mlx.training.service import FineTuneConfig

        return svc.create_job(
            "m", "/tmp/ds", FineTuneConfig(execution_mode="thread"), "a"
        )

    def test_reject_marks_failed_and_persists(self, svc, patched_gate, tmp_path):
        from fusion_mlx.training.service import JobStatus

        svc._running = True
        job = self._job(svc)
        svc._process_queue()
        assert job.status == JobStatus.FAILED
        assert job.terminal is True
        assert "safety line" in job.error
        assert svc._current_job_id is None
        assert job.job_id not in svc._queue
        data = json.loads((tmp_path / "jobs.json").read_text())
        persisted = [j for j in data if j["job_id"] == job.job_id]
        assert persisted and persisted[0]["status"] == "failed"

    def test_deferred_stays_queued_and_schedules_recheck(self, svc, patched_gate, stub):
        from fusion_mlx.training.service import JobStatus

        stub._decision = _deferred()
        svc._running = True
        svc._loop = _FakeLoop()
        job = self._job(svc)
        svc._process_queue()
        assert job.status == JobStatus.QUEUED
        assert svc._queue[0] == job.job_id
        assert svc._admission_recheck_scheduled is True
        assert svc._loop.calls[0][0] == ADMISSION_RECHECK_SECONDS

    @pytest.mark.asyncio
    async def test_deferred_recheck_admits_later(
        self, svc, patched_gate, stub, monkeypatch
    ):
        from unittest.mock import AsyncMock

        from fusion_mlx.training.service import JobStatus

        svc._running = True
        svc._loop = asyncio.get_running_loop()
        svc._run_job = AsyncMock()
        job = self._job(svc)

        stub._decision = _deferred()
        svc._process_queue()
        assert job.status == JobStatus.QUEUED

        stub._decision = HeavyAdmissionDecision(admitted=True)
        svc._process_queue()
        assert job.status == JobStatus.RUNNING
        await asyncio.sleep(0)
        svc._run_job.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_admitted_runs_job(self, svc, patched_gate, stub, monkeypatch):
        from unittest.mock import AsyncMock

        from fusion_mlx.training.service import JobStatus

        stub._decision = HeavyAdmissionDecision(admitted=True)
        svc._running = True
        svc._loop = asyncio.get_running_loop()
        svc._run_job = AsyncMock()
        job = self._job(svc)
        svc._process_queue()
        assert job.status == JobStatus.RUNNING
        await asyncio.sleep(0)
        svc._run_job.assert_awaited_once()

    def test_not_running_skips_gate(self, svc, patched_gate, stub):
        svc._running = False
        self._job(svc)
        svc._process_queue()
        assert stub.budget_calls == []


class TestVLMService(_ServiceFixtureBase):
    @pytest.fixture
    def svc(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "fusion_mlx.training.vlm_service.ADAPTER_BASE_DIR", tmp_path
        )
        from fusion_mlx.training.vlm_service import VLMFineTuneService

        return VLMFineTuneService()

    def _job(self, svc):
        from fusion_mlx.training.vlm_service import VLMFineTuneConfig

        return svc.create_job(
            "m", "/tmp/ds", VLMFineTuneConfig(execution_mode="thread"), "a"
        )

    def test_reject_marks_failed_without_persist(self, svc, patched_gate, tmp_path):
        from fusion_mlx.training.service import JobStatus

        svc._running = True
        job = self._job(svc)
        svc._process_queue()
        assert job.status == JobStatus.FAILED
        assert "safety line" in job.error
        # VLM never persists jobs.
        assert not (tmp_path / "vlm_jobs.json").exists()

    @pytest.mark.asyncio
    async def test_admitted_runs_job(self, svc, patched_gate, stub):
        from unittest.mock import AsyncMock

        from fusion_mlx.training.service import JobStatus
        from fusion_mlx.training.vlm_service import VLMFineTuneConfig

        stub._decision = HeavyAdmissionDecision(admitted=True)
        svc._running = True
        svc._loop = asyncio.get_running_loop()
        svc._run_job = AsyncMock()
        job = svc.create_job(
            "m", "/tmp/ds", VLMFineTuneConfig(execution_mode="thread"), "a"
        )
        svc._process_queue()
        assert job.status == JobStatus.RUNNING
        await asyncio.sleep(0)
        svc._run_job.assert_awaited_once()


class TestGRPOService(_ServiceFixtureBase):
    @pytest.fixture
    def svc(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "fusion_mlx.training.grpo_service.ADAPTER_BASE_DIR", tmp_path
        )
        from fusion_mlx.training.grpo_service import GRPOService

        return GRPOService()

    def _job(self, svc):
        from fusion_mlx.training.grpo import GRPOConfig

        return svc.create_job("m", ["p1", "p2"], GRPOConfig(), "a")

    def test_reject_marks_failed_and_persists(self, svc, patched_gate, tmp_path):
        from fusion_mlx.training.service import JobStatus

        svc._running = True
        job = self._job(svc)
        svc._process_queue()
        assert job.status == JobStatus.FAILED
        assert "safety line" in job.error
        assert svc._current_job_id is None
        data = json.loads((tmp_path / "grpo_jobs.json").read_text())
        persisted = [j for j in data if j["job_id"] == job.job_id]
        assert persisted and persisted[0]["status"] == "failed"

    @pytest.mark.asyncio
    async def test_admitted_runs_job(self, svc, patched_gate, stub):
        from unittest.mock import AsyncMock

        from fusion_mlx.training.grpo import GRPOConfig
        from fusion_mlx.training.service import JobStatus

        stub._decision = HeavyAdmissionDecision(admitted=True)
        svc._running = True
        svc._loop = asyncio.get_running_loop()
        svc._run_job = AsyncMock()
        job = svc.create_job("m", ["p1", "p2"], GRPOConfig(), "a")
        svc._process_queue()
        assert job.status == JobStatus.RUNNING
        await asyncio.sleep(0)
        svc._run_job.assert_awaited_once()
