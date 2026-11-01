# SPDX-License-Identifier: Apache-2.0
"""#1473 item (1): training memory budget preflight."""

from __future__ import annotations

from pathlib import Path

import pytest

from fusion_mlx.training import mem_preflight
from fusion_mlx.training.mem_preflight import (
    ACT_HEADROOM_FLOOR_BYTES,
    ENV_TRAIN_MEM_BUDGET_GB,
    estimate_training_peak_bytes,
    resolve_training_budget_bytes,
    train_mem_budget_ceiling_bytes,
)


def _make_model_dir(tmp_path: Path, size_bytes: int) -> Path:
    d = tmp_path / "model"
    d.mkdir()
    f = d / "model.safetensors"
    # Sparse file: stat().st_size reports the full size without writing
    # real bytes (a 10GB test weight would otherwise fill the disk).
    f.write_bytes(b"\0")
    with open(f, "rb+") as fh:
        fh.truncate(size_bytes)
    return d


# -- estimate_training_peak_bytes ------------------------------------------


def test_peak_estimate_without_reference(tmp_path):
    # 10GB weights, no reference: peak = weights + headroom(0.35x).
    d = _make_model_dir(tmp_path, 10 * 1024**3)
    peak = estimate_training_peak_bytes(d, needs_reference=False)
    # estimate_model_size adds a 5% runtime-buffer overhead.
    weights = int(10 * 1024**3 * 1.05)
    expected = weights + max(ACT_HEADROOM_FLOOR_BYTES, int(weights * 0.35))
    assert peak == expected


def test_peak_estimate_with_reference_doubles_weights(tmp_path):
    d = _make_model_dir(tmp_path, 10 * 1024**3)
    with_ref = estimate_training_peak_bytes(d, needs_reference=True)
    without_ref = estimate_training_peak_bytes(d, needs_reference=False)
    weights = int(10 * 1024**3 * 1.05)
    assert with_ref - without_ref == weights


def test_peak_estimate_headroom_floor_for_small_model(tmp_path):
    # 100MB model: headroom floor (4GB) dominates over 0.35x.
    d = _make_model_dir(tmp_path, 100 * 1024**2)
    peak = estimate_training_peak_bytes(d, needs_reference=False)
    weights = int(100 * 1024**2 * 1.05)
    assert peak == weights + ACT_HEADROOM_FLOOR_BYTES


# -- train_mem_budget_ceiling_bytes ----------------------------------------


def test_ceiling_unset_returns_zero(monkeypatch):
    monkeypatch.delenv(ENV_TRAIN_MEM_BUDGET_GB, raising=False)
    assert train_mem_budget_ceiling_bytes() == 0


def test_ceiling_env_gb(monkeypatch):
    monkeypatch.setenv(ENV_TRAIN_MEM_BUDGET_GB, "64")
    assert train_mem_budget_ceiling_bytes() == 64 * 1024**3


def test_ceiling_env_invalid_falls_back_to_zero(monkeypatch):
    monkeypatch.setenv(ENV_TRAIN_MEM_BUDGET_GB, "not-a-number")
    assert train_mem_budget_ceiling_bytes() == 0


def test_ceiling_env_zero_disables(monkeypatch):
    monkeypatch.setenv(ENV_TRAIN_MEM_BUDGET_GB, "0")
    assert train_mem_budget_ceiling_bytes() == 0


# -- resolve_training_budget_bytes -----------------------------------------


@pytest.fixture(autouse=True)
def _no_env_budget(monkeypatch):
    monkeypatch.delenv(ENV_TRAIN_MEM_BUDGET_GB, raising=False)


def test_resolve_no_model_path_uses_default_budget():
    from fusion_mlx.admission import get_heavy_admission_gate

    gate = get_heavy_admission_gate()
    budget, reason = resolve_training_budget_bytes(
        job_id="job-x",
        memory_limit_gb=0.0,
        model_path=None,
        needs_reference=False,
    )
    assert reason is None
    assert budget == gate.training_budget_bytes(0.0)


def test_resolve_estimate_failure_falls_back(monkeypatch):
    # Estimation error must NOT fail the job — fall back to the legacy
    # budget loudly (fail-closed applies to a KNOWN oversize).
    from fusion_mlx.admission import get_heavy_admission_gate

    def _boom(path):
        raise ValueError("No model weights found")

    monkeypatch.setattr(mem_preflight, "estimate_training_peak_bytes", _boom)
    gate = get_heavy_admission_gate()
    budget, reason = resolve_training_budget_bytes(
        job_id="job-x",
        memory_limit_gb=0.0,
        model_path="/tmp/whatever",
        needs_reference=False,
    )
    assert reason is None
    assert budget == gate.training_budget_bytes(0.0)


def test_resolve_passes_when_projected_under_ceiling(monkeypatch, tmp_path):
    # Env ceiling 100GB, resident 10GB, job peak ~15GB: pass, budget = peak.
    d = _make_model_dir(tmp_path, 10 * 1024**3)
    monkeypatch.setenv(ENV_TRAIN_MEM_BUDGET_GB, "100")
    monkeypatch.setattr(
        "fusion_mlx.admission.get_heavy_admission_gate",
        lambda: _FakeGate(usage=10 * 1024**3),
    )
    budget, reason = resolve_training_budget_bytes(
        job_id="job-ok",
        memory_limit_gb=0.0,
        model_path=str(d),
        needs_reference=False,
    )
    assert reason is None
    assert budget == estimate_training_peak_bytes(str(d), needs_reference=False)


def test_resolve_rejects_when_projected_over_ceiling(monkeypatch, tmp_path):
    # Env ceiling 20GB, resident 10GB, job peak ~15GB: projected 25GB > 20GB
    # -> fail-closed with a reason naming the job.
    d = _make_model_dir(tmp_path, 10 * 1024**3)
    monkeypatch.setenv(ENV_TRAIN_MEM_BUDGET_GB, "20")
    monkeypatch.setattr(
        "fusion_mlx.admission.get_heavy_admission_gate",
        lambda: _FakeGate(usage=10 * 1024**3),
    )
    budget, reason = resolve_training_budget_bytes(
        job_id="job-big",
        memory_limit_gb=0.0,
        model_path=str(d),
        needs_reference=False,
    )
    assert budget == 0
    assert reason is not None
    assert "job-big" in reason
    assert "exceeds ceiling" in reason


def test_resolve_rejects_even_with_explicit_job_budget(monkeypatch, tmp_path):
    # Operator set memory_limit_gb, but the ceiling check is machine
    # safety — it runs regardless of the operator's claimed budget.
    d = _make_model_dir(tmp_path, 10 * 1024**3)
    monkeypatch.setenv(ENV_TRAIN_MEM_BUDGET_GB, "20")
    monkeypatch.setattr(
        "fusion_mlx.admission.get_heavy_admission_gate",
        lambda: _FakeGate(usage=10 * 1024**3),
    )
    budget, reason = resolve_training_budget_bytes(
        job_id="job-override",
        memory_limit_gb=1.0,
        model_path=str(d),
        needs_reference=False,
    )
    assert budget == 0
    assert reason is not None


def test_resolve_explicit_job_budget_honored_when_passing(monkeypatch, tmp_path):
    d = _make_model_dir(tmp_path, 10 * 1024**3)
    monkeypatch.setattr(
        "fusion_mlx.admission.get_heavy_admission_gate",
        lambda: _FakeGate(usage=10 * 1024**3),
    )
    budget, reason = resolve_training_budget_bytes(
        job_id="job-cfg",
        memory_limit_gb=2.0,
        model_path=str(d),
        needs_reference=False,
    )
    assert reason is None
    assert budget == 2 * 1024**3


def test_resolve_no_ceiling_available_skips_check(monkeypatch, tmp_path):
    # No env knob, no enforcer wired: check skipped, budget = peak.
    d = _make_model_dir(tmp_path, 10 * 1024**3)
    monkeypatch.setattr(
        "fusion_mlx.admission.get_heavy_admission_gate",
        lambda: _FakeGate(usage=10 * 1024**3),
    )
    monkeypatch.setattr(mem_preflight, "_enforcer_ceiling_bytes", lambda: 0)
    budget, reason = resolve_training_budget_bytes(
        job_id="job-noceiling",
        memory_limit_gb=0.0,
        model_path=str(d),
        needs_reference=False,
    )
    assert reason is None
    assert budget == estimate_training_peak_bytes(str(d), needs_reference=False)


def test_resolve_enforcer_ceiling_used_when_env_unset(monkeypatch, tmp_path):
    # Env unset: enforcer final ceiling is the fallback ceiling.
    d = _make_model_dir(tmp_path, 10 * 1024**3)
    monkeypatch.setattr(
        "fusion_mlx.admission.get_heavy_admission_gate",
        lambda: _FakeGate(usage=10 * 1024**3),
    )
    # Job peak ~14.2GB + usage 10GB = 24.2GB > 25GB ceiling? No: 24.2 <= 25.
    monkeypatch.setattr(mem_preflight, "_enforcer_ceiling_bytes", lambda: 25 * 1024**3)
    budget, reason = resolve_training_budget_bytes(
        job_id="job-enf",
        memory_limit_gb=0.0,
        model_path=str(d),
        needs_reference=False,
    )
    assert reason is None
    # Same job with a tighter enforcer ceiling: rejected.
    monkeypatch.setattr(mem_preflight, "_enforcer_ceiling_bytes", lambda: 20 * 1024**3)
    budget, reason = resolve_training_budget_bytes(
        job_id="job-enf",
        memory_limit_gb=0.0,
        model_path=str(d),
        needs_reference=False,
    )
    assert budget == 0
    assert reason is not None


class _FakeGate:
    def __init__(self, usage: int):
        self._usage = usage

    def training_budget_bytes(self, memory_limit_gb: float = 0.0) -> int:
        if memory_limit_gb and memory_limit_gb > 0:
            return int(memory_limit_gb * 1024**3)
        return int(128 * 1024**3 * 0.60)

    def current_usage_bytes(self) -> int:
        return self._usage

    def admit_training(self, budget_bytes):
        from fusion_mlx.admission import HeavyAdmissionDecision

        return HeavyAdmissionDecision(admitted=True, reason="fake gate")


# -- service wiring ---------------------------------------------------------


def test_dpo_service_rejects_job_at_preflight(monkeypatch, tmp_path):
    # Wiring: _process_queue fails the job closed with the preflight
    # reason and never claims the execution slot (#1473).
    from fusion_mlx.training.dpo import DPOConfig
    from fusion_mlx.training.dpo_service import DPOJob, DPOService, JobStatus

    d = _make_model_dir(tmp_path, 10 * 1024**3)
    monkeypatch.setenv(ENV_TRAIN_MEM_BUDGET_GB, "20")
    monkeypatch.setattr(
        "fusion_mlx.admission.get_heavy_admission_gate",
        lambda: _FakeGate(usage=10 * 1024**3),
    )

    service = DPOService.__new__(DPOService)
    service._running = True
    service._current_job_id = None
    service._loop = None
    service._persist_jobs = lambda: None
    service._notify_job = lambda job: None
    service._resolve_model_path = lambda mid: str(d)
    cfg = DPOConfig(method="dpo", iters=1)
    job = DPOJob(
        job_id="wire-1",
        model_id="m1",
        preference_pairs=[{"prompt": "p", "chosen": "c", "rejected": "r"}],
        config=cfg,
        adapter_name="a",
        adapter_path=str(tmp_path / "a"),
    )
    service._jobs = {"wire-1": job}
    service._queue = ["wire-1"]
    service._process_queue()
    assert job.status == JobStatus.FAILED
    assert "memory preflight" in (job.error or "")
    assert service._current_job_id is None


def test_dpo_service_passes_preflight_and_admits(monkeypatch, tmp_path):
    # Under the ceiling: job proceeds past the preflight to the gate.
    # Gate admits (projected under line) and the job claims the slot —
    # asserted via status RUNNING before the executor thread starts.
    import asyncio

    from fusion_mlx.training.dpo import DPOConfig
    from fusion_mlx.training.dpo_service import DPOJob, DPOService, JobStatus

    d = _make_model_dir(tmp_path, 1 * 1024**3)
    monkeypatch.setenv(ENV_TRAIN_MEM_BUDGET_GB, "100")
    monkeypatch.setattr(
        "fusion_mlx.admission.get_heavy_admission_gate",
        lambda: _FakeGate(usage=1 * 1024**3),
    )

    service = DPOService.__new__(DPOService)
    service._running = True
    service._current_job_id = None
    service._persist_jobs = lambda: None
    service._notify_job = lambda job: None
    service._resolve_model_path = lambda mid: str(d)
    service._pending_tasks = set()
    # Real loop so asyncio.ensure_future works; _execute_dpo is stubbed.
    service._loop = asyncio.new_event_loop()
    try:
        service._execute_dpo = lambda job: None
        cfg = DPOConfig(method="dpo", iters=1)
        job = DPOJob(
            job_id="wire-2",
            model_id="m1",
            preference_pairs=[{"prompt": "p", "chosen": "c", "rejected": "r"}],
            config=cfg,
            adapter_name="a",
            adapter_path=str(tmp_path / "a"),
        )
        service._jobs = {"wire-2": job}
        service._queue = ["wire-2"]
        service._process_queue()
        assert job.status == JobStatus.RUNNING
        assert service._current_job_id == "wire-2"
        # Drain the executor tasks so the loop can close cleanly.
        if service._pending_tasks:
            service._loop.run_until_complete(
                asyncio.wait(set(service._pending_tasks), timeout=5.0)
            )
    finally:
        service._loop.close()
        service._loop = None
