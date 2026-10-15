# SPDX-License-Identifier: Apache-2.0
"""#1184 chaos shape: fine-tune training load concurrent with serving.

Injection: the #1187 heavy admission gate sees a training budget under a
resident serving set that crowds the safety line (physical memory and
usage monkeypatched — no real model load). The training job must be
gated (deferred while paused / rejected with a FAILED error over the
line), laya-style probes must defer rather than poison within the media
budget, and every queued job must carry an explicit decision.
"""

import time

import pytest

from fusion_mlx import engine_core as ec
from fusion_mlx.admission import get_heavy_admission_gate, reset_heavy_admission_gate
from fusion_mlx.executor_watchdog import ExecutorWatchdog
from tests.chaos.conftest import DETECTION_BOUND_S, RECOVERY_BOUND_S

PHYS = 128 * 2**30


@pytest.fixture()
def gate(monkeypatch):
    reset_heavy_admission_gate()
    g = get_heavy_admission_gate()
    monkeypatch.setattr(g, "physical_bytes", lambda: PHYS)
    return g


def test_training_rejected_over_safety_line(gate, monkeypatch):
    # Resident serving set 100GB + 60% default training budget: projected
    # 176.8GB > line 108.8GB -> explicit rejection, never a silent queue.
    monkeypatch.setattr(gate, "current_usage_bytes", lambda: 100 * 2**30)
    t0 = time.monotonic()
    decision = gate.admit_training(gate.training_budget_bytes())
    assert decision.admitted is False
    assert decision.deferred is False
    assert "safety line" in decision.reason
    assert time.monotonic() - t0 < DETECTION_BOUND_S


def test_training_deferred_during_serving_window(gate, monkeypatch):
    # Paused serving window: training defers (stays queued) instead of
    # starting and saturating the GPU (#1187 pause endpoint shape).
    monkeypatch.setattr(gate, "current_usage_bytes", lambda: 10 * 2**30)
    gate.pause("training")
    decision = gate.admit_training(gate.training_budget_bytes())
    assert decision.admitted is False
    assert decision.deferred is True
    assert "paused" in decision.reason


def test_recovery_resume_admits_and_rechecks(gate, monkeypatch):
    # Recovery contract: the serving window ends -> resume() admits
    # training and pokes every queued service's _process_queue.
    monkeypatch.setattr(gate, "current_usage_bytes", lambda: 10 * 2**30)
    gate.pause("training")
    assert gate.admit_training(gate.training_budget_bytes()).deferred is True

    class _Queued:
        def __init__(self):
            self.rechecks = 0

        def _process_queue(self):
            self.rechecks += 1

    queued = _Queued()
    gate.add_recheck_target(queued)
    t0 = time.monotonic()
    snap = gate.resume()
    assert snap["paused"] is False
    assert queued.rechecks == 1, "queued service not poked on resume"
    decision = gate.admit_training(gate.training_budget_bytes())
    assert decision.admitted is True, decision.reason
    assert time.monotonic() - t0 < RECOVERY_BOUND_S


def test_media_rejected_never_deferred(gate, monkeypatch):
    # Media kinds never defer — over the line they are rejected so the
    # route maps to an explicit 507 (no queue, no hang).
    monkeypatch.setattr(gate, "current_usage_bytes", lambda: 100 * 2**30)
    decision = gate.admit_media("image", budget_bytes=20 * 2**30)
    assert decision.admitted is False
    assert decision.deferred is False


def test_laya_probe_defers_not_poisons(monkeypatch, caplog):
    # #1184 core shape: while training holds the GPU, laya probe
    # heartbeats go stale — the watchdog must DEFER (training active),
    # not poison the laya executor, within the media defer budget.
    monkeypatch.setattr(ec, "_LLM_WATCHDOG_MEDIA_DEFER_S", 5.0)
    fired = []
    wd = ExecutorWatchdog(
        name="laya",
        window_s=0.5,
        on_hang=lambda name: fired.append(name),
        defer_check=lambda: True,
    )
    wd.begin_job("decide_probe")
    time.sleep(0.7)
    with caplog.at_level("WARNING"):
        assert wd.check() is False
    assert fired == []
    assert any("deferring" in r.message for r in caplog.records)
    # Training ends: the defer clears and the next probe is healthy again.
    wd.defer_check = lambda: False
    wd.heartbeat()
    wd.end_job()
    wd.begin_job("decide_probe_2")
    wd.heartbeat()
    assert wd.check() is False


def test_zero_silent_loss_training_queue(gate, monkeypatch):
    # Spike of training jobs while over the line: every job gets an
    # explicit decision (admitted/rejected-with-reason) — none dropped.
    monkeypatch.setattr(gate, "current_usage_bytes", lambda: 100 * 2**30)
    decisions = [gate.admit_training(gate.training_budget_bytes()) for _ in range(20)]
    assert len(decisions) == 20
    for d in decisions:
        assert d.admitted is False
        assert d.reason, "silent drop: rejected without reason"
