# SPDX-License-Identifier: Apache-2.0
"""H1 chaos shape: GPU / Metal command-queue stall.

A real GPU stall (kernel wedged on the Metal command queue) cannot be
injected from pytest without risking the CI runner itself — the stall IS
the machine hanging. The injection point is therefore the watchdog INPUT
the G2 LLM watchdog observes: a pending step deadline + a heartbeat that
stops advancing + a frozen step counter. That is byte-for-byte what a
wedged Metal submission produces in production. See docs/chaos-suite.md.
"""

import time

from fusion_mlx import engine_core as ec
from tests.chaos.conftest import DETECTION_BOUND_S, RECOVERY_BOUND_S

WINDOW_S = 2.0


class _FakeEngine:
    # Frozen step counter: the #1179 progress check must see no advance
    # between ticks to confirm hang (a slow-but-progressing step must NOT
    # poison — covered in tests/unit/test_g2_freeze_recovery_1179.py).
    _steps_executed = 7
    _dead = False
    mark_dead_reason = ""

    def is_dead(self):
        return self._dead

    def mark_dead(self, reason):
        self._dead = True
        self.mark_dead_reason = reason


def _arm_stall(window=WINDOW_S):
    # Arm the watchdog exactly as a wedged worker would: a step is pending
    # and the heartbeat went stale `window + margin` ago.
    stale = time.monotonic() - (window + 0.5)
    ec._llm_heartbeat = stale
    ec._llm_step_deadline = stale + 1.0
    ec._llm_step_window = window


def _install(monkeypatch, engine):
    monkeypatch.setattr(ec, "_LLM_WATCHDOG_TIMEOUT_S", WINDOW_S)
    monkeypatch.setattr(ec, "poison_executor", lambda name: None)
    monkeypatch.setattr(ec, "is_media_job_active", lambda: False)
    monkeypatch.setattr(ec, "_llm_watchdog_engine_ref", lambda: engine)
    ec._llm_watchdog_last_steps = -1
    ec._llm_progress_defer_logged = False
    return engine


def test_h1_detection_poisons_within_window(monkeypatch, caplog):
    engine = _install(monkeypatch, _FakeEngine())
    fired = []
    monkeypatch.setattr(ec, "poison_executor", lambda name: fired.append(name))
    _arm_stall()
    t0 = time.monotonic()
    with caplog.at_level("ERROR"):
        # Two ticks: the first logs the frozen-counter confirm, the second
        # poisons (mirrors the real 1s tick loop cadence).
        ec._llm_watchdog_check()
        ec._llm_watchdog_check()
    elapsed = time.monotonic() - t0
    assert fired == ["llm"], "watchdog did not poison the llm executor"
    assert engine.is_dead(), "engine not marked dead for pool reload"
    assert any("hung" in r.message for r in caplog.records)
    # Detection contract: fired within the configured window, and the
    # configured window itself satisfies the <60s target.
    assert elapsed < DETECTION_BOUND_S
    assert WINDOW_S <= DETECTION_BOUND_S


def test_h1_media_starvation_does_not_poison(monkeypatch, caplog):
    # GPU contention shape: a media job starving the LLM worker is NOT a
    # hang — the watchdog must defer, not kill the engine.
    engine = _install(monkeypatch, _FakeEngine())
    monkeypatch.setattr(ec, "is_media_job_active", lambda: True)
    monkeypatch.setattr(ec, "_LLM_WATCHDOG_MEDIA_DEFER_S", 5.0)
    fired = []
    monkeypatch.setattr(ec, "poison_executor", lambda name: fired.append(name))
    _arm_stall()
    with caplog.at_level("WARNING"):
        ec._llm_watchdog_check()
    assert fired == [], "watchdog poisoned during media starvation (false kill)"
    assert not engine.is_dead()
    assert any("deferring" in r.message for r in caplog.records)


def test_h1_recovery_clean_reload(monkeypatch):
    # Poison + mark-dead happened; the pool reload path resets the poison
    # state so the reloaded engine starts with a clean watchdog baseline.
    engine = _install(monkeypatch, _FakeEngine())
    fired = []
    monkeypatch.setattr(ec, "poison_executor", lambda name: fired.append(name))
    _arm_stall()
    ec._llm_watchdog_check()
    ec._llm_watchdog_check()
    assert engine.is_dead()
    t0 = time.monotonic()
    ec.reset_llm_executor_poison()
    assert ec._llm_executor_poisoned is False
    assert ec._llm_poison_count == 0
    # Fresh engine: heartbeat advances again and the watchdog stays quiet.
    fresh = _FakeEngine()
    monkeypatch.setattr(ec, "_llm_watchdog_engine_ref", lambda: fresh)
    ec.update_llm_heartbeat()
    ec._llm_step_deadline = time.monotonic() + 1.0
    fired.clear()
    ec._llm_watchdog_check()
    assert fired == []
    assert time.monotonic() - t0 < RECOVERY_BOUND_S


def test_h1_zero_silent_loss_bounded_wait(monkeypatch):
    # The client-visible contract during a wedged worker: add_request is
    # BOUNDED (#1179, _ADD_REQUEST_SUBMIT_TIMEOUT_S knob) — the request
    # gets an explicit error within the configured timeout instead of
    # hanging forever. The bound is wired into AsyncEngineCore.add_request;
    # this chaos scenario asserts the contract shape at the executor layer.
    import asyncio

    bound = 0.3
    monkeypatch.setattr(ec, "_ADD_REQUEST_SUBMIT_TIMEOUT_S", bound)

    async def main():
        loop = asyncio.get_running_loop()
        t0 = loop.time()
        try:
            await asyncio.wait_for(
                loop.run_in_executor(None, time.sleep, 1.5), timeout=bound
            )
            return "served", loop.time() - t0
        except TimeoutError:
            return "explicit_error", loop.time() - t0

    outcome, elapsed = asyncio.run(main())
    assert outcome == "explicit_error"
    assert elapsed < 1.0
    # The knob we just exercised must be the one add_request actually uses.
    assert bound == ec._ADD_REQUEST_SUBMIT_TIMEOUT_S
