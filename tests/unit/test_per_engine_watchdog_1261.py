# SPDX-License-Identifier: Apache-2.0
"""#1261: per-engine LLM watchdog — a wedged engine must be detectable and
recoverable while sibling engines stay healthy.

Covers:
- per-engine heartbeat evaluation (engine_core.py _llm_watchdog_check)
- targeted recovery: mark dead + cancel loop task + fail contexts
- legacy global path preserved when no per-engine inputs exist
- dead-engine teardown override in EnginePool.unload_engine_async
- EngineCore loop-death done-callback
- /metrics per-engine step lag + dead flag
"""

from __future__ import annotations

import asyncio
import time
import types

import pytest

from fusion_mlx import engine_core as ec


class _WedgedEngine:
    # Fake EngineCore carrying the #1261 per-engine watchdog inputs.
    def __init__(self, eid: str = "wedge-1"):
        self._engine_id = eid
        self._hb = 0.0
        self._step_window = 0.0
        self._step_deadline = 0.0
        self._steps_executed = 7
        self._running = True
        self._engine_dead = False
        self.dead_reason = None
        self.failed_contexts: list[str] = []
        self._loop: asyncio.AbstractEventLoop | None = None
        self._task: asyncio.Task | None = None

    def is_dead(self):
        return self._engine_dead

    def mark_dead(self, reason):
        self._engine_dead = True
        self.dead_reason = reason

    def is_running(self):
        return self._running

    def _fail_unfinished_contexts(self, error_msg):
        self.failed_contexts.append(error_msg)
        return len(self.failed_contexts)


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    # Tests below assign ec._LLM_WATCHDOG_TIMEOUT_S directly; restore it so
    # the value never leaks into later suites (breaks window-scaling math).
    monkeypatch.setattr(ec, "_LLM_WATCHDOG_TIMEOUT_S", ec._LLM_WATCHDOG_TIMEOUT_S)
    with ec._llm_engine_ref_lock:
        ec._llm_watchdog_engine_refs.clear()
        ec._llm_watchdog_engine_ref = None
    ec._llm_watchdog_last_steps_by_id.clear()
    ec.reset_llm_executor_poison()
    ec.stop_llm_watchdog()
    yield
    with ec._llm_engine_ref_lock:
        ec._llm_watchdog_engine_refs.clear()
        ec._llm_watchdog_engine_ref = None
    ec._llm_watchdog_last_steps_by_id.clear()
    ec.reset_llm_executor_poison()
    ec.stop_llm_watchdog()


def _arm_wedged(engine: _WedgedEngine, window: float = 2.0) -> None:
    # Arm the watchdog exactly as a wedged worker would: a step is pending,
    # this engine's own heartbeat went stale, step counter frozen.
    stale = time.monotonic() - (window + 0.5)
    engine._hb = stale
    engine._step_deadline = stale + 1.0
    engine._step_window = window


class TestPerEngineDetection:
    def test_wedged_engine_detected_while_sibling_advances(self):
        # The core #1261 shape: 27B wedged (stale hb, frozen steps), 4B
        # healthy (fresh hb). The shared pre-#1261 heartbeat was masked by
        # the sibling; per-engine evaluation must fire on the wedged one.
        wedged = _WedgedEngine("wedge-a")
        healthy = _WedgedEngine("healthy-b")
        _arm_wedged(wedged)
        healthy._hb = time.monotonic()
        healthy._step_deadline = time.monotonic() + 10.0
        healthy._step_window = 300.0
        ec.register_llm_engine_for_watchdog(wedged)
        ec.register_llm_engine_for_watchdog(healthy)
        ec._LLM_WATCHDOG_TIMEOUT_S = 2.0
        ec._llm_watchdog_check()
        ec._llm_watchdog_check()
        assert wedged.is_dead(), "wedged engine not marked dead"
        assert "#1261" in (wedged.dead_reason or "")
        assert not healthy.is_dead(), "healthy sibling was touched"
        assert ec.is_llm_executor_poisoned() is False, (
            "per-engine recovery must not set the process-global poison "
            "flag — it would fast-fail every healthy sibling's loop"
        )

    def test_frozen_step_counter_confirms_before_recovery(self):
        engine = _WedgedEngine()
        ec.register_llm_engine_for_watchdog(engine)
        ec._LLM_WATCHDOG_TIMEOUT_S = 2.0
        _arm_wedged(engine)
        # First tick: step counter visible for the first time — deferred
        # (slow vs hung disambiguation, #1179 contract per engine).
        ec._llm_watchdog_check()
        assert not engine.is_dead()
        # Second tick: counter unchanged — hang confirmed.
        ec._llm_watchdog_check()
        assert engine.is_dead()

    def test_progressing_engine_not_recovered(self):
        engine = _WedgedEngine()
        ec.register_llm_engine_for_watchdog(engine)
        ec._LLM_WATCHDOG_TIMEOUT_S = 2.0
        _arm_wedged(engine)
        ec._llm_watchdog_check()
        engine._steps_executed += 4
        engine._hb = time.monotonic() - 3.0
        ec._llm_watchdog_check()
        assert not engine.is_dead()

    def test_idle_engine_without_pending_step_skipped(self):
        engine = _WedgedEngine()
        ec.register_llm_engine_for_watchdog(engine)
        ec._LLM_WATCHDOG_TIMEOUT_S = 2.0
        engine._hb = time.monotonic() - 999.0
        engine._step_deadline = 0.0
        ec._llm_watchdog_check()
        ec._llm_watchdog_check()
        assert not engine.is_dead()

    def test_dead_engine_not_refired(self):
        engine = _WedgedEngine()
        engine._engine_dead = True
        ec.register_llm_engine_for_watchdog(engine)
        ec._LLM_WATCHDOG_TIMEOUT_S = 2.0
        _arm_wedged(engine)
        ec._llm_watchdog_check()
        assert ec._llm_poison_count == 0


class TestTargetedRecovery:
    async def test_recovery_cancels_task_and_fails_contexts(self):
        engine = _WedgedEngine()
        ec.register_llm_engine_for_watchdog(engine)
        ec._LLM_WATCHDOG_TIMEOUT_S = 2.0

        async def _parked():
            await asyncio.Event().wait()

        engine._loop = asyncio.get_running_loop()
        engine._task = asyncio.create_task(_parked())
        _arm_wedged(engine)
        ec._llm_watchdog_check()
        ec._llm_watchdog_check()
        assert engine.is_dead()
        # Let the call_soon_threadsafe callbacks drain.
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert engine._task.cancelled() or engine._task.done()
        assert engine.failed_contexts, "unfinished contexts not failed"
        assert "hung" in engine.failed_contexts[0]

    def test_recovery_without_loop_marks_dead_only(self):
        engine = _WedgedEngine()
        ec.register_llm_engine_for_watchdog(engine)
        ec._LLM_WATCHDOG_TIMEOUT_S = 2.0
        _arm_wedged(engine)
        ec._llm_watchdog_check()
        ec._llm_watchdog_check()
        assert engine.is_dead()

    def test_poison_count_increments_per_wedge_event(self):
        engine = _WedgedEngine()
        ec.register_llm_engine_for_watchdog(engine)
        ec._LLM_WATCHDOG_TIMEOUT_S = 2.0
        _arm_wedged(engine)
        ec._llm_watchdog_check()
        ec._llm_watchdog_check()
        assert ec._llm_poison_count == 1

    def test_reset_clears_per_engine_progress_map(self):
        ec._llm_watchdog_last_steps_by_id["x"] = 5
        ec.reset_llm_executor_poison()
        assert ec._llm_watchdog_last_steps_by_id == {}


class TestRegistryLifecycle:
    def test_unregister_single_engine_keeps_others(self):
        a = _WedgedEngine("a")
        b = _WedgedEngine("b")
        ec.register_llm_engine_for_watchdog(a)
        ec.register_llm_engine_for_watchdog(b)
        ec.unregister_llm_engine_from_watchdog(a)
        assert ec.llm_watchdog_has_engines()
        ec.unregister_llm_engine_from_watchdog(b)
        assert not ec.llm_watchdog_has_engines()

    def test_noarg_unregister_clears_all(self):
        ec.register_llm_engine_for_watchdog(_WedgedEngine("a"))
        ec.unregister_llm_engine_from_watchdog()
        assert not ec.llm_watchdog_has_engines()
        assert ec._llm_watchdog_engine_ref is None

    def test_weakref_teardown_pruned(self):
        engine = _WedgedEngine()
        ec.register_llm_engine_for_watchdog(engine)
        del engine
        # Pruned on the next check; must not crash.
        ec._llm_watchdog_check()


class TestLoopDeathCallback:
    def test_unexpected_loop_exit_marks_dead(self):
        core = ec.EngineCore.__new__(ec.EngineCore)
        core._engine_dead = False
        core._running = True

        task = types.SimpleNamespace(
            cancelled=lambda: False,
            exception=lambda: RuntimeError("boom"),
        )
        core._on_loop_task_done(task)
        assert core._engine_dead
        assert "boom" in (core._engine_dead_reason or "")

    def test_normal_stop_not_marked_dead(self):
        core = ec.EngineCore.__new__(ec.EngineCore)
        core._engine_dead = False
        core._running = False
        task = types.SimpleNamespace(cancelled=lambda: False, exception=lambda: None)
        core._on_loop_task_done(task)
        assert not core._engine_dead

    def test_cancelled_task_not_marked_dead(self):
        core = ec.EngineCore.__new__(ec.EngineCore)
        core._engine_dead = False
        core._running = True
        task = types.SimpleNamespace(cancelled=lambda: True, exception=lambda: None)
        core._on_loop_task_done(task)
        assert not core._engine_dead


class _LegacyFake:
    def __init__(self):
        self.dead = False
        self.reason = None

    def is_dead(self):
        return self.dead

    def mark_dead(self, reason):
        self.dead = True
        self.reason = reason


class TestLegacyPathPreserved:
    def test_legacy_poison_still_fires_without_registry(self):
        # No per-engine inputs registered → the pre-#1261 global contract
        # (poison_executor + mark dead via the singular ref) holds.
        engine = _LegacyFake()
        ec._LLM_WATCHDOG_TIMEOUT_S = 2.0
        with ec._llm_engine_ref_lock:
            ec._llm_watchdog_engine_ref = lambda: engine
        with ec._llm_heartbeat_lock:
            stale = time.monotonic() - 5.0
            ec._llm_heartbeat = stale
            ec._llm_step_deadline = stale + 1.0
            ec._llm_step_window = 2.0
        try:
            ec._llm_watchdog_check()
            ec._llm_watchdog_check()
            assert ec.is_llm_executor_poisoned() is True
            assert engine.is_dead()
        finally:
            with ec._llm_engine_ref_lock:
                ec._llm_watchdog_engine_ref = None
            with ec._llm_heartbeat_lock:
                ec._llm_heartbeat = 0.0
                ec._llm_step_deadline = 0.0
                ec._llm_step_window = 0.0


class TestDeadEngineTeardownOverride:
    @staticmethod
    def _make_entry(model_id: str, engine):
        from fusion_mlx.pool.engine_pool import EngineEntry

        return EngineEntry(
            model_id=model_id,
            model_path=f"/tmp/{model_id}",
            model_type="llm",
            engine_type="batched",
            estimated_size=1,
        )

    async def test_unload_proceeds_despite_stale_leases(self):
        # #1261: a dead engine's leases are wedge victims being failed —
        # the teardown must proceed instead of deadlocking behind them.
        from fusion_mlx.pool.engine_pool import EnginePool

        pool = EnginePool()
        dead_engine = _WedgedEngine("dead-1")
        dead_engine._engine_dead = True
        entry = self._make_entry("wedged-model", dead_engine)
        entry.engine = dead_engine
        entry.in_use = 15  # the wedge signature from the incident
        pool._entries["wedged-model"] = entry

        detached: list[str] = []

        async def _fake_detach(model_id):
            detached.append(model_id)
            return 0

        pool._detach_engine = _fake_detach
        await pool.unload_engine_async("wedged-model", with_settle=False)
        assert detached == ["wedged-model"], "teardown deadlocked behind stale leases"

    async def test_live_engine_still_defers(self):
        # R-6 use-after-stop guard untouched for LIVE engines.
        from fusion_mlx.pool.engine_pool import EnginePool

        pool = EnginePool()
        live_engine = _WedgedEngine("live-1")
        entry = self._make_entry("live-model", live_engine)
        entry.engine = live_engine
        entry.in_use = 2
        pool._entries["live-model"] = entry

        detached: list[str] = []

        async def _fake_detach(model_id):
            detached.append(model_id)
            return 0

        pool._detach_engine = _fake_detach
        await pool.unload_engine_async("live-model", with_settle=False)
        assert detached == []
        assert entry.pending_unload_reason is not None


class TestMetrics:
    def test_step_lag_and_dead_flag_render(self, monkeypatch):
        from fusion_mlx import server as server_mod
        from fusion_mlx.pool.engine_pool import EngineEntry
        from fusion_mlx.routes_internal.metrics import _render_pool_metrics

        wedged = _WedgedEngine("lag-1")
        wedged._hb = time.monotonic() - 42.0
        wedged._engine_dead = True
        entry = EngineEntry(
            model_id="m",
            model_path="/tmp/m",
            model_type="llm",
            engine_type="batched",
            estimated_size=1,
        )
        entry.engine = wedged

        class _Pool:
            model_count = 1
            loaded_model_count = 1
            current_model_memory = 0
            _entries = {"m": entry}

        monkeypatch.setitem(server_mod._server_state, "engine_pool", _Pool())
        lines = _render_pool_metrics()
        text = "\n".join(lines)
        assert "fusion_mlx_engine_step_lag_seconds" in text
        assert "fusion_mlx_engine_dead" in text
        lag_line = next(
            l for l in lines if l.startswith("fusion_mlx_engine_step_lag_seconds")
        )
        assert 'model="m"' in lag_line
        value = float(lag_line.rsplit(" ", 1)[1])
        assert 40.0 <= value <= 60.0
