# SPDX-License-Identifier: Apache-2.0
"""G2/R-2: LLM executor poison + watchdog unit tests."""

import threading
import time

import pytest

from fusion_mlx import engine_core as ec
from fusion_mlx.engine_core import (
    is_llm_executor_poisoned,
    poison_executor,
    register_llm_engine_for_watchdog,
    reset_llm_executor_poison,
    start_llm_watchdog,
    stop_llm_watchdog,
    unregister_llm_engine_from_watchdog,
    update_llm_heartbeat,
)


class _FastEvent:
    def __init__(self):
        self._flag = threading.Event()

    def set(self):
        self._flag.set()

    def clear(self):
        self._flag.clear()

    def is_set(self):
        return self._flag.is_set()

    def wait(self, timeout=None):
        return self._flag.wait(0.05)


@pytest.fixture(autouse=True)
def _clean_poison_state(monkeypatch):
    reset_llm_executor_poison()
    stop_llm_watchdog()
    with ec._llm_heartbeat_lock:
        ec._llm_heartbeat = 0.0
        ec._llm_step_deadline = 0.0
    monkeypatch.setattr(ec, "_llm_watchdog_stop", _FastEvent())
    yield
    stop_llm_watchdog()
    reset_llm_executor_poison()
    with ec._llm_heartbeat_lock:
        ec._llm_step_deadline = 0.0


class TestLLMPoisonFlag:
    def test_default_not_poisoned(self):
        assert is_llm_executor_poisoned() is False

    def test_poison_executor_sets_flag(self):
        poison_executor("llm")
        assert is_llm_executor_poisoned() is True

    def test_reset_clears_flag(self):
        poison_executor("llm")
        assert is_llm_executor_poisoned() is True
        reset_llm_executor_poison()
        assert is_llm_executor_poisoned() is False


class TestHeartbeat:
    def test_update_sets_nonzero(self):
        update_llm_heartbeat()
        with ec._llm_heartbeat_lock:
            assert ec._llm_heartbeat > 0.0

    def test_update_advances(self):
        update_llm_heartbeat()
        with ec._llm_heartbeat_lock:
            t1 = ec._llm_heartbeat
        time.sleep(0.01)
        update_llm_heartbeat()
        with ec._llm_heartbeat_lock:
            assert ec._llm_heartbeat > t1


class TestWatchdog:
    def test_watchdog_starts_and_stops(self):
        start_llm_watchdog()
        assert ec._llm_watchdog_thread is not None
        assert ec._llm_watchdog_thread.is_alive()
        stop_llm_watchdog()

    def test_no_trigger_with_fresh_heartbeat(self):
        ec._LLM_WATCHDOG_TIMEOUT_S = 2.0
        start_llm_watchdog()
        with ec._llm_heartbeat_lock:
            ec._llm_step_deadline = time.monotonic() + 10.0
        update_llm_heartbeat()
        time.sleep(0.3)
        assert is_llm_executor_poisoned() is False
        stop_llm_watchdog()

    def test_triggers_on_stale_heartbeat(self):
        ec._LLM_WATCHDOG_TIMEOUT_S = 0.2
        start_llm_watchdog()
        with ec._llm_heartbeat_lock:
            ec._llm_step_deadline = time.monotonic() + 10.0
            ec._llm_heartbeat = time.monotonic() - 5.0
        time.sleep(0.6)
        assert is_llm_executor_poisoned() is True
        stop_llm_watchdog()

    def test_no_trigger_without_pending_step(self):
        ec._LLM_WATCHDOG_TIMEOUT_S = 0.2
        with ec._llm_heartbeat_lock:
            ec._llm_step_deadline = 0.0
            ec._llm_heartbeat = time.monotonic() - 999.0
        start_llm_watchdog()
        time.sleep(0.5)
        assert is_llm_executor_poisoned() is False
        stop_llm_watchdog()


class _FakeEngine:
    # Stand-in for EngineCore exposing is_dead / mark_dead so the watchdog's
    # mark-dead-on-hung wiring (#862) can be tested without a real MLX engine.
    def __init__(self):
        self._dead = False
        self.reason = None

    def is_dead(self):
        return self._dead

    def mark_dead(self, reason):
        self._dead = True
        self.reason = reason


class TestWatchdogAutoRebuild:
    # G2 auto-rebuild (#862): on hung-worker detection the watchdog must mark
    # the registered engine dead (so EnginePool's EF-1 lazy reload triggers)
    # in addition to setting the poison flag.

    def test_watchdog_marks_registered_engine_dead(self):
        ec._LLM_WATCHDOG_TIMEOUT_S = 0.2
        engine = _FakeEngine()
        register_llm_engine_for_watchdog(engine)
        try:
            start_llm_watchdog()
            with ec._llm_heartbeat_lock:
                ec._llm_step_deadline = time.monotonic() + 10.0
                ec._llm_heartbeat = time.monotonic() - 5.0
            time.sleep(0.6)
            assert is_llm_executor_poisoned() is True
            assert engine.is_dead() is True
            assert engine.reason is not None
            assert "G2" in engine.reason
            stop_llm_watchdog()
        finally:
            unregister_llm_engine_from_watchdog()

    def test_no_mark_dead_when_no_engine_registered(self):
        ec._LLM_WATCHDOG_TIMEOUT_S = 0.2
        unregister_llm_engine_from_watchdog()
        start_llm_watchdog()
        with ec._llm_heartbeat_lock:
            ec._llm_step_deadline = time.monotonic() + 10.0
            ec._llm_heartbeat = time.monotonic() - 5.0
        time.sleep(0.6)
        # Poison flag still set (fast-fail), but no engine to mark — no crash.
        assert is_llm_executor_poisoned() is True
        stop_llm_watchdog()

    def test_weakref_survives_engine_teardown(self):
        engine = _FakeEngine()
        register_llm_engine_for_watchdog(engine)
        ref = ec._llm_watchdog_engine_ref
        assert ref is not None and ref() is engine
        del engine
        # Weakref dies with the engine — watchdog tick must not crash.
        ec._LLM_WATCHDOG_TIMEOUT_S = 0.2
        start_llm_watchdog()
        with ec._llm_heartbeat_lock:
            ec._llm_step_deadline = time.monotonic() + 10.0
            ec._llm_heartbeat = time.monotonic() - 5.0
        time.sleep(0.6)
        assert is_llm_executor_poisoned() is True
        stop_llm_watchdog()
        unregister_llm_engine_from_watchdog()

    def test_reset_poison_clears_for_rebuild(self):
        poison_executor("llm")
        assert is_llm_executor_poisoned() is True
        reset_llm_executor_poison()
        assert is_llm_executor_poisoned() is False
