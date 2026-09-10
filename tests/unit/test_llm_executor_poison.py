# SPDX-License-Identifier: Apache-2.0
"""G2/R-2: LLM executor poison + watchdog unit tests."""

import threading
import time

import pytest

from fusion_mlx import engine_core as ec
from fusion_mlx.engine_core import (
    is_llm_executor_poisoned,
    poison_executor,
    reset_llm_executor_poison,
    start_llm_watchdog,
    stop_llm_watchdog,
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
