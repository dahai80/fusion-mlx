# SPDX-License-Identifier: Apache-2.0
"""Tests for #1176: G2 watchdog recovery, timeout, memory, log rotation.

Bug 1: check_llm_executor_healthy triggers async engine reload when
       poisoned — without it the guard 503s every request forever
       (pool never sees is_dead() to trigger EF-1 lazy reload).
Bug 2: _LLM_WATCHDOG_TIMEOUT_S default raised from 120s to 300s so
       slow-but-progressing prefills (19k tokens under memory pressure)
       don't false-trigger the watchdog.
Bug 3: consecutive poison counter — after N poison events the stuck
       Metal threads hold unreclaimable GPU memory; fatal_exit so the
       supervisor restarts clean. Counter resets on successful reload.
Bug 4: start.sh rotates server.log to server.log.prev instead of
       truncating, preserving the crash trace for post-mortem.
"""

from __future__ import annotations

import inspect
import textwrap


def _import_engine_core():
    from fusion_mlx import engine_core

    return engine_core


def _import_guards():
    from fusion_mlx.api import _guards

    return _guards


# ---------------------------------------------------------------------------
# Bug 1: guard triggers async reload
# ---------------------------------------------------------------------------


class TestGuardTriggersReload:
    def test_trigger_function_exists(self):
        g = _import_guards()
        assert hasattr(g, "_trigger_poisoned_engine_reload")

    def test_guard_calls_trigger(self):
        g = _import_guards()
        src = textwrap.dedent(inspect.getsource(g.check_llm_executor_healthy))
        assert "_trigger_poisoned_engine_reload" in src, (
            "#1176 Bug 1: check_llm_executor_healthy must call "
            "_trigger_poisoned_engine_reload when poisoned"
        )

    def test_trigger_uses_lock_to_prevert_duplicates(self):
        g = _import_guards()
        src = inspect.getsource(g._trigger_poisoned_engine_reload)
        assert "_poison_reload_lock" in src, (
            "#1176 Bug 1: reload trigger must use a lock to prevent "
            "duplicate reload tasks"
        )

    def test_trigger_schedules_async_task(self):
        g = _import_guards()
        src = inspect.getsource(g._trigger_poisoned_engine_reload)
        assert (
            "create_task" in src or "run_in_executor" in src
        ), "#1176 Bug 1: reload trigger must schedule an async task"

    def test_trigger_finds_dead_engines(self):
        g = _import_guards()
        src = inspect.getsource(g._trigger_poisoned_engine_reload)
        assert "is_dead" in src, (
            "#1176 Bug 1: reload trigger must check is_dead() to find "
            "engines needing unload"
        )

    def test_trigger_unloads_dead_engines(self):
        g = _import_guards()
        src = inspect.getsource(g._trigger_poisoned_engine_reload)
        assert "unload_engine_async" in src, (
            "#1176 Bug 1: reload trigger must call unload_engine_async "
            "to evict dead engines"
        )


# ---------------------------------------------------------------------------
# Bug 2: watchdog timeout raised to 300s
# ---------------------------------------------------------------------------


class TestWatchdogTimeoutRaised:
    def test_default_timeout_is_300(self):
        # Check source, not runtime value — other tests in the suite
        # (test_llm_executor_poison, test_g2_media_defer) mutate
        # _LLM_WATCHDOG_TIMEOUT_S directly without restoring it, so the
        # runtime value is unreliable depending on test ordering.
        ec = _import_engine_core()
        src = inspect.getsource(ec)
        assert (
            '"300"' in src
        ), "#1176 Bug 2: default watchdog timeout must be 300s in source"

    def test_env_override_documented_in_source(self):
        ec = _import_engine_core()
        src = inspect.getsource(ec)
        assert "FUSION_LLM_WATCHDOG_TIMEOUT" in src, (
            "#1176 Bug 2: _LLM_WATCHDOG_TIMEOUT_S must be overridable "
            "via FUSION_LLM_WATCHDOG_TIMEOUT env var"
        )
        assert '"300"' in src, "#1176 Bug 2: default must be 300 in source"


# ---------------------------------------------------------------------------
# Bug 3: poison counter + fatal_exit threshold
# ---------------------------------------------------------------------------


class TestPoisonCounterAndFatalExit:
    def test_counter_variable_exists(self):
        ec = _import_engine_core()
        assert hasattr(ec, "_llm_poison_count")

    def test_threshold_variable_exists(self):
        ec = _import_engine_core()
        assert hasattr(ec, "_LLM_POISON_FATAL_THRESHOLD")
        assert ec._LLM_POISON_FATAL_THRESHOLD == 3

    def test_reset_clears_counter(self):
        ec = _import_engine_core()
        ec._llm_poison_count = 5
        ec.reset_llm_executor_poison()
        assert (
            ec._llm_poison_count == 0
        ), "#1176 Bug 3: reset_llm_executor_poison must zero the counter"

    def test_watchdog_tick_increments_counter(self):
        ec = _import_engine_core()
        src = textwrap.dedent(inspect.getsource(ec._llm_watchdog_check))
        assert (
            "_llm_poison_count" in src
        ), "#1176 Bug 3: watchdog tick must increment _llm_poison_count"

    def test_watchdog_tick_calls_fatal_exit_at_threshold(self):
        ec = _import_engine_core()
        src = textwrap.dedent(inspect.getsource(ec._llm_watchdog_check))
        assert "fatal_exit" in src, (
            "#1176 Bug 3: watchdog tick must call fatal_exit when "
            "poison count exceeds threshold"
        )

    def test_watchdog_tick_has_global_decl_for_counter(self):
        ec = _import_engine_core()
        src = inspect.getsource(ec._llm_watchdog_check)
        global_lines = [
            line for line in src.split("\n") if line.strip().startswith("global ")
        ]
        assert any(
            "_llm_poison_count" in line for line in global_lines
        ), "#1176 Bug 3: _llm_watchdog_check must declare global _llm_poison_count"


# ---------------------------------------------------------------------------
# Bug 4: start.sh rotates log instead of truncating
# ---------------------------------------------------------------------------


class TestStartShLogRotation:
    def _read_start_sh(self):
        with open("start.sh") as f:
            return f.read()

    def test_no_bare_truncate_of_server_log(self):
        content = self._read_start_sh()
        assert ': > "${LOG_DIR}/server.log"' not in content, (
            "#1176 Bug 4: start.sh must not truncate server.log with "
            "': >' — this destroys the crash trace"
        )

    def test_rotates_to_prev(self):
        content = self._read_start_sh()
        assert "server.log.prev" in content, (
            "#1176 Bug 4: start.sh must rotate server.log to "
            "server.log.prev before starting a new server"
        )

    def test_uses_mv_not_truncate(self):
        content = self._read_start_sh()
        assert (
            'mv "${LOG_DIR}/server.log" "${LOG_DIR}/server.log.prev"' in content
        ), "#1176 Bug 4: start.sh must use mv to rotate, not truncate"
