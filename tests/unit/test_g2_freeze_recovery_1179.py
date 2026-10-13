# SPDX-License-Identifier: Apache-2.0
"""Tests for #1179: G2 freeze — watchdog false-positive + recovery wedge.

Fix A: watchdog window scales with the pending prefill chunk size, and a
       cross-tick step-counter confirmation defers poison while the worker
       is still completing steps (slow, not hung).
Fix B: add_request bounds its wait on the single-worker MLX executor so a
       wedged/grinding worker fails the request fast instead of hanging
       every client forever (the add_request-tail freeze signature).
Fix C: EngineCore.close() runs scheduler teardown on a throwaway thread
       when the engine is dead — submitting to the abandoned executor
       queued behind the stuck step and fatal_exit'd the whole server.
Fix D: the watchdog schedules the dead-engine unload itself via the event
       loop, so recovery does not depend on new HTTP traffic arriving.
Fix E: /metrics renders in a worker thread under a hard timeout; a blocked
       render returns 503 instead of freezing the event loop.
"""

from __future__ import annotations

import asyncio
import inspect
import threading
import time
from types import SimpleNamespace

import pytest


def _import_engine_core():
    from fusion_mlx import engine_core

    return engine_core


def _import_metrics():
    from fusion_mlx.routes_internal import metrics

    return metrics


# ---------------------------------------------------------------------------
# Fix A: watchdog window scaling for large prefill chunks
# ---------------------------------------------------------------------------


class TestWatchdogWindowScaling:
    def test_window_fn_exists(self):
        ec = _import_engine_core()
        assert hasattr(ec, "_llm_watchdog_window_for_pending")

    def test_base_window_when_scheduler_none(self):
        ec = _import_engine_core()
        assert ec._llm_watchdog_window_for_pending(None) == ec._LLM_WATCHDOG_TIMEOUT_S

    def test_base_window_when_config_missing(self):
        ec = _import_engine_core()
        sched = SimpleNamespace(config=None)
        assert ec._llm_watchdog_window_for_pending(sched) == ec._LLM_WATCHDOG_TIMEOUT_S

    def test_base_window_when_chunk_zero(self):
        ec = _import_engine_core()
        sched = SimpleNamespace(config=SimpleNamespace(max_num_batched_tokens=0))
        assert ec._llm_watchdog_window_for_pending(sched) == ec._LLM_WATCHDOG_TIMEOUT_S

    def test_scales_with_chunk(self):
        ec = _import_engine_core()
        # 16000-token chunk at the 50 tok/s contention floor + 300 base.
        sched = SimpleNamespace(config=SimpleNamespace(max_num_batched_tokens=16000))
        expected = 16000 / ec._LLM_WATCHDOG_PREFILL_TOK_PER_S + 300.0
        assert ec._llm_watchdog_window_for_pending(sched) == pytest.approx(expected)

    def test_capped_at_max(self):
        ec = _import_engine_core()
        sched = SimpleNamespace(config=SimpleNamespace(max_num_batched_tokens=10**9))
        assert (
            ec._llm_watchdog_window_for_pending(sched) == ec._LLM_WATCHDOG_MAX_TIMEOUT_S
        )

    def test_tick_uses_dynamic_window(self):
        ec = _import_engine_core()
        src = inspect.getsource(ec._llm_watchdog_tick)
        assert "_llm_step_window" in src, (
            "#1179 Fix A: the tick must compare staleness against the "
            "per-burst window, not the static constant"
        )
        assert "now - hb > window" in src

    def test_engine_loop_arms_window(self):
        ec = _import_engine_core()
        src = inspect.getsource(ec.EngineCore._engine_loop)
        assert "_llm_watchdog_window_for_pending" in src, (
            "#1179 Fix A: _engine_loop must arm the scaled window before "
            "submitting the burst"
        )

    def test_max_timeout_env_override(self):
        ec = _import_engine_core()
        src = inspect.getsource(ec)
        assert "FUSION_LLM_WATCHDOG_MAX_TIMEOUT" in src


# ---------------------------------------------------------------------------
# Fix A (layer 2): cross-tick step-counter confirmation
# ---------------------------------------------------------------------------


class TestWatchdogProgressConfirm:
    def test_progress_globals_exist(self):
        ec = _import_engine_core()
        assert hasattr(ec, "_llm_watchdog_last_steps")
        assert hasattr(ec, "_llm_progress_defer_logged")

    def test_tick_reads_steps_counter(self):
        ec = _import_engine_core()
        src = inspect.getsource(ec._llm_watchdog_tick)
        assert "_steps_executed" in src, (
            "#1179: the tick must confirm the step counter froze before "
            "poisoning — a step that just completed means slow, not hung"
        )
        assert "_llm_watchdog_last_steps" in src

    def test_reset_clears_progress_state(self):
        ec = _import_engine_core()
        ec._llm_watchdog_last_steps = 4242
        ec._llm_progress_defer_logged = True
        ec.reset_llm_executor_poison()
        assert ec._llm_watchdog_last_steps == -1
        assert ec._llm_progress_defer_logged is False


# ---------------------------------------------------------------------------
# Fix B: add_request bounded wait
# ---------------------------------------------------------------------------


class TestAddRequestBoundedWait:
    def test_timeout_constant_exists(self):
        ec = _import_engine_core()
        assert hasattr(ec, "_ADD_REQUEST_SUBMIT_TIMEOUT_S")
        assert ec._ADD_REQUEST_SUBMIT_TIMEOUT_S > 0

    def test_add_request_wraps_wait_for(self):
        ec = _import_engine_core()
        src = inspect.getsource(ec.EngineCore.add_request)
        assert "asyncio.wait_for" in src, (
            "#1179 Fix B: the executor submit must be bounded or a wedged "
            "worker hangs every new request forever"
        )
        assert "_ADD_REQUEST_SUBMIT_TIMEOUT_S" in src

    def test_timeout_branch_aborts_late_insert(self):
        ec = _import_engine_core()
        src = inspect.getsource(ec.EngineCore.add_request)
        assert "abort_request" in src, (
            "#1179 Fix B: the timeout branch must enqueue the deferred "
            "abort so a late scheduler insert is cleaned up"
        )

    def _make_stub_engine(self, stall_s: float):
        import concurrent.futures

        ec = _import_engine_core()
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        aborted = []
        cleaned = []

        class StubScheduler:
            def add_request(self, request):
                time.sleep(stall_s)

            def abort_request(self, rid):
                aborted.append(rid)

        stub = SimpleNamespace(
            _running=True,
            _active_contexts={},
            _finished_at={},
            scheduler=StubScheduler(),
            _mlx_executor=executor,
            _cleanup_request=lambda rid: cleaned.append(rid),
            _wake_engine_loop=lambda: None,
            config=SimpleNamespace(stream_interval=1),
        )
        return ec, stub, aborted, cleaned, executor

    def test_timeout_fails_fast(self):
        # Worker stalls 1.0s; the bounded wait gives up at 0.2s. The late
        # scheduler insert is then aborted on the same worker (runs once
        # the stall elapses) — assert both the fast failure and the
        # eventual abort.
        ec, stub, aborted, cleaned, executor = self._make_stub_engine(stall_s=1.0)
        ec._ADD_REQUEST_SUBMIT_TIMEOUT_S = 0.2
        try:
            with pytest.raises(RuntimeError, match="admission timed out"):
                asyncio.run(ec.EngineCore.add_request(stub, "hello", request_id="r1"))
        finally:
            ec._ADD_REQUEST_SUBMIT_TIMEOUT_S = float(
                __import__("os").environ.get("FUSION_LLM_ADD_REQUEST_TIMEOUT", "120")
            )
            executor.shutdown(wait=True)
        assert cleaned == ["r1"], "#1179 Fix B: ctx must be cleaned on timeout"
        assert aborted == ["r1"], "#1179 Fix B: late insert must be aborted"

    def test_fast_path_returns_request_id(self):
        ec, stub, aborted, cleaned, executor = self._make_stub_engine(stall_s=0.0)
        ec._ADD_REQUEST_SUBMIT_TIMEOUT_S = 5.0
        try:
            rid = asyncio.run(ec.EngineCore.add_request(stub, "hello", request_id="r2"))
        finally:
            ec._ADD_REQUEST_SUBMIT_TIMEOUT_S = float(
                __import__("os").environ.get("FUSION_LLM_ADD_REQUEST_TIMEOUT", "120")
            )
            executor.shutdown(wait=True)
        assert rid == "r2"
        assert "r2" in stub._active_contexts, "ctx tracked for the consumer"
        assert aborted == []


# ---------------------------------------------------------------------------
# Fix C: close() teardown on a throwaway thread when the engine is dead
# ---------------------------------------------------------------------------


class TestCloseTeardownFreshThread:
    def test_close_has_fresh_thread_branch(self):
        ec = _import_engine_core()
        src = inspect.getsource(ec.EngineCore.close)
        assert "self._engine_dead" in src, (
            "#1179 Fix C: teardown must bypass the abandoned executor for a "
            "dead engine"
        )
        assert "threading.Thread" in src

    def test_dead_engine_teardown_skips_busy_executor(self):
        ec = _import_engine_core()
        ran_on = {}

        def _record(name):
            def fn():
                ran_on[name] = threading.current_thread().name

            return fn

        executor_submits = []

        class BusyExecutor:
            def submit(self, fn, *a, **kw):
                executor_submits.append(fn)
                raise RuntimeError("executor is shut down")

        class StubSched:
            paged_ssd_cache_manager = None
            shutdown = staticmethod(_record("shutdown"))
            deep_reset = staticmethod(_record("deep_reset"))

        stub = SimpleNamespace(
            _running=False,
            _wake_event=None,
            _task=None,
            _owns_model=False,
            _closed=False,
            _engine_dead=True,
            _engine_id="deadbeef-cafe",
            scheduler=StubSched(),
            _mlx_executor=BusyExecutor(),
            _active_contexts={},
            _finished_at={},
            _fail_unfinished_contexts=lambda msg: 0,
            _mlx_stream=None,
            model=None,
            tokenizer=None,
        )
        ec.EngineCore.close(stub)
        assert set(ran_on) == {
            "shutdown",
            "deep_reset",
        }, "#1179 Fix C: both teardown fns must run for a dead engine"
        for name, thread_name in ran_on.items():
            assert thread_name.startswith("mlx-teardown-"), (
                f"#1179 Fix C: {name} must run on a throwaway thread, got "
                f"{thread_name}"
            )
        assert executor_submits == [], (
            "#1179 Fix C: no submits may go to the abandoned executor when "
            "the engine is dead"
        )


# ---------------------------------------------------------------------------
# Fix D: watchdog schedules recovery without new traffic
# ---------------------------------------------------------------------------


class TestWatchdogSchedulesRecovery:
    def test_tick_schedules_reload_after_mark_dead(self):
        ec = _import_engine_core()
        src = inspect.getsource(ec._llm_watchdog_tick)
        assert "call_soon_threadsafe" in src, (
            "#1179 Fix D: the watchdog must schedule the dead-engine unload "
            "itself; the request-path guard may never fire while every "
            "request hangs"
        )
        assert "_trigger_poisoned_engine_reload" in src


# ---------------------------------------------------------------------------
# Fix E: /metrics render offloaded with a hard timeout
# ---------------------------------------------------------------------------


class TestMetricsRenderOffload:
    def test_handler_uses_wait_for(self):
        m = _import_metrics()
        src = inspect.getsource(m.prometheus_metrics)
        assert (
            "run_in_executor" in src
        ), "#1179 Fix E: render must run off the event loop"
        assert (
            "asyncio.wait_for" in src
        ), "#1179 Fix E: render must be bounded by a timeout"

    def test_slow_render_returns_503(self, monkeypatch):
        m = _import_metrics()

        def slow_render():
            time.sleep(1.0)
            return "never"

        monkeypatch.setattr(m, "render_prometheus_metrics", slow_render)
        monkeypatch.setattr(m, "_METRICS_RENDER_TIMEOUT_S", 0.1)
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(m.prometheus_metrics(_auth=True))
        assert exc_info.value.status_code == 503

    def test_fast_render_returns_body(self, monkeypatch):
        m = _import_metrics()
        monkeypatch.setattr(m, "render_prometheus_metrics", lambda: "OK")
        resp = asyncio.run(m.prometheus_metrics(_auth=True))
        assert resp.body == b"OK"

    def test_render_still_completes_inline(self):
        # The sync render function itself stays a plain function so any
        # internal caller (tests, admin) is unaffected.
        m = _import_metrics()
        assert not inspect.iscoroutinefunction(m.render_prometheus_metrics)
