# SPDX-License-Identifier: Apache-2.0
"""Tests for #1172/#1174: default TTL, eviction hang, wrong victim.

#1172: models without ttl_seconds in model_settings.json stay resident
       forever after generation, crowding out subsequent workloads. Fix:
       _resolve_effective_ttl returns _DEFAULT_TTL_SECONDS (600) as a
       last-resort fallback so non-pinned models always auto-evict.
#1174 Bug 1a: abort_all_requests ran mx.synchronize() on the event loop
       thread, blocking all engines' HTTP responses during eviction.
       Fix: moved to loop.run_in_executor(self._mlx_executor, ...).
#1174 Bug 1b: generate() awaited ctx.finished_event.wait() with no
       timeout — permanent hang if the event is never set. Fix: added
       asyncio.wait_for with FUSION_MLX_GENERATE_TIMEOUT_S (default 600).
#1174 Bug 2: soft pressure with no idle victim evicted an innocent
       bystander by aborting its active requests. Fix: under soft
       pressure, if no idle evictable model exists, warn + wait instead
       of aborting active requests. Hard pressure still falls through to
       the intentional busy-victim abort path.
"""

from __future__ import annotations

import ast
import inspect
import textwrap


def _import_engine_pool():
    from fusion_mlx.pool import engine_pool

    return engine_pool


def _import_engine_core():
    from fusion_mlx import engine_core

    return engine_core


def _import_memory_enforcer():
    from fusion_mlx.pool import memory_enforcer

    return memory_enforcer


# ---------------------------------------------------------------------------
# #1172: default TTL fallback
# ---------------------------------------------------------------------------


class TestDefaultTtlFallback:
    def _make_pool(self, settings_manager=None, global_idle=None):
        ep = _import_engine_pool()
        pool = ep.EnginePool.__new__(ep.EnginePool)
        pool._settings_manager = settings_manager
        pool._global_idle_timeout_seconds = lambda: global_idle
        return pool

    def test_default_ttl_constant_exists(self):
        ep = _import_engine_pool()
        assert hasattr(ep.EnginePool, "_DEFAULT_TTL_SECONDS")
        assert ep.EnginePool._DEFAULT_TTL_SECONDS == 600

    def test_returns_default_when_no_settings_manager(self):
        pool = self._make_pool(settings_manager=None)
        assert pool._resolve_effective_ttl("flux-1-dev") == 600

    def test_returns_default_when_get_settings_not_callable(self):
        pool = self._make_pool(settings_manager=object())
        assert pool._resolve_effective_ttl("flux-1-dev") == 600

    def test_returns_default_when_get_settings_raises(self):
        class Broken:
            def get_settings(self, model_id):
                raise RuntimeError("db down")

        pool = self._make_pool(settings_manager=Broken(), global_idle=None)
        assert pool._resolve_effective_ttl("flux-1-dev") == 600

    def test_returns_default_when_no_per_model_and_no_global(self):
        class Settings:
            ttl_seconds = None

        class Manager:
            def get_settings(self, model_id):
                return Settings()

        pool = self._make_pool(settings_manager=Manager(), global_idle=None)
        assert pool._resolve_effective_ttl("flux-1-dev") == 600

    def test_per_model_ttl_takes_precedence(self):
        class Settings:
            ttl_seconds = 120

        class Manager:
            def get_settings(self, model_id):
                return Settings()

        pool = self._make_pool(settings_manager=Manager(), global_idle=300)
        assert pool._resolve_effective_ttl("flux-1-dev") == 120

    def test_global_idle_takes_precedence_over_default(self):
        class Settings:
            ttl_seconds = None

        class Manager:
            def get_settings(self, model_id):
                return Settings()

        pool = self._make_pool(settings_manager=Manager(), global_idle=180)
        assert pool._resolve_effective_ttl("flux-1-dev") == 180

    def test_never_returns_none(self):
        pool = self._make_pool(settings_manager=None, global_idle=None)
        result = pool._resolve_effective_ttl("any-model")
        assert result is not None
        assert result > 0


# ---------------------------------------------------------------------------
# #1174 Bug 1a: abort_all_requests must not block the event loop
# ---------------------------------------------------------------------------


class TestAbortAllRequestsOffEventLoop:
    def _abort_method_ast(self):
        ec = _import_engine_core()
        src = textwrap.dedent(inspect.getsource(ec.EngineCore.abort_all_requests))
        mod = ast.parse(src)
        return mod.body[0]

    def test_uses_run_in_executor(self):
        func = self._abort_method_ast()
        found = False
        for node in ast.walk(func):
            if isinstance(node, ast.Call):
                f = node.func
                if isinstance(f, ast.Attribute) and f.attr == "run_in_executor":
                    found = True
        assert found, (
            "#1174 Bug 1a: abort_all_requests must use "
            "loop.run_in_executor for mx.synchronize()"
        )

    def test_mlx_synchronize_not_called_directly(self):
        func = self._abort_method_ast()
        inner_func_ids = set()
        for node in ast.walk(func):
            if isinstance(node, ast.FunctionDef):
                inner_func_ids.add(node)
        direct_sync = []
        for node in ast.walk(func):
            if isinstance(node, ast.Call):
                f = node.func
                if isinstance(f, ast.Attribute) and f.attr == "synchronize":
                    inside_inner = False
                    for inner in inner_func_ids:
                        if node in ast.walk(inner):
                            inside_inner = True
                            break
                    if not inside_inner:
                        direct_sync.append(node)
        assert not direct_sync, (
            "#1174 Bug 1a: mx.synchronize() called at method top level in "
            "abort_all_requests — must be inside an inner function passed "
            "to run_in_executor"
        )

    def test_sync_wrapped_in_inner_function(self):
        func = self._abort_method_ast()
        inner_funcs = [n for n in ast.walk(func) if isinstance(n, ast.FunctionDef)]
        assert inner_funcs, (
            "#1174 Bug 1a: expected an inner function (e.g. _sync_clear_gc) "
            "wrapping mx.synchronize() for run_in_executor"
        )
        sync_in_inner = False
        for inner in inner_funcs:
            for node in ast.walk(inner):
                if isinstance(node, ast.Call):
                    f = node.func
                    if isinstance(f, ast.Attribute) and f.attr == "synchronize":
                        sync_in_inner = True
        assert sync_in_inner, (
            "#1174 Bug 1a: mx.synchronize() must be inside the inner "
            "function passed to run_in_executor"
        )


# ---------------------------------------------------------------------------
# #1174 Bug 1b: generate() must have a watchdog timeout
# ---------------------------------------------------------------------------


class TestGenerateWatchdogTimeout:
    def _generate_method_ast(self):
        ec = _import_engine_core()
        src = textwrap.dedent(inspect.getsource(ec.EngineCore.generate))
        mod = ast.parse(src)
        return mod.body[0]

    def test_uses_wait_for(self):
        func = self._generate_method_ast()
        found = False
        for node in ast.walk(func):
            if isinstance(node, ast.Call):
                f = node.func
                if isinstance(f, ast.Attribute) and f.attr == "wait_for":
                    found = True
                if isinstance(f, ast.Name) and f.id == "wait_for":
                    found = True
        assert found, (
            "#1174 Bug 1b: generate() must use asyncio.wait_for on "
            "ctx.finished_event.wait()"
        )

    def test_reads_timeout_from_env(self):
        func = self._generate_method_ast()
        src = ast.unparse(func) if hasattr(ast, "unparse") else ""
        assert "FUSION_MLX_GENERATE_TIMEOUT_S" in src, (
            "#1174 Bug 1b: generate() must read timeout from "
            "FUSION_MLX_GENERATE_TIMEOUT_S env var"
        )

    def test_handles_timeout_error(self):
        func = self._generate_method_ast()
        handlers = [n for n in ast.walk(func) if isinstance(n, ast.ExceptHandler)]
        has_timeout = any(
            (isinstance(h.type, ast.Name) and h.type.id == "TimeoutError")
            or (
                isinstance(h.type, ast.Tuple)
                and any(
                    isinstance(t, ast.Name) and t.id == "TimeoutError"
                    for t in h.type.elts
                )
            )
            for h in handlers
        )
        assert has_timeout, (
            "#1174 Bug 1b: generate() must catch TimeoutError and "
            "abort + raise RuntimeError"
        )

    def test_timeout_handler_calls_abort_and_cleanup(self):
        func = self._generate_method_ast()
        timeout_handler = None
        for node in ast.walk(func):
            if isinstance(node, ast.ExceptHandler):
                if isinstance(node.type, ast.Name) and node.type.id == "TimeoutError":
                    timeout_handler = node
                    break
        assert timeout_handler is not None, "No TimeoutError handler found"
        body_src = ast.unparse(ast.Module(body=timeout_handler.body, type_ignores=[]))
        assert (
            "abort_request" in body_src
        ), "#1174 Bug 1b: TimeoutError handler must call abort_request"
        assert (
            "_cleanup_request" in body_src
        ), "#1174 Bug 1b: TimeoutError handler must call _cleanup_request"
        assert (
            "RuntimeError" in body_src
        ), "#1174 Bug 1b: TimeoutError handler must raise RuntimeError"


# ---------------------------------------------------------------------------
# #1174 Bug 2: soft pressure must not abort active requests on bystander
# ---------------------------------------------------------------------------


class TestSoftPressureNoVictimDoesNotAbort:
    def _enforce_step_ast(self):
        me = _import_memory_enforcer()
        src = textwrap.dedent(
            inspect.getsource(me.ProcessMemoryEnforcer._check_and_enforce)
        )
        mod = ast.parse(src)
        return mod.body[0]

    def test_soft_pressure_breaks_without_evicting(self):
        func = self._enforce_step_ast()
        src = ast.unparse(func)
        assert 'new_level != "hard"' in src or "new_level != 'hard'" in src, (
            "#1174 Bug 2: soft-pressure no-victim path must guard on "
            "new_level != 'hard' so hard pressure still reaches the "
            "busy-victim fall-through"
        )

    def test_soft_no_victim_logs_warning(self):
        func = self._enforce_step_ast()
        src = ast.unparse(func)
        assert "Soft memory pressure" in src or "no idle evictable" in src, (
            "#1174 Bug 2: soft-pressure no-victim path must log a clear "
            "warning (operator needs to know memory will stay high)"
        )

    def test_hard_pressure_still_falls_through(self):
        func = self._enforce_step_ast()
        src = ast.unparse(func)
        assert "_find_lru_busy_non_pinned_victim_locked" in src, (
            "#1174 Bug 2: hard-pressure busy-victim path must remain "
            "reachable when no idle victim exists"
        )

    def test_soft_no_victim_does_not_call_abort_all(self):
        func = self._enforce_step_ast()
        src = ast.unparse(func)
        idx_soft_guard = src.find("new_level !=")
        assert idx_soft_guard != -1, "soft-pressure guard not found"
        idx_break = src.find("break", idx_soft_guard)
        assert idx_break != -1, "soft-pressure guard must break"
        soft_block = src[idx_soft_guard:idx_break]
        # #1174 Bug 2 (amended by #1473): soft pressure must not abort
        # active bystanders IMMEDIATELY. The #1473 hard tier may abort
        # after sustained polls, but only behind the escalation counter
        # guard — an unguarded abort_all_requests in the soft block is
        # the original bug.
        idx_abort = soft_block.find("abort_all_requests")
        if idx_abort != -1:
            idx_escalation = soft_block.find("_soft_busy_cycles >=")
            assert idx_escalation != -1 and idx_escalation < idx_abort, (
                "#1174 Bug 2: abort_all_requests in the soft-pressure "
                "no-victim path must be gated behind the #1473 "
                "sustained-poll escalation guard"
            )
