# SPDX-License-Identifier: Apache-2.0
"""Tests for #1169/#1170/#1171: worker hang, poison 503, VRAM allocator.

#1171: OOM retry path must not self-deadlock on the single-worker llm
       executor (submit().result() from the worker itself).
#1169: chat/completions/anthropic/ollama routes fast-fail 503 when the
       LLM executor is poisoned (G2 watchdog fired).
#1170: VramAllocator facade + tier budgets + background TDM scheduler
       (already wired in _check_ttl).
"""

from __future__ import annotations

import inspect

import pytest


def _import_sched_batch():
    from fusion_mlx.scheduler import sched_batch

    return sched_batch


def _import_engine_core():
    from fusion_mlx import engine_core

    return engine_core


def _import_guards():
    from fusion_mlx.api import _guards

    return _guards


def _import_engine_pool():
    from fusion_mlx.pool import engine_pool

    return engine_pool


class TestOomRetryNoSelfDeadlock:
    def _oom_retry_ast(self):
        # Parse _step_prefill_chunk and extract the OOM except-handler body
        # as a list of AST nodes, so we inspect executable calls — not
        # comment text (comments mention the prior broken pattern).
        import ast

        sb = _import_sched_batch()
        src = inspect.getsource(sb._step_prefill_chunk)
        mod = ast.parse(src)
        func = mod.body[0]
        oom_handler_body = []
        for node in ast.walk(func):
            if isinstance(node, ast.ExceptHandler):
                # The OOM retry handler is the one containing clear_cache
                handler_src = ast.unparse(node) if hasattr(ast, "unparse") else ""
                if "clear_cache" in handler_src:
                    oom_handler_body = node.body
                    break
        return oom_handler_body

    def test_no_cross_executor_submit_in_oom_retry(self):
        import ast

        body = self._oom_retry_ast()
        assert body, "OOM except-handler with clear_cache not found"
        for node in ast.walk(ast.Module(body=body, type_ignores=[])):
            if isinstance(node, ast.Call):
                # No .submit() attribute access in the retry path
                func = node.func
                if isinstance(func, ast.Attribute) and func.attr == "submit":
                    pytest.fail("#1171: .submit() found in OOM retry — self-deadlock")
                if isinstance(func, ast.Name) and func.id == "get_mlx_executor":
                    pytest.fail("#1171: get_mlx_executor() call in OOM retry path")

    def test_oom_retry_clears_cache_inline(self):
        import ast

        body = self._oom_retry_ast()
        assert body, "OOM except-handler not found"
        found_clear = False
        for node in ast.walk(ast.Module(body=body, type_ignores=[])):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Attribute) and func.attr == "clear_cache":
                    found_clear = True
        assert found_clear, "OOM retry must call mx.clear_cache() inline"

    def test_oom_retry_skips_mx_synchronize(self):
        import ast

        body = self._oom_retry_ast()
        assert body, "OOM except-handler not found"
        for node in ast.walk(ast.Module(body=body, type_ignores=[])):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Attribute) and func.attr == "synchronize":
                    pytest.fail(
                        "#1171: mx.synchronize() in OOM retry — blocks forever "
                        "on a wedged command buffer"
                    )


class TestPoison503Guard:
    def test_check_llm_executor_healthy_exists(self):
        g = _import_guards()
        assert hasattr(g, "check_llm_executor_healthy")
        assert callable(g.check_llm_executor_healthy)

    def test_poisoned_openai_raises_503(self):
        from fastapi import HTTPException

        g = _import_guards()
        ec = _import_engine_core()
        ec._llm_executor_poisoned = True
        try:
            with pytest.raises(HTTPException) as exc_info:
                g.check_llm_executor_healthy("test-model", adapter="openai")
            assert exc_info.value.status_code == 503
            detail = exc_info.value.detail
            assert detail["error"]["type"] == "server_busy"
            assert exc_info.value.headers["Retry-After"] == "5"
        finally:
            ec._llm_executor_poisoned = False

    def test_poisoned_anthropic_raises_503(self):
        from fastapi import HTTPException

        g = _import_guards()
        ec = _import_engine_core()
        ec._llm_executor_poisoned = True
        try:
            with pytest.raises(HTTPException) as exc_info:
                g.check_llm_executor_healthy("test-model", adapter="anthropic")
            assert exc_info.value.status_code == 503
            detail = exc_info.value.detail
            assert detail["error"]["type"] == "resource_exhausted"
        finally:
            ec._llm_executor_poisoned = False

    def test_not_poisoned_passes_through(self):
        g = _import_guards()
        ec = _import_engine_core()
        ec._llm_executor_poisoned = False
        # Should not raise
        g.check_llm_executor_healthy("test-model", adapter="openai")

    def test_chat_route_calls_poison_check(self):
        from fusion_mlx.api.openai import chat

        src = inspect.getsource(chat.chat_completions)
        assert (
            "check_llm_executor_healthy" in src
        ), "#1169: chat_completions route must call check_llm_executor_healthy."

    def test_completions_route_calls_poison_check(self):
        import importlib

        comp_mod = importlib.import_module("fusion_mlx.api.openai.completions")
        src = inspect.getsource(comp_mod.completions)
        assert "check_llm_executor_healthy" in src

    def test_anthropic_route_calls_poison_check(self):
        from fusion_mlx.api import anthropic_routes

        src = inspect.getsource(anthropic_routes.anthropic_messages)
        assert "check_llm_executor_healthy" in src

    def test_ollama_chat_route_calls_poison_check(self):
        from fusion_mlx.api import ollama_routes

        src = inspect.getsource(ollama_routes.api_chat)
        assert "check_llm_executor_healthy" in src

    def test_ollama_generate_route_calls_poison_check(self):
        from fusion_mlx.api import ollama_routes

        src = inspect.getsource(ollama_routes.api_generate)
        assert "check_llm_executor_healthy" in src


class TestVramAllocator:
    def test_vram_allocator_class_exists(self):
        ep = _import_engine_pool()
        assert hasattr(ep, "VramAllocator")
        assert hasattr(ep.VramAllocator, "tiers")
        assert hasattr(ep.VramAllocator, "set_budget")
        assert hasattr(ep.VramAllocator, "check_load")
        assert hasattr(ep.VramAllocator, "swap_idle_p1")

    def test_tier_budget_methods_exist(self):
        ep = _import_engine_pool()
        assert hasattr(ep.EnginePool, "set_tier_budget")
        assert hasattr(ep.EnginePool, "tier_resident_bytes")
        assert hasattr(ep.EnginePool, "check_tier_budget")

    def test_check_tier_budget_unbounded_default(self):
        ep = _import_engine_pool()
        pool = ep.EnginePool.__new__(ep.EnginePool)
        pool._tier_budgets_bytes = {"P0": 0, "P1": 0}
        pool._entries = {}
        ok, reason = pool.check_tier_budget("P0_DYNAMIC", 10 * 1024**3)
        assert ok
        assert reason == "unbounded"

    def test_check_tier_budget_pinned_bypass(self):
        ep = _import_engine_pool()
        pool = ep.EnginePool.__new__(ep.EnginePool)
        pool._tier_budgets_bytes = {"P0": 1, "P1": 1}
        pool._entries = {}
        ok, reason = pool.check_tier_budget("P0_NEVER_EVICT", 10 * 1024**3)
        assert ok
        assert reason == "pinned-bypass"

    def test_check_tier_budget_rejects_overrun(self):
        ep = _import_engine_pool()
        pool = ep.EnginePool.__new__(ep.EnginePool)
        pool._tier_budgets_bytes = {"P0": 0, "P1": 20 * 1024**3}
        pool._entries = {}

        # Simulate 15GB already resident in P1
        class _FakeEntry:
            def __init__(self, tier, size, loaded):
                self.priority_tier = tier
                self.actual_size = size
                self.estimated_size = size
                self.engine = object() if loaded else None

        pool._entries = {
            "diffusion-A": _FakeEntry("P1_TDM", 15 * 1024**3, True),
        }
        ok, reason = pool.check_tier_budget("P1_TDM", 10 * 1024**3)
        assert not ok
        assert "budget exceeded" in reason

    def test_check_tier_budget_allows_within_budget(self):
        ep = _import_engine_pool()
        pool = ep.EnginePool.__new__(ep.EnginePool)
        pool._tier_budgets_bytes = {"P0": 0, "P1": 30 * 1024**3}
        pool._entries = {}

        class _FakeEntry:
            def __init__(self, tier, size, loaded):
                self.priority_tier = tier
                self.actual_size = size
                self.estimated_size = size
                self.engine = object() if loaded else None

        pool._entries = {
            "diffusion-A": _FakeEntry("P1_TDM", 15 * 1024**3, True),
        }
        ok, reason = pool.check_tier_budget("P1_TDM", 10 * 1024**3)
        assert ok
        assert reason == "ok"

    def test_set_tier_budget(self):
        ep = _import_engine_pool()
        pool = ep.EnginePool.__new__(ep.EnginePool)
        pool._tier_budgets_bytes = {"P0": 0, "P1": 0}
        pool._entries = {}
        assert pool.set_tier_budget("P1_TDM", 25 * 1024**3)
        assert pool._tier_budgets_bytes["P1"] == 25 * 1024**3
        assert pool.set_tier_budget("P0_DYNAMIC", 80 * 1024**3)
        assert pool._tier_budgets_bytes["P0"] == 80 * 1024**3

    def test_set_tier_budget_invalid_tier(self):
        ep = _import_engine_pool()
        pool = ep.EnginePool.__new__(ep.EnginePool)
        pool._tier_budgets_bytes = {"P0": 0, "P1": 0}
        pool._entries = {}
        assert not pool.set_tier_budget("P9", 100)

    def test_vram_allocator_facade_delegates(self):
        ep = _import_engine_pool()
        pool = ep.EnginePool.__new__(ep.EnginePool)
        pool._tier_budgets_bytes = {"P0": 0, "P1": 0}
        pool._entries = {}
        alloc = ep.VramAllocator(pool)
        ok, reason = alloc.check_load("P0_DYNAMIC", 1024)
        assert ok
        assert alloc.resident_bytes() == {"P0": 0, "P1": 0}


class TestTdmBackgroundScheduler:
    def test_check_ttl_calls_swap_idle_tdm(self):
        # #1170 gap 3: background scheduler must call swap_idle_tdm_models.
        from fusion_mlx.pool import memory_enforcer

        src = inspect.getsource(memory_enforcer.ProcessMemoryEnforcer._check_ttl)
        assert (
            "swap_idle_tdm_models" in src
        ), "#1170 gap 3: _check_ttl must call swap_idle_tdm_models (background TDM scheduler)."

    def test_enforcement_loop_calls_check_ttl(self):
        from fusion_mlx.pool import memory_enforcer

        src = inspect.getsource(memory_enforcer.ProcessMemoryEnforcer._enforcement_loop)
        assert (
            "_check_ttl" in src
        ), "_enforcement_loop must call _check_ttl periodically (TDM background scheduler)."
