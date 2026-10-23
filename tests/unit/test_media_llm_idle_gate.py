# SPDX-License-Identifier: Apache-2.0
# #1333: media job LLM-idle defer gate — wait_for_llm_idle +
# llm_engines_busy. No real engines or GPU needed.
from unittest.mock import patch

from fusion_mlx import engine_core
from fusion_mlx.engines.image_gen import wait_for_llm_idle


class _FakeEngine:
    def __init__(self, active):
        self._active = active

    def get_stats_snapshot(self):
        return {"active_requests": self._active}


class TestWaitForLlmIdle:
    async def test_disabled_returns_zero(self):
        waited = await wait_for_llm_idle(0.0, poll_s=0.01)
        assert waited == 0.0

    async def test_llm_idle_starts_immediately(self):
        with patch.object(engine_core, "llm_engines_busy", return_value=False):
            waited = await wait_for_llm_idle(60.0, poll_s=0.01)
        assert waited == 0.0

    async def test_waits_until_llm_drains(self):
        calls = {"n": 0}

        def busy():
            calls["n"] += 1
            return calls["n"] < 3  # busy for first 2 polls, idle after

        with patch.object(engine_core, "llm_engines_busy", side_effect=busy):
            waited = await wait_for_llm_idle(60.0, poll_s=0.01)
        assert calls["n"] == 3
        assert 0.01 <= waited < 60.0

    async def test_cap_reached_starts_anyway(self):
        with patch.object(engine_core, "llm_engines_busy", return_value=True):
            waited = await wait_for_llm_idle(0.05, poll_s=0.01)
        assert 0.04 <= waited <= 0.1


class TestLlmEnginesBusy:
    def test_no_engines_not_busy(self):
        with patch.dict(engine_core.__dict__, {"_llm_watchdog_engine_refs": {}}):
            assert engine_core.llm_engines_busy() is False

    def test_active_engine_is_busy(self):
        refs = {"a": lambda: _FakeEngine(2)}
        with patch.dict(engine_core.__dict__, {"_llm_watchdog_engine_refs": refs}):
            assert engine_core.llm_engines_busy() is True

    def test_idle_engines_not_busy(self):
        refs = {
            "a": lambda: _FakeEngine(0),
            "b": lambda: _FakeEngine(0),
        }
        with patch.dict(engine_core.__dict__, {"_llm_watchdog_engine_refs": refs}):
            assert engine_core.llm_engines_busy() is False

    def test_dead_weakref_and_failing_engine_skipped(self):
        def _boom():
            raise RuntimeError("snapshot failed")

        class _Broken:
            get_stats_snapshot = _boom

        refs = {
            "dead": lambda: None,
            "broken": lambda: _Broken(),
            "idle": lambda: _FakeEngine(0),
        }
        with patch.dict(engine_core.__dict__, {"_llm_watchdog_engine_refs": refs}):
            assert engine_core.llm_engines_busy() is False
