# SPDX-License-Identifier: Apache-2.0
"""Tests for PR-J: C++ EngineRunner + Python inline fallback.

Native path tests are skipped when the shim extension is not built (CI /
non-macOS). Fallback path tests always run.
"""

from __future__ import annotations

import threading

import pytest

from fusion_mlx.shim.fast import (
    _InlineEngineRunner,
    engine_runner,
    is_engine_runner_enabled,
    is_native_available,
)


def _enable(monkeypatch):
    monkeypatch.setenv("FUSION_ENGINE_RUNNER", "1")


def _disable(monkeypatch):
    monkeypatch.setenv("FUSION_ENGINE_RUNNER", "0")


class TestSwitch:
    def test_default_off(self, monkeypatch):
        monkeypatch.delenv("FUSION_ENGINE_RUNNER", raising=False)
        assert is_engine_runner_enabled() is False

    def test_env_on(self, monkeypatch):
        _enable(monkeypatch)
        assert is_engine_runner_enabled() is True


class TestInlineFallback:
    def test_disabled_returns_inline(self, monkeypatch):
        _disable(monkeypatch)
        r = engine_runner()
        assert isinstance(r, _InlineEngineRunner)
        assert r.is_running() is False

    def test_disabled_submit_runs_on_calling_thread(self, monkeypatch):
        _disable(monkeypatch)
        r = engine_runner()
        seen = []
        code, msg = r.submit(lambda: seen.append(threading.get_ident()))
        assert (code, msg) == (0, "")
        assert seen == [threading.get_ident()]

    def test_disabled_start_is_noop(self, monkeypatch):
        _disable(monkeypatch)
        r = engine_runner()
        assert r.start() is False
        r.stop()
        assert r.is_running() is False

    def test_disabled_submit_propagates_exception(self, monkeypatch):
        _disable(monkeypatch)
        r = engine_runner()
        with pytest.raises(ValueError):
            r.submit(lambda: (_ for _ in ()).throw(ValueError("boom")))

    def test_inline_stats_shape(self):
        r = _InlineEngineRunner(qos=2)
        s = r.stats()
        assert s == {
            "submitted": 0,
            "completed": 0,
            "failed": 0,
            "thread_started": 0,
            "thread_stopped": 0,
            "qos_class": 2,
        }

    def test_enabled_without_native_still_inline(self, monkeypatch):
        _enable(monkeypatch)
        if is_native_available():
            pytest.skip("native available — engine_runner returns native runner")
        r = engine_runner()
        assert isinstance(r, _InlineEngineRunner)


@pytest.mark.skipif(not is_native_available(), reason="shim _ext not built")
class TestNativeEngineRunner:
    def test_lifecycle(self, monkeypatch):
        _enable(monkeypatch)
        r = engine_runner()
        assert not isinstance(r, _InlineEngineRunner)
        try:
            assert r.start() is True
            assert r.is_running() is True
        finally:
            r.stop()
        assert r.is_running() is False

    def test_submit_runs_on_worker_thread(self, monkeypatch):
        _enable(monkeypatch)
        r = engine_runner()
        try:
            r.start()
            seen = []
            code, msg = r.submit(lambda: seen.append(threading.get_ident()))
            assert (code, msg) == (0, "")
            assert seen[0] != threading.get_ident()
        finally:
            r.stop()

    def test_submit_captures_exception(self, monkeypatch):
        _enable(monkeypatch)
        r = engine_runner()
        try:
            r.start()
            code, msg = r.submit(
                lambda: (_ for _ in ()).throw(ValueError("runner boom"))
            )
            assert code != 0
            assert "runner boom" in msg
        finally:
            r.stop()

    def test_stats_counters(self, monkeypatch):
        _enable(monkeypatch)
        r = engine_runner()
        try:
            r.start()
            r.submit(lambda: None)
            r.submit(lambda: (_ for _ in ()).throw(ValueError("x")))
            s = r.stats()
            assert s["submitted"] == 2
            assert s["completed"] == 1
            assert s["failed"] == 1
            assert s["thread_started"] == 1
        finally:
            r.stop()
        assert r.stats()["thread_stopped"] == 1

    def test_inline_after_stop(self, monkeypatch):
        _enable(monkeypatch)
        r = engine_runner()
        r.start()
        r.stop()
        seen = []
        code, msg = r.submit(lambda: seen.append(threading.get_ident()))
        assert (code, msg) == (0, "")
        assert seen == [threading.get_ident()]

    def test_disabled_uses_inline_even_with_native(self, monkeypatch):
        _disable(monkeypatch)
        r = engine_runner()
        assert isinstance(r, _InlineEngineRunner)

    def test_multi_submit_sequential(self, monkeypatch):
        _enable(monkeypatch)
        r = engine_runner()
        try:
            r.start()
            out = []
            for i in range(8):
                code, _ = r.submit(lambda i=i: out.append(i))
                assert code == 0
            assert out == list(range(8))
        finally:
            r.stop()
