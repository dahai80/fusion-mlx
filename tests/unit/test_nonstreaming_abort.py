# Hard-pressure abort for non-streaming engines (image/video/audio).
#
# Bug (observed 2026-09-15, server.log): hard memory pressure picked busy
# image victim 'AITRADER--FLUX2-dev-mlx-8bit' but BaseNonStreamingEngine had
# no abort_all_requests — hasattr() was False, aborted stayed 0, the job
# kept running, and the enforcer logged the same warning every ~1s (1467x)
# until the process died. Three layers fixed here:
# 1. BaseNonStreamingEngine.abort_all_requests cancels activity tasks.
# 2. In-process image path shields the executor future so cancel does not
#    mark the engine idle while the worker thread still runs (otherwise the
#    pool would unload the model mid-generation and crash).
# 3. Enforcer throttles the per-poll warning and escalates to ERROR.

import asyncio
import logging
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from fusion_mlx.engines.base import BaseNonStreamingEngine


class _Engine(BaseNonStreamingEngine):
    @property
    def model_name(self) -> str:
        return "fake-engine"

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    def get_stats(self) -> dict:
        return {}


class TestBaseAbort:
    async def test_abort_cancels_activity_task(self):
        eng = _Engine()
        started = asyncio.Event()

        async def job():
            activity_id = eng._begin_activity("job")
            started.set()
            try:
                await asyncio.sleep(30)
            finally:
                eng._end_activity(activity_id)

        task = asyncio.create_task(job())
        await started.wait()
        assert eng.has_active_requests()

        aborted = await eng.abort_all_requests()
        assert aborted == 1
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not eng.has_active_requests()

    async def test_abort_does_not_recancel_cancelled_task(self):
        # Repeated hard-pressure polls must not re-cancel a task already
        # unwinding (a second cancel would break shield-wait cleanup).
        eng = _Engine()
        started = asyncio.Event()

        async def job():
            activity_id = eng._begin_activity("job")
            started.set()
            try:
                await asyncio.sleep(30)
            finally:
                eng._end_activity(activity_id)

        task = asyncio.create_task(job())
        await started.wait()
        assert await eng.abort_all_requests() == 1
        assert await eng.abort_all_requests() == 0
        with pytest.raises(asyncio.CancelledError):
            await task

    async def test_abort_with_no_activities_returns_zero(self):
        eng = _Engine()
        assert await eng.abort_all_requests() == 0

    async def test_abort_skips_activity_without_task(self):
        # _begin_activity outside a task (plain worker thread) tracks no
        # task; abort must not crash or count it.
        eng = _Engine()
        activity_id = await asyncio.to_thread(eng._begin_activity, "sync-job")
        assert await eng.abort_all_requests() == 0
        eng._end_activity(activity_id)


class _FakeImage:
    def save(self, buf, format="PNG"):
        buf.write(b"fakepng")


class TestImageGenCancelShield:
    async def test_cancel_holds_activity_until_thread_finishes(self, monkeypatch):
        from fusion_mlx.engines import image_gen as ig

        monkeypatch.setenv("FUSION_IMAGE_SUBPROCESS", "0")
        monkeypatch.setattr(ig, "get_image_gen_timeout", lambda *a, **k: 60.0)

        eng = ig.ImageGenEngine("/tmp/fake-model", variant="txt2img")

        entered = threading.Event()
        release = threading.Event()

        def fake_generate_image(**kwargs):
            entered.set()
            assert release.wait(10), "test thread never released"
            return SimpleNamespace(image=_FakeImage())

        eng._flux = SimpleNamespace(generate_image=fake_generate_image)

        task = asyncio.create_task(eng.generate("a cat"))
        assert await asyncio.to_thread(entered.wait, 10)

        aborted = await eng.abort_all_requests()
        assert aborted == 1

        # The worker thread is still inside generate_image: the engine must
        # stay busy so the pool cannot unload the model mid-generation.
        assert eng.has_active_requests()

        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not eng.has_active_requests()

    async def test_cancel_kills_subprocess_job_and_frees_activity(self, monkeypatch):
        # Subprocess mode: cancelling the activity task must kill the
        # worker (run_image_job finally) and end the activity.
        from fusion_mlx.engines import image_gen as ig

        monkeypatch.setenv("FUSION_IMAGE_SUBPROCESS", "1")

        eng = ig.ImageGenEngine("/tmp/fake-model", variant="txt2img")
        # generate() fast-fails when _flux is None; subprocess mode never
        # touches it, a bare sentinel is enough.
        eng._flux = SimpleNamespace()

        killed = threading.Event()

        class _FakeMgr:
            async def run_image_job(self, **kwargs):
                try:
                    await asyncio.sleep(30)
                except asyncio.CancelledError:
                    killed.set()
                    raise

        monkeypatch.setattr(ig.ImageGenEngine, "_media_mgr", _FakeMgr(), raising=False)

        task = asyncio.create_task(eng.generate("a cat"))
        await asyncio.sleep(0.1)
        assert eng.has_active_requests()

        aborted = await eng.abort_all_requests()
        assert aborted == 1
        with pytest.raises(asyncio.CancelledError):
            await task
        assert killed.is_set()
        assert not eng.has_active_requests()


class TestEnforcerHardBusyThrottle:
    def _make_enforcer(self, abort_result):
        from fusion_mlx.pool.memory_enforcer import ProcessMemoryEnforcer

        entry = SimpleNamespace(
            model_id="m",
            engine=SimpleNamespace(
                abort_all_requests=AsyncMock(return_value=abort_result)
            ),
            is_pinned=False,
            is_loading=False,
            in_use=1,
            pending_unload_reason=None,
            abort_requested=False,
            last_access=0.0,
        )
        pool = Mock()
        pool._entries = {"m": entry}
        pool._lock = asyncio.Lock()
        pool._find_lru_victim = Mock(return_value=None)
        pool._find_pending_unload_ready_locked = Mock(return_value=None)
        pool._mark_pending_unload_locked = Mock(return_value=True)
        pool._unload_pending_if_idle_locked = AsyncMock(return_value=False)

        enf = ProcessMemoryEnforcer(engine_pool=pool)
        enf._current_usage_bytes = Mock(return_value=99)
        enf._get_hard_limit_bytes = Mock(return_value=100)
        enf._shrink_hot_cache_for_pressure = Mock(return_value=0)
        enf._is_emergency_pressure = Mock(return_value=False)
        enf._walk_store_cache_caps = Mock()
        enf._propagate_memory_limit = Mock()
        enf._find_lru_busy_non_pinned_victim_locked = Mock(return_value="m")
        return enf, entry

    async def test_throttle_first_then_every_60th(self, caplog):
        enf, entry = self._make_enforcer(0)
        with caplog.at_level(logging.WARNING, logger="fusion_mlx.pool.memory_enforcer"):
            for _ in range(61):
                await enf._check_and_enforce()
        warns = [
            r for r in caplog.records if "requested abort/unload" in r.getMessage()
        ]
        # cycle 1 and cycle 60 only — not 61 identical warnings
        assert len(warns) == 2, [r.getMessage() for r in warns]
        assert "stuck_cycles=1" in warns[0].getMessage()
        assert "stuck_cycles=60" in warns[1].getMessage()
        assert entry.engine.abort_all_requests.await_count == 61

    async def test_escalates_to_error_at_300_cycles(self, caplog):
        enf, entry = self._make_enforcer(0)
        with caplog.at_level(logging.WARNING, logger="fusion_mlx.pool.memory_enforcer"):
            for _ in range(300):
                await enf._check_and_enforce()
        errors = [
            r
            for r in caplog.records
            if r.levelno == logging.ERROR and "stuck on busy victim" in r.getMessage()
        ]
        assert len(errors) == 1
        assert "FUSION_IMAGE_SUBPROCESS=1" in errors[0].getMessage()

    async def test_abort_actually_cancelled_resets_nothing_but_counts(self, caplog):
        # aborted>0 (real cancel) still throttles the log; the counter is
        # about the victim staying busy, not the aborted count.
        enf, entry = self._make_enforcer(1)
        with caplog.at_level(logging.WARNING, logger="fusion_mlx.pool.memory_enforcer"):
            for _ in range(3):
                await enf._check_and_enforce()
        warns = [
            r for r in caplog.records if "requested abort/unload" in r.getMessage()
        ]
        assert len(warns) == 1
        assert "aborted=1" in warns[0].getMessage()
