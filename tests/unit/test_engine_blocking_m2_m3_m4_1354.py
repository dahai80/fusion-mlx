# SPDX-License-Identifier: Apache-2.0
"""#1354 audit M2/M3/M4: admission and teardown must not block callers.

M2: add_request's post-timeout / post-cancel abort used fut.result(timeout=5),
    freezing the event loop (the #1174 silent-hang signature).
M3: admission's SSD preload wait blocked the single MLX worker for up to 10s;
    a timeout now skips prefix reuse for that request.
M4: aclose() cancelled the engine loop task and immediately tore down the
    scheduler/model under it; it must await the task's exit first.
"""

import asyncio
import collections
import concurrent.futures
import time
from types import SimpleNamespace

import pytest

from fusion_mlx import engine_core
from fusion_mlx.engine_core import EngineCore
from fusion_mlx.request import Request, SamplingParams
from fusion_mlx.scheduler import sched_admission
from fusion_mlx.scheduler.config import SchedulingPolicy


def _make_request(rid="req-1354"):
    request = Request(
        request_id=rid,
        prompt="hi",
        sampling_params=SamplingParams(max_tokens=8),
    )
    request.prompt_token_ids = [1, 2, 3, 4, 5, 6, 7, 8]
    request.num_prompt_tokens = 8
    return request


class _Ticking:
    def __init__(self):
        self.count = 0
        self._task = None

    def start(self):
        async def _run():
            while True:
                self.count += 1
                await asyncio.sleep(0.005)

        self._task = asyncio.create_task(_run())

    async def stop(self):
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass


class TestM2AbortDoesNotBlockLoop:
    def _core(self, executor, scheduler):
        return SimpleNamespace(
            _running=True,
            scheduler=scheduler,
            _mlx_executor=executor,
            _active_contexts={},
            config=SimpleNamespace(stream_interval=1),
            _cleanup_request=lambda rid: None,
            _wake_engine_loop=lambda: None,
        )

    async def test_post_timeout_abort_yields_to_event_loop(self, monkeypatch):
        monkeypatch.setattr(engine_core, "_ADD_REQUEST_SUBMIT_TIMEOUT_S", 0.05)
        ticks = _Ticking()
        ticks.start()
        window = [0, 0]

        def _slow_add(request):
            time.sleep(0.2)

        def _slow_abort(rid):
            window[0] = ticks.count
            time.sleep(0.2)
            window[1] = ticks.count

        scheduler = SimpleNamespace(add_request=_slow_add, abort_request=_slow_abort)
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        core = self._core(executor, scheduler)
        try:
            with pytest.raises(RuntimeError, match="admission timed out"):
                await EngineCore.add_request(core, "hi", request_id="m2-timeout")
        finally:
            await ticks.stop()
            executor.shutdown(wait=True)
        assert window[1] - window[0] >= 10, (
            f"event loop ticked only {window[1] - window[0]} times during the "
            "blocking abort — fut.result() is back"
        )

    async def test_post_cancel_abort_yields_to_event_loop(self, monkeypatch):
        ticks = _Ticking()
        ticks.start()
        window = [0, 0]

        def _cancelled_add(request):
            raise asyncio.CancelledError()

        def _slow_abort(rid):
            window[0] = ticks.count
            time.sleep(0.2)
            window[1] = ticks.count

        scheduler = SimpleNamespace(
            add_request=_cancelled_add, abort_request=_slow_abort
        )
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        core = self._core(executor, scheduler)
        try:
            with pytest.raises(asyncio.CancelledError):
                await EngineCore.add_request(core, "hi", request_id="m2-cancel")
        finally:
            await ticks.stop()
            executor.shutdown(wait=True)
        assert window[1] - window[0] >= 10


class _FakeBlockAwareCache:
    def __init__(self, block_table):
        self.block_table = block_table
        self.cleared = []
        self.reconstructed = []
        self.preload_fut = concurrent.futures.Future()

    def fetch_cache(
        self,
        rid,
        tokens,
        extra_keys=None,
        extra_key_token_start=None,
        extra_key_ranges=None,
    ):
        return self.block_table, list(tokens)

    def preload_blocks_async(self, block_table):
        return self.preload_fut

    def reconstruct_cache(self, block_table, promote_to_hot_cache=True):
        self.reconstructed.append(block_table)
        return None

    def clear_request_entry_no_free(self, rid):
        self.cleared.append(rid)


def _block_table():
    return SimpleNamespace(num_tokens=8, block_ids=[1, 2])


class TestM3PreloadTimeoutSkipsReuse:
    def _scheduler(self, cache):
        return SimpleNamespace(
            _prefix_cache_prepared=set(),
            block_aware_cache=cache,
            _try_cow_donate=lambda req: False,
            _find_relevant_inflight_store=lambda req: None,
            _has_unfinished_store_futures=lambda: False,
            _bypass_hot_cache_under_pressure=lambda: False,
        )

    def test_prepare_skips_reconstruct_on_timeout(self, monkeypatch):
        monkeypatch.setattr(sched_admission, "_SSD_PRELOAD_TIMEOUT_S", 0.05)
        cache = _FakeBlockAwareCache(_block_table())
        scheduler = self._scheduler(cache)
        request = _make_request("m3-prepare")

        t0 = time.monotonic()
        sched_admission._prepare_prefix_cache_for_request(scheduler, request)
        elapsed = time.monotonic() - t0

        assert elapsed < 2.0
        assert cache.reconstructed == []
        assert cache.cleared == ["m3-prepare"]
        assert request.remaining_tokens == request.prompt_token_ids
        assert "m3-prepare" in scheduler._prefix_cache_prepared

    def test_add_request_skips_reconstruct_on_timeout(self, monkeypatch):
        monkeypatch.setattr(sched_admission, "_SSD_PRELOAD_TIMEOUT_S", 0.05)
        cache = _FakeBlockAwareCache(_block_table())
        scheduler = self._scheduler(cache)
        scheduler.requests = {}
        scheduler.waiting = collections.deque()
        scheduler.running = collections.deque()
        scheduler.config = SimpleNamespace(max_waiting=32, policy=SchedulingPolicy.FCFS)
        request = _make_request("m3-add")

        t0 = time.monotonic()
        sched_admission.add_request(scheduler, request)
        elapsed = time.monotonic() - t0

        assert elapsed < 2.0
        assert cache.reconstructed == []
        assert cache.cleared == ["m3-add"]
        assert request.remaining_tokens == request.prompt_token_ids
        assert list(scheduler.waiting) == [request]


class TestM4AcloseAwaitsLoopExit:
    def _core(self, task, close):
        return SimpleNamespace(
            _closed=False,
            _task=task,
            _engine_id="engine-m4-1354",
            close=close,
        )

    async def test_aclose_waits_for_loop_cancellation(self, monkeypatch):
        monkeypatch.setattr(engine_core, "_ENGINE_TASK_EXIT_TIMEOUT_S", 1.0)
        order = []

        async def _engine_loop():
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                order.append("loop_exited")
                raise

        task = asyncio.create_task(_engine_loop())
        await asyncio.sleep(0)

        def _close():
            order.append("close")

        await EngineCore.aclose(self._core(task, _close))
        assert order == ["loop_exited", "close"]
        assert task.cancelled()

    async def test_aclose_tears_down_after_loop_timeout(self, monkeypatch):
        monkeypatch.setattr(engine_core, "_ENGINE_TASK_EXIT_TIMEOUT_S", 0.02)
        order = []

        async def _slow_cancel():
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                await asyncio.sleep(0.1)
                order.append("loop_exited")
                raise

        task = asyncio.create_task(_slow_cancel())
        await asyncio.sleep(0)

        def _close():
            order.append("close")

        await EngineCore.aclose(self._core(task, _close))
        assert order == ["close"]
        await asyncio.gather(task, return_exceptions=True)
        assert order == ["close", "loop_exited"]
