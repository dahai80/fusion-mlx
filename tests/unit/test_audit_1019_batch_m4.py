# SPDX-License-Identifier: Apache-2.0
# Regression tests for the fourth M-series batch of
# audit/fusion-mlx-codeing-audit-report-1019.md:
#   M4 engine loop cancelled without waiting, M11 SSD lazy-restore pin leak.

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from fusion_mlx.cache.paged_cache import PagedCacheManager, compute_block_hash
from fusion_mlx.engine_core import EngineCore


class _FakeSSD:
    def __init__(self, present):
        self.present = set(present)
        self.calls = 0

    def has_block(self, block_hash):
        self.calls += 1
        return block_hash in self.present


class TestLazyRestoreRequiresClaim:
    # M11: a non-claiming lookup restored a block from SSD, took the lazy pin
    # (ref_count=1) and never released it — the block became unevictable while
    # the prefix index kept hitting a block whose data was never loaded.

    def _manager(self, tokens=(1, 2, 3, 4)):
        mgr = PagedCacheManager(
            block_size=4, max_blocks=16, model_name="m", initial_blocks=16
        )
        block_hash = compute_block_hash(None, list(tokens), model_name="m")
        mgr._paged_ssd_cache_manager = _FakeSSD([block_hash])
        return mgr, block_hash

    def test_non_claiming_lookup_does_not_restore(self):
        mgr, block_hash = self._manager()
        baseline_allocated = mgr.stats.allocated_blocks

        blocks, num_tokens = mgr.get_computed_blocks([1, 2, 3, 4])

        assert blocks == []
        assert num_tokens == 0
        assert mgr._paged_ssd_cache_manager.calls == 0
        assert mgr.cached_block_hash_to_block.get_block(block_hash) is None
        assert mgr.stats.allocated_blocks == baseline_allocated

    def test_claiming_lookup_restores_with_single_pin(self):
        mgr, block_hash = self._manager()

        blocks, num_tokens = mgr.get_computed_blocks([1, 2, 3, 4], claim=True)

        assert num_tokens == 4
        assert len(blocks) == 1
        # The lazy-restore pin transfers to the caller instead of stacking.
        assert blocks[0].ref_count == 1


class TestEngineCloseSettlesLoopTask:
    # M4: close() cancelled the engine loop task and nulled scheduler/model
    # without waiting for the coroutine to unwind.

    def _bare_core(self):
        core = EngineCore.__new__(EngineCore)
        core._closed = False
        core._closing = False
        core._running = True
        core._wake_event = None
        core._owns_model = False
        core._engine_id = "test-engine"
        core._engine_dead = True
        core._mlx_executor = None
        core._mlx_stream = None
        core.scheduler = None
        core._active_contexts = {}
        core._finished_at = {}
        core._fail_unfinished_contexts = lambda *a, **k: None
        core.model = object()
        core.tokenizer = object()
        return core

    def test_close_marks_closing_then_cancels_then_settles(self, monkeypatch):
        core = self._bare_core()
        order = []

        class _Task:
            def cancel(self):
                order.append(("cancel", core._closing))

            def done(self):
                return True

        core._task = _Task()
        monkeypatch.setattr(
            EngineCore,
            "_settle_engine_task",
            staticmethod(
                lambda task, timeout: order.append(("settle", timeout)) or True
            ),
        )

        core.close()

        assert order[0] == ("cancel", True)
        assert order[1][0] == "settle"
        assert core._closing is True
        assert core._closed is True
        assert core.scheduler is None
        assert core.model is None

    def test_close_is_idempotent(self, monkeypatch):
        core = self._bare_core()
        core._task = None
        monkeypatch.setattr(
            EngineCore, "_settle_engine_task", staticmethod(lambda *a: True)
        )

        core.close()
        core.close()

        assert core._closed is True

    def test_settle_waits_from_another_thread(self):
        loop = asyncio.new_event_loop()
        started = threading.Event()

        async def _sleeper():
            started.set()
            await asyncio.sleep(30)

        holder: dict[str, asyncio.Task] = {}
        thread = threading.Thread(target=loop.run_forever, daemon=True)
        thread.start()
        try:
            loop.call_soon_threadsafe(
                lambda: holder.setdefault("task", loop.create_task(_sleeper()))
            )
            assert started.wait(5.0)
            task = holder["task"]
            assert not task.done()

            settled = EngineCore._settle_engine_task(task, 5.0)

            assert settled is True
        finally:
            loop.call_soon_threadsafe(loop.stop)
            thread.join(5.0)
            loop.close()

    @pytest.mark.asyncio
    async def test_settle_does_not_block_own_loop(self):
        async def _sleeper():
            await asyncio.sleep(30)

        task = asyncio.create_task(_sleeper())
        try:
            t0 = time.monotonic()
            settled = EngineCore._settle_engine_task(task, 5.0)
            elapsed = time.monotonic() - t0
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        # Cannot wait on its own loop: returns immediately, _closing covers it.
        assert settled is False
        assert elapsed < 1.0


if __name__ == "__main__":  # pragma: no cover — convenience only
    pytest.main([__file__, "-v"])
