# SPDX-License-Identifier: Apache-2.0
"""Regression tests for issue #1009.

#1009: in-place LoRA swap (``FUSION_LORA_INPLACE_SWAP=1``, default off) leaked
the swap lock + in_use count + swap object when the base engine was
concurrently unloaded during the swap window.

Two gaps:
1. ``release_engine`` only called ``_release_inplace_adapter`` when
   ``base_entry.engine is not None``. If the base was torn down mid-lease
   (engine is None), it fell through to the normal adapter-key path — which
   never touched the base's in_use, never released ``_adapter_swap_locks[base]``,
   and never popped ``_active_swap[base]``. Result: in_use永久 +1 (LRU/TTL
   skip) + lock held forever (every subsequent adapter request hung on
   ``lock.acquire()``).
2. ``_detach_engine`` did not clean ``_active_swap`` — the swap object held a
   model reference, pinning weights so gc/clear_cache in the settle barrier
   could not reclaim them.

Fix (#1009): ``release_engine`` always calls ``_release_inplace_adapter`` for
inplace adapter releases (the helper handles None engine/entry gracefully);
``_detach_engine`` pops ``_active_swap`` + releases a held swap lock.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from fusion_mlx.pool.engine_pool import EngineEntry, EnginePool


def _pool(monkeypatch) -> EnginePool:
    monkeypatch.setenv("FUSION_LORA_INPLACE_SWAP", "1")
    pool = EnginePool()
    pool._get_final_ceiling = lambda: 0
    return pool


def _base_entry(model_id: str = "qwen-base", engine_alive: bool = True) -> EngineEntry:
    entry = EngineEntry(
        model_id=model_id,
        model_path=f"/models/{model_id}",
        model_type="llm",
        engine_type="batched",
        estimated_size=1000,
    )
    if engine_alive:
        engine = MagicMock()
        engine._model = MagicMock(name="model")
        engine.stop = AsyncMock()
        engine.safe_evict = AsyncMock()
        engine.has_active_requests = MagicMock(return_value=False)
        engine.is_dead = MagicMock(return_value=False)
        engine._reset_activity_tracking = MagicMock()
        entry.engine = engine
    else:
        entry.engine = None
    return entry


class TestReleaseEngineInplaceBaseUnloaded:
    # #1009 fix #1: release_engine must release the swap lease even when the
    # base engine was concurrently unloaded (engine is None).
    @pytest.mark.asyncio
    async def test_base_engine_none_still_decrements_in_use(self, monkeypatch):
        pool = _pool(monkeypatch)
        entry = _base_entry(engine_alive=False)
        entry.in_use = 1
        pool._entries["qwen-base"] = entry
        swap = MagicMock()
        swap.restore = MagicMock()
        pool._active_swap["qwen-base"] = swap

        await pool.release_engine("qwen-base", adapter_path="/adapters/fixA")

        assert entry.in_use == 0
        assert "qwen-base" not in pool._active_swap

    @pytest.mark.asyncio
    async def test_base_engine_none_releases_swap_lock(self, monkeypatch):
        pool = _pool(monkeypatch)
        entry = _base_entry(engine_alive=False)
        entry.in_use = 1
        pool._entries["qwen-base"] = entry
        swap = MagicMock()
        swap.restore = MagicMock()
        pool._active_swap["qwen-base"] = swap
        lock = asyncio.Lock()
        pool._adapter_swap_locks["qwen-base"] = lock
        await lock.acquire()

        await pool.release_engine("qwen-base", adapter_path="/adapters/fixA")

        assert not lock.locked()

    @pytest.mark.asyncio
    async def test_base_engine_none_does_not_fall_through_to_adapter_key(
        self, monkeypatch
    ):
        # The old bug fell through to the adapter-key path, which looked up a
        # non-existent derived entry and never touched the base. Verify no
        # derived adapter entry is created/touched.
        pool = _pool(monkeypatch)
        entry = _base_entry(engine_alive=False)
        entry.in_use = 1
        pool._entries["qwen-base"] = entry
        pool._active_swap["qwen-base"] = MagicMock(restore=MagicMock())

        await pool.release_engine("qwen-base", adapter_path="/adapters/fixA")

        # No derived adapter entry should exist.
        assert "qwen-base\x00/adapters/fixA" not in pool._entries
        assert "qwen-base::/adapters/fixA" not in pool._entries
        assert entry.in_use == 0

    @pytest.mark.asyncio
    async def test_base_engine_alive_path_unchanged(self, monkeypatch):
        # The normal path (engine alive) must still restore + decrement.
        pool = _pool(monkeypatch)
        entry = _base_entry(engine_alive=True)
        entry.in_use = 1
        pool._entries["qwen-base"] = entry
        swap = MagicMock()
        swap.restore = MagicMock()
        pool._active_swap["qwen-base"] = swap

        await pool.release_engine("qwen-base", adapter_path="/adapters/fixA")

        swap.restore.assert_called_once()
        assert entry.in_use == 0
        assert "qwen-base" not in pool._active_swap


class TestDetachEngineDropsActiveSwap:
    # #1009 fix #2: _detach_engine must pop _active_swap + release a held
    # swap lock so weights are not pinned after unload.
    @pytest.mark.asyncio
    async def test_detach_pops_active_swap(self, monkeypatch):
        pool = _pool(monkeypatch)
        entry = _base_entry(engine_alive=True)
        pool._entries["qwen-base"] = entry
        swap = MagicMock()
        pool._active_swap["qwen-base"] = swap

        await pool._detach_engine("qwen-base")

        assert "qwen-base" not in pool._active_swap
        assert entry.engine is None

    @pytest.mark.asyncio
    async def test_detach_releases_held_swap_lock(self, monkeypatch):
        pool = _pool(monkeypatch)
        entry = _base_entry(engine_alive=True)
        pool._entries["qwen-base"] = entry
        pool._active_swap["qwen-base"] = MagicMock()
        lock = asyncio.Lock()
        pool._adapter_swap_locks["qwen-base"] = lock
        await lock.acquire()

        await pool._detach_engine("qwen-base")

        assert not lock.locked()

    @pytest.mark.asyncio
    async def test_detach_no_swap_is_safe(self, monkeypatch):
        pool = _pool(monkeypatch)
        entry = _base_entry(engine_alive=True)
        pool._entries["qwen-base"] = entry

        await pool._detach_engine("qwen-base")

        assert "qwen-base" not in pool._active_swap
        assert entry.engine is None


if __name__ == "__main__":  # pragma: no cover — convenience only
    pytest.main([__file__, "-v"])
