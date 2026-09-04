# SPDX-License-Identifier: Apache-2.0
"""Verify concurrent multi-model serving (#796).

#796 asks for concurrent Fast + Slow model residency: a small fast model and a
large slow model both resident, serving requests without evicting each other.
EnginePool already supports this via:
  - register_engine / get_engine keeping multiple entries loaded at once
  - set_pinned(True) preventing LRU eviction of a model
  - in_use lease (acquire/release) preventing eviction mid-request
  - unload_if_idle_unpinned refusing pinned or in-use models

These tests exercise that with fake engines (no MLX load), headless.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

import pytest

from fusion_mlx.pool.engine_pool import EngineEntry, EnginePool

logger = logging.getLogger(__name__)


@dataclass
class _FakeEngine:
    model_id: str
    stopped: bool = False

    async def stop(self):
        self.stopped = True


def _register(pool: EnginePool, model_id: str, pinned: bool = False) -> _FakeEngine:
    engine = _FakeEngine(model_id=model_id)
    entry = EngineEntry(
        model_id=model_id,
        model_path=f"/fake/{model_id}",
        model_type="llm",
        engine_type="batched",
        estimated_size=1_000_000_000,
    )
    entry.engine = engine
    entry.is_pinned = pinned
    entry.last_access = 1.0
    pool._entries[model_id] = entry
    pool._current_model_memory += entry.estimated_size
    return engine


@pytest.fixture
def pool():
    return EnginePool()


# ---------------------------------------------------------------- residency


@pytest.mark.asyncio
async def test_fast_and_slow_both_resident(pool):
    # The core #796 scenario: a fast and a slow model loaded simultaneously.
    fast = _register(pool, "fast-0.5b")
    slow = _register(pool, "slow-32b")
    assert pool.get_entry("fast-0.5b").engine is fast
    assert pool.get_entry("slow-32b").engine is slow
    assert pool.loaded_model_count == 2


# ---------------------------------------------------------------- pinning


@pytest.mark.asyncio
async def test_pinned_model_not_evicted(pool):
    fast = _register(pool, "fast-0.5b", pinned=True)
    # unload_if_idle_unpinned must refuse a pinned model even when idle.
    evicted = await pool.unload_if_idle_unpinned("fast-0.5b")
    assert evicted is False
    assert pool.get_entry("fast-0.5b").engine is fast
    assert not fast.stopped


@pytest.mark.asyncio
async def test_unpinned_idle_model_can_be_unloaded(pool):
    _register(pool, "fast-0.5b", pinned=False)
    evicted = await pool.unload_if_idle_unpinned("fast-0.5b")
    assert evicted is True
    assert pool.get_entry("fast-0.5b").engine is None


@pytest.mark.asyncio
async def test_set_pinned_toggles_protection(pool):
    _register(pool, "fast-0.5b", pinned=False)
    assert pool.set_pinned("fast-0.5b", True) is True
    assert pool.get_entry("fast-0.5b").is_pinned is True
    # Now pinned -> unload refused.
    assert await pool.unload_if_idle_unpinned("fast-0.5b") is False
    # Unpin -> unload succeeds.
    assert pool.set_pinned("fast-0.5b", False) is True
    assert await pool.unload_if_idle_unpinned("fast-0.5b") is True


# ---------------------------------------------------------------- in-use lease


@pytest.mark.asyncio
async def test_in_use_lease_blocks_unload(pool):
    _register(pool, "fast-0.5b", pinned=False)
    entry = pool.get_entry("fast-0.5b")
    entry.in_use = 1  # simulate an in-flight request
    # Busy (leased) model must not be evicted even if unpinned + idle-looking.
    assert await pool.unload_if_idle_unpinned("fast-0.5b") is False
    assert entry.engine is not None
    entry.in_use = 0
    assert await pool.unload_if_idle_unpinned("fast-0.5b") is True


# ---------------------------------------------------------------- concurrency


@pytest.mark.asyncio
async def test_concurrent_acquire_both_models(pool):
    # Pin both so the test is deterministic about residency, then acquire both
    # concurrently to prove two models can serve in parallel without one
    # evicting the other.
    _register(pool, "fast-0.5b", pinned=True)
    _register(pool, "slow-32b", pinned=True)

    results: dict[str, bool] = {}

    async def _use(model_id: str):
        async with pool.acquire(model_id):
            entry = pool.get_entry(model_id)
            assert entry.in_use >= 1
            assert entry.engine is not None
            await asyncio.sleep(0)
            results[model_id] = True

    await asyncio.gather(_use("fast-0.5b"), _use("slow-32b"))
    assert results == {"fast-0.5b": True, "slow-32b": True}
    # Both still resident after concurrent use.
    assert pool.loaded_model_count == 2
    # Leases released after the context managers exit.
    assert pool.get_entry("fast-0.5b").in_use == 0
    assert pool.get_entry("slow-32b").in_use == 0


@pytest.mark.asyncio
async def test_release_releases_lease(pool):
    _register(pool, "fast-0.5b", pinned=True)
    entry = pool.get_entry("fast-0.5b")
    entry.in_use = 1
    await pool.release_engine("fast-0.5b")
    assert entry.in_use == 0
