# SPDX-License-Identifier: Apache-2.0
# #1354 audit H12: admission eviction used to pick a victim with the advisory
# _find_lru_victim() (no lock) and only then call unload_engine_async, so two
# concurrent admission loops could select the same idle model and run two
# teardowns (two stop() calls) against one engine. The fix claims the victim
# under the pool lock (is_unloading + unloading_count) and makes
# unload_engine_async skip an entry another teardown already claimed.

from __future__ import annotations

import asyncio

import pytest

from fusion_mlx.pool.engine_pool import EngineEntry, EnginePool


class _FakeEngine:
    def __init__(self, model_id, gate=None):
        self.model_id = model_id
        self.stop_calls = 0
        self.started = asyncio.Event()
        self._gate = gate

    async def stop(self):
        self.stop_calls += 1
        self.started.set()
        if self._gate is not None:
            await self._gate.wait()


def _register(pool: EnginePool, model_id: str, last_access: float = 1.0) -> _FakeEngine:
    engine = _FakeEngine(model_id)
    entry = EngineEntry(
        model_id=model_id,
        model_path=f"/fake/{model_id}",
        model_type="llm",
        engine_type="batched",
        estimated_size=100 * 1024 * 1024,
    )
    entry.engine = engine
    entry.last_access = last_access
    pool._entries[model_id] = entry
    return engine


@pytest.fixture
def pool():
    return EnginePool()


@pytest.mark.asyncio
async def test_claim_is_exclusive(pool):
    _register(pool, "llm-a")
    _register(pool, "llm-b")

    first = await pool._claim_lru_victim()
    assert first is not None
    entry = pool.get_entry(first)
    assert entry.is_unloading is True
    assert entry.unloading_count == 1

    second = await pool._claim_lru_victim()
    assert second is not None and second != first

    third = await pool._claim_lru_victim()
    assert third is None


@pytest.mark.asyncio
async def test_find_lru_victim_skips_claimed_entry(pool):
    _register(pool, "llm-a", last_access=1.0)
    _register(pool, "llm-b", last_access=2.0)
    assert pool._find_lru_victim() == "llm-a"

    pool.get_entry("llm-a").is_unloading = True
    assert pool._find_lru_victim() == "llm-b"


@pytest.mark.asyncio
async def test_cooldown_prefers_other_candidate(pool):
    _register(pool, "llm-a", last_access=1.0)
    _register(pool, "llm-b", last_access=2.0)
    # llm-a is the LRU but was evicted a moment ago: prefer llm-b instead of
    # thrashing the reload that just lost the race.
    pool.get_entry("llm-a").last_evicted_at = asyncio.get_running_loop().time() - 0.5
    assert pool._find_lru_victim() == "llm-b"

    # Once the cooldown lapses, plain LRU order returns.
    pool.get_entry("llm-a").last_evicted_at = 0.0
    assert pool._find_lru_victim() == "llm-a"


@pytest.mark.asyncio
async def test_cooldown_never_blocks_only_candidate(pool):
    _register(pool, "llm-a", last_access=1.0)
    pool.get_entry("llm-a").last_evicted_at = asyncio.get_running_loop().time()
    assert pool._find_lru_victim() == "llm-a"


@pytest.mark.asyncio
async def test_concurrent_unload_stops_engine_once(pool):
    gate = asyncio.Event()
    _register(pool, "llm-a")
    engine = pool.get_entry("llm-a").engine
    engine._gate = gate

    first = asyncio.create_task(pool.unload_engine_async("llm-a", with_settle=False))
    await asyncio.wait_for(engine.started.wait(), timeout=5.0)

    # Second teardown for the same model: must skip, not call stop() again.
    await pool.unload_engine_async("llm-a", with_settle=False)
    assert engine.stop_calls == 1

    gate.set()
    await asyncio.wait_for(first, timeout=5.0)
    assert engine.stop_calls == 1
    entry = pool.get_entry("llm-a")
    assert entry.engine is None
    assert entry.is_unloading is False
    assert entry.unloading_count == 0


@pytest.mark.asyncio
async def test_claimed_victim_unload_clears_claim(pool):
    _register(pool, "llm-a")
    victim = await pool._claim_lru_victim()
    assert victim == "llm-a"
    await pool.unload_engine_async(victim, with_settle=False, _marker_already_set=True)
    entry = pool.get_entry("llm-a")
    assert entry.engine is None
    assert entry.is_unloading is False
    assert entry.unloading_count == 0


@pytest.mark.asyncio
async def test_stranded_marker_without_claim_is_recovered(pool):
    # #1001: a stranded is_unloading flag (no live claim) must still be
    # recoverable by a later unload, otherwise the model is stuck forever.
    _register(pool, "llm-a")
    entry = pool.get_entry("llm-a")
    entry.is_unloading = True

    await pool.unload_engine_async("llm-a", with_settle=False)
    assert entry.engine is None
    assert entry.is_unloading is False
    assert entry.unloading_count == 0
