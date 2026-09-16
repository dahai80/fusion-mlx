# SPDX-License-Identifier: Apache-2.0
"""Unit tests for EnginePool.admit_media_job kind extension (P0底座)."""

import pytest

from fusion_mlx.pool.engine_pool import EnginePool

_GB = 1024**3


class _FakeEnforcer:
    def __init__(self, ane_resident=0, ane_budget=0):
        self._ane_resident = ane_resident
        self._ane_budget = ane_budget

    def get_ane_resident_bytes(self):
        return self._ane_resident

    def get_ane_memory_budget(self):
        return self._ane_budget


class _FakePool:
    """Minimal stand-in exposing only what admit_media_job calls."""

    def __init__(self, ceiling, usage, victim=None, enforcer=None):
        self._ceiling = ceiling
        self._usage = usage
        self._victim = victim
        self._process_memory_enforcer = enforcer
        self.unloaded = []
        self.evictions = []
        self._unloaded_victims = set()

    def _current_ceiling(self):
        return self._ceiling

    def _admission_current_usage(self, *, exclude_entry_key=None):
        # each unload frees ~3GB so the loop converges
        return max(0, self._usage - len(self.unloaded) * 3 * _GB)

    def _find_lru_victim(self):
        if self._unloaded_victims:
            return None
        return self._victim

    def _record_eviction(self, reason):
        self.evictions.append(reason)

    async def unload_engine_async(self, mid, **kwargs):
        self.unloaded.append(mid)
        self._unloaded_victims.add(mid)


@pytest.mark.asyncio
async def test_kind_image_default_fits():
    pool = _FakePool(ceiling=10 * _GB, usage=2 * _GB)
    ok = await EnginePool.admit_media_job(pool, required_bytes=2 * _GB)
    assert ok is True
    assert pool.unloaded == []


@pytest.mark.asyncio
async def test_kind_image_evicts_to_fit():
    pool = _FakePool(ceiling=10 * _GB, usage=9 * _GB, victim="llm-a")
    ok = await EnginePool.admit_media_job(pool, required_bytes=2 * _GB)
    assert ok is True
    assert pool.unloaded == ["llm-a"]


@pytest.mark.asyncio
async def test_kind_image_no_victim_returns_false():
    pool = _FakePool(ceiling=10 * _GB, usage=9 * _GB, victim=None)
    ok = await EnginePool.admit_media_job(pool, required_bytes=5 * _GB)
    assert ok is False


@pytest.mark.asyncio
async def test_kind_ane_fits():
    enf = _FakeEnforcer(ane_resident=2 * _GB, ane_budget=16 * _GB)
    pool = _FakePool(ceiling=64 * _GB, usage=4 * _GB, enforcer=enf)
    ok = await EnginePool.admit_media_job(pool, required_bytes=2 * _GB, kind="ane")
    assert ok is True


@pytest.mark.asyncio
async def test_kind_ane_over_budget_returns_false():
    enf = _FakeEnforcer(ane_resident=14 * _GB, ane_budget=16 * _GB)
    pool = _FakePool(ceiling=64 * _GB, usage=0, enforcer=enf, victim=None)
    # current = usage(0) + ane_resident(14) = 14; required 4 → 18 > 16 budget
    ok = await EnginePool.admit_media_job(pool, required_bytes=4 * _GB, kind="ane")
    assert ok is False


@pytest.mark.asyncio
async def test_kind_ane_no_victim_returns_false():
    enf = _FakeEnforcer(ane_resident=14 * _GB, ane_budget=16 * _GB)
    pool = _FakePool(ceiling=64 * _GB, usage=0, enforcer=enf, victim=None)
    ok = await EnginePool.admit_media_job(pool, required_bytes=4 * _GB, kind="ane")
    assert ok is False


@pytest.mark.asyncio
async def test_kind_kernel_uses_image_path():
    pool = _FakePool(ceiling=10 * _GB, usage=2 * _GB)
    ok = await EnginePool.admit_media_job(pool, required_bytes=1 * _GB, kind="kernel")
    assert ok is True


@pytest.mark.asyncio
async def test_ceiling_zero_returns_true():
    pool = _FakePool(ceiling=0, usage=999 * _GB)
    ok = await EnginePool.admit_media_job(pool, required_bytes=5 * _GB)
    assert ok is True


@pytest.mark.asyncio
async def test_find_lru_victim_skips_ane_pinned():
    """Verify the edited _find_lru_victim skips ane_pinned entries.

    Uses a real EnginePool but only exercises _find_lru_victim by constructing
    entries directly on _entries.
    """
    pool = EnginePool.__new__(EnginePool)
    pool._entries = {}

    class _E:
        def __init__(self, pinned, ane, last, in_use=0, engine=object()):
            self.is_pinned = pinned
            self.ane_pinned = ane
            self.last_access = last
            self.in_use = in_use
            self.engine = engine

    pool._entries["llm-a"] = _E(pinned=False, ane=False, last=1.0)
    pool._entries["ane-b"] = _E(pinned=False, ane=True, last=0.5)
    pool._entries["llm-c"] = _E(pinned=False, ane=False, last=2.0)

    def _has_active(entry):
        return False

    pool._entry_has_active_requests = _has_active
    victim = pool._find_lru_victim()
    # ane-b has oldest last_access but is ANE-pinned → skipped; llm-a is next
    assert victim == "llm-a"
