# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ANE memory safety invariants (P0底座)."""

import pytest

from fusion_mlx.pool.memory_enforcer import ProcessMemoryEnforcer

_GB = 1024**3


@pytest.fixture
def enforcer(monkeypatch):
    monkeypatch.setattr("fusion_mlx.pool.memory_enforcer.get_phys_footprint", lambda: 0)
    pool = type("P", (), {})()
    return ProcessMemoryEnforcer(
        engine_pool=pool,
        memory_guard_tier="custom",
        memory_guard_custom_ceiling_gb=64.0,
        poll_interval=999,
    )


def test_ane_resident_not_invisible(enforcer, monkeypatch):
    """ANE resident must raise _current_usage_bytes (no blind spot)."""
    import fusion_mlx.pool.memory_enforcer as mod

    fake_mx = type("MX", (), {})()
    fake_mx.get_active_memory = lambda: 10 * _GB
    monkeypatch.setattr(mod, "mx", fake_mx)
    monkeypatch.setattr(mod, "get_phys_footprint", lambda: 8 * _GB)
    monkeypatch.setattr(enforcer, "_has_active_requests", lambda: False)
    before = enforcer._current_usage_bytes()
    enforcer.register_ane_resident("m1", 4 * _GB)
    after = enforcer._current_usage_bytes()
    assert after - before == 4 * _GB


def test_iosurface_no_double_count(enforcer, monkeypatch):
    """IOSurface shared subtracted once so the same物理页 isn't counted twice."""
    import fusion_mlx.pool.memory_enforcer as mod

    fake_mx = type("MX", (), {})()
    fake_mx.get_active_memory = lambda: 10 * _GB
    monkeypatch.setattr(mod, "mx", fake_mx)
    monkeypatch.setattr(mod, "get_phys_footprint", lambda: 8 * _GB)
    monkeypatch.setattr(enforcer, "_has_active_requests", lambda: False)
    enforcer.register_ane_resident("m1", 4 * _GB)
    before = enforcer._current_usage_bytes()
    # registering the SAME shared bytes again should not increase usage
    enforcer.register_iosurface_shared(2 * _GB)
    after_first = enforcer._current_usage_bytes()
    assert after_first == before - 2 * _GB
    # idempotent: re-register same value, no further change
    enforcer.register_iosurface_shared(2 * _GB)
    after_second = enforcer._current_usage_bytes()
    assert after_second == after_first


@pytest.mark.asyncio
async def test_ane_over_budget_rejected_via_admit(monkeypatch):
    """admit_media_job(kind='ane') returns False when ANE over budget."""
    from fusion_mlx.pool.engine_pool import EnginePool

    class _Enf:
        def get_ane_resident_bytes(self):
            return 15 * _GB

        def get_ane_memory_budget(self):
            return 16 * _GB

    class _Pool:
        _process_memory_enforcer = _Enf()

        def _current_ceiling(self):
            return 64 * _GB

        def _admission_current_usage(self, *, exclude_entry_key=None):
            return 0

        def _find_lru_victim(self):
            return None

        def _record_eviction(self, reason):
            pass

        async def unload_engine_async(self, mid, **kw):
            pass

    pool = _Pool()
    ok = await EnginePool.admit_media_job(pool, required_bytes=4 * _GB, kind="ane")
    assert ok is False


def test_ane_budget_default_below_static(enforcer):
    """ANE budget default (15%) must be strictly below the static ceiling."""
    budget = enforcer.get_ane_memory_budget()
    static = enforcer._get_static_ceiling()
    assert 0 < budget < static


def test_concurrent_register_unregister_no_deadlock(enforcer):
    import threading

    done = threading.Event()

    def _work():
        for i in range(100):
            enforcer.register_ane_resident(f"m{i % 4}", i * 1024)
            enforcer.unregister_ane_resident(f"m{i % 4}")
        done.set()

    t = threading.Thread(target=_work)
    t.start()
    t.join(timeout=5)
    assert done.is_set(), "concurrent register/unregister deadlocked"
