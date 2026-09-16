# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ProcessMemoryEnforcer ANE memory awareness (P0底座)."""

import threading

import pytest

from fusion_mlx.pool.memory_enforcer import ProcessMemoryEnforcer


@pytest.fixture
def enforcer(monkeypatch):
    """Build an enforcer without a real engine pool / event loop.

    prefill_memory_guard=True so ceilings are live. Mock MLX + phys footprint
    so _current_usage_bytes is deterministic.
    """
    monkeypatch.setattr("fusion_mlx.pool.memory_enforcer.get_phys_footprint", lambda: 0)
    pool = type("P", (), {})()
    enf = ProcessMemoryEnforcer(
        engine_pool=pool,
        memory_guard_tier="custom",
        memory_guard_custom_ceiling_gb=64.0,
        poll_interval=999,
    )
    return enf


_GB = 1024**3


def test_register_ane_resident(enforcer):
    enforcer.register_ane_resident("m1", _GB)
    assert enforcer.get_ane_resident_bytes() == _GB


def test_register_multiple_accumulates(enforcer):
    enforcer.register_ane_resident("m1", _GB)
    enforcer.register_ane_resident("m2", 2 * _GB)
    assert enforcer.get_ane_resident_bytes() == 3 * _GB


def test_unregister_ane_resident(enforcer):
    enforcer.register_ane_resident("m1", _GB)
    enforcer.register_ane_resident("m2", 2 * _GB)
    enforcer.unregister_ane_resident("m1")
    assert enforcer.get_ane_resident_bytes() == 2 * _GB


def test_unregister_unknown_noop(enforcer):
    enforcer.unregister_ane_resident("nonexistent")
    assert enforcer.get_ane_resident_bytes() == 0


def test_register_overwrites_same_engine(enforcer):
    enforcer.register_ane_resident("m1", _GB)
    enforcer.register_ane_resident("m1", 3 * _GB)
    assert enforcer.get_ane_resident_bytes() == 3 * _GB


def test_get_metal_wired_bytes(monkeypatch, enforcer):
    import fusion_mlx.pool.memory_enforcer as mod

    fake_mx = type("MX", (), {})()
    fake_mx.get_active_memory = lambda: 5 * _GB
    monkeypatch.setattr(mod, "mx", fake_mx)
    assert enforcer.get_metal_wired_bytes() == 5 * _GB


def test_get_metal_wired_bytes_safe_on_error(monkeypatch, enforcer):
    import fusion_mlx.pool.memory_enforcer as mod

    fake_mx = type("MX", (), {})()

    def _boom():
        raise RuntimeError("no metal")

    fake_mx.get_active_memory = _boom
    monkeypatch.setattr(mod, "mx", fake_mx)
    assert enforcer.get_metal_wired_bytes() == 0


def test_current_usage_includes_ane(monkeypatch, enforcer):
    """_current_usage_bytes = max(active, phys) + ane_resident - shared."""
    import fusion_mlx.pool.memory_enforcer as mod

    fake_mx = type("MX", (), {})()
    fake_mx.get_active_memory = lambda: 10 * _GB
    monkeypatch.setattr(mod, "mx", fake_mx)
    monkeypatch.setattr(mod, "get_phys_footprint", lambda: 8 * _GB)
    monkeypatch.setattr(enforcer, "_has_active_requests", lambda: False)
    enforcer.register_ane_resident("m1", 4 * _GB)
    assert enforcer._current_usage_bytes() == 10 * _GB + 4 * _GB


def test_current_usage_subtracts_iosurface_shared(monkeypatch, enforcer):
    import fusion_mlx.pool.memory_enforcer as mod

    fake_mx = type("MX", (), {})()
    fake_mx.get_active_memory = lambda: 10 * _GB
    monkeypatch.setattr(mod, "mx", fake_mx)
    monkeypatch.setattr(mod, "get_phys_footprint", lambda: 8 * _GB)
    monkeypatch.setattr(enforcer, "_has_active_requests", lambda: False)
    enforcer.register_ane_resident("m1", 4 * _GB)
    enforcer.register_iosurface_shared(3 * _GB)
    assert enforcer._current_usage_bytes() == 10 * _GB + 4 * _GB - 3 * _GB


def test_ane_memory_budget_default_15pct(enforcer):
    budget = enforcer.get_ane_memory_budget()
    static = enforcer._get_static_ceiling()
    assert budget == int(static * 0.15)


def test_ane_memory_budget_override(enforcer):
    enforcer.set_ane_memory_budget_gb(16.0)
    assert enforcer.get_ane_memory_budget() == 16 * _GB


def test_threadsafe_concurrent_register(enforcer):
    errors = []

    def _worker(i):
        try:
            for j in range(50):
                enforcer.register_ane_resident(f"m{i}", j * 1024)
                enforcer.unregister_ane_resident(f"m{i}")
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=_worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert not errors
    assert enforcer.get_ane_resident_bytes() == 0
