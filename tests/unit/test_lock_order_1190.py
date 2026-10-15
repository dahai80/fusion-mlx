# SPDX-License-Identifier: Apache-2.0
"""Tests for lock-order tracking + #1190 TOCTOU fixes (debug-gated)."""

import asyncio
from pathlib import Path
from unittest.mock import Mock

import pytest

from fusion_mlx.pool.lock_order import (
    LockOrderTracker,
    LockOrderViolation,
    TrackedAsyncLock,
    TrackedThreadLock,
    reset_lock_order_tracker,
    snapshot_held,
)

_MEMORY_ENFORCER_PATH = (
    Path(__file__).resolve().parents[2] / "fusion_mlx" / "pool" / "memory_enforcer.py"
)


def _enable(monkeypatch):
    monkeypatch.setenv("FUSION_MLX_LOCK_DEBUG", "1")
    monkeypatch.delenv("FUSION_MLX_LOCK_DEBUG_STRICT", raising=False)


@pytest.fixture(autouse=True)
def _clean_tracker():
    reset_lock_order_tracker()
    yield
    reset_lock_order_tracker()


async def test_disabled_by_default_no_overhead(monkeypatch):
    monkeypatch.delenv("FUSION_MLX_LOCK_DEBUG", raising=False)
    tracker = LockOrderTracker()
    assert tracker.enabled() is False
    assert tracker.register("pool.main") is None
    assert tracker.check_acquire("pool.main") is None
    assert tracker.check_release("pool.main") is None
    assert snapshot_held() == {}
    lock = TrackedAsyncLock("pool.main")
    async with lock:
        assert snapshot_held() == {}
    assert lock.locked() is False


async def test_order_violation_logged(caplog, monkeypatch):
    _enable(monkeypatch)
    lock = TrackedAsyncLock("pool.adapter_swap")
    async with lock:
        LockOrderTracker().check_acquire("pool.main")
    assert any(
        "LOCK ORDER VIOLATION" in r.message and r.levelname == "ERROR"
        for r in caplog.records
    )


async def test_strict_mode_raises(monkeypatch):
    monkeypatch.setenv("FUSION_MLX_LOCK_DEBUG", "1")
    monkeypatch.setenv("FUSION_MLX_LOCK_DEBUG_STRICT", "1")
    tracker = LockOrderTracker()
    tracker.check_acquire("pool.main")
    tracker.mark_acquired("pool.main")
    with pytest.raises(LockOrderViolation):
        tracker.check_acquire("enforcer.state")


async def test_correct_order_no_violation(caplog, monkeypatch):
    _enable(monkeypatch)
    tracker = LockOrderTracker()
    lock_state = TrackedThreadLock("enforcer.state")
    lock_pool = TrackedAsyncLock("pool.main")
    with lock_state:
        LockOrderTracker().check_acquire("pool.main")
        async with lock_pool:
            pass
    assert not any("LOCK ORDER VIOLATION" in r.message for r in caplog.records)
    assert snapshot_held() == {}


async def test_tracked_locks_behave_like_std():
    lock = TrackedAsyncLock("pool.main")
    assert isinstance(lock, asyncio.Lock)
    async with lock:
        assert lock.locked()
    await asyncio.wait_for(lock.acquire(), timeout=0.5)
    lock.release()
    assert not lock.locked()

    tlock = TrackedThreadLock("pool.pinned")
    assert tlock.acquire(timeout=0.5) is True
    assert tlock.locked() is True
    tlock.release()
    assert tlock.locked() is False
    with tlock as ctx:
        assert ctx is tlock
        assert tlock.locked()
    assert tlock.locked() is False
    with tlock:
        got = await asyncio.get_running_loop().run_in_executor(
            None, lambda: tlock.acquire(blocking=True, timeout=0.1)
        )
        assert got is False


async def test_reset_and_snapshot(monkeypatch):
    _enable(monkeypatch)
    tracker = LockOrderTracker()
    tracker.mark_acquired("pool.main")
    held = snapshot_held()
    assert any("pool.main" in v for v in held.values())
    reset_lock_order_tracker()
    assert snapshot_held() == {}


def _make_enforcer():
    from fusion_mlx.pool.memory_enforcer import ProcessMemoryEnforcer

    pool = Mock()
    pool.loaded_model_bytes = 0
    pool.get_loaded_model_bytes = Mock(return_value=0)
    pool._entries = {}
    return ProcessMemoryEnforcer(
        engine_pool=pool,
        memory_guard_tier="balanced",
        memory_guard_custom_ceiling_gb=0.0,
        poll_interval=1.0,
    )


def test_enforcer_state_lock_is_tracked():
    enforcer = _make_enforcer()
    state_lock = enforcer._state_lock
    assert getattr(state_lock, "_lock_order_name", None) == "enforcer.state"
    assert isinstance(state_lock, TrackedThreadLock)


def test_enforcer_last_evicted_write_is_locked():
    src = _MEMORY_ENFORCER_PATH.read_text()
    lines = src.splitlines()
    anchors = [
        i
        for i, line in enumerate(lines)
        if "self._last_evicted_at[victim]" in line and "time.monotonic()" in line
    ]
    assert anchors, "cooldown write site not found"
    for idx in anchors:
        window = lines[max(0, idx - 5) : idx + 1]
        assert any(
            "with self._state_lock" in w for w in window
        ), f"cooldown write at line {idx + 1} not under _state_lock"


def test_enforcer_prune_discard_is_locked():
    src = _MEMORY_ENFORCER_PATH.read_text()
    lines = src.splitlines()
    start = next(i for i, line in enumerate(lines) if "def prune_model_state" in line)
    body = lines[start : start + 10]
    discard_idx = next(
        i for i, line in enumerate(body) if "_eviction_marked.discard" in line
    )
    with_idx = next(i for i, line in enumerate(body) if "with self._state_lock" in line)
    assert (
        with_idx < discard_idx
    ), "_eviction_marked.discard must sit inside the _state_lock block"


@pytest.mark.parametrize("value", ["1", "true", "ON", "True"])
async def test_env_gate_values(monkeypatch, value):
    monkeypatch.setenv("FUSION_MLX_LOCK_DEBUG", value)
    lock = TrackedAsyncLock("pool.adapter_swap")
    async with lock:
        assert any("pool.adapter_swap" in v for v in snapshot_held().values())
