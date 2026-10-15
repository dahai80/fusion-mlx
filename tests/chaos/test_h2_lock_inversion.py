# SPDX-License-Identifier: Apache-2.0
"""H2 chaos shape: lock-order inversion (deep-bug deadlock prevention).

Injection: acquire two TrackedLocks in the WRONG nested order under
FUSION_MLX_LOCK_DEBUG=1 + FUSION_MLX_LOCK_DEBUG_STRICT=1 (#1190). The
tracker must detect the inversion before the deadlock can form (strict
mode raises LockOrderViolation; non-strict logs). Recovery = releasing in
order leaves no residual held state.
"""

import asyncio
import time

import pytest

from fusion_mlx.pool.lock_order import (
    LockOrderViolation,
    TrackedAsyncLock,
    TrackedThreadLock,
    reset_lock_order_tracker,
    snapshot_held,
)
from tests.chaos.conftest import DETECTION_BOUND_S, RECOVERY_BOUND_S


@pytest.fixture()
def strict(monkeypatch):
    monkeypatch.setenv("FUSION_MLX_LOCK_DEBUG", "1")
    monkeypatch.setenv("FUSION_MLX_LOCK_DEBUG_STRICT", "1")
    reset_lock_order_tracker()
    return None


def test_h2_async_inversion_detected_before_deadlock(strict):
    # WRONG order: pool.pinned held, then pool.main acquired — the exact
    # nesting that deadlocks against the declared global order.
    t0 = time.monotonic()
    outer = TrackedAsyncLock("pool.pinned")
    inner = TrackedAsyncLock("pool.main")

    async def invert():
        async with outer:
            async with inner:  # violates: pool.main is EARLIER
                return "acquired"

    with pytest.raises(LockOrderViolation):
        asyncio.run(invert())
    assert time.monotonic() - t0 < DETECTION_BOUND_S


def test_h2_thread_inversion_detected(strict):
    t0 = time.monotonic()
    outer = TrackedThreadLock("pool.discover_sync")
    inner = TrackedThreadLock("pool.main")
    with outer:
        with pytest.raises(LockOrderViolation):
            with inner:
                pass
    assert time.monotonic() - t0 < DETECTION_BOUND_S


def test_h2_correct_order_passes(strict):
    # Declared order (outermost-first): pool.main before pool.pinned.
    outer = TrackedAsyncLock("pool.main")
    inner = TrackedAsyncLock("pool.pinned")

    async def correct():
        async with outer:
            async with inner:
                return "ok"

    assert asyncio.run(correct()) == "ok"
    assert snapshot_held() == {}


def test_h2_logging_without_strict(monkeypatch, caplog):
    # Debug on, strict off: the violation is logged (detection) but does
    # not break the holder — matches the default-off prod posture when a
    # debug build surfaces an inversion during a real request.
    monkeypatch.setenv("FUSION_MLX_LOCK_DEBUG", "1")
    monkeypatch.delenv("FUSION_MLX_LOCK_DEBUG_STRICT", raising=False)
    reset_lock_order_tracker()
    outer = TrackedThreadLock("pool.pinned")
    inner = TrackedThreadLock("pool.main")
    with caplog.at_level("ERROR"):
        with outer:
            with inner:
                pass
    assert any("LOCK ORDER VIOLATION" in r.message for r in caplog.records)


def test_h2_recovery_no_residual_state(strict):
    # After a detected inversion the thread releases everything; the
    # tracker must show a clean slate so the next acquisition starts fresh
    # (a stale held-stack would false-positive the NEXT request).
    outer = TrackedThreadLock("pool.adapter_swap")
    inner = TrackedThreadLock("enforcer.state")
    with outer:
        with pytest.raises(LockOrderViolation):
            with inner:
                pass
    # outer released via context exit even though inner raised
    assert snapshot_held() == {}
    t0 = time.monotonic()
    good = TrackedThreadLock("enforcer.state")
    with good:
        assert snapshot_held() != {}
    assert time.monotonic() - t0 < RECOVERY_BOUND_S


def test_h2_cross_task_concurrency_no_false_positive(strict):
    # Two tasks holding different locks concurrently is LEGITIMATE — the
    # tracker keys by (thread, task) so there must be no cross-task
    # false inversion.
    a = TrackedAsyncLock("pool.main")
    b = TrackedAsyncLock("pool.pinned")

    async def hold_a():
        async with a:
            await asyncio.sleep(0.05)

    async def hold_b():
        async with b:
            await asyncio.sleep(0.05)

    async def main():
        await asyncio.gather(hold_a(), hold_b())

    asyncio.run(main())
    assert snapshot_held() == {}
