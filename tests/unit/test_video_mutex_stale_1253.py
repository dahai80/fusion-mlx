# SPDX-License-Identifier: Apache-2.0
"""#1253: stale video-mutex re-arm.

A mutex held by an owner asyncio task that is DONE (failed/cancelled without
reaching end_task) cannot be mid-generation — re-arm and admit instead of
503ing on a generation that will never complete. A live owner (or no owner,
i.e. sync callers) is a real PRD §3.2 conflict and still fails fast.
"""

import asyncio

import pytest

from fusion_mlx.scheduler.video_unified_scheduler import (
    VideoMutexBusyError,
    VideoUnifiedScheduler,
)


def test_stale_mutex_rearmed_when_owner_task_done():
    sched = VideoUnifiedScheduler()
    held = {}

    async def holder():
        sched.acquire("ltx2_5")
        # Simulate a failed/cancelled job that leaks the mutex: the task ends
        # WITHOUT calling release(). end_task's guarded finally makes this
        # hard to reach in practice; this pins the safety net.
        held["done"] = True

    asyncio.run(holder())
    # The holder task is done; the mutex is still held. Next acquire must
    # re-arm instead of raising VideoMutexBusyError forever.
    asyncio.run(holder_admits(sched))
    sched.release("ltx2_5")
    assert held["done"]


async def holder_admits(sched):
    # Runs as its own task; acquire succeeds via the #1253 stale re-arm.
    sched.acquire("ltx2_5")
    assert sched._active_model == "ltx2_5"
    sched.release("ltx2_5")


def test_live_owner_blocks_other_model():
    sched = VideoUnifiedScheduler()

    async def main():
        holder = asyncio.create_task(live_holder(sched))
        await asyncio.sleep(0.05)
        assert sched._owner_task is holder
        with pytest.raises(VideoMutexBusyError):
            sched.acquire("wan2")
        # Same-model retry while the first generation is LIVE is also a real
        # conflict (the first job will complete and free the mutex).
        with pytest.raises(VideoMutexBusyError):
            sched.acquire("ltx2_5")
        holder.cancel()
        with pytest.raises(asyncio.CancelledError):
            await holder

    asyncio.run(main())
    # Cancelled holder: engine.generate's finally always calls end_task in
    # production; here release ran via the holder's own finally, so the
    # scheduler is free again.
    assert sched._active_model is None


async def live_holder(sched):
    sched.acquire("ltx2_5")
    try:
        await asyncio.sleep(5)
    finally:
        sched.release("ltx2_5")


def test_release_frees_mutex_even_when_cache_teardown_raises():
    sched = VideoUnifiedScheduler()
    sched.acquire("ltx2_5")
    real_cache = sched._cache
    assert real_cache is not None

    def _boom():
        raise RuntimeError("simulated dequant cache teardown failure")

    real_cache.release = _boom
    # Must not raise, and must free the mutex regardless.
    sched.release("ltx2_5")
    sched.acquire("ltx2_5")
    sched.release("ltx2_5")


def test_stale_rearm_resets_owner_and_aborts():
    sched = VideoUnifiedScheduler()
    from fusion_mlx.scheduler.video_unified_scheduler import (
        _VIDEO_ABORT_EVENT,
        _VIDEO_ABORT_REASON,
    )

    async def holder():
        sched.acquire("ltx2_5")
        _VIDEO_ABORT_REASON.append("stale abort from a dead generation")

    asyncio.run(holder())
    sched.acquire("wan2")
    # Re-arm clears the stale abort state so the new generation starts clean.
    assert not _VIDEO_ABORT_EVENT.is_set()
    assert _VIDEO_ABORT_REASON == []
    sched.release("wan2")
