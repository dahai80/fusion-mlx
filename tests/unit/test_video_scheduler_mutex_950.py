# SPDX-License-Identifier: Apache-2.0
"""#950: video scheduler dual-model mutex — concurrent different-model
request must NOT kill the in-flight generation.

Prior bug: RLock (reentrant) let the asyncio event-loop thread re-acquire
the lock, bypassing timeout=0, hitting "mutex invariant violated" RuntimeError
-> 500 + premature release -> aborted the running task.

Fix: non-reentrant Lock + VideoMutexBusyError -> 503 Retry-After.
"""

import threading

import pytest

from fusion_mlx.scheduler.video_unified_scheduler import (
    VideoMutexBusyError,
    VideoUnifiedScheduler,
)


def test_second_model_raises_busy_not_invariant_violation():
    sched = VideoUnifiedScheduler()
    sched.acquire("model_a")
    try:
        with pytest.raises(VideoMutexBusyError) as exc_info:
            sched.acquire("model_b")
        assert "model_a" in str(exc_info.value)
        assert "model_b" in str(exc_info.value)
    finally:
        sched.release("model_a")


def test_same_model_reacquire_after_release():
    sched = VideoUnifiedScheduler()
    sched.acquire("model_a")
    sched.release("model_a")
    # lock is free now — same model can re-acquire
    sched.acquire("model_a")
    sched.release("model_a")


def test_release_clears_active_model():
    sched = VideoUnifiedScheduler()
    sched.acquire("model_a")
    assert sched._active_model == "model_a"
    sched.release("model_a")
    assert sched._active_model is None


def test_lock_is_non_reentrant():
    sched = VideoUnifiedScheduler()
    # Verify the underlying lock is a non-reentrant Lock, not RLock.
    # RLock would allow re-acquire on the same thread (the #950 bug).
    assert not isinstance(sched._video_mtx, type(threading.RLock()))


def test_concurrent_thread_cannot_steal_lock():
    sched = VideoUnifiedScheduler()
    sched.acquire("model_a")
    barrier = threading.Event()
    result = {}

    def try_acquire_b():
        barrier.set()
        try:
            sched.acquire("model_b")
            result["ok"] = True
        except VideoMutexBusyError as e:
            result["busy"] = str(e)

    t = threading.Thread(target=try_acquire_b)
    t.start()
    barrier.wait(timeout=2)
    t.join(timeout=5)
    assert "busy" in result
    assert "ok" not in result
    sched.release("model_a")
