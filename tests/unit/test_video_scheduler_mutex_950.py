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
    MemoryLevel,
    VideoMemoryPressureError,
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


# #951: mid-denoise memory guard — check_step_pressure must abort the
# generation (VideoMemoryPressureError -> 507) when sustained L3 circuit
# pressure cannot be reclaimed, instead of letting ProcessMemoryEnforcer
# fatal_exit kill the whole server.


def test_check_step_pressure_ok_when_below_l3(monkeypatch):
    sched = VideoUnifiedScheduler()
    monkeypatch.setattr(sched, "_current_bytes", lambda: 50 * 1024**3)
    assert sched.check_step_pressure() is MemoryLevel.OK


def test_check_step_pressure_aborts_when_sustained_l3(monkeypatch):
    sched = VideoUnifiedScheduler()
    # Sustained L3: probe stays >=98GB even after emergency_reclaim.
    monkeypatch.setattr(sched, "_current_bytes", lambda: 99 * 1024**3)
    with pytest.raises(VideoMemoryPressureError) as exc_info:
        sched.check_step_pressure()
    assert "98" in str(exc_info.value) or "red line" in str(exc_info.value)


def test_check_step_pressure_recovers_after_reclaim(monkeypatch):
    sched = VideoUnifiedScheduler()
    calls = {"n": 0}

    def _fake_bytes():
        calls["n"] += 1
        # first probe (pre-reclaim): L3; second probe (post-reclaim): OK
        return 99 * 1024**3 if calls["n"] == 1 else 50 * 1024**3

    monkeypatch.setattr(sched, "_current_bytes", _fake_bytes)
    # emergency_reclaim must not crash (no real cache allocated).
    level = sched.check_step_pressure()
    assert level is MemoryLevel.OK


# #951-downstream: external abort signal (enforcer 1s poll / parent watchdog
# orphan path) must convert a process-killing jetsam/fatal_exit into a clean
# 507 VideoMemoryPressureError at the next step boundary.
def test_signal_video_abort_raises_when_generating():
    from fusion_mlx.scheduler import video_unified_scheduler as mod

    mod._singleton = None
    sched = mod.get_video_scheduler()
    mod._VIDEO_ABORT_EVENT.clear()
    mod._VIDEO_ABORT_REASON.clear()
    sched.acquire("ltx_video_legacy")
    try:
        # enforcer/watchdog arms the event from another thread
        armed = mod.signal_video_abort("enforcer emergency pressure test")
        assert armed is True
        assert mod.is_video_generating() is True
        with pytest.raises(VideoMemoryPressureError) as exc_info:
            sched.check_abort()
        assert "enforcer emergency pressure test" in str(exc_info.value)
        # event cleared after raising
        assert mod._VIDEO_ABORT_EVENT.is_set() is False
    finally:
        sched.release("ltx_video_legacy")
        mod._singleton = None


def test_signal_video_abort_noop_when_not_generating():
    from fusion_mlx.scheduler import video_unified_scheduler as mod

    mod._singleton = None
    mod._VIDEO_ABORT_EVENT.clear()
    mod._VIDEO_ABORT_REASON.clear()
    # no generation in-flight -> signal returns False (caller proceeds hard)
    armed = mod.signal_video_abort("orphan but no video")
    assert armed is False
    # check_abort is a no-op (no active model, but event not set anyway)
    assert mod._VIDEO_ABORT_EVENT.is_set() is False


def test_check_step_pressure_checks_abort_event_first():
    # #951-downstream: check_step_pressure must check the external abort event
    # BEFORE its own footprint probe — the enforcer (1s) sees the spike during
    # a step eval; this probe runs at the boundary where footprint momentarily
    # dipped below L3 (MLX cache released between steps). Without checking the
    # event first, the 49-frame run escapes the guard.
    from fusion_mlx.scheduler import video_unified_scheduler as mod

    mod._singleton = None
    sched = mod.get_video_scheduler()
    mod._VIDEO_ABORT_EVENT.clear()
    mod._VIDEO_ABORT_REASON.clear()
    sched.acquire("ltx_video_legacy")
    try:
        mod.signal_video_abort("watchdog orphan")
        # force footprint to read OK (below L3) — proves abort is checked first
        sched._current_bytes = lambda: 0
        with pytest.raises(VideoMemoryPressureError):
            sched.check_step_pressure()
    finally:
        sched.release("ltx_video_legacy")
        mod._singleton = None


def test_abort_event_cleared_on_acquire_release_isolation():
    from fusion_mlx.scheduler import video_unified_scheduler as mod

    mod._singleton = None
    sched = mod.get_video_scheduler()
    # stale abort from a prior task
    mod._VIDEO_ABORT_EVENT.set()
    mod._VIDEO_ABORT_REASON.append("stale")
    sched.acquire("model_a")
    try:
        assert mod._VIDEO_ABORT_EVENT.is_set() is False
    finally:
        sched.release("model_a")
    assert mod._VIDEO_ABORT_EVENT.is_set() is False
    mod._singleton = None


def test_wait_video_generation_done_returns_when_released():
    from fusion_mlx.scheduler import video_unified_scheduler as mod

    mod._singleton = None
    sched = mod.get_video_scheduler()
    sched.acquire("model_a")

    def _release_after():
        import time as _t

        _t.sleep(0.4)
        sched.release("model_a")

    t = threading.Thread(target=_release_after)
    t.start()
    ok = mod.wait_video_generation_done(timeout=3.0)
    t.join()
    assert ok is True
    mod._singleton = None
