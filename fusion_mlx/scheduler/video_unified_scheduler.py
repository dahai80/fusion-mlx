# SPDX-License-Identifier: Apache-2.0
"""Unified video scheduler (PRD v1 §3.2-§3.4 / §4.1).

THE core new safety layer. Wraps the existing ProcessMemoryEnforcer
ceiling machinery with the PRD-mandated three-level video-specific
circuit breaker, NF4 Metal dequant cache, forced GC, and dual-model
mutex. Every video generation task (LTX or H3) flows through here so
the 128G UMA box never hits kernel OOM.

Three-level protection (PRD §3.4):
  L1 warn  (>=90GB): reduce sampling steps, disable fine upsampling
  L2 protect(>=95GB): drop resolution, disable joint audio
  L3 circuit(>=98GB): mx.metal.clear_cache() + full tensor GC, <=1s drop

NF4 dequant cache (PRD §3.3):
  DiT core weights → pre-dequantized resident cache (avoid per-step
  30-40x repeat dequant reads). Small weights/bias → Metal inline
  instant dequant. Cache lifetime = one generation task, released on
  task end.

Dual-model mutex (PRD §3.2):
  Only ONE video model resident at a time. LLM/video mutual exclusion
  is already handled by EnginePool; this adds the video↔video lock so
  LTX and H3 never co-resident.

Memory red line (PRD §3.2): global peak <= 98GB (was 105GB).
"""

from __future__ import annotations

import gc
import logging
import threading
import time
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import mlx.core as mx

logger = logging.getLogger(__name__)

# PRD v1 §3.2 — 128G UMA safety red line. Was 105GB, tightened to 98GB
# leaving >=28GB system buffer so macOS never compresses/jetsams.
_RED_LINE_GB = 98
_L1_WARN_GB = 90
_L2_PROTECT_GB = 95
_L3_CIRCUIT_GB = 98

_GB = 1024**3


class VideoMutexBusyError(RuntimeError):
    # #950: raised when a second video model requests the mutex while another
    # is resident + generating. Maps to HTTP 503 Retry-After (NOT 500) so the
    # client retries and the in-flight generation is NOT killed. The prior
    # design used threading.RLock (reentrant) which let the asyncio event-loop
    # thread re-acquire the lock, bypassing the timeout=0 guard and hitting
    # the "mutex invariant violated" RuntimeError — that crashed the server
    # and aborted the running task.
    pass


class VideoMemoryPressureError(RuntimeError):
    # #951: raised mid-denoise when sustained memory pressure crosses the L3
    # circuit red line (>=98GB) and emergency_reclaim cannot bring it back
    # below the ceiling. Aborts the in-flight generation with a clean 507
    # Retry-After so the SERVER survives — instead of the ProcessMemoryEnforcer
    # fatal_exit killing the whole process (+ any co-tenant LLM). The legacy
    # 2B DiT has no temporal upsampler so per-step attention memory scales
    # linearly with latent frames; a long/high-res request can cross the
    # ceiling mid-denoise even when begin_task's pre-probe saw OK memory.
    #
    # #951-downstream: also raised when an EXTERNAL abort signal is set —
    # either by the ProcessMemoryEnforcer (1s poll granularity, finer than
    # the per-step ~8s check_step_pressure boundary) or by the parent
    # watchdog when macOS jetsam kills the supervisor shell mid-generation.
    # Without this, a 49-frame 1344x768 run dies at step ~26 because jetsam
    # kills the parent shell -> the serve child self-SIGTERMs (orphan path)
    # before the next step-boundary probe fires. The abort event is the
    # 1s-granularity trip wire that reaches the video thread at the next
    # step boundary and converts the death into a clean 507.
    pass


# #951-downstream: process-global abort event. Set by the enforcer (emergency
# pressure while a video generation is in-flight) or by the parent watchdog
# (orphaned mid-generation). The video backend's step callback checks this at
# each denoise step boundary via check_abort() and raises
# VideoMemoryPressureError if set — converting a process-killing jetsam /
# fatal_exit into a clean 507 Retry-After. threading.Event is thread-safe
# and safe to set from the enforcer/watchdog threads.
_VIDEO_ABORT_EVENT = threading.Event()
_VIDEO_ABORT_REASON: list[str] = []


def signal_video_abort(reason: str) -> bool:
    # Returns True if this call newly armed the abort (i.e. a video generation
    # is in-flight and will pick it up). False if no generation is in-flight
    # (caller — enforcer/watchdog — can then proceed with its own hard path).
    sched = _singleton
    generating = sched is not None and sched._active_model is not None
    if not generating:
        return False
    _VIDEO_ABORT_REASON.append(reason)
    _VIDEO_ABORT_EVENT.set()
    logger.warning(
        "video abort signaled (in-flight model=%s): %s",
        sched._active_model,
        reason,
    )
    return True


def is_video_generating() -> bool:
    sched = _singleton
    return sched is not None and sched._active_model is not None


def wait_video_generation_done(timeout: float = 12.0) -> bool:
    # Used by the parent watchdog orphan path: after signaling abort, wait for
    # the video thread to raise VideoMemoryPressureError and exit the
    # generation (releasing the mutex) before self-terminating. Returns True
    # if the generation ended within the grace window.
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not is_video_generating():
            return True
        time.sleep(0.25)
    return not is_video_generating()


class MemoryLevel(IntEnum):
    OK = 0
    L1_WARN = 1
    L2_PROTECT = 2
    L3_CIRCUIT = 3


@dataclass
class DegradationPlan:
    level: MemoryLevel
    reduce_steps: bool = False
    disable_upsample: bool = False
    drop_resolution: bool = False
    disable_audio: bool = False
    force_gc: bool = False
    reason: str = ""

    def apply_to(self, params: Any) -> Any:
        if self.reduce_steps and getattr(params, "num_inference_steps", None):
            old = params.num_inference_steps
            params.num_inference_steps = max(8, old // 2)
            logger.warning(
                "L1 degrade: steps %d -> %d (%s)",
                old,
                params.num_inference_steps,
                self.reason,
            )
        if self.disable_upsample:
            params.no_compile = True
            logger.warning("L1 degrade: fine upsampling off (%s)", self.reason)
        if self.drop_resolution and getattr(params, "height", None):
            params.height = max(384, params.height // 2)
            params.width = max(384, getattr(params, "width", 384) // 2)
            logger.warning("L2 degrade: resolution halved (%s)", self.reason)
        if self.disable_audio:
            params.audio = False
            logger.warning("L2 degrade: joint audio off (%s)", self.reason)
        return params


@dataclass
class _DequantCacheEntry:
    key: str
    array: mx.array
    bytes_: int
    ts: float


class NF4DequantCache:
    """PRD §3.3 — Metal inline dequant + pre-dequant resident cache.

    DiT core weights stay pre-dequantized for the task lifetime (avoid
    30-40 step repeat reads). Small weights dequant inline per call.
    """

    def __init__(self, budget_gb: float = 24.0):
        self._entries: dict[str, _DequantCacheEntry] = {}
        self._budget = int(budget_gb * _GB)
        self._used = 0
        self._mtx = threading.Lock()

    def get_or_dequant(
        self, key: str, dequant_fn: Any, *, resident: bool = True
    ) -> mx.array:
        with self._mtx:
            hit = self._entries.get(key)
            if hit is not None:
                logger.debug("dequant cache hit: %s", key)
                return hit.array
        arr = dequant_fn()
        mx.eval(arr)
        if not resident:
            return arr
        size = int(arr.size * arr.itemsize) if arr.itemsize > 0 else 0
        with self._mtx:
            self._evict_if_needed(size)
            self._entries[key] = _DequantCacheEntry(key, arr, size, time.time())
            self._used += size
            logger.info(
                "dequant cache insert: %s (%.2f GB, total %.2f GB)",
                key,
                size / _GB,
                self._used / _GB,
            )
        return arr

    def _evict_if_needed(self, incoming: int) -> None:
        while self._used + incoming > self._budget and self._entries:
            _, ev = self._entries.pop(next(iter(self._entries)))
            self._used -= ev.bytes_
            logger.debug("dequant cache evict: %s", ev.key)

    def release(self) -> None:
        with self._mtx:
            n = len(self._entries)
            self._entries.clear()
            self._used = 0
        if n:
            logger.info("dequant cache released (%d entries)", n)
        gc.collect()
        # #gap6: version-compatible cache clear (mx.clear_cache w/ metal fallback).
        from fusion_mlx.utils.proc_memory import clear_metal_cache

        try:
            clear_metal_cache()
        except Exception as exc:
            logger.debug("dequant cache metal clear failed: %s", exc)


class VideoUnifiedScheduler:
    """PRD v1 §4.1 — the single entry point every video task passes
    through. Holds the video↔video mutex, probes memory, emits the
    DegradationPlan, owns the NF4 dequant cache for the task lifetime.
    """

    def __init__(self, red_line_gb: int = _RED_LINE_GB):
        self.red_line_bytes = red_line_gb * _GB
        # #950: non-reentrant Lock (not RLock). The asyncio event-loop thread
        # calls begin_task (acquire) for model A, then while A generates in an
        # executor thread, B's request calls begin_task on the same event-loop
        # thread. RLock would re-acquire (reentrant) and hit the invariant
        # check -> 500 + premature release killing A. Lock + timeout=0 fails
        # immediately -> VideoMutexBusyError -> 503 Retry-After (A survives).
        self._video_mtx = threading.Lock()
        self._active_model: str | None = None
        self._cache: NF4DequantCache | None = None
        logger.info("VideoUnifiedScheduler ready (red_line=%dGB)", red_line_gb)

    # -- memory probe ----------------------------------------------------
    def _current_bytes(self) -> int:
        try:
            from fusion_mlx.utils.proc_memory import get_phys_footprint

            fp = get_phys_footprint()
            if fp and fp > 0:
                return int(fp)
        except Exception:
            pass
        try:
            return int(mx.metal.get_active_memory())
        except Exception:
            return 0

    @staticmethod
    def _clear_cache() -> None:
        # #gap6: mx.metal.clear_cache deprecated in MLX 0.32+; use the
        # version-compatible helper (mx.clear_cache with metal fallback).
        from fusion_mlx.utils.proc_memory import clear_metal_cache

        clear_metal_cache()

    def probe_level(self) -> MemoryLevel:
        b = self._current_bytes()
        gb = b / _GB
        if gb >= _L3_CIRCUIT_GB:
            return MemoryLevel.L3_CIRCUIT
        if gb >= _L2_PROTECT_GB:
            return MemoryLevel.L2_PROTECT
        if gb >= _L1_WARN_GB:
            return MemoryLevel.L1_WARN
        return MemoryLevel.OK

    def plan(self) -> DegradationPlan:
        lvl = self.probe_level()
        if lvl is MemoryLevel.L3_CIRCUIT:
            self.emergency_reclaim()
            return DegradationPlan(
                level=lvl,
                reduce_steps=True,
                disable_upsample=True,
                drop_resolution=True,
                disable_audio=True,
                force_gc=True,
                reason=f"peak >= {_L3_CIRCUIT_GB}GB, emergency reclaim fired",
            )
        if lvl is MemoryLevel.L2_PROTECT:
            return DegradationPlan(
                level=lvl,
                drop_resolution=True,
                disable_audio=True,
                reason=f"peak >= {_L2_PROTECT_GB}GB",
            )
        if lvl is MemoryLevel.L1_WARN:
            return DegradationPlan(
                level=lvl,
                reduce_steps=True,
                disable_upsample=True,
                reason=f"peak >= {_L1_WARN_GB}GB",
            )
        return DegradationPlan(level=MemoryLevel.OK)

    def check_abort(self) -> None:
        # #951-downstream: raise VideoMemoryPressureError if an external actor
        # (enforcer 1s poll / parent watchdog orphan path) armed the abort
        # event during the last denoise step. Called at every step boundary
        # by the backend's step callback (before the per-step pressure probe).
        if _VIDEO_ABORT_EVENT.is_set():
            reason = (
                _VIDEO_ABORT_REASON[-1]
                if _VIDEO_ABORT_REASON
                else ("external abort signal (enforcer/watchdog)")
            )
            _VIDEO_ABORT_EVENT.clear()
            _VIDEO_ABORT_REASON.clear()
            raise VideoMemoryPressureError(
                f"video generation aborted by external memory signal: {reason} "
                f"(#951-downstream). The server stays alive; reduce "
                f"num_frames / resolution and retry."
            )

    # -- PRD §3.4 L3: 1s emergency reclaim -------------------------------
    def emergency_reclaim(self) -> None:
        t0 = time.monotonic()
        gc.collect()
        try:
            self._clear_cache()
        except Exception:
            pass
        if self._cache is not None:
            self._cache.release()
            self._cache = None
        elapsed = time.monotonic() - t0
        logger.warning("emergency reclaim done in %.3fs", elapsed)

    def check_step_pressure(self) -> MemoryLevel:
        # #951: mid-denoise memory guard. Called between denoise steps (via the
        # backend's step callback). If pressure is L3_CIRCUIT (>=98GB), fire
        # emergency_reclaim first (clear Metal cache + release dequant cache +
        # GC — may drop enough to continue). Re-probe: if STILL L3, raise
        # VideoMemoryPressureError so the backend aborts the generation (clean
        # 507) instead of the ProcessMemoryEnforcer fatal_exit killing the
        # whole server. Returns the post-reclaim level on success.
        #
        # #951-downstream: check the external abort event FIRST — the enforcer
        # (1s poll) or the parent watchdog may have armed it between step
        # boundaries when this probe's own footprint sample is momentarily
        # below L3 (MLX releases cache between steps). Without this, a 49-frame
        # run whose pressure spikes DURING a step eval (not at the boundary)
        # escapes the per-step guard and dies to jetsam/fatal_exit.
        self.check_abort()
        lvl = self.probe_level()
        if lvl is not MemoryLevel.L3_CIRCUIT:
            return lvl
        logger.warning(
            "mid-denoise L3 circuit pressure (%.1fGB) — emergency reclaim",
            self._current_bytes() / _GB,
        )
        self.emergency_reclaim()
        post = self.probe_level()
        if post is MemoryLevel.L3_CIRCUIT:
            raise VideoMemoryPressureError(
                f"sustained memory pressure {self._current_bytes() / _GB:.1f}GB "
                f">= {_L3_CIRCUIT_GB}GB red line mid-denoise after emergency "
                f"reclaim; aborting generation to keep the server alive (#951). "
                f"Reduce num_frames / resolution and retry."
            )
        return post

    # -- dual-model mutex (PRD §3.2) -------------------------------------
    def acquire(self, model_name: str) -> None:
        # #950: non-reentrant Lock + timeout=0. If another video model is
        # resident (lock held), fail fast with VideoMutexBusyError -> 503
        # Retry-After. The in-flight generation is NOT killed.
        if not self._video_mtx.acquire(timeout=0):
            other = self._active_model or "unknown"
            raise VideoMutexBusyError(
                f"video model {model_name} cannot load: {other} is resident "
                f"(dual-model mutex, PRD §3.2). Retry after the current "
                f"generation completes."
            )
        try:
            self._active_model = model_name
            self._cache = NF4DequantCache()
            # #951-downstream: clear any stale abort from a prior task so the
            # new generation starts clean.
            _VIDEO_ABORT_EVENT.clear()
            _VIDEO_ABORT_REASON.clear()
            logger.info("video scheduler acquired by %s", model_name)
        except Exception:
            self._video_mtx.release()
            raise

    def release(self, model_name: str) -> None:
        if self._cache is not None:
            self._cache.release()
            self._cache = None
        gc.collect()
        try:
            self._clear_cache()
        except Exception:
            pass
        self._active_model = None
        # #951-downstream: disarm the abort event so a waiter (parent watchdog)
        # sees the generation as ended.
        _VIDEO_ABORT_EVENT.clear()
        _VIDEO_ABORT_REASON.clear()
        try:
            self._video_mtx.release()
        except RuntimeError:
            pass
        logger.info("video scheduler released by %s", model_name)

    # -- task lifecycle --------------------------------------------------
    def begin_task(self, model_name: str, params: Any) -> Any:
        self.acquire(model_name)
        plan = self.plan()
        if plan.level is not MemoryLevel.OK:
            params = plan.apply_to(params)
        return params

    def end_task(self, model_name: str) -> None:
        self.release(model_name)

    @property
    def dequant_cache(self) -> NF4DequantCache:
        if self._cache is None:
            raise RuntimeError("no active task — call acquire/begin_task first")
        return self._cache

    def snapshot(self) -> dict:
        return {
            "active_model": self._active_model,
            "peak_gb": round(self._current_bytes() / _GB, 2),
            "level": self.probe_level().name,
            "red_line_gb": self.red_line_bytes // _GB,
        }


_singleton: VideoUnifiedScheduler | None = None
_mtx = threading.Lock()


def get_video_scheduler() -> VideoUnifiedScheduler:
    global _singleton
    if _singleton is None:
        with _mtx:
            if _singleton is None:
                _singleton = VideoUnifiedScheduler()
    return _singleton
