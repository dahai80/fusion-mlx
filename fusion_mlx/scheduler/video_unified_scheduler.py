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
        try:
            mx.metal.clear_cache()
        except Exception:
            pass


class VideoUnifiedScheduler:
    """PRD v1 §4.1 — the single entry point every video task passes
    through. Holds the video↔video mutex, probes memory, emits the
    DegradationPlan, owns the NF4 dequant cache for the task lifetime.
    """

    def __init__(self, red_line_gb: int = _RED_LINE_GB):
        self.red_line_bytes = red_line_gb * _GB
        self._video_mtx = threading.RLock()
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

    # -- PRD §3.4 L3: 1s emergency reclaim -------------------------------
    def emergency_reclaim(self) -> None:
        t0 = time.monotonic()
        gc.collect()
        try:
            mx.metal.clear_cache()
        except Exception:
            pass
        if self._cache is not None:
            self._cache.release()
            self._cache = None
        elapsed = time.monotonic() - t0
        logger.warning("emergency reclaim done in %.3fs", elapsed)

    # -- dual-model mutex (PRD §3.2) -------------------------------------
    def acquire(self, model_name: str) -> None:
        if not self._video_mtx.acquire(timeout=0):
            other = self._active_model or "unknown"
            raise RuntimeError(
                f"video model {model_name} cannot load: {other} is resident "
                f"(dual-model mutex, PRD §3.2). Queue and retry."
            )
        try:
            if self._active_model is not None and self._active_model != model_name:
                raise RuntimeError(
                    f"mutex invariant violated: {self._active_model} resident, "
                    f"requested {model_name}"
                )
            self._active_model = model_name
            self._cache = NF4DequantCache()
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
            mx.metal.clear_cache()
        except Exception:
            pass
        self._active_model = None
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
