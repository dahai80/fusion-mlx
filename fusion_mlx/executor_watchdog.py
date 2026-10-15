# SPDX-License-Identifier: Apache-2.0
# Generalized executor heartbeat/poison watchdog (#1189). One implementation,
# per-executor instances: image / video / audio / laya (llm keeps its own
# G2 watchdog in engine_core.py — do not register "llm" here).
#
# Pattern (mirrors the llm watchdog semantics): a job begins before work is
# dispatched to the executor, advances its heartbeat as progress happens
# (per step / per chunk), and ends in a finally. A shared daemon tick thread
# checks every live watchdog: an inflight job whose heartbeat has been stale
# for longer than the window is hung (Python cannot cancel a running thread),
# so on_hang fires ONCE for that job — typically poison_executor(pool_type),
# which swaps a fresh worker into the registry and fast-fails later requests.

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger(__name__)

# Default staleness windows (seconds), env-overridable. A per-instance
# window_s argument always wins (tests use sub-second windows).
_DEFAULT_WINDOWS: dict[str, tuple[str, float]] = {
    "image": ("FUSION_IMAGE_WATCHDOG_TIMEOUT", 600.0),
    "video": ("FUSION_VIDEO_WATCHDOG_TIMEOUT", 600.0),
    "audio": ("FUSION_AUDIO_WATCHDOG_TIMEOUT", 300.0),
    "laya": ("FUSION_LAYA_WATCHDOG_TIMEOUT", 120.0),
}
_FALLBACK_WINDOW_S = 600.0


def _default_window_s(name: str) -> float:
    entry = _DEFAULT_WINDOWS.get(name)
    if entry is None:
        return _FALLBACK_WINDOW_S
    env_key, default = entry
    raw = os.environ.get(env_key)
    if not raw:
        return default
    try:
        val = float(raw)
    except ValueError:
        logger.warning(
            "%s=%r is not a number, using default %.0fs", env_key, raw, default
        )
        return default
    if val <= 0:
        logger.warning("%s=%r <= 0, using default %.0fs", env_key, raw, default)
        return default
    return val


class ExecutorWatchdog:
    def __init__(
        self,
        name: str,
        window_s: float | None = None,
        on_hang: Callable[[str], Any] | None = None,
        defer_check: Callable[[], bool] | None = None,
    ):
        self.name = name
        self.window_s = window_s if window_s is not None else _default_window_s(name)
        self.on_hang = on_hang
        self.defer_check = defer_check
        self._lock = threading.Lock()
        self._inflight = False
        self._job_id = 0
        self._job_label = ""
        self._start = 0.0
        self._heartbeat = 0.0
        # One-shot fire per wedged job: the tick thread must not re-fire
        # while the same stuck job is still inflight.
        self._fired_job_id = 0
        self._fire_count = 0
        # Media-defer state (image + audio only — video/laya pass no
        # defer_check). Same shape as the llm watchdog's G2-SR defer.
        self._defer_start = 0.0
        self._defer_logged = False
        self._last_end = 0.0
        self._job_count = 0

    def begin_job(self, label: str = "") -> int:
        with self._lock:
            self._job_id += 1
            self._inflight = True
            self._job_label = label
            self._job_count += 1
            now = time.monotonic()
            self._start = now
            self._heartbeat = now
            self._defer_start = 0.0
            self._defer_logged = False
            return self._job_id

    def heartbeat(self) -> None:
        with self._lock:
            self._heartbeat = time.monotonic()

    def end_job(self) -> None:
        with self._lock:
            self._inflight = False
            self._job_label = ""
            self._last_end = time.monotonic()

    def _media_defer_budget_s(self) -> float:
        # Reuse the llm watchdog's media-defer budget (#1189) — the same
        # "GPU starvation, not hang" hazard applies to image/audio jobs that
        # share the Metal command queue with the llm worker. Lazy import to
        # avoid a cycle at module load.
        from .engine_core import _LLM_WATCHDOG_MEDIA_DEFER_S

        return _LLM_WATCHDOG_MEDIA_DEFER_S

    def _deferred(self, now: float) -> bool:
        if self.defer_check is None:
            return False
        try:
            active = bool(self.defer_check())
        except Exception:
            logger.debug("watchdog %s defer_check raised", self.name, exc_info=True)
            return False
        if not active:
            self._defer_start = 0.0
            self._defer_logged = False
            return False
        if self._defer_start == 0.0:
            self._defer_start = now
        budget = self._media_defer_budget_s()
        if now - self._defer_start < budget:
            if not self._defer_logged:
                self._defer_logged = True
                logger.warning(
                    "#1189 %s watchdog: heartbeat stale but media job "
                    "active — deferring poison (GPU starvation suspected, "
                    "not hang). Budget %.0fs.",
                    self.name,
                    budget,
                )
            return True
        logger.error(
            "#1189 %s watchdog: media job active for %.0fs — exceeded "
            "media defer budget (%.0fs). Poisoning; the media job may "
            "be stuck.",
            self.name,
            now - self._defer_start,
            budget,
        )
        return False

    def check(self) -> bool:
        # One watchdog evaluation. Split out of the tick loop so tests can
        # drive it synchronously (same rationale as _llm_watchdog_check).
        with self._lock:
            if not self._inflight:
                return False
            job_id = self._job_id
            if self._fired_job_id == job_id:
                return False
            now = time.monotonic()
            if now - max(self._start, self._heartbeat) <= self.window_s:
                return False
            deferred = self._deferred(now)
        if deferred:
            return False
        with self._lock:
            # Re-validate: the job may have ended while the defer check ran.
            if not self._inflight or self._job_id != job_id:
                return False
            if self._fired_job_id == job_id:
                return False
            self._fired_job_id = job_id
            self._fire_count += 1
        stale = now - max(self._start, self._heartbeat)
        logger.error(
            "#1189 %s watchdog: executor job '%s' hung for %.0fs (heartbeat "
            "stale, window %.0fs). Firing on_hang — poison + executor "
            "replacement. The stuck thread is abandoned.",
            self.name,
            self._job_label,
            stale,
            self.window_s,
        )
        if self.on_hang is not None:
            try:
                self.on_hang(self.name)
            except Exception:
                logger.exception(
                    "#1189 %s watchdog: on_hang callback failed", self.name
                )
        try:
            from .server_metrics import get_server_metrics

            get_server_metrics().record_watchdog_poison(self.name)
        except Exception:
            logger.debug(
                "#1189 %s watchdog: poison metric recording failed",
                self.name,
                exc_info=True,
            )
        return True

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            now = time.monotonic()
            if self._inflight:
                age = now - max(self._start, self._heartbeat)
            elif self._last_end > 0.0:
                age = now - self._last_end
            else:
                age = None
            return {
                "executor": self.name,
                "inflight": self._inflight,
                "job": self._job_label,
                "heartbeat_age_s": age,
                "window_s": self.window_s,
                "fire_count": self._fire_count,
                "job_count": self._job_count,
            }


class _WatchHandle:
    def __init__(self, watchdog: ExecutorWatchdog):
        self._watchdog = watchdog

    def heartbeat(self) -> None:
        self._watchdog.heartbeat()


@contextmanager
def executor_watch(name: str, window_s: float | None = None):
    # Exception-safe job bracket: begin on enter, end on exit. The yielded
    # handle exposes .heartbeat() for per-step/per-chunk progress.
    wd = get_executor_watchdog(name, window_s=window_s)
    wd.begin_job(f"{name}-job")
    try:
        yield _WatchHandle(wd)
    finally:
        wd.end_job()


_watchdogs: dict[str, ExecutorWatchdog] = {}
_registry_lock = threading.Lock()
_tick_thread: threading.Thread | None = None
_tick_stop = threading.Event()
_TICK_INTERVAL_S = 1.0


def _tick() -> None:
    while not _tick_stop.is_set():
        _tick_stop.wait(_TICK_INTERVAL_S)
        if _tick_stop.is_set():
            break
        with _registry_lock:
            watchers = list(_watchdogs.values())
        for wd in watchers:
            try:
                wd.check()
            except Exception:
                logger.exception("#1189 watchdog tick failed: %s", wd.name)


def _ensure_tick_thread_locked() -> None:
    global _tick_thread
    if _tick_thread is not None and _tick_thread.is_alive():
        return
    _tick_stop.clear()
    _tick_thread = threading.Thread(
        target=_tick, name="mlx-executor-watchdog", daemon=True
    )
    _tick_thread.start()
    logger.info(
        "#1189 executor watchdog tick thread started (interval=%.0fs)",
        _TICK_INTERVAL_S,
    )


def get_executor_watchdog(
    name: str,
    window_s: float | None = None,
    on_hang: Callable[[str], Any] | None = None,
    defer_check: Callable[[], bool] | None = None,
) -> ExecutorWatchdog:
    # Per-executor singleton. First call fixes the on_hang/defer_check
    # wiring; later callers may pass overrides but the existing instance
    # wins (all wiring sites pass the same callbacks for a given name).
    with _registry_lock:
        wd = _watchdogs.get(name)
        if wd is None:
            wd = ExecutorWatchdog(
                name,
                window_s=window_s,
                on_hang=on_hang,
                defer_check=defer_check,
            )
            _watchdogs[name] = wd
            _ensure_tick_thread_locked()
        return wd


def reset_executor_watchdogs() -> None:
    # Test hook: drop all instances so a test can construct fresh watchdogs
    # with tight windows. The tick thread keeps running (it iterates an
    # empty registry until a new watchdog is created).
    with _registry_lock:
        _watchdogs.clear()


def executor_watchdog_snapshots() -> list[dict[str, Any]]:
    with _registry_lock:
        watchers = list(_watchdogs.values())
    return [wd.snapshot() for wd in watchers]
