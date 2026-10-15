# SPDX-License-Identifier: Apache-2.0
"""Active file-descriptor watchdog (#1192).

Counts the process's open FDs, breaks them down by type (socket / file /
pipe / other), warns at a configurable usage ratio and rejects NEW streams
at a higher ratio (existing streams are never touched). Leak attribution
compares per-type counts across the sample history ring. Everything here is
wrapped so the watchdog can never break a request — worst case is an empty
snapshot.
"""

import logging
import os
import resource
import stat
import threading
import time

logger = logging.getLogger(__name__)

_FD_SAMPLE_MIN_INTERVAL_S = 1.0
_FD_INFO_LOG_INTERVAL_S = 60.0
_FD_WARN_INTERVAL_S = 300.0
_FD_HISTORY_MAX = 60
_FD_TYPE_SOCKET = "socket"
_FD_TYPE_FILE = "file"
_FD_TYPE_PIPE = "pipe"
_FD_TYPE_OTHER = "other"
_FD_TYPE_GONE = "gone"


class FDPressureError(RuntimeError):
    pass


def _classify_fd(fd: int) -> str:
    try:
        st = os.fstat(fd)
    except OSError:
        return _FD_TYPE_GONE
    if stat.S_ISSOCK(st.st_mode):
        return _FD_TYPE_SOCKET
    if stat.S_ISREG(st.st_mode):
        return _FD_TYPE_FILE
    if stat.S_ISFIFO(st.st_mode):
        return _FD_TYPE_PIPE
    return _FD_TYPE_OTHER


def _list_open_fds() -> list[int]:
    try:
        if os.path.isdir("/proc/self/fd"):
            names = os.listdir("/proc/self/fd")
        else:
            names = os.listdir("/dev/fd")
    except OSError as exc:
        logger.debug("fd watchdog: fd listing failed: %s", exc)
        return []
    fds: list[int] = []
    for name in names:
        try:
            fds.append(int(name))
        except ValueError:
            continue
    return fds


def _resolve_fd_limit() -> int:
    try:
        soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    except Exception as exc:
        logger.debug("fd watchdog: getrlimit failed: %s", exc)
        return 0
    if soft == resource.RLIM_INFINITY:
        return 0
    return int(soft)


def _env_ratio(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        logger.warning(
            "fd watchdog: invalid %s=%r, falling back to %s", name, raw, default
        )
        return default
    if not 0.0 < value <= 1.0:
        logger.warning(
            "fd watchdog: %s=%r out of (0, 1], falling back to %s", name, raw, default
        )
        return default
    return value


class FDWatchdog:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._last_sample: dict[str, int] | None = None
        self._cached_snapshot: dict | None = None
        self._cached_at = 0.0
        self._limit = 0
        self._limit_resolved = False
        self._last_warn = 0.0
        self._last_info = 0.0
        self._pressure_rejected_total = 0
        self._warn_total = 0
        self._history: list[tuple[float, dict[str, int]]] = []
        self._reject_ratio = _env_ratio("FUSION_MLX_FD_REJECT_RATIO", 0.90)
        self._warn_ratio = _env_ratio("FUSION_MLX_FD_WARN_RATIO", 0.80)

    def sample(self, force: bool = False) -> dict:
        now = time.monotonic()
        if not force and self._cached_snapshot is not None:
            if now - self._cached_at < _FD_SAMPLE_MIN_INTERVAL_S:
                return self._cached_snapshot

        total = 0
        by_type: dict[str, int] = {}
        fds = _list_open_fds()
        for fd in fds:
            total += 1
            kind = _classify_fd(fd)
            by_type[kind] = by_type.get(kind, 0) + 1
        if not self._limit_resolved:
            self._limit = _resolve_fd_limit()
            self._limit_resolved = True
        lim = self._limit
        ratio = (total / lim) if lim > 0 else 0.0

        snapshot = {
            "total": total,
            "limit": lim,
            "by_type": by_type,
            "ratio": ratio,
        }
        with self._lock:
            self._last_sample = by_type
            self._cached_snapshot = snapshot
            self._cached_at = now
            self._history.append((time.time(), dict(by_type)))
            if len(self._history) > _FD_HISTORY_MAX:
                del self._history[: len(self._history) - _FD_HISTORY_MAX]
        self._maybe_warn(ratio, total, lim)
        if now - self._last_info >= _FD_INFO_LOG_INTERVAL_S:
            self._last_info = now
            logger.info(
                "fd watchdog: %d open fds (limit=%s, ratio=%.0f%%) by_type=%s",
                total,
                lim if lim > 0 else "unlimited",
                ratio * 100.0,
                by_type,
            )
        return snapshot

    def _maybe_warn(self, ratio: float, used: int, limit: int) -> None:
        if limit <= 0 or not (self._warn_ratio <= ratio < self._reject_ratio):
            return
        now = time.monotonic()
        if now - self._last_warn <= _FD_WARN_INTERVAL_S:
            return
        self._last_warn = now
        self._warn_total += 1
        logger.warning(
            "fd pressure %.0f%% (%d/%d): approaching stream rejection at "
            "%.0f%% — check fd leak attribution via /metrics",
            ratio * 100.0,
            used,
            limit,
            self._reject_ratio * 100.0,
        )
        try:
            from .server_metrics import get_server_metrics

            get_server_metrics().record_fd_warn()
        except Exception as exc:
            logger.debug("fd watchdog: warn metric record failed: %s", exc)

    def check_stream_admission(self) -> None:
        snap = self.sample()
        lim = snap["limit"]
        ratio = snap["ratio"]
        if lim <= 0 or ratio < self._reject_ratio:
            return
        total = snap["total"]
        with self._lock:
            self._pressure_rejected_total += 1
        logger.error(
            "FD pressure %.0f%% (%d/%d): rejecting NEW stream "
            "(existing streams unaffected)",
            ratio * 100.0,
            total,
            lim,
        )
        try:
            from .server_metrics import get_server_metrics

            get_server_metrics().record_fd_pressure_rejection()
        except Exception as exc:
            logger.debug("fd watchdog: rejection metric record failed: %s", exc)
        raise FDPressureError(
            f"fd pressure {ratio * 100.0:.0f}% ({total}/{lim}): "
            f"rejecting NEW stream (existing streams unaffected)"
        )

    def attribution(self) -> dict:
        with self._lock:
            history = list(self._history)
        if len(history) < 2:
            return {
                "window_seconds": 0.0,
                "delta_by_type": {},
                "history": history,
            }
        (t0, counts0), (t1, counts1) = history[0], history[-1]
        dt = t1 - t0
        keys = set(counts0) | set(counts1)
        delta_by_type = {key: counts1.get(key, 0) - counts0.get(key, 0) for key in keys}
        return {
            "window_seconds": dt,
            "delta_by_type": delta_by_type,
            "history": history,
        }

    def snapshot(self) -> dict:
        snap = self.sample()
        result = dict(snap)
        result["warn_total"] = self._warn_total
        result["pressure_rejected_total"] = self._pressure_rejected_total
        result["attribution"] = self.attribution()
        return result


_watchdog: FDWatchdog | None = None
_watchdog_lock = threading.Lock()


def get_fd_watchdog() -> FDWatchdog:
    global _watchdog
    with _watchdog_lock:
        if _watchdog is None:
            _watchdog = FDWatchdog()
        return _watchdog


def reset_fd_watchdog() -> None:
    global _watchdog
    with _watchdog_lock:
        _watchdog = None


def ensure_stream_fd_capacity() -> None:
    # #1192: FD pressure gate — reject NEW streams at >=90% fd usage.
    # Converts FDPressureError to HTTP 503 at the single shared call site
    # so each route gate stays one line; check_stream_admission itself
    # stays framework-free for direct callers and tests.
    try:
        get_fd_watchdog().check_stream_admission()
    except FDPressureError as exc:
        from fastapi import HTTPException

        raise HTTPException(status_code=503, detail=str(exc)) from exc
