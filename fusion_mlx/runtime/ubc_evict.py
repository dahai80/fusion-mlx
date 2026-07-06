# SPDX-License-Identifier: Apache-2.0
"""macOS Unified Buffer Cache (UBC) eviction helper.

On macOS, mmap(MAP_SHARED, PROT_READ) pages remain resident in UBC
after munmap/close. For large models (GLM-5.2 ~200GB), this doubles
memory pressure during load (mmap mirror + materialised weights ~= 2x),
tripping Jetsam.

The fix: msync(addr, len, MS_INVALIDATE) on a MAP_SHARED|PROT_READ
mapping flushes the UBC mirror. Safe for read-only mappings — no dirty
data to write back, purely an eviction.

No-op on non-Darwin platforms. Never raises — every error path logs
and returns 0.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import logging
import os
import sys
import threading
import time
from collections.abc import Iterable

logger = logging.getLogger(__name__)

_PROT_READ: int = 0x01
_MAP_SHARED: int = 0x0001
_MS_INVALIDATE: int = 0x0002
_MMAP_FAILED: int = ctypes.c_void_p(-1).value

_libc_lock = threading.Lock()
_libc: ctypes.CDLL | None = None


def _get_libc() -> ctypes.CDLL | None:
    global _libc
    if sys.platform != "darwin":
        return None
    if _libc is not None:
        return _libc
    with _libc_lock:
        if _libc is not None:
            return _libc
        try:
            libname = ctypes.util.find_library("c") or "libSystem.dylib"
            lib = ctypes.CDLL(libname, use_errno=True)
            lib.mmap.argtypes = [
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_longlong,
            ]
            lib.mmap.restype = ctypes.c_void_p
            lib.msync.argtypes = [
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_int,
            ]
            lib.msync.restype = ctypes.c_int
            lib.munmap.argtypes = [
                ctypes.c_void_p,
                ctypes.c_size_t,
            ]
            lib.munmap.restype = ctypes.c_int
            _libc = lib
        except OSError as e:
            logger.debug("ubc_evict: libc load failed: %s", e)
            _libc = None
    return _libc


_counter_lock = threading.Lock()
_ubc_evicted_bytes_total: int = 0
_ubc_evict_calls_total: int = 0
_ubc_evict_failed_total: int = 0


def _bump_counter(evicted: int, *, failed: bool) -> None:
    global _ubc_evicted_bytes_total, _ubc_evict_calls_total, _ubc_evict_failed_total
    with _counter_lock:
        _ubc_evict_calls_total += 1
        if failed:
            _ubc_evict_failed_total += 1
        elif evicted > 0:
            _ubc_evicted_bytes_total += int(evicted)


def ubc_evict(path: str) -> int:
    if sys.platform != "darwin":
        logger.debug("ubc_evict no-op on %s", sys.platform)
        _bump_counter(0, failed=False)
        return 0

    libc = _get_libc()
    if libc is None:
        logger.debug("ubc_evict: libc unavailable, no-op")
        _bump_counter(0, failed=True)
        return 0

    try:
        size = os.path.getsize(path)
    except (FileNotFoundError, NotADirectoryError, PermissionError) as e:
        logger.warning("ubc_evict: cannot stat %s: %s", path, e)
        _bump_counter(0, failed=True)
        return 0
    except OSError as e:
        logger.warning("ubc_evict: stat %s failed: %s", path, e)
        _bump_counter(0, failed=True)
        return 0
    if size <= 0:
        logger.debug("ubc_evict: %s is empty, no-op", path)
        _bump_counter(0, failed=False)
        return 0

    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError as e:
        logger.warning("ubc_evict: open %s failed: %s", path, e)
        _bump_counter(0, failed=True)
        return 0

    try:
        ctypes.set_errno(0)
        addr = libc.mmap(None, size, _PROT_READ, _MAP_SHARED, fd, 0)
        if addr is None or addr == 0 or addr == _MMAP_FAILED:
            err = ctypes.get_errno()
            logger.warning(
                "ubc_evict: mmap %s failed errno=%d (%s)",
                path,
                err,
                os.strerror(err) if err else "unknown",
            )
            _bump_counter(0, failed=True)
            return 0
        msync_ok = False
        munmap_ok = False
        try:
            ctypes.set_errno(0)
            rc = libc.msync(addr, size, _MS_INVALIDATE)
            if rc != 0:
                err = ctypes.get_errno()
                logger.warning(
                    "ubc_evict: msync(MS_INVALIDATE) %s rc=%d errno=%d (%s)",
                    path,
                    rc,
                    err,
                    os.strerror(err) if err else "unknown",
                )
            else:
                msync_ok = True
        finally:
            ctypes.set_errno(0)
            munmap_rc = libc.munmap(addr, size)
            if munmap_rc != 0:
                err = ctypes.get_errno()
                logger.warning(
                    "ubc_evict: munmap %s rc=%d errno=%d (%s)",
                    path,
                    munmap_rc,
                    err,
                    os.strerror(err) if err else "unknown",
                )
            else:
                munmap_ok = True
    finally:
        try:
            os.close(fd)
        except OSError:
            pass

    if not (msync_ok and munmap_ok):
        _bump_counter(0, failed=True)
        return 0
    _bump_counter(size, failed=False)
    return size


def ubc_evict_paths(paths: Iterable[str]) -> int:
    if sys.platform != "darwin":
        logger.debug("ubc_evict_paths no-op on %s", sys.platform)
        return 0
    total = 0
    t0 = time.monotonic()
    for p in paths:
        bytes_evicted = ubc_evict(str(p))
        if bytes_evicted > 0:
            logger.info(
                "ubc_evict: evicted %.1f MB from UBC for %s",
                bytes_evicted / (1024 * 1024),
                p,
            )
        total += bytes_evicted
    if total > 0:
        logger.info(
            "ubc_evict: pass complete total_mb=%.1f elapsed_s=%.2f",
            total / (1024 * 1024),
            time.monotonic() - t0,
        )
    return total


def snapshot() -> dict[str, int]:
    with _counter_lock:
        return {
            "ubc_evicted_bytes_total": _ubc_evicted_bytes_total,
            "ubc_evict_calls_total": _ubc_evict_calls_total,
            "ubc_evict_failed_total": _ubc_evict_failed_total,
        }


_UBC_EVICTED_HELP = (
    "Cumulative bytes that the macOS UBC eviction helper has asked "
    "the kernel to discard via msync(MS_INVALIDATE). Non-zero only "
    "on Darwin."
)


def render_prometheus_lines() -> list[str]:
    stats = snapshot()
    evicted = int(stats.get("ubc_evicted_bytes_total", 0))
    return [
        f"# HELP fusion_mlx_ubc_evicted_bytes_total {_UBC_EVICTED_HELP}",
        "# TYPE fusion_mlx_ubc_evicted_bytes_total counter",
        f'fusion_mlx_ubc_evicted_bytes_total{{path_kind="safetensors"}} {evicted}',
    ]


def reset_for_tests() -> None:
    global _ubc_evicted_bytes_total, _ubc_evict_calls_total, _ubc_evict_failed_total
    with _counter_lock:
        _ubc_evicted_bytes_total = 0
        _ubc_evict_calls_total = 0
        _ubc_evict_failed_total = 0


__all__ = [
    "ubc_evict",
    "ubc_evict_paths",
    "snapshot",
    "render_prometheus_lines",
    "reset_for_tests",
]
