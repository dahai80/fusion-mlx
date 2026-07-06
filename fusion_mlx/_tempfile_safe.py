# SPDX-License-Identifier: Apache-2.0
import atexit
import contextlib
import logging
import os
import tempfile
import threading
from collections.abc import Iterator

logger = logging.getLogger(__name__)

_pending_paths: set[str] = set()
_pending_lock = threading.Lock()
_atexit_registered = False


def _atexit_reap_all() -> None:
    with _pending_lock:
        snapshot = list(_pending_paths)
        _pending_paths.clear()
    for path in snapshot:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
        except OSError:
            pass


def _ensure_atexit_registered() -> None:
    global _atexit_registered
    if _atexit_registered:
        return
    with _pending_lock:
        if _atexit_registered:
            return
        atexit.register(_atexit_reap_all)
        _atexit_registered = True


class _TempfileHandle:
    __slots__ = ("_path", "_released")

    def __init__(self, path: str) -> None:
        self._path = path
        self._released = False

    @property
    def path(self) -> str:
        return self._path

    @property
    def released(self) -> bool:
        return self._released

    def release(self) -> str:
        with _pending_lock:
            if self._released:
                return self._path
            _pending_paths.discard(self._path)
            self._released = True
        return self._path

    def __fspath__(self) -> str:
        return self._path

    def __str__(self) -> str:
        return self._path

    def __repr__(self) -> str:
        state = "released" if self._released else "pending"
        return f"_TempfileHandle({self._path!r}, {state})"


@contextlib.contextmanager
def managed_tempfile_path(
    *,
    prefix: str = "tmp",
    suffix: str = "",
    dir: str | None = None,
) -> Iterator[_TempfileHandle]:
    fd, path = tempfile.mkstemp(prefix=prefix, suffix=suffix, dir=dir)
    try:
        try:
            os.close(fd)
        except OSError:
            pass

        _ensure_atexit_registered()
        with _pending_lock:
            _pending_paths.add(path)
    except BaseException:
        try:
            _ensure_atexit_registered()
        except BaseException:
            pass
        with _pending_lock:
            _pending_paths.add(path)
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            os.unlink(path)
        except OSError:
            pass
        else:
            with _pending_lock:
                _pending_paths.discard(path)
        raise

    handle = _TempfileHandle(path)
    try:
        yield handle
    finally:
        with _pending_lock:
            prev_released = handle._released
            if not prev_released:
                handle._released = True
        should_unlink = not prev_released
        if should_unlink:
            unlinked = False
            try:
                os.unlink(path)
                unlinked = True
            except FileNotFoundError:
                unlinked = True
            except OSError:
                pass
            if unlinked:
                with _pending_lock:
                    _pending_paths.discard(path)


def safe_tempdir(*args, **kwargs):
    return tempfile.TemporaryDirectory(*args, **kwargs)


def _pending_snapshot() -> set[str]:
    with _pending_lock:
        return set(_pending_paths)


__all__ = ["managed_tempfile_path", "safe_tempdir"]
