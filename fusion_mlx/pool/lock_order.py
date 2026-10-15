# SPDX-License-Identifier: Apache-2.0
"""Debug-only lock-order checker (issue #1190, deep-bug H2 deadlock prevention).

Declares the global lock acquisition order for fusion_mlx and, when
FUSION_MLX_LOCK_DEBUG is enabled, verifies at runtime that nested lock
acquisitions follow it. Disabled by default: every check is one boolean
test and the wrapper classes delegate straight through.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import traceback

logger = logging.getLogger(__name__)

# Global order, outermost first. A thread/task must never acquire a lock
# that appears EARLIER in this list while already holding one that appears
# LATER (that nesting direction is what deadlocks).
_LOCK_ORDER = [
    "server.app_init",
    "enforcer.state",
    "pool.main",
    "pool.discover_sync",
    "pool.pinned",
    "pool.adapter_swap",
]

_ORDER_INDEX = {name: i for i, name in enumerate(_LOCK_ORDER)}


class LockOrderViolation(RuntimeError):
    pass


def _lock_debug_enabled() -> bool:
    return os.getenv("FUSION_MLX_LOCK_DEBUG", "").strip().lower() in (
        "1",
        "true",
        "on",
    )


def _lock_debug_strict() -> bool:
    return os.getenv("FUSION_MLX_LOCK_DEBUG_STRICT", "").strip() == "1"


def _task_key() -> str:
    task = None
    try:
        task = asyncio.current_task()
    except RuntimeError:
        task = None
    if task is not None:
        return f"{threading.get_ident()}:{id(task)}"
    return f"{threading.get_ident()}:sync"


class LockOrderTracker:
    _instance = None
    _singleton_guard = threading.Lock()

    def __new__(cls) -> LockOrderTracker:
        if cls._instance is None:
            with cls._singleton_guard:
                if cls._instance is None:
                    inst = super().__new__(cls)
                    inst._stacks: dict[str, list[str]] = {}
                    inst._stacks_guard = threading.Lock()
                    cls._instance = inst
        return cls._instance

    @staticmethod
    def enabled() -> bool:
        return _lock_debug_enabled()

    def register(self, name: str) -> None:
        if not self.enabled():
            return
        logger.debug("lock_order: registered lock '%s'", name)

    def check_acquire(self, name: str) -> None:
        if not self.enabled():
            return
        idx = _ORDER_INDEX.get(name)
        if idx is None:
            return
        key = _task_key()
        with self._stacks_guard:
            held = list(self._stacks.get(key, ()))
        later = [h for h in held if h in _ORDER_INDEX and _ORDER_INDEX[h] > idx]
        if not later:
            return
        stack = "".join(traceback.format_stack()[:8])
        logger.error(
            "LOCK ORDER VIOLATION: acquiring '%s' while already holding %s "
            "(declared order outermost-first: %s); owner=%s; held from:\n%s",
            name,
            later,
            _LOCK_ORDER,
            key,
            stack,
        )
        if _lock_debug_strict():
            raise LockOrderViolation(
                f"acquiring '{name}' while holding {later} violates "
                f"declared order {_LOCK_ORDER}"
            )

    def mark_acquired(self, name: str) -> None:
        if not self.enabled():
            return
        if name not in _ORDER_INDEX:
            return
        key = _task_key()
        with self._stacks_guard:
            self._stacks.setdefault(key, []).append(name)

    def check_release(self, name: str) -> None:
        if not self.enabled():
            return
        key = _task_key()
        unbalanced = False
        with self._stacks_guard:
            stack = self._stacks.get(key)
            if stack and name in stack:
                stack.reverse()
                stack.remove(name)
                stack.reverse()
                if not stack:
                    self._stacks.pop(key, None)
            else:
                unbalanced = True
        if unbalanced:
            logger.warning(
                "lock_order: release without matching acquire for '%s' "
                "(owner=%s); possible unbalanced acquire/release",
                name,
                key,
            )

    def snapshot_held(self) -> dict[str, list[str]]:
        with self._stacks_guard:
            return {k: list(v) for k, v in self._stacks.items()}

    def reset(self) -> None:
        with self._stacks_guard:
            self._stacks.clear()


_tracker = LockOrderTracker()


def reset_lock_order_tracker() -> None:
    _tracker.reset()


def snapshot_held() -> dict[str, list[str]]:
    return _tracker.snapshot_held()


class TrackedAsyncLock(asyncio.Lock):
    def __init__(self, name: str):
        super().__init__()
        self._lock_order_name = name
        _tracker.register(name)

    async def acquire(self) -> bool:
        _tracker.check_acquire(self._lock_order_name)
        acquired = await super().acquire()
        if acquired:
            _tracker.mark_acquired(self._lock_order_name)
        return acquired

    def release(self) -> None:
        _tracker.check_release(self._lock_order_name)
        super().release()


class TrackedThreadLock:
    def __init__(self, name: str):
        self._lock_order_name = name
        self._lock = threading.Lock()
        _tracker.register(name)

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        _tracker.check_acquire(self._lock_order_name)
        got = self._lock.acquire(blocking, timeout)
        if got:
            _tracker.mark_acquired(self._lock_order_name)
        return got

    def release(self) -> None:
        _tracker.check_release(self._lock_order_name)
        self._lock.release()

    def locked(self) -> bool:
        return self._lock.locked()

    def __enter__(self) -> TrackedThreadLock:
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.release()
        return False
