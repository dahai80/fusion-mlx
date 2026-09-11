# SPDX-License-Identifier: Apache-2.0
"""D2.8/G13: multi-cache-layer atomic rollback semantics.

Composite generation writes fan out across several cache layers
(paged hot, paged_ssd cold, radix prefix, response). A failure mid-write
leaves partial blocks in some layers — a torn cache. Subsequent lookups
hit stale/partial data, or worse, a half-demoted block surfaces in hot
after cold already saved a different revision.

This module provides transactional rollback: snapshot the affected keys
before the risky write, and on failure evict every snapshot key from
every registered leaf cache so all layers agree on "this key does not
exist". On success the snapshot is discarded (the writes are intended).

Design:
  - Leaf caches register an ``evict_fn(key) -> bool``. paged/paged_ssd/
    radix expose ``evict``; response_cache exposes ``invalidate`` — the
    handle normalizes both to one contract.
  - ``_TRIM_LOCK`` (RLock) serializes rollback against concurrent
    trim/evict/demotion so two threads cannot race a block out from
    under each other and double-count or miss a leaf.
  - ``rollback_token()`` returns a context manager. On ``__exit__`` with
    an exception, it rolls back; on clean exit it commits (no-op — the
    writes stay). This is the same semantics as a DB transaction.

I/O errors during rollback are logged but NEVER re-raised — rollback is
a best-effort consistency repair, not a path that should mask the
original exception. The original exception propagates regardless.
"""

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CacheLeafHandle:
    name: str
    evict_fn: Callable[[Any], bool]

    def evict(self, key: Any) -> bool:
        try:
            return bool(self.evict_fn(key))
        except Exception as e:
            logger.warning(
                "cache-rollback: leaf '%s' evict failed for key=%r: %s",
                self.name,
                key,
                e,
            )
            return False


@dataclass
class RollbackToken:
    keys: list[Any] = field(default_factory=list)
    leaves: list[CacheLeafHandle] = field(default_factory=list)
    ts: float = 0.0
    rolled_back: bool = False


class CacheRollbackManager:
    def __init__(self) -> None:
        self._leaves: list[CacheLeafHandle] = []
        self._lock = threading.RLock()

    def register_leaf(self, name: str, evict_fn: Callable[[Any], bool]) -> None:
        with self._lock:
            for h in self._leaves:
                if h.name == name:
                    h.evict_fn = evict_fn
                    logger.debug("cache-rollback: re-registered leaf '%s'", name)
                    return
            self._leaves.append(CacheLeafHandle(name=name, evict_fn=evict_fn))
            logger.info(
                "cache-rollback: registered leaf '%s' (%d total)",
                name,
                len(self._leaves),
            )

    def leaves(self) -> list[CacheLeafHandle]:
        with self._lock:
            return list(self._leaves)

    def snapshot(self, keys: list[Any]) -> RollbackToken:
        token = RollbackToken(
            keys=list(keys),
            leaves=self.leaves(),
            ts=time.monotonic(),
        )
        logger.debug(
            "cache-rollback: snapshot %d keys across %d leaves",
            len(token.keys),
            len(token.leaves),
        )
        return token

    def rollback(self, token: RollbackToken) -> int:
        if token.rolled_back:
            logger.debug("cache-rollback: token already rolled back, skipping")
            return 0
        total = 0
        with self._lock:
            for key in token.keys:
                for leaf in token.leaves:
                    if leaf.evict(key):
                        total += 1
                        logger.debug(
                            "cache-rollback: evicted key=%r from leaf '%s'",
                            key,
                            leaf.name,
                        )
            token.rolled_back = True
        logger.info(
            "cache-rollback: rolled back %d keys x %d leaves (%d evictions)",
            len(token.keys),
            len(token.leaves),
            total,
        )
        return total

    def rollback_keys(self, keys: list[Any]) -> int:
        return self.rollback(self.snapshot(keys))

    def rollback_context(self, keys: list[Any]):
        token = self.snapshot(keys)
        return _RollbackContext(self, token)


class _RollbackContext:
    def __init__(self, manager: CacheRollbackManager, token: RollbackToken) -> None:
        self._manager = manager
        self._token = token

    def __enter__(self) -> RollbackToken:
        return self._token

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if exc_type is not None:
            try:
                evicted = self._manager.rollback(self._token)
                logger.warning(
                    "cache-rollback: rolled back after %s (%d evictions)",
                    exc_type.__name__ if exc_type else "exception",
                    evicted,
                )
            except Exception as rollback_err:
                logger.error(
                    "cache-rollback: rollback itself failed (original exc preserved): %s",
                    rollback_err,
                )
        return False
