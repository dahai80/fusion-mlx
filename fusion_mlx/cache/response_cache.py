# SPDX-License-Identifier: Apache-2.0
"""Response-level cache for deterministic LLM outputs.

Caches complete ChatCompletion responses keyed by request fingerprint.
Only caches temperature=0 (deterministic) requests by default;
higher temperatures opt-in via X-Cache: force header.

Usage from routes:
    from ..cache.response_cache import get_response_cache, CachePolicy

    cache = get_response_cache()
    key = cache.fingerprint(request)
    policy = cache.resolve_policy(request.temperature, http_headers)
    if policy != CachePolicy.BYPASS:
        hit = cache.get(key)
        if hit and policy != CachePolicy.WRITE_ONLY:
            return hit  # cache HIT

    result = await engine.chat(...)

    if policy != CachePolicy.NO_STORE:
        cache.put(key, result)
"""

import hashlib
import json
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class CachePolicy(Enum):
    FORCE = "force"
    ONLY_IF_CACHED = "only-if-cached"
    BYPASS = "bypass"
    WRITE_ONLY = "write-only"
    NO_STORE = "no-store"


@dataclass
class ResponseCacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size_bytes: int = 0
    entry_count: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "hit_rate": round(self.hit_rate, 4),
            "size_bytes": self.size_bytes,
            "entry_count": self.entry_count,
        }


@dataclass
class _CacheEntry:
    key: str
    value: Any
    size_bytes: int
    created_at: float
    last_access: float
    ttl: float


class ResponseCache:
    """LRU response cache with TTL and size limits."""

    def __init__(
        self,
        max_entries: int = 256,
        max_entry_bytes: int = 65536,
        default_ttl: float = 3600.0,
        max_total_bytes: int = 64 * 1024 * 1024,
    ):
        self._max_entries = max_entries
        self._max_entry_bytes = max_entry_bytes
        self._default_ttl = default_ttl
        self._max_total_bytes = max_total_bytes
        self._store: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._lock = threading.Lock()
        self._stats = ResponseCacheStats()

    @property
    def stats(self) -> ResponseCacheStats:
        return self._stats

    def fingerprint(
        self,
        model: str,
        messages: list[dict],
        *,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
        tools: list[dict] | None = None,
        response_format: dict | None = None,
        seed: int | None = None,
        grammar: str | None = None,
        adapters: str | None = None,
        **extra: Any,
    ) -> str:
        parts = [
            model or "",
            adapters or "",
            json.dumps(messages, sort_keys=True, separators=(",", ":")),
            str(temperature or 0.0),
            str(top_p or 1.0),
            str(max_tokens or -1),
            json.dumps(stop or [], sort_keys=True, separators=(",", ":")),
            json.dumps(tools or [], sort_keys=True, separators=(",", ":")),
            json.dumps(response_format or {}, sort_keys=True, separators=(",", ":")),
            str(seed or 0),
            str(grammar or ""),
        ]
        blob = "|".join(parts)
        return hashlib.sha256(blob.encode()).hexdigest()

    def resolve_policy(
        self,
        temperature: float | None,
        headers: dict[str, str] | None = None,
    ) -> CachePolicy:
        headers = headers or {}
        cache_header = (
            headers.get("x-cache", headers.get("X-Cache", "")).lower().strip()
        )
        if cache_header == "bypass":
            return CachePolicy.BYPASS
        if cache_header == "force":
            return CachePolicy.FORCE
        if cache_header == "only-if-cached":
            return CachePolicy.ONLY_IF_CACHED
        if cache_header == "no-store":
            return CachePolicy.NO_STORE
        temp = temperature if temperature is not None else 0.7
        if temp == 0.0:
            return CachePolicy.FORCE
        return CachePolicy.BYPASS

    def get(self, key: str) -> Any | None:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._stats.misses += 1
                return None
            now = time.monotonic()
            if now - entry.created_at > entry.ttl:
                self._evict(key)
                self._stats.misses += 1
                logger.debug("Response cache STALE key=%s", key[:12])
                return None
            self._store.move_to_end(key)
            entry.last_access = now
            self._stats.hits += 1
            logger.debug(
                "Response cache HIT key=%s age=%.1fs",
                key[:12],
                now - entry.created_at,
            )
            return entry.value

    def put(
        self,
        key: str,
        value: Any,
        ttl: float | None = None,
    ) -> bool:
        try:
            raw = json.dumps(value, separators=(",", ":"))
        except (TypeError, ValueError):
            logger.debug("Response cache SKIP key=%s (not JSON-serializable)", key[:12])
            return False
        size = len(raw.encode("utf-8"))
        if size > self._max_entry_bytes:
            logger.debug(
                "Response cache SKIP key=%s size=%d > max=%d",
                key[:12],
                size,
                self._max_entry_bytes,
            )
            return False
        with self._lock:
            now = time.monotonic()
            if key in self._store:
                old = self._store[key]
                self._stats.size_bytes -= old.size_bytes
                del self._store[key]
            while (
                len(self._store) >= self._max_entries
                or (self._stats.size_bytes + size) > self._max_total_bytes
            ):
                if not self._store:
                    break
                evict_key, evict_entry = self._store.popitem(last=False)
                self._stats.size_bytes -= evict_entry.size_bytes
                self._stats.evictions += 1
            entry = _CacheEntry(
                key=key,
                value=value,
                size_bytes=size,
                created_at=now,
                last_access=now,
                ttl=ttl or self._default_ttl,
            )
            self._store[key] = entry
            self._stats.size_bytes += size
            self._stats.entry_count = len(self._store)
            logger.debug("Response cache PUT key=%s size=%d", key[:12], size)
            return True

    def _evict(self, key: str) -> None:
        entry = self._store.pop(key, None)
        if entry:
            self._stats.size_bytes -= entry.size_bytes
            self._stats.evictions += 1

    def invalidate(self, key: str) -> bool:
        with self._lock:
            entry = self._store.pop(key, None)
            if entry:
                self._stats.size_bytes -= entry.size_bytes
                self._stats.entry_count = len(self._store)
                return True
            return False

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self._stats.size_bytes = 0
            self._stats.entry_count = 0

    def get_by_response_id(self, response_id: str) -> Any | None:
        with self._lock:
            for entry in self._store.values():
                resp = entry.value
                if isinstance(resp, dict) and resp.get("id") == response_id:
                    entry.last_access = time.monotonic()
                    self._store.move_to_end(entry.key)
                    return resp
        return None


_global_cache: ResponseCache | None = None
_cache_lock = threading.Lock()


def get_response_cache() -> ResponseCache:
    global _global_cache
    if _global_cache is None:
        with _cache_lock:
            if _global_cache is None:
                _global_cache = ResponseCache()
                logger.info(
                    "Response cache initialized max_entries=%d ttl=%.0fs",
                    _global_cache._max_entries,
                    _global_cache._default_ttl,
                )
    return _global_cache


def reset_response_cache() -> None:
    global _global_cache
    with _cache_lock:
        if _global_cache is not None:
            _global_cache.clear()
        _global_cache = None
