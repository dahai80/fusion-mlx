# SPDX-License-Identifier: Apache-2.0
"""In-memory per-session token usage tracker (issue #226).

Thread-safe singleton. LRU-bounded to prevent unbounded growth.
No persistence: stats live for the lifetime of the server process.

Sessions are scoped by ``(principal, session_id)`` so callers cannot read or
mutate another principal's session (IDOR fix). The public ``/v1`` API currently
authenticates a single shared key, so there is one principal in practice;
scoping is still applied for forward-safety and to cover no-key dev mode
where callers are distinguished by subnet bucket id.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

_MAX_SESSIONS = 4096
_STATE_ENV = "FUSION_SESSION_STATE_DIR"
_STATE_FILENAME = "sessions.json"
_SNAPSHOT_INTERVAL_S = 5.0

# Principal used when a caller does not supply one. Keeps the tracker usable
# by callers that pre-date the principal-scoping change (e.g. direct helper
# calls without an HTTP request in scope).
DEFAULT_PRINCIPAL = "default"


@dataclass
class SessionStats:
    principal: str
    session_id: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    total_tokens: int = 0
    request_count: int = 0
    max_context_tokens: int | None = None
    last_active: float = field(default_factory=time.monotonic)

    def to_dict(self) -> dict:
        return {
            "principal": self.principal,
            "session_id": self.session_id,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cached_tokens": self.cached_tokens,
            "total_tokens": self.total_tokens,
            "request_count": self.request_count,
            "max_context_tokens": self.max_context_tokens,
            "last_active": self.last_active,
        }


class SessionTracker:
    """Thread-safe per-session token usage aggregator with LRU eviction.

    #754: optional JSON snapshot persistence. When a state dir is configured
    (FUSION_SESSION_STATE_DIR), session stats are periodically snapshotted
    to disk and rehydrated on startup so a failover/restart does not lose
    cumulative token-usage context. Persistence is off by default (single-
    process dev keeps the original in-memory-only behavior).
    """

    def __init__(
        self,
        max_sessions: int = _MAX_SESSIONS,
        state_dir: str | None = None,
    ) -> None:
        self._max_sessions = max_sessions
        self._lock = threading.Lock()
        self._sessions: OrderedDict[tuple[str, str], SessionStats] = OrderedDict()
        self._state_dir = state_dir
        self._last_snapshot = 0.0
        if state_dir:
            os.makedirs(state_dir, exist_ok=True)
            self._rehydrate()

    @staticmethod
    def _key(principal: str | None, session_id: str) -> tuple[str, str]:
        return (principal or DEFAULT_PRINCIPAL, session_id)

    def record(
        self,
        session_id: str,
        *,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cached_tokens: int = 0,
        principal: str | None = None,
    ) -> SessionStats:
        if not session_id:
            return SessionStats(principal=principal or DEFAULT_PRINCIPAL, session_id="")
        pt = max(0, int(prompt_tokens or 0))
        ct = max(0, int(completion_tokens or 0))
        cct = max(0, int(cached_tokens or 0))
        key = self._key(principal, session_id)
        with self._lock:
            stats = self._sessions.get(key)
            if stats is None:
                stats = SessionStats(principal=key[0], session_id=session_id)
                self._sessions[key] = stats
                self._evict_locked()
            stats.prompt_tokens += pt
            stats.completion_tokens += ct
            stats.cached_tokens += cct
            stats.total_tokens = stats.prompt_tokens + stats.completion_tokens
            stats.request_count += 1
            stats.last_active = time.monotonic()
            self._sessions.move_to_end(key)
            result = SessionStats(**stats.__dict__)
        self._maybe_snapshot()
        return result

    def get(
        self, session_id: str, *, principal: str | None = None
    ) -> SessionStats | None:
        key = self._key(principal, session_id)
        with self._lock:
            stats = self._sessions.get(key)
            if stats is None:
                return None
            return SessionStats(**stats.__dict__)

    def set_max_context(
        self,
        session_id: str,
        max_context_tokens: int | None,
        *,
        principal: str | None = None,
    ) -> bool:
        if not session_id:
            return False
        if max_context_tokens is not None and max_context_tokens < 0:
            return False
        key = self._key(principal, session_id)
        with self._lock:
            stats = self._sessions.get(key)
            if stats is None:
                stats = SessionStats(principal=key[0], session_id=session_id)
                self._sessions[key] = stats
                self._evict_locked()
            stats.max_context_tokens = (
                int(max_context_tokens) if max_context_tokens is not None else None
            )
            self._sessions.move_to_end(key)
        self._maybe_snapshot()
        return True

    def _rehydrate(self) -> None:
        path = os.path.join(self._state_dir, _STATE_FILENAME)
        if not os.path.exists(path):
            return
        try:
            with open(path, encoding="utf-8") as fh:
                rows = json.load(fh)
            loaded = 0
            for row in rows:
                key = (
                    row.get("principal", DEFAULT_PRINCIPAL),
                    row.get("session_id", ""),
                )
                if not key[1]:
                    continue
                stats = SessionStats(
                    principal=key[0],
                    session_id=key[1],
                    prompt_tokens=int(row.get("prompt_tokens", 0)),
                    completion_tokens=int(row.get("completion_tokens", 0)),
                    cached_tokens=int(row.get("cached_tokens", 0)),
                    total_tokens=int(row.get("total_tokens", 0)),
                    request_count=int(row.get("request_count", 0)),
                    max_context_tokens=row.get("max_context_tokens"),
                    last_active=float(row.get("last_active", time.monotonic())),
                )
                self._sessions[key] = stats
                loaded += 1
            logger.info(
                "session tracker rehydrated %d session(s) from %s", loaded, path
            )
        except Exception:
            logger.warning(
                "session tracker rehydrate failed for %s", path, exc_info=True
            )

    def _maybe_snapshot(self) -> None:
        if not self._state_dir:
            return
        now = time.monotonic()
        if now - self._last_snapshot < _SNAPSHOT_INTERVAL_S:
            return
        self._last_snapshot = now
        path = os.path.join(self._state_dir, _STATE_FILENAME)
        tmp = path + ".tmp"
        try:
            with self._lock:
                rows = [s.to_dict() for s in self._sessions.values()]
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(rows, fh)
            os.replace(tmp, path)
            logger.debug("session tracker snapshot wrote %d session(s)", len(rows))
        except Exception:
            logger.warning("session tracker snapshot failed", exc_info=True)

    def list_sessions(self) -> list[SessionStats]:
        with self._lock:
            return [SessionStats(**s.__dict__) for s in self._sessions.values()]

    def reset(self) -> None:
        with self._lock:
            self._sessions.clear()

    def _evict_locked(self) -> None:
        evicted = 0
        while len(self._sessions) > self._max_sessions:
            _, _ = self._sessions.popitem(last=False)
            evicted += 1
        if evicted:
            logger.warning("session tracker evicted %d oldest session(s)", evicted)


_tracker: SessionTracker | None = None
_tracker_lock = threading.Lock()


def get_session_tracker() -> SessionTracker:
    global _tracker
    if _tracker is None:
        with _tracker_lock:
            if _tracker is None:
                state_dir = os.environ.get(_STATE_ENV, "").strip() or None
                _tracker = SessionTracker(state_dir=state_dir)
    return _tracker


def record_chat_session(
    session_id: str | None,
    *,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    cached_tokens: int = 0,
    principal: str | None = None,
) -> None:
    if not session_id:
        return
    get_session_tracker().record(
        session_id,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cached_tokens=cached_tokens,
        principal=principal,
    )


def set_session_max_context(
    session_id: str,
    max_context_tokens: int | None,
    *,
    principal: str | None = None,
) -> bool:
    return get_session_tracker().set_max_context(
        session_id, max_context_tokens, principal=principal
    )


def reset_session_tracker_for_tests() -> None:
    get_session_tracker().reset()
