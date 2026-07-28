# SPDX-License-Identifier: Apache-2.0
"""In-memory per-session token usage tracker (issue #226).

Thread-safe singleton. LRU-bounded to prevent unbounded growth.
No persistence: stats live for the lifetime of the server process.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

_MAX_SESSIONS = 4096


@dataclass
class SessionStats:
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
    """Thread-safe per-session token usage aggregator with LRU eviction."""

    def __init__(self, max_sessions: int = _MAX_SESSIONS) -> None:
        self._max_sessions = max_sessions
        self._lock = threading.Lock()
        self._sessions: OrderedDict[str, SessionStats] = OrderedDict()

    def record(
        self,
        session_id: str,
        *,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cached_tokens: int = 0,
    ) -> SessionStats:
        if not session_id:
            return SessionStats(session_id="")
        pt = max(0, int(prompt_tokens or 0))
        ct = max(0, int(completion_tokens or 0))
        cct = max(0, int(cached_tokens or 0))
        with self._lock:
            stats = self._sessions.get(session_id)
            if stats is None:
                stats = SessionStats(session_id=session_id)
                self._sessions[session_id] = stats
                self._evict_locked()
            stats.prompt_tokens += pt
            stats.completion_tokens += ct
            stats.cached_tokens += cct
            stats.total_tokens = stats.prompt_tokens + stats.completion_tokens
            stats.request_count += 1
            stats.last_active = time.monotonic()
            self._sessions.move_to_end(session_id)
            return stats

    def get(self, session_id: str) -> SessionStats | None:
        with self._lock:
            stats = self._sessions.get(session_id)
            if stats is None:
                return None
            return SessionStats(**stats.__dict__)

    def set_max_context(self, session_id: str, max_context_tokens: int | None) -> bool:
        if not session_id:
            return False
        if max_context_tokens is not None and max_context_tokens < 0:
            return False
        with self._lock:
            stats = self._sessions.get(session_id)
            if stats is None:
                stats = SessionStats(session_id=session_id)
                self._sessions[session_id] = stats
                self._evict_locked()
            stats.max_context_tokens = (
                int(max_context_tokens) if max_context_tokens is not None else None
            )
            self._sessions.move_to_end(session_id)
            return True

    def list_sessions(self) -> list[SessionStats]:
        with self._lock:
            return [SessionStats(**s.__dict__) for s in self._sessions.values()]

    def reset(self) -> None:
        with self._lock:
            self._sessions.clear()

    def _evict_locked(self) -> None:
        evicted = 0
        while len(self._sessions) > self._max_sessions:
            sid, _ = self._sessions.popitem(last=False)
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
                _tracker = SessionTracker()
    return _tracker


def record_chat_session(
    session_id: str | None,
    *,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    cached_tokens: int = 0,
) -> None:
    if not session_id:
        return
    get_session_tracker().record(
        session_id,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cached_tokens=cached_tokens,
    )


def set_session_max_context(session_id: str, max_context_tokens: int | None) -> bool:
    return get_session_tracker().set_max_context(session_id, max_context_tokens)


def reset_session_tracker_for_tests() -> None:
    get_session_tracker().reset()
