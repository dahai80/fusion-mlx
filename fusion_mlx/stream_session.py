# SPDX-License-Identifier: Apache-2.0
"""Resumable streaming session store (issue #801).

A ``StreamSession`` survives across HTTP connections so a client that drops
mid-generation (crash, network drop, proxy timeout) can reconnect via
``GET /v1/streams/lookup`` and receive the full generated text. A background
producer task drives ``engine.stream_chat`` and appends formatted SSE chunks
into the session buffer; each HTTP connection is a consumer that replays the
buffer from index 0 then live-tails the producer until the session completes.

The store is process-local (single-node). Sessions expire after ``_TTL`` and
are reaped lazily on access.
"""

import asyncio
import logging
import time

logger = logging.getLogger(__name__)

_TTL = 3600.0


class StreamSession:
    def __init__(self, session_id: str, principal: str | None = None) -> None:
        self.session_id = session_id
        # #801 IDOR: the principal that started this stream. Lookup by any
        # other caller is rejected (403) so one valid API-key holder cannot
        # read another's generated text by guessing the session id.
        self.principal = principal
        self.events: list[str] = []
        self.complete = False
        self.finish_reason: str | None = None
        self.error: str | None = None
        self.created_at = time.monotonic()
        self._new_data = asyncio.Event()
        self._seq = 0
        # Strong ref to the background producer task so it is not GC'd, and
        # so a consumer can detect a dead producer (avoids an eternal hang
        # if the producer ends without setting complete).
        self.producer_task: asyncio.Task | None = None

    def append(self, chunk: str) -> None:
        self.events.append(chunk)
        self._seq += 1
        self._new_data.set()

    def mark_complete(self, finish_reason: str | None = None) -> None:
        self.complete = True
        self.finish_reason = finish_reason
        self._new_data.set()

    def mark_error(self, err: str) -> None:
        self.error = err
        self.complete = True
        self._new_data.set()

    def expired(self) -> bool:
        return (time.monotonic() - self.created_at) > _TTL

    def producer_finished(self) -> bool:
        # True only when a tracked producer task exists AND is done. Lets a
        # consumer break out of _new_data.wait() instead of hanging forever
        # if the producer died without marking complete/error. Returns False
        # when no task is tracked (untracked/legacy session) so live-tail
        # behavior is preserved.
        return self.producer_task is not None and self.producer_task.done()


class StreamSessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, StreamSession] = {}

    def get(self, session_id: str) -> StreamSession | None:
        sess = self._sessions.get(session_id)
        if sess is None:
            return None
        if sess.expired():
            self._sessions.pop(session_id, None)
            logger.info("stream_session expired: %s", session_id)
            return None
        return sess

    def create(self, session_id: str, principal: str | None = None) -> StreamSession:
        sess = StreamSession(session_id, principal=principal)
        self._sessions[session_id] = sess
        logger.info("stream_session created: %s principal=%s", session_id, principal)
        return sess

    def drop(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def active(self) -> list[str]:
        return [sid for sid, s in self._sessions.items() if not s.expired()]

    def count_for_principal(self, principal: str | None) -> int:
        # #801 DoS guard: cap concurrent live sessions per caller so one
        # principal cannot spawn unbounded background producers.
        return sum(
            1
            for s in self._sessions.values()
            if not s.expired() and s.principal == principal
        )


_store: StreamSessionStore | None = None


def get_store() -> StreamSessionStore:
    global _store
    if _store is None:
        _store = StreamSessionStore()
    return _store


def reset_store_for_tests() -> StreamSessionStore:
    global _store
    _store = StreamSessionStore()
    return _store
