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

# Audit M1: the buffer used to be unbounded, so one long-lived stream could
# accumulate tens of MB of formatted SSE per session for the full TTL.
# Overflow fails the session LOUDLY (mark_error + producer cancel) rather
# than silently dropping the oldest chunks, which would hand a reconnecting
# client a truncated answer it cannot detect.
_MAX_EVENTS = 200_000
_MAX_BUFFER_BYTES = 64 * 1024 * 1024


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
        # Approximate size of the buffered SSE body in bytes (len() of a str
        # is a close-enough proxy; the cap only needs to be in the right
        # order of magnitude).
        self._buffered_bytes = 0
        # Strong ref to the background producer task so it is not GC'd, and
        # so a consumer can detect a dead producer (avoids an eternal hang
        # if the producer ends without setting complete).
        self.producer_task: asyncio.Task | None = None

    def append(self, chunk: str) -> None:
        if self.complete:
            # Terminal already (normal finish, error, or overflow): further
            # chunks cannot be replayed meaningfully and must not grow the
            # buffer past its cap.
            return
        self.events.append(chunk)
        self._buffered_bytes += len(chunk)
        self._seq += 1
        self._new_data.set()
        if not self.complete and (
            len(self.events) > _MAX_EVENTS or self._buffered_bytes > _MAX_BUFFER_BYTES
        ):
            logger.error(
                "stream_session %s buffer overflow: events=%d bytes=%d "
                "(caps %d / %d) — failing session and cancelling producer",
                self.session_id,
                len(self.events),
                self._buffered_bytes,
                _MAX_EVENTS,
                _MAX_BUFFER_BYTES,
            )
            self.mark_error(
                f"stream buffer exceeded caps "
                f"(events={len(self.events)} bytes={self._buffered_bytes})"
            )
            self.cancel_producer()

    def cancel_producer(self) -> None:
        task = self.producer_task
        if task is not None and not task.done():
            task.cancel()
            logger.info("stream_session %s: producer task cancelled", self.session_id)

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
            # Audit M1: an expired session used to be dropped while its
            # producer kept generating and appending forever.
            sess.cancel_producer()
            logger.info("stream_session expired: %s", session_id)
            return None
        return sess

    def _reap_expired(self) -> None:
        # Audit M1: expiry was only evaluated on access, so a session no one
        # ever looked up again was never reclaimed. Sweep on every create so
        # abandoned sessions release their buffer + producer without traffic.
        if not self._sessions:
            return
        expired = [sid for sid, s in self._sessions.items() if s.expired()]
        for sid in expired:
            sess = self._sessions.pop(sid, None)
            if sess is not None:
                sess.cancel_producer()
        if expired:
            logger.info("stream_session reap: removed %d expired", len(expired))

    def create(self, session_id: str, principal: str | None = None) -> StreamSession:
        self._reap_expired()
        sess = StreamSession(session_id, principal=principal)
        self._sessions[session_id] = sess
        logger.info("stream_session created: %s principal=%s", session_id, principal)
        return sess

    def drop(self, session_id: str) -> None:
        sess = self._sessions.pop(session_id, None)
        if sess is not None:
            sess.cancel_producer()

    def active(self) -> list[str]:
        return [sid for sid, s in self._sessions.items() if not s.expired()]

    def count_for_principal(self, principal: str | None) -> int:
        # #801 DoS guard: cap concurrent live sessions per caller so one
        # principal cannot spawn unbounded background producers.
        # Audit 1019 L35: expired sessions used to be skipped by the filter
        # but left in the dict, so the cap only worked until the store grew
        # — and every call scanned the whole expired backlog. Reap first.
        self._reap_expired()
        return sum(1 for s in self._sessions.values() if s.principal == principal)


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
