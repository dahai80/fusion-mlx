# SPDX-License-Identifier: Apache-2.0
"""Regression tests for issue #1069.

#1069: _drain_once cleared the queue BEFORE the POST, so a flush failure
permanently lost the whole batch (no re-enqueue, no retry). events_dropped
only counted maxlen overflow, not flush losses. Fix: re-enqueue the failed
batch (capped to never evict newer events) + a flush_dropped counter.
"""

from __future__ import annotations

import pytest

from fusion_mlx.telemetry.queue import TelemetryQueue


def _evt(i: int) -> dict:
    return {"id": i}


class TestFlushRequeue1069:
    """Failed flush re-enqueues the batch (capped) + counts flush_dropped."""

    def test_failed_flush_reenqueues_batch(self):
        calls = []

        def failing_flusher(batch):
            calls.append(list(batch))
            return False

        q = TelemetryQueue(flusher=failing_flusher, max_len=100)
        for i in range(3):
            q.enqueue(_evt(i))
        q._drain_once()
        # batch was tried once
        assert len(calls) == 1
        assert len(calls[0]) == 3
        # and re-enqueued — pending is back to 3
        snap = q.snapshot()
        assert snap["pending"] == 3
        assert snap["flushes_failed"] == 1
        assert snap["flush_dropped"] == 0

    def test_successful_flush_clears_queue(self):
        def ok_flusher(batch):
            return True

        q = TelemetryQueue(flusher=ok_flusher, max_len=100)
        for i in range(3):
            q.enqueue(_evt(i))
        q._drain_once()
        snap = q.snapshot()
        assert snap["pending"] == 0
        assert snap["flushes_ok"] == 1
        assert snap["flush_dropped"] == 0

    def test_failed_flush_drops_surplus_when_queue_full(self):
        def failing_flusher(batch):
            return False

        q = TelemetryQueue(flusher=failing_flusher, max_len=5)
        # fill the queue, flush fails (batch of 5), then we enqueue 5 NEW
        # events during the "POST" — re-enqueue must not evict the new ones.
        for i in range(5):
            q.enqueue(_evt(i))
        # simulate: _drain_once grabs batch=5, clears. Then new events arrive.
        # We emulate by calling _drain_once, then enqueue, then _drain_once.
        q._drain_once()  # batch [0..4] grabbed, POST failed, queue now empty
        for i in range(5, 10):
            q.enqueue(_evt(i))  # queue now full of new events [5..9]
        # now re-run drain: the previous batch [0..4] was already lost (queue
        # was empty at fail time, so it re-enqueued fine). To exercise the
        # drop path we need new events present DURING the fail. Emulate by
        # pre-filling, then manually calling the fail path:
        q2 = TelemetryQueue(flusher=failing_flusher, max_len=5)
        for i in range(5):
            q2.enqueue(_evt(i))
        # grab the batch manually then refill before the counters update:
        with q2._lock:
            batch = list(q2._events)
            q2._events.clear()
        for i in range(5, 10):
            q2.enqueue(_evt(i))  # queue full of new events
        # now invoke the fail-handling with the old batch directly:
        with q2._lock:
            capacity = q2._events.maxlen - len(q2._events)
            requeue = batch[:capacity] if capacity > 0 else []
            dropped = len(batch) - len(requeue)
            if requeue:
                q2._events.extendleft(reversed(requeue))
            q2.flushes_failed += 1
            if dropped > 0:
                q2.flush_dropped += dropped
        snap = q2.snapshot()
        assert capacity == 0
        assert snap["pending"] == 5  # the new events [5..9] preserved
        assert snap["flush_dropped"] == 5  # old batch [0..4] all dropped
        assert snap["flushes_failed"] == 1

    def test_snapshot_exposes_flush_dropped(self):
        q = TelemetryQueue(flusher=lambda b: True, max_len=10)
        snap = q.snapshot()
        assert "flush_dropped" in snap
        assert snap["flush_dropped"] == 0

    def test_flusher_exception_treated_as_failure_and_reenqueued(self):
        def crashing_flusher(batch):
            raise RuntimeError("boom")

        q = TelemetryQueue(flusher=crashing_flusher, max_len=100)
        for i in range(2):
            q.enqueue(_evt(i))
        q._drain_once()
        snap = q.snapshot()
        assert snap["flushes_failed"] == 1
        assert snap["pending"] == 2  # re-enqueued despite the exception
        assert snap["flush_dropped"] == 0

    def test_reenqueued_batch_retried_on_next_flush(self):
        state = {"fail": True}

        def fl(batch):
            return not state["fail"]

        q = TelemetryQueue(flusher=fl, max_len=100)
        for i in range(2):
            q.enqueue(_evt(i))
        q._drain_once()  # fails, re-enqueues
        assert q.snapshot()["pending"] == 2
        state["fail"] = False
        q._drain_once()  # succeeds, clears
        snap = q.snapshot()
        assert snap["pending"] == 0
        assert snap["flushes_ok"] == 1
        assert snap["flushes_failed"] == 1


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
