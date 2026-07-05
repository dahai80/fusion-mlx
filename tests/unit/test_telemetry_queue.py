# SPDX-License-Identifier: Apache-2.0
import logging
import threading
import time

from fusion_mlx.telemetry.queue import (
    FLUSH_INTERVAL_S,
    FLUSH_THRESHOLD,
    MAX_QUEUE_LEN,
    TelemetryQueue,
)

logger = logging.getLogger(__name__)


def test_enqueue_buffers_until_threshold():
    flushed: list[list[dict]] = []

    def flusher(batch):
        flushed.append(batch)
        return True

    q = TelemetryQueue(flusher=flusher, flush_interval_s=60.0, flush_threshold=5)
    q.start()
    try:
        for i in range(4):
            q.enqueue({"i": i})
        time.sleep(0.05)
        assert flushed == []

        q.enqueue({"i": 4})
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and not flushed:
            time.sleep(0.01)
        assert flushed, "daemon never flushed on threshold cross"
        assert len(flushed[0]) == 5
    finally:
        q.shutdown(timeout=0.5)


def test_drops_oldest_when_over_capacity():
    captured: list[dict] = []
    started = threading.Event()

    def flusher(batch):
        started.set()
        captured.extend(batch)
        return True

    q = TelemetryQueue(
        flusher=flusher,
        max_len=3,
        flush_interval_s=60.0,
        flush_threshold=999,
    )
    q.start()
    try:
        for i in range(6):
            q.enqueue({"i": i})
        snap = q.snapshot()
        assert snap["pending"] == 3
        assert snap["enqueued_total"] == 6
        assert snap["dropped_total"] == 3
    finally:
        q.shutdown(timeout=0.5)
    assert started.is_set()
    assert [e["i"] for e in captured] == [3, 4, 5]


def test_shutdown_drains_remaining_events():
    captured: list[dict] = []

    def flusher(batch):
        captured.extend(batch)
        return True

    q = TelemetryQueue(flusher=flusher, flush_interval_s=60.0, flush_threshold=999)
    q.start()
    q.enqueue({"a": 1})
    q.enqueue({"a": 2})
    q.shutdown(timeout=1.0)
    assert [e["a"] for e in captured] == [1, 2]


def test_shutdown_does_not_orphan_thread_for_restart_when_join_times_out():
    release = threading.Event()

    def slow_flusher(batch):
        release.wait(timeout=5.0)
        return True

    q = TelemetryQueue(flusher=slow_flusher, flush_interval_s=60.0, flush_threshold=1)
    q.start()
    q.enqueue({"x": 1})
    time.sleep(0.1)

    q.shutdown(timeout=0.1)
    original_thread = q._thread
    assert original_thread is not None
    assert original_thread.is_alive()

    before_named = [t for t in threading.enumerate() if t.name == "rapid-mlx-telemetry"]
    q.start()
    after_named = [t for t in threading.enumerate() if t.name == "rapid-mlx-telemetry"]
    assert q._thread is original_thread, "start() replaced live daemon"
    assert len(after_named) == len(before_named), (
        f"start() spawned a second telemetry daemon "
        f"(before={len(before_named)}, after={len(after_named)})"
    )

    release.set()
    if q._thread is not None:
        q._thread.join(timeout=2.0)


def test_shutdown_returns_within_budget_even_if_flusher_hangs():
    release = threading.Event()

    def slow_flusher(batch):
        release.wait(timeout=5.0)
        return True

    q = TelemetryQueue(flusher=slow_flusher, flush_interval_s=60.0, flush_threshold=1)
    q.start()
    try:
        q.enqueue({"x": 1})
        time.sleep(0.1)

        t0 = time.monotonic()
        q.shutdown(timeout=0.2)
        elapsed = time.monotonic() - t0
        assert elapsed < 0.5, f"shutdown blocked {elapsed:.2f}s on slow flusher"
    finally:
        release.set()
        if q._thread is not None:
            q._thread.join(timeout=2.0)


def test_flusher_exception_increments_failed_not_crash():
    def bad_flusher(batch):
        raise RuntimeError("synthetic")

    q = TelemetryQueue(flusher=bad_flusher, flush_interval_s=60.0, flush_threshold=1)
    q.start()
    q.enqueue({"x": 1})
    time.sleep(0.2)
    snap = q.snapshot()
    assert snap["flushes_failed"] >= 1
    assert snap["flushes_ok"] == 0
    q.shutdown(timeout=0.5)
    q.start()
    q.enqueue({"x": 2})
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        snap = q.snapshot()
        if snap["flushes_failed"] >= 2:
            break
        time.sleep(0.01)
    q.shutdown(timeout=0.5)

    snap_after = q.snapshot()
    assert snap_after["flushes_failed"] >= 2, (
        f"second lifecycle did not attempt a flush "
        f"(flushes_failed={snap_after['flushes_failed']}); "
        "either the daemon never restarted or the second batch was dropped"
    )
    assert snap_after["pending"] == 0, (
        f"second lifecycle left {snap_after['pending']} event(s) stuck "
        "in the queue -- the drain on shutdown is broken"
    )


def test_shutdown_called_twice_does_not_double_budget():
    release = threading.Event()

    def slow_flusher(batch):
        release.wait(timeout=5.0)
        return True

    q = TelemetryQueue(flusher=slow_flusher, flush_interval_s=60.0, flush_threshold=1)
    q.start()
    try:
        q.enqueue({"x": 1})
        time.sleep(0.1)

        t0 = time.monotonic()
        q.shutdown(timeout=0.3)
        q.shutdown(timeout=0.3)
        elapsed = time.monotonic() - t0
        assert elapsed < 0.5, (
            f"second shutdown re-spent the budget: "
            f"elapsed={elapsed:.2f}s (expected <0.5s, latched no-op)"
        )
    finally:
        release.set()
        if q._thread is not None:
            q._thread.join(timeout=2.0)


def test_start_clears_shutdown_latch_for_restart():
    captured: list[dict] = []

    def flusher(batch):
        captured.extend(batch)
        return True

    q = TelemetryQueue(flusher=flusher, flush_interval_s=60.0, flush_threshold=999)
    q.start()
    q.enqueue({"a": 1})
    q.shutdown(timeout=1.0)
    assert [e["a"] for e in captured] == [1]

    captured.clear()
    q.start()
    q.enqueue({"a": 2})
    q.shutdown(timeout=1.0)
    assert [e["a"] for e in captured] == [
        2
    ], "second lifecycle did not drain -- shutdown latch leaked across start()"


def test_start_preserves_wake_for_events_enqueued_pre_start():
    flushed: list[list[dict]] = []
    flushed_event = threading.Event()

    def flusher(batch):
        flushed.append(batch)
        flushed_event.set()
        return True

    q = TelemetryQueue(flusher=flusher, flush_interval_s=60.0, flush_threshold=2)
    q.enqueue({"i": 0})
    q.enqueue({"i": 1})
    q.start()
    try:
        assert flushed_event.wait(timeout=2.0), (
            "start() lost the threshold wake -- daemon did not flush "
            "pre-start over-threshold batch within 2 s"
        )
        assert flushed and len(flushed[0]) == 2
    finally:
        q.shutdown(timeout=0.5)


def test_start_is_idempotent():
    q = TelemetryQueue(flusher=lambda b: True)
    q.start()
    q.start()
    q.shutdown(timeout=0.1)


def test_concurrent_start_does_not_spawn_duplicate_daemons():
    q = TelemetryQueue(flusher=lambda b: True)
    started = threading.Event()

    def racer():
        started.wait(timeout=2.0)
        q.start()

    threads = [threading.Thread(target=racer) for _ in range(8)]
    for t in threads:
        t.start()
    started.set()
    for t in threads:
        t.join(timeout=2.0)

    named = [t for t in threading.enumerate() if t.name == "rapid-mlx-telemetry"]
    assert (
        len(named) == 1
    ), f"concurrent start() spawned {len(named)} daemons (want exactly 1)"
    q.shutdown(timeout=0.5)


def test_snapshot_shape():
    q = TelemetryQueue(flusher=lambda b: True)
    snap = q.snapshot()
    assert set(snap) == {
        "pending",
        "enqueued_total",
        "dropped_total",
        "flushes_ok",
        "flushes_failed",
    }


def test_module_defaults_are_sane():
    assert MAX_QUEUE_LEN == 100
    assert FLUSH_INTERVAL_S == 60.0
    assert FLUSH_THRESHOLD == 10
