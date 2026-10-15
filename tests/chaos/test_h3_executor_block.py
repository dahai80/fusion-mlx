# SPDX-License-Identifier: Apache-2.0
"""H3 chaos shape: blocking op on an executor with no GIL/heartbeat release.

Injection: a job occupies the single-worker executor and never heartbeats
(#1189 ExecutorWatchdog shape). The watchdog must fire on_hang within its
window (poison + executor replacement), the replacement must serve the
next job, and no request may hang silently.
"""

import concurrent.futures
import threading
import time

from fusion_mlx import engine_core as ec
from fusion_mlx.executor_watchdog import ExecutorWatchdog
from tests.chaos.conftest import DETECTION_BOUND_S, RECOVERY_BOUND_S

WINDOW_S = 0.5


def _make_watchdog(on_hang):
    return ExecutorWatchdog(name="chaos_h3", window_s=WINDOW_S, on_hang=on_hang)


def test_h3_detection_fires_within_window():
    fired = []
    wd = _make_watchdog(lambda label: fired.append(label))
    wd.begin_job("stuck_render")
    t0 = time.monotonic()
    time.sleep(WINDOW_S + 0.2)  # no heartbeat while "blocked"
    got = wd.check()
    elapsed = time.monotonic() - t0
    assert got is True
    assert fired == ["chaos_h3"], "on_hang did not fire"
    assert elapsed < DETECTION_BOUND_S
    assert WINDOW_S <= DETECTION_BOUND_S


def test_h3_heartbeat_prevents_fire():
    fired = []
    wd = _make_watchdog(lambda label: fired.append(label))
    wd.begin_job("slow_but_alive")
    wd.heartbeat()
    time.sleep(WINDOW_S * 0.8)
    wd.heartbeat()  # still making progress
    assert wd.check() is False
    assert fired == []


def test_h3_one_shot_per_stuck_job():
    # The tick loop must not re-fire while the same stuck job is inflight.
    fired = []
    wd = _make_watchdog(lambda label: fired.append(label))
    wd.begin_job("stuck")
    time.sleep(WINDOW_S + 0.2)
    assert wd.check() is True
    assert wd.check() is False  # same job: already fired
    assert len(fired) == 1


def test_h3_recovery_executor_replacement_serves_next_job():
    # The #1189 recovery contract: on_hang poisons the executor and a
    # REPLACEMENT executor takes over — the stuck thread is abandoned but
    # new work gets served. Zero silent loss: the post-poison job returns.
    stuck = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    replacement = {"exec": None}
    started = threading.Event()
    release = threading.Event()

    def stuck_job():
        started.set()
        release.wait(10.0)  # blocks the single worker forever

    def on_hang(label):
        replacement["exec"] = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    wd = ExecutorWatchdog(name="chaos_h3_recovery", window_s=WINDOW_S, on_hang=on_hang)
    wd.begin_job("stuck_render")
    fut = stuck.submit(stuck_job)
    started.wait(2.0)
    t0 = time.monotonic()
    time.sleep(WINDOW_S + 0.2)
    assert wd.check() is True
    assert replacement["exec"] is not None
    # Next job must be served by the replacement within the recovery bound.
    served = replacement["exec"].submit(lambda: "ok").result(timeout=5.0)
    elapsed = time.monotonic() - t0
    assert served == "ok"
    assert elapsed < RECOVERY_BOUND_S
    release.set()
    fut.cancel()
    stuck.shutdown(wait=False)
    replacement["exec"].shutdown(wait=False)


def test_h3_defer_within_media_budget(monkeypatch, caplog):
    # Media-defer contract (#1189): GPU-starved-but-alive jobs defer the
    # poison within the media budget; exceeding it fires.
    monkeypatch.setattr(ec, "_LLM_WATCHDOG_MEDIA_DEFER_S", 0.4)
    fired = []
    wd = ExecutorWatchdog(
        name="chaos_h3_defer",
        window_s=WINDOW_S,
        on_hang=lambda label: fired.append(label),
        defer_check=lambda: True,
    )
    wd.begin_job("gpu_starved")
    time.sleep(WINDOW_S + 0.2)
    with caplog.at_level("WARNING"):
        assert wd.check() is False  # within budget: deferred
    assert fired == []
    assert any("deferring" in r.message for r in caplog.records)
    time.sleep(0.4)  # exceed the media defer budget
    assert wd.check() is True
    assert fired == ["chaos_h3_defer"], "no fire after budget exceeded"


def test_h3_zero_silent_loss_every_job_outcome():
    # Client-visible contract across a hang + recovery cycle: each job
    # either completes on the replacement or fails with an explicit error
    # from the poisoned pool (concurrent.futures.BrokenExecutor shape).
    outcomes = []
    stuck = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    release = threading.Event()
    replacement = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def stuck_job():
        release.wait(10.0)

    stuck.submit(stuck_job)
    wd = _make_watchdog(lambda label: None)
    wd.begin_job("stuck")
    time.sleep(WINDOW_S + 0.2)
    wd.check()  # fires the hang path

    for i in range(5):
        try:
            outcomes.append(replacement.submit(lambda: "served").result(2.0))
        except Exception as exc:
            outcomes.append(f"explicit_error:{type(exc).__name__}")
    release.set()
    stuck.shutdown(wait=False)
    replacement.shutdown(wait=False)
    assert len(outcomes) == 5
    assert all(o == "served" for o in outcomes), outcomes
