# SPDX-License-Identifier: Apache-2.0
"""Fault-injection tests for the generalized executor watchdog (#1189).

Covers the four new modalities (image / video / audio / laya — llm keeps its
own G2 watchdog). Each fault-injection test drives ExecutorWatchdog.check()
synchronously (same anti-flake rationale as _llm_watchdog_check) against the
REAL engine_core poison machinery: hang → on_hang → poison_executor → flag
set → metric recorded → end_job + reset → next job admitted.
"""

from __future__ import annotations

import inspect
import threading
import time

import pytest

from fusion_mlx import engine_core
from fusion_mlx.executor_watchdog import (
    ExecutorWatchdog,
    executor_watch,
    get_executor_watchdog,
    reset_executor_watchdogs,
)
from fusion_mlx.server_metrics import get_server_metrics

_WINDOW_S = 0.4
_SLEEP_S = 0.6


@pytest.fixture(autouse=True)
def _clean_state():
    reset_executor_watchdogs()
    engine_core.reset_image_executor_poison()
    engine_core.reset_video_executor_poison()
    engine_core.reset_audio_executor_poison()
    engine_core.reset_laya_executor_poison()
    engine_core.set_media_job_active(False)
    yield
    reset_executor_watchdogs()
    engine_core.reset_image_executor_poison()
    engine_core.reset_video_executor_poison()
    engine_core.reset_audio_executor_poison()
    engine_core.reset_laya_executor_poison()
    engine_core.set_media_job_active(False)


def _make_watchdog(modality: str) -> ExecutorWatchdog:
    # Real setter: engine_core.poison_executor (the same on_hang target the
    # wiring sites use).
    return ExecutorWatchdog(
        modality,
        window_s=_WINDOW_S,
        on_hang=lambda _name: engine_core.poison_executor(modality),
    )


@pytest.mark.parametrize(
    ("modality", "is_poisoned", "reset_poison"),
    [
        (
            "image",
            engine_core.is_image_executor_poisoned,
            engine_core.reset_image_executor_poison,
        ),
        (
            "video",
            engine_core.is_video_executor_poisoned,
            engine_core.reset_video_executor_poison,
        ),
        (
            "audio",
            engine_core.is_audio_executor_poisoned,
            engine_core.reset_audio_executor_poison,
        ),
        (
            "laya",
            engine_core.is_laya_executor_poisoned,
            engine_core.reset_laya_executor_poison,
        ),
    ],
)
def test_hang_poisons_modality_and_recovers(modality, is_poisoned, reset_poison):
    wd = _make_watchdog(modality)
    sm = get_server_metrics()
    poisons_before = sm.watchdog_poisons_by_executor.get(modality, 0)

    assert not is_poisoned()
    wd.begin_job(f"{modality}-hung-job")
    # Within the window: no fire.
    assert wd.check() is False
    assert not is_poisoned()

    time.sleep(_SLEEP_S)
    # Past the window: fires exactly once, sets the REAL poison flag and
    # records the poison metric.
    assert wd.check() is True
    assert is_poisoned()
    assert sm.watchdog_poisons_by_executor.get(modality, 0) == poisons_before + 1
    # No re-fire for the same wedged job.
    assert wd.check() is False
    assert sm.watchdog_poisons_by_executor.get(modality, 0) == poisons_before + 1

    # Recovery: end_job + the real reset clears the flag; the next job is
    # admitted and tracked from a clean baseline.
    wd.end_job()
    reset_poison()
    assert not is_poisoned()
    wd.begin_job(f"{modality}-next-job")
    assert wd.check() is False


def test_executor_watch_heartbeat_survives_long_job():
    fired = threading.Event()
    reset_executor_watchdogs()
    wd_ref = {}

    def _on_hang(_name):
        fired.set()

    # Heartbeat every 0.1s for 0.6s with a 0.4s window: the job must never
    # look stale, so on_hang never fires.
    with executor_watch("image", window_s=_WINDOW_S) as handle:
        wd_ref["wd"] = get_executor_watchdog("image")
        wd_ref["wd"].on_hang = _on_hang
        deadline = time.monotonic() + 0.6
        while time.monotonic() < deadline:
            time.sleep(0.1)
            handle.heartbeat()
            assert get_executor_watchdog("image").check() is False
    assert not fired.is_set()
    assert get_executor_watchdog("image").snapshot()["inflight"] is False


def test_media_defer_blocks_then_budget_expiry_fires(monkeypatch):
    fired = threading.Event()
    monkeypatch.setattr(engine_core, "_LLM_WATCHDOG_MEDIA_DEFER_S", 1.0)
    wd = ExecutorWatchdog(
        "image",
        window_s=_WINDOW_S,
        on_hang=lambda _n: fired.set(),
        defer_check=engine_core.is_media_job_active,
    )
    engine_core.set_media_job_active(True)
    wd.begin_job("deferred-job")
    time.sleep(_SLEEP_S)
    # Media job active + within the (patched 1s) defer budget: no fire.
    assert wd.check() is False
    assert not fired.is_set()
    # Simulate the defer budget expiring (defer start pushed past the budget).
    wd._defer_start = time.monotonic() - 2.0
    assert wd.check() is True
    assert fired.is_set()


def test_metrics_render_exposes_watchdog_series():
    from fusion_mlx.routes_internal.metrics import render_prometheus_metrics

    reset_executor_watchdogs()
    wd = get_executor_watchdog("image", window_s=_WINDOW_S)
    wd.begin_job("render-test")
    get_server_metrics().record_watchdog_poison("image")
    body = render_prometheus_metrics()
    assert 'fusion_mlx_watchdog_poison_total{executor="image"}' in body
    assert 'fusion_mlx_watchdog_heartbeat_age_seconds{executor="image"}' in body
    assert "fusion_mlx_watchdog_poison_total " in body
    wd.end_job()


def test_watchdog_snapshot_shape():
    wd = ExecutorWatchdog("laya", window_s=_WINDOW_S)
    snap = wd.snapshot()
    assert snap["inflight"] is False
    assert snap["heartbeat_age_s"] is None
    wd.begin_job("predict")
    snap = wd.snapshot()
    assert snap["inflight"] is True
    assert snap["job"] == "predict"
    assert snap["heartbeat_age_s"] is not None
    wd.end_job()
    assert wd.snapshot()["inflight"] is False


def test_daemon_tick_thread_fires_on_hang():
    # One test exercises the real daemon tick thread path; all other tests
    # drive check() synchronously to avoid scheduler flakiness.
    fired = threading.Event()
    reset_executor_watchdogs()
    wd = get_executor_watchdog("laya", window_s=0.3, on_hang=lambda _n: fired.set())
    wd.begin_job("daemon-test")
    assert fired.wait(timeout=6.0), "daemon tick thread did not fire on_hang"
    wd.end_job()


def _source(path: str) -> str:
    import fusion_mlx

    root = fusion_mlx.__path__[0]
    with open(f"{root}/{path}") as f:
        return f.read()


def test_source_pins_watchdog_wiring():
    img_src = _source("engines/image_gen.py")
    assert "executor_watchdog" in img_src
    assert "begin_job" in img_src
    assert "_wd_sync_cb" in img_src

    vid_src = _source("engines/video.py")
    assert "executor_watchdog" in vid_src
    assert "begin_job" in vid_src
    assert "_wd_step_cb" in vid_src

    tts_src = _source("engines/tts.py")
    assert "executor_watchdog" in tts_src
    assert "begin_job" in tts_src
    assert "heartbeat()" in tts_src

    laya_src = _source("api/decisions_routes.py")
    assert "executor_watchdog" in laya_src
    assert "_ensure_laya_executor_healthy" in laya_src
    assert "begin_job" in laya_src

    audio_src = _source("api/audio_routes.py")
    assert audio_src.count("_ensure_audio_executor_healthy()") >= 5

    # inspect.getsource cross-check on the watchdog module itself.
    src = inspect.getsource(ExecutorWatchdog)
    assert "begin_job" in src and "heartbeat" in src and "check" in src
