# SPDX-License-Identifier: Apache-2.0
"""#1468: queue-wait visibility on the shared global executors. A task that
starts after a long queue wait logs a WARNING; a new submit arriving while
all workers are busy and the oldest running task is ancient logs an ERROR
naming the possible wedged holder. Both signals are event-driven (no timers)
and fire through the get_executor() proxy with zero call-site changes.
"""

from __future__ import annotations

import concurrent.futures
import logging
import threading
import time

import pytest

import fusion_mlx.engine_core as ec
from fusion_mlx.engine_core import _TrackedExecutor, get_executor


def _make_proxy(max_workers: int = 1) -> _TrackedExecutor:
    inner = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
    return _TrackedExecutor(inner, "test-pool", max_workers)


@pytest.fixture(autouse=True)
def _tight_thresholds(monkeypatch):
    monkeypatch.setattr(ec, "_QUEUE_WAIT_WARN_S", 0.2)
    monkeypatch.setattr(ec, "_EXEC_WEDGE_WARN_S", 0.4)


def test_queue_wait_warning_on_slow_start(caplog):
    proxy = _make_proxy()
    release = threading.Event()
    try:
        blocker = proxy.submit(release.wait)
        with caplog.at_level(logging.WARNING, logger="fusion_mlx.engine_core"):
            second = proxy.submit(lambda: "done")
            time.sleep(0.35)
            release.set()
            assert second.result(timeout=5) == "done"
        warn_records = [
            r
            for r in caplog.records
            if "queue wait" in r.getMessage() and "test-pool" in r.getMessage()
        ]
        assert warn_records, "expected a queue-wait WARNING for the starved task"
        msg = warn_records[0].getMessage()
        assert "(#1468)" in msg
        assert "busy or starved" in msg
        assert blocker.result(timeout=5)
    finally:
        proxy.shutdown(wait=True)


def test_no_warning_when_wait_under_threshold(caplog):
    proxy = _make_proxy()
    try:
        with caplog.at_level(logging.WARNING, logger="fusion_mlx.engine_core"):
            assert proxy.submit(lambda: 1).result(timeout=5) == 1
        assert not [
            r for r in caplog.records if "queue wait" in r.getMessage()
        ], "fast task must not warn"
    finally:
        proxy.shutdown(wait=True)


def test_wedge_error_fires_once_per_stuck_task(caplog):
    proxy = _make_proxy()
    release = threading.Event()
    try:
        blocker = proxy.submit(release.wait)
        time.sleep(0.45)
        with caplog.at_level(logging.ERROR, logger="fusion_mlx.engine_core"):
            queued_1 = proxy.submit(lambda: "a")
            time.sleep(0.05)
            queued_2 = proxy.submit(lambda: "b")
            time.sleep(0.05)
            release.set()
            assert queued_1.result(timeout=5) == "a"
            assert queued_2.result(timeout=5) == "b"
        wedge_records = [
            r
            for r in caplog.records
            if "possible wedged" in r.getMessage() and "test-pool" in r.getMessage()
        ]
        assert len(wedge_records) == 1, "wedge ERROR must fire exactly once"
        msg = wedge_records[0].getMessage()
        assert "(#1468)" in msg
        assert "queued behind it" in msg
        assert blocker.result(timeout=5)
    finally:
        proxy.shutdown(wait=True)


def test_no_wedge_error_when_worker_idle(caplog):
    proxy = _make_proxy()
    try:
        time.sleep(0.45)
        with caplog.at_level(logging.ERROR, logger="fusion_mlx.engine_core"):
            assert proxy.submit(lambda: 1).result(timeout=5) == 1
        assert not [
            r for r in caplog.records if "possible wedged" in r.getMessage()
        ], "no running task means no wedge"
    finally:
        proxy.shutdown(wait=True)


def test_wedge_error_needs_all_workers_busy(caplog):
    # max_workers=2: one long-running task leaves a free worker — a new
    # submit must NOT fire the wedge ERROR.
    proxy = _make_proxy(max_workers=2)
    release = threading.Event()
    try:
        blocker = proxy.submit(release.wait)
        time.sleep(0.45)
        with caplog.at_level(logging.ERROR, logger="fusion_mlx.engine_core"):
            assert proxy.submit(lambda: "ok").result(timeout=5) == "ok"
        assert not [
            r for r in caplog.records if "possible wedged" in r.getMessage()
        ], "free worker available — not a wedge"
        release.set()
        assert blocker.result(timeout=5)
    finally:
        proxy.shutdown(wait=True)


def test_get_executor_returns_proxy_and_tracks_state(monkeypatch):
    monkeypatch.setattr(ec, "_global_executors", {})
    monkeypatch.setattr(ec, "_global_executor_proxies", {})
    p1 = get_executor("test-pool-x")
    assert isinstance(p1, _TrackedExecutor)
    # Same raw executor -> same proxy instance.
    assert get_executor("test-pool-x") is p1
    try:
        assert p1.submit(lambda: "v").result(timeout=5) == "v"
    finally:
        p1.shutdown(wait=True)


def test_poison_replacement_gets_fresh_proxy(monkeypatch):
    # poison_executor swaps a fresh raw executor into _global_executors;
    # get_executor must hand out a NEW proxy wrapping the new raw executor.
    monkeypatch.setattr(ec, "_global_executors", {})
    monkeypatch.setattr(ec, "_global_executor_proxies", {})
    p1 = get_executor("test-pool-y")
    fresh = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        ec._global_executors["test-pool-y"] = fresh
        p2 = get_executor("test-pool-y")
        assert p2 is not p1
        assert p2._inner is fresh
        assert p2.submit(lambda: "v").result(timeout=5) == "v"
    finally:
        p1.shutdown(wait=True)
        fresh.shutdown(wait=True)


def test_env_float_parsing(monkeypatch, caplog):
    monkeypatch.setenv("FUSION_TEST_1468_KNOB", "not-a-number")
    with caplog.at_level(logging.WARNING, logger="fusion_mlx.engine_core"):
        assert ec._env_float("FUSION_TEST_1468_KNOB", 7.0) == 7.0
    assert any("is not a number" in r.getMessage() for r in caplog.records)
    monkeypatch.setenv("FUSION_TEST_1468_KNOB", "0")
    assert ec._env_float("FUSION_TEST_1468_KNOB", 7.0) == 7.0
    monkeypatch.setenv("FUSION_TEST_1468_KNOB", "12.5")
    assert ec._env_float("FUSION_TEST_1468_KNOB", 7.0) == 12.5
    monkeypatch.delenv("FUSION_TEST_1468_KNOB")
    assert ec._env_float("FUSION_TEST_1468_KNOB", 7.0) == 7.0
