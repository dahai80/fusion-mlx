# SPDX-License-Identifier: Apache-2.0
"""H4 chaos shape: file-descriptor leak (open sockets without close).

Injection: open real socket FDs and never close them, with the watchdog's
resolved limit monkeypatched tight so the leak crosses the warn/reject
ratios without needing a million FDs (#1192 FDWatchdog). Detection =
warn + per-type attribution; recovery = closing the leaked FDs lets NEW
streams through again; existing streams are never touched.
"""

import socket
import time

import pytest

from fusion_mlx import fd_watchdog as fw
from fusion_mlx.fd_watchdog import FDPressureError, ensure_stream_fd_capacity
from tests.chaos.conftest import DETECTION_BOUND_S, RECOVERY_BOUND_S


class _Leak:
    def __init__(self, count):
        self.socks = [
            socket.socket(socket.AF_INET, socket.SOCK_STREAM) for _ in range(count)
        ]

    def close(self):
        for s in self.socks:
            try:
                s.close()
            except OSError:
                pass
        self.socks = []


@pytest.fixture()
def tight_limit(monkeypatch):
    # Real sockets + a tight synthetic limit: base_fd_count + margin so
    # ~40 leaked sockets push the ratio over both thresholds.
    base = len(fw._list_open_fds())
    limit = base + 50
    monkeypatch.setattr(fw, "_resolve_fd_limit", lambda: limit)
    return limit


def test_h4_leak_detected_by_type(tight_limit):
    fw.reset_fd_watchdog()
    wd = fw.get_fd_watchdog()
    before = wd.sample(force=True)
    leak = _Leak(40)
    try:
        t0 = time.monotonic()
        after = wd.sample(force=True)
        delta = after["by_type"].get("socket", 0) - before["by_type"].get("socket", 0)
        assert delta >= 30, "socket leak not attributed by type"
        assert after["ratio"] > before["ratio"]
        assert time.monotonic() - t0 < DETECTION_BOUND_S
        attr = wd.attribution()
        assert attr["delta_by_type"].get("socket", 0) >= 30
    finally:
        leak.close()


def test_h4_new_stream_rejected_at_pressure(tight_limit):
    fw.reset_fd_watchdog()
    leak = _Leak(50)  # base+50 limit: 50 leaked = 100% usage
    try:
        with pytest.raises(FDPressureError):
            fw.get_fd_watchdog().check_stream_admission()
        # HTTP gate maps the same condition to an explicit 503.
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            ensure_stream_fd_capacity()
        assert exc_info.value.status_code == 503
        snap = fw.get_fd_watchdog().snapshot()
        assert snap["pressure_rejected_total"] >= 1
    finally:
        leak.close()


def test_h4_recovery_after_close(tight_limit):
    fw.reset_fd_watchdog()
    leak = _Leak(50)
    try:
        with pytest.raises(FDPressureError):
            fw.get_fd_watchdog().check_stream_admission()
    finally:
        leak.close()
    t0 = time.monotonic()
    # Watchdog state resets (fresh singleton = a restarted process shape);
    # the admission check must pass again — recovery.
    fw.reset_fd_watchdog()
    fw.get_fd_watchdog().check_stream_admission()  # must NOT raise
    assert time.monotonic() - t0 < RECOVERY_BOUND_S


def test_h4_existing_streams_never_rejected(tight_limit):
    # The reject gate applies to NEW streams only: an already-open socket
    # stays usable while the gate rejects new admissions.
    fw.reset_fd_watchdog()
    held = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    leak = _Leak(50)
    try:
        with pytest.raises(FDPressureError):
            fw.get_fd_watchdog().check_stream_admission()
        # The held (existing) stream is still alive and usable: SO_TYPE
        # succeeds on any open socket and only fails on a closed fd.
        assert held.getsockopt(socket.SOL_SOCKET, socket.SO_TYPE) == socket.SOCK_STREAM
    finally:
        leak.close()
        held.close()


def test_h4_zero_silent_loss_under_spike(tight_limit):
    # Spike of concurrent stream admissions under pressure: every one
    # either passes or raises the explicit 503-mappable error — none
    # hangs, none silently disappears.
    import asyncio

    fw.reset_fd_watchdog()
    leak = _Leak(50)
    outcomes = []
    try:

        async def try_stream(i):
            try:
                ensure_stream_fd_capacity()
                return f"{i}:admitted"
            except Exception as exc:
                return f"{i}:rejected:{type(exc).__name__}"

        async def main():
            tasks = [try_stream(i) for i in range(30)]
            return await asyncio.wait_for(asyncio.gather(*tasks), timeout=10.0)

        outcomes = asyncio.run(main())
    finally:
        leak.close()
    assert len(outcomes) == 30
    assert all(":admitted" in o or ":rejected:HTTPException" in o for o in outcomes)
    rejected = sum(1 for o in outcomes if "rejected" in o)
    admitted = sum(1 for o in outcomes if "admitted" in o)
    assert rejected + admitted == 30
