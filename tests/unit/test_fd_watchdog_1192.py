# SPDX-License-Identifier: Apache-2.0
"""#1192: active FD watchdog tests."""

import logging
import os
import socket as socket_mod

import pytest

from fusion_mlx import fd_watchdog as fd_mod
from fusion_mlx.fd_watchdog import (
    FDPressureError,
    FDWatchdog,
    _classify_fd,
    get_fd_watchdog,
    reset_fd_watchdog,
)


@pytest.fixture(autouse=True)
def _fresh_watchdog():
    reset_fd_watchdog()
    yield
    reset_fd_watchdog()


def test_classify_fd(tmp_path):
    f = tmp_path / "probe.txt"
    f.write_text("x")
    with open(f) as fh:
        assert _classify_fd(fh.fileno()) == "file"
    a, b = socket_mod.socketpair()
    try:
        assert _classify_fd(a.fileno()) == "socket"
    finally:
        a.close()
        b.close()
    r, w = os.pipe()
    try:
        assert _classify_fd(r) == "pipe"
    finally:
        os.close(r)
        os.close(w)


def test_classify_fd_gone():
    fd = os.open("/dev/null", os.O_RDONLY)
    os.close(fd)
    assert _classify_fd(fd) == "gone"


def test_sample_counts_fds():
    wd = get_fd_watchdog()
    snap = wd.sample(force=True)
    assert snap["total"] >= 1
    assert set(snap["by_type"]).issubset({"socket", "file", "pipe", "other", "gone"})
    lim = snap["limit"]
    if lim > 0:
        assert 0.0 < snap["ratio"] <= 1.0
    else:
        assert snap["ratio"] == 0.0


def test_throttle():
    wd = get_fd_watchdog()
    s1 = wd.sample()
    s2 = wd.sample()
    assert s1 is s2
    s3 = wd.sample(force=True)
    assert s3 is not s1


def test_warn_at_80pct(monkeypatch, caplog):
    wd = FDWatchdog()
    monkeypatch.setattr(fd_mod.resource, "getrlimit", lambda _: (100, 100))
    monkeypatch.setattr(fd_mod, "_list_open_fds", lambda: list(range(85)))
    monkeypatch.setattr(fd_mod, "_classify_fd", lambda fd: "file")
    with caplog.at_level(logging.WARNING, logger="fusion_mlx.fd_watchdog"):
        snap = wd.sample(force=True)
    assert snap["total"] == 85
    assert snap["limit"] == 100
    assert snap["ratio"] == pytest.approx(0.85)
    assert any(
        rec.levelno == logging.WARNING and "fd pressure" in rec.getMessage()
        for rec in caplog.records
    )
    assert wd._warn_total == 1


def test_no_warn_below_threshold(monkeypatch):
    wd = FDWatchdog()
    monkeypatch.setattr(fd_mod.resource, "getrlimit", lambda _: (100, 100))
    monkeypatch.setattr(fd_mod, "_list_open_fds", lambda: list(range(50)))
    monkeypatch.setattr(fd_mod, "_classify_fd", lambda fd: "file")
    snap = wd.sample(force=True)
    assert snap["ratio"] == pytest.approx(0.50)
    assert wd._warn_total == 0


def test_reject_at_90pct(monkeypatch):
    import fusion_mlx.server_metrics as metrics_mod

    sm = metrics_mod.ServerMetrics()
    monkeypatch.setattr(metrics_mod, "get_server_metrics", lambda: sm)
    wd = FDWatchdog()
    monkeypatch.setattr(fd_mod.resource, "getrlimit", lambda _: (100, 100))
    monkeypatch.setattr(fd_mod, "_list_open_fds", lambda: list(range(95)))
    monkeypatch.setattr(fd_mod, "_classify_fd", lambda fd: "file")
    with pytest.raises(FDPressureError):
        wd.check_stream_admission()
    assert wd._pressure_rejected_total == 1
    assert sm.fd_pressure_rejections_total == 1
    snap = wd.sample(force=True)
    assert snap["total"] == 95


def test_existing_streams_unaffected(monkeypatch):
    import fusion_mlx.server_metrics as metrics_mod

    sm = metrics_mod.ServerMetrics()
    monkeypatch.setattr(metrics_mod, "get_server_metrics", lambda: sm)
    wd = FDWatchdog()
    monkeypatch.setattr(fd_mod.resource, "getrlimit", lambda _: (100, 100))
    monkeypatch.setattr(fd_mod, "_list_open_fds", lambda: list(range(95)))
    monkeypatch.setattr(fd_mod, "_classify_fd", lambda fd: "file")
    with pytest.raises(FDPressureError):
        wd.check_stream_admission()
    snap = wd.sample(force=True)
    assert snap["total"] == 95
    attr = wd.attribution()
    assert attr["window_seconds"] >= 0.0
    assert sum(attr["delta_by_type"].values()) == 0


def test_attribution_window():
    wd = FDWatchdog()
    wd._history = [
        (1000.0, {"file": 10}),
        (1060.0, {"file": 25}),
    ]
    attr = wd.attribution()
    assert attr["window_seconds"] == pytest.approx(60.0)
    assert attr["delta_by_type"]["file"] == 15


def test_unlimited_limit_skips_gate(monkeypatch):
    import resource as resource_mod

    wd = FDWatchdog()
    monkeypatch.setattr(
        resource_mod,
        "getrlimit",
        lambda _: (resource_mod.RLIM_INFINITY, resource_mod.RLIM_INFINITY),
    )
    monkeypatch.setattr(fd_mod, "_list_open_fds", lambda: list(range(500)))
    monkeypatch.setattr(fd_mod, "_classify_fd", lambda fd: "file")
    wd.check_stream_admission()
    assert wd.sample(force=True)["limit"] == 0


def test_invalid_env_fallback(monkeypatch):
    monkeypatch.setenv("FUSION_MLX_FD_REJECT_RATIO", "junk")
    wd = FDWatchdog()
    assert wd._reject_ratio == 0.90


def test_valid_env_override(monkeypatch):
    monkeypatch.setenv("FUSION_MLX_FD_REJECT_RATIO", "0.5")
    monkeypatch.setenv("FUSION_MLX_FD_WARN_RATIO", "0.4")
    wd = FDWatchdog()
    assert wd._reject_ratio == 0.5
    assert wd._warn_ratio == 0.4


def test_out_of_range_env_fallback(monkeypatch):
    monkeypatch.setenv("FUSION_MLX_FD_WARN_RATIO", "1.5")
    wd = FDWatchdog()
    assert wd._warn_ratio == 0.80


_GATE_SITES = [
    ("fusion_mlx/api/openai/streaming.py", 1),
    ("fusion_mlx/api/openai/resume.py", 2),
    ("fusion_mlx/api/openai/markitdown.py", 1),
    ("fusion_mlx/api/anthropic_routes.py", 1),
    ("fusion_mlx/api/audio_routes.py", 1),
    ("fusion_mlx/api/openclaw_routes.py", 1),
]


@pytest.mark.parametrize(
    "rel_path,min_sites",
    [(p, n) for p, n in _GATE_SITES],
)
def test_stream_route_gate_call_sites(rel_path, min_sites):
    import pathlib

    pkg_root = pathlib.Path(fd_mod.__file__).resolve().parent.parent
    text = (pkg_root / rel_path).read_text()
    assert "ensure_stream_fd_capacity" in text
    assert text.count("ensure_stream_fd_capacity()") >= min_sites
    assert "#1192" in text


def test_ensure_stream_fd_capacity_maps_503(monkeypatch):
    from fastapi import HTTPException

    wd = FDWatchdog()
    monkeypatch.setattr(fd_mod, "_watchdog", wd)
    monkeypatch.setattr(fd_mod.resource, "getrlimit", lambda _: (100, 100))
    monkeypatch.setattr(fd_mod, "_list_open_fds", lambda: list(range(95)))
    monkeypatch.setattr(fd_mod, "_classify_fd", lambda fd: "file")
    with pytest.raises(HTTPException) as excinfo:
        fd_mod.ensure_stream_fd_capacity()
    assert excinfo.value.status_code == 503
    assert "rejecting NEW stream" in str(excinfo.value.detail)


def test_ensure_stream_fd_capacity_passes_below_threshold(monkeypatch):
    wd = FDWatchdog()
    monkeypatch.setattr(fd_mod, "_watchdog", wd)
    monkeypatch.setattr(fd_mod.resource, "getrlimit", lambda _: (100, 100))
    monkeypatch.setattr(fd_mod, "_list_open_fds", lambda: list(range(50)))
    monkeypatch.setattr(fd_mod, "_classify_fd", lambda fd: "file")
    fd_mod.ensure_stream_fd_capacity()
