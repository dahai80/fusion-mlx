# SPDX-License-Identifier: Apache-2.0
"""Defect 4 -- macOS UBC eviction helper regression coverage.

Exercises ``fusion_mlx.runtime.ubc_evict``:

* **Darwin happy path** -- build a multi-MB random file, force it into
  the Unified Buffer Cache via mmap+touch, then assert that
  ``ubc_evict`` returns ``Pages free`` to within a noise tolerance of
  the pre-mmap baseline (i.e. the kernel actually released our pages).
* **Cross-platform no-op** -- monkeypatch ``sys.platform = "linux"``,
  assert the helper returns 0 and does not touch libc.
* **Missing file** -- assert the helper returns 0 + logs a warning.
* **Zero-byte file** -- assert the helper returns 0 + logs at DEBUG (no
  warning) and does not invoke ``mmap``.
* **Prometheus rendering** -- counter is monotonic, renders as a single
  HELP/TYPE/sample line triple, label is ``path_kind="safetensors"``.

The Darwin path runs only on Darwin (`pytest.mark.skipif`); on Linux /
Windows CI we still cover the no-op + error + Prometheus paths.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

import fusion_mlx.runtime.ubc_evict as ubc_module
from fusion_mlx.runtime.ubc_evict import (
    render_prometheus_lines,
    reset_for_tests,
    snapshot,
    ubc_evict,
    ubc_evict_paths,
)

logger = logging.getLogger(__name__)


@pytest.fixture(autouse=True)
def _reset_counters():
    """Zero the counters around every test for isolation."""
    reset_for_tests()
    yield
    reset_for_tests()


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------


def _vm_stat_pages_free() -> int:
    out = subprocess.run(
        ["vm_stat"], capture_output=True, text=True, timeout=5, check=True
    ).stdout
    for line in out.splitlines():
        m = re.match(r"Pages free:\s+(\d+)", line)
        if m:
            return int(m.group(1))
    raise RuntimeError("vm_stat missing 'Pages free' line")


def _page_size() -> int:
    """Return the kernel page size (16 KiB on Apple Silicon, 4 KiB on x86)."""
    return os.sysconf("SC_PAGE_SIZE")


# ---------------------------------------------------------------------
# Darwin happy path
# ---------------------------------------------------------------------


@pytest.mark.skipif(sys.platform != "darwin", reason="UBC eviction is macOS-only")
def test_ubc_evict_darwin_releases_pages(tmp_path):
    """On Darwin, ubc_evict releases UBC-resident pages back to the free pool."""
    pg = _page_size()
    size = 100 * 1024 * 1024  # 100 MB
    expected_pages = size // pg

    payload = tmp_path / "ubc_payload.bin"
    subprocess.run(
        [
            "dd",
            "if=/dev/urandom",
            f"of={payload}",
            "bs=1m",
            f"count={size // (1024 * 1024)}",
            "status=none",
        ],
        check=True,
    )
    assert payload.stat().st_size == size

    import mmap as _mmap_mod

    fd = os.open(payload, os.O_RDONLY)
    try:
        mm = _mmap_mod.mmap(fd, length=size, prot=_mmap_mod.PROT_READ)
        try:
            for off in range(0, size, pg):
                _ = mm[off]
        finally:
            mm.close()
    finally:
        os.close(fd)

    pre_evict = _vm_stat_pages_free()
    bytes_evicted = ubc_evict(str(payload))
    post_evict = _vm_stat_pages_free()

    assert bytes_evicted == size
    free_delta_pages = post_evict - pre_evict
    assert free_delta_pages >= expected_pages // 2, (
        f"Expected at least {expected_pages // 2} pages back to free, "
        f"got delta={free_delta_pages}"
    )
    snap = snapshot()
    assert snap["ubc_evicted_bytes_total"] == size
    assert snap["ubc_evict_calls_total"] == 1
    assert snap["ubc_evict_failed_total"] == 0


# ---------------------------------------------------------------------
# Cross-platform no-op
# ---------------------------------------------------------------------


def test_ubc_evict_noop_on_linux(monkeypatch, tmp_path, caplog):
    """ubc_evict returns 0 on non-Darwin and never touches libc."""
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(
        ubc_module,
        "_get_libc",
        lambda: (_ for _ in ()).throw(
            AssertionError("_get_libc must not run on non-Darwin")
        ),
    )

    payload = tmp_path / "p.bin"
    payload.write_bytes(b"x" * 4096)

    with caplog.at_level(logging.DEBUG, logger=ubc_module.logger.name):
        result = ubc_evict(str(payload))
    assert result == 0

    snap = snapshot()
    assert snap["ubc_evicted_bytes_total"] == 0
    assert snap["ubc_evict_calls_total"] == 1
    assert snap["ubc_evict_failed_total"] == 0
    assert any("no-op" in r.message for r in caplog.records)


def test_ubc_evict_paths_noop_on_linux(monkeypatch, tmp_path):
    """ubc_evict_paths skips the iteration entirely on non-Darwin."""
    monkeypatch.setattr(sys, "platform", "linux")
    payload = tmp_path / "p.bin"
    payload.write_bytes(b"x" * 4096)

    assert ubc_evict_paths([str(payload), str(payload)]) == 0
    assert snapshot()["ubc_evict_calls_total"] == 0


# ---------------------------------------------------------------------
# Error paths -- never raise
# ---------------------------------------------------------------------


@pytest.mark.skipif(
    sys.platform != "darwin", reason="munmap simulation needs Darwin libc"
)
def test_ubc_evict_munmap_failure_is_not_reported_as_success(tmp_path, caplog):
    """A munmap failure must NOT publish the file size as evicted."""
    payload = tmp_path / "p.bin"
    subprocess.run(
        ["dd", "if=/dev/urandom", f"of={payload}", "bs=1m", "count=1", "status=none"],
        check=True,
    )

    real_libc = ubc_module._get_libc()
    assert real_libc is not None

    class _StubLibc:
        def __init__(self, real):
            self.mmap = real.mmap
            self.msync = real.msync

        def munmap(self, addr, size):
            import ctypes
            ctypes.set_errno(22)  # EINVAL
            return -1

    monkey_libc = _StubLibc(real_libc)

    with caplog.at_level(logging.WARNING, logger=ubc_module.logger.name):
        ubc_module._libc = monkey_libc
        try:
            result = ubc_evict(str(payload))
        finally:
            ubc_module._libc = real_libc

    assert result == 0, (
        "munmap failure must NOT report file size as evicted"
    )
    snap = snapshot()
    assert snap["ubc_evict_failed_total"] == 1
    assert snap["ubc_evicted_bytes_total"] == 0
    assert any("munmap" in r.message for r in caplog.records)


@pytest.mark.skipif(sys.platform != "darwin", reason="error path checks libc on Darwin")
def test_ubc_evict_missing_file_returns_zero(tmp_path, caplog):
    """Missing file: returns 0, logs WARNING, counts as failure."""
    missing = tmp_path / "does_not_exist.bin"
    with caplog.at_level(logging.WARNING, logger=ubc_module.logger.name):
        result = ubc_evict(str(missing))
    assert result == 0
    assert any(
        "cannot stat" in r.message or "stat" in r.message for r in caplog.records
    )
    snap = snapshot()
    assert snap["ubc_evict_failed_total"] == 1
    assert snap["ubc_evicted_bytes_total"] == 0


@pytest.mark.skipif(
    sys.platform != "darwin", reason="zero-byte path checks Darwin libc"
)
def test_ubc_evict_zero_byte_file_returns_zero(tmp_path, caplog):
    """Zero-byte file: returns 0 cleanly, logs DEBUG, NOT a failure."""
    empty = tmp_path / "empty.bin"
    empty.touch()
    with caplog.at_level(logging.DEBUG, logger=ubc_module.logger.name):
        result = ubc_evict(str(empty))
    assert result == 0
    snap = snapshot()
    assert snap["ubc_evict_calls_total"] == 1
    assert snap["ubc_evict_failed_total"] == 0
    assert any("empty" in r.message for r in caplog.records)


# ---------------------------------------------------------------------
# Prometheus rendering
# ---------------------------------------------------------------------


def test_render_prometheus_lines_shape():
    """Three lines: HELP, TYPE, single sample with the path_kind label."""
    lines = render_prometheus_lines()
    assert len(lines) == 3
    assert lines[0].startswith("# HELP fusion_mlx_ubc_evicted_bytes_total ")
    assert lines[1] == "# TYPE fusion_mlx_ubc_evicted_bytes_total counter"
    assert re.fullmatch(
        r'fusion_mlx_ubc_evicted_bytes_total\{path_kind="safetensors"\} \d+',
        lines[2],
    )


def test_render_prometheus_lines_zero_by_default():
    """Fresh process: sample is 0 -- series MUST exist even before any load."""
    lines = render_prometheus_lines()
    assert lines[2].endswith(" 0")


# ---------------------------------------------------------------------
# Counter monotonicity
# ---------------------------------------------------------------------


@pytest.mark.skipif(
    sys.platform != "darwin", reason="Counter ticks only on Darwin successes"
)
def test_counter_monotonic_across_calls(tmp_path):
    """Counter accumulates across successive successful evictions."""
    files = []
    for i in range(3):
        p = tmp_path / f"s{i}.bin"
        subprocess.run(
            ["dd", "if=/dev/urandom", f"of={p}", "bs=1m", "count=1", "status=none"],
            check=True,
        )
        files.append(p)

    total = ubc_evict_paths([str(p) for p in files])
    assert total == 3 * 1024 * 1024
    snap = snapshot()
    assert snap["ubc_evicted_bytes_total"] == 3 * 1024 * 1024
    assert snap["ubc_evict_calls_total"] == 3
    assert snap["ubc_evict_failed_total"] == 0


# ---------------------------------------------------------------------
# Reset hook is test-only -- production callers must NOT reset.
# ---------------------------------------------------------------------


def test_reset_for_tests_clears_state(monkeypatch):
    """reset_for_tests zeros every counter."""
    monkeypatch.setattr(sys, "platform", "linux")
    payload = Path("/tmp/non_existent_for_reset_test")
    ubc_evict(str(payload))

    assert snapshot()["ubc_evict_calls_total"] == 1
    reset_for_tests()
    assert snapshot() == {
        "ubc_evicted_bytes_total": 0,
        "ubc_evict_calls_total": 0,
        "ubc_evict_failed_total": 0,
    }
