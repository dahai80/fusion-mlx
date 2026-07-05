# SPDX-License-Identifier: Apache-2.0
"""Unit tests for stub modules: _parent_watchdog, _download_gate.

These modules are stubs in the fusion-mlx build. Tests verify their
no-op interface contract so they are not accidentally broken by refactors.
"""

from __future__ import annotations

import logging

from fusion_mlx._download_gate import (
    confirm_or_abort,
    estimate_repo_size_bytes,
    is_repo_cached,
)
from fusion_mlx._parent_watchdog import install_parent_watchdog, resolve_expected_ppid

# =========================================================================
# _parent_watchdog stub
# =========================================================================


class TestParentWatchdogStub:
    """Parent watchdog is a no-op stub in this build."""

    def test_install_does_not_raise(self):
        # Should not raise regardless of arguments
        install_parent_watchdog(12345)
        install_parent_watchdog(0)
        install_parent_watchdog(-1)

    def test_install_with_interval_does_not_raise(self):
        install_parent_watchdog(12345, interval=5.0)

    def test_resolve_returns_same_value(self):
        assert resolve_expected_ppid(42) == 42
        assert resolve_expected_ppid(0) == 0

    def test_resolve_none_returns_none(self):
        assert resolve_expected_ppid(None) is None

    def test_install_logs_debug_message(self, caplog):
        caplog.set_level(logging.DEBUG)
        install_parent_watchdog(99999)
        assert any("Parent watchdog skipped" in r.message for r in caplog.records)


# =========================================================================
# _download_gate stub
# =========================================================================


class TestDownloadGateStub:
    """Download gate is a no-op stub in this build."""

    def test_confirm_or_abort_does_not_raise(self):
        confirm_or_abort("test-model")
        confirm_or_abort("test-model", estimated_bytes=1_000_000_000)

    def test_estimate_repo_size_returns_none(self):
        result = estimate_repo_size_bytes("any-model-name")
        assert result is None

    def test_estimate_repo_size_empty_string(self):
        result = estimate_repo_size_bytes("")
        assert result is None

    def test_is_repo_cached_returns_false(self):
        result = is_repo_cached("any-model-name")
        assert result is False

    def test_is_repo_cached_empty_string(self):
        result = is_repo_cached("")
        assert result is False
