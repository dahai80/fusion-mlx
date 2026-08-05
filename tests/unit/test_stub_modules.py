# SPDX-License-Identifier: Apache-2.0
"""Unit tests for stub modules: _download_gate.

``_download_gate`` is a no-op stub in this build; tests pin its no-op
interface so it is not accidentally broken by refactors.

Note: ``_parent_watchdog`` was a stub when this file was created but is
now a real implementation (commit d152bb9). Its contract is covered by
``test_parent_watchdog.py``. The former ``TestParentWatchdogStub`` class
was removed because it installed the watchdog with a bogus ppid and no
``on_orphan`` mock, so the real implementation's default orphan callback
SIGTERM-killed the test runner. See ``test_parent_watchdog.py`` for the
correct install-with-mock pattern.
"""

from __future__ import annotations

from fusion_mlx._download_gate import (
    confirm_or_abort,
    estimate_repo_size_bytes,
    is_repo_cached,
)

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
