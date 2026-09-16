# SPDX-License-Identifier: Apache-2.0
"""Unit tests for custom_kernels/lifecycle.py (P0底座)."""

import logging

import pytest

from fusion_mlx.custom_kernels.lifecycle import logger as _lifecycle_logger
from fusion_mlx.custom_kernels.lifecycle import with_kernel_scope


@pytest.fixture
def fake_mx(monkeypatch):
    """Inject a fake mlx.core into the lifecycle module namespace."""
    calls = {"clear_cache": 0}

    class _MX:
        @staticmethod
        def get_active_memory():
            return 1000

        @staticmethod
        def clear_cache():
            calls["clear_cache"] += 1

    import fusion_mlx.custom_kernels.lifecycle as mod

    monkeypatch.setattr(mod, "_safe_get_active_memory", lambda: 1000)
    monkeypatch.setattr(
        mod,
        "_safe_clear_cache",
        lambda: calls.__setitem__("clear_cache", calls["clear_cache"] + 1),
    )
    return calls


def test_scope_calls_clear_cache_on_exit(fake_mx):
    with with_kernel_scope("test"):
        pass
    assert fake_mx["clear_cache"] == 1


def test_scope_clears_cache_on_exception(fake_mx):
    with pytest.raises(ValueError):
        with with_kernel_scope("boom"):
            raise ValueError("inner")
    assert fake_mx["clear_cache"] == 1


def test_scope_clear_cache_failure_swallowed(monkeypatch):
    """If the underlying mx.clear_cache raises, _safe_clear_cache swallows
    it so the scope's finally never masks the original path."""
    # Inject a fake mlx.core whose clear_cache raises into the module's
    # _safe_clear_cache (it imports mlx lazily) by patching sys.modules.
    import sys
    import types

    import fusion_mlx.custom_kernels.lifecycle as mod

    def _boom():
        raise RuntimeError("clear failed")

    fake_mx = types.ModuleType("mlx.core")
    fake_mx.clear_cache = _boom
    fake_mx.get_active_memory = lambda: 0
    fake_mx_parent = types.ModuleType("mlx")
    fake_mx_parent.core = fake_mx
    monkeypatch.setitem(sys.modules, "mlx", fake_mx_parent)
    monkeypatch.setitem(sys.modules, "mlx.core", fake_mx)
    # _safe_clear_cache must not raise
    mod._safe_clear_cache()
    # and scope must complete cleanly
    with with_kernel_scope("bad"):
        pass


def test_scope_logs_baseline_peak(caplog, fake_mx):
    # #0916: under the full suite a prior test leaves the root logger (or this
    # module's logger) above DEBUG, so caplog.at_level("DEBUG") on root alone
    # doesn't capture the module-level DEBUG record. Pin the specific logger's
    # level via caplog.set_level so the record is captured regardless of
    # cross-module logging pollution.
    with caplog.at_level(logging.DEBUG, logger=_lifecycle_logger.name):
        with with_kernel_scope("logged"):
            pass
    assert any("kernel_scope[logged]" in r.message for r in caplog.records)
