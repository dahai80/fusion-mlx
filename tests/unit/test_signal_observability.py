# SPDX-License-Identifier: Apache-2.0
"""Unit tests for fusion_mlx._signal_observability — signal handler chain.

Pure-logic surface: _signal_name, _on_signal (prior callable/SIG_DFL/SIGHUP paths),
install_signal_observability (non-main-thread skip, faulthandler enable,
per-signal install + idempotent), _reset_for_tests, _get_installed_handlers.
"""

from __future__ import annotations

import signal
import threading
from unittest.mock import MagicMock, patch

import pytest

from fusion_mlx import _signal_observability as mod


class TestSignalName:
    def test_known_signal(self):
        assert mod._signal_name(signal.SIGTERM) == "SIGTERM"

    def test_unknown_returns_fallback(self):
        # Use a large unlikely signal number
        result = mod._signal_name(99999)
        assert "signal 99999" in result


class TestOnSignal:
    def test_callable_prior_chains(self, monkeypatch):
        monkeypatch.setattr(mod, "_prior_handlers", {15: lambda s, f: None})
        with patch("faulthandler.dump_traceback"):
            mod._on_signal(15, None)  # SIGTERM=15

    def test_callable_prior_raises_propagates(self, monkeypatch):
        def bad_prior(s, f):
            raise RuntimeError("boom")

        monkeypatch.setattr(mod, "_prior_handlers", {15: bad_prior})
        with patch("faulthandler.dump_traceback"):
            with pytest.raises(RuntimeError, match="boom"):
                mod._on_signal(15, None)

    def test_sig_dfl_sighup_returns(self, monkeypatch):
        # SIGHUP with SIG_DFL prior → return (no re-raise)
        monkeypatch.setattr(mod, "_prior_handlers", {signal.SIGHUP: signal.SIG_DFL})
        with patch("faulthandler.dump_traceback"):
            mod._on_signal(signal.SIGHUP, None)  # should not raise

    def test_sig_dfl_sigterm_chains(self, monkeypatch):
        # SIGTERM with SIG_DFL → signal.signal + signal.raise_signal
        monkeypatch.setattr(mod, "_prior_handlers", {signal.SIGTERM: signal.SIG_DFL})
        with patch("signal.signal"):
            with patch("signal.raise_signal"):
                with patch("faulthandler.dump_traceback"):
                    mod._on_signal(signal.SIGTERM, None)


class TestInstallSignalObservability:
    def test_non_main_thread_returns_false(self):
        with patch.object(
            threading, "current_thread", return_value=MagicMock(name="worker")
        ):
            assert mod.install_signal_observability() is False

    def test_main_thread_installs_handlers(self, monkeypatch):
        mod._reset_for_tests()
        with patch("faulthandler.enable"):
            with patch("signal.signal", return_value=signal.SIG_DFL):
                result = mod.install_signal_observability(
                    observed_signals=(signal.SIGTERM,)
                )
                assert result is True
        mod._reset_for_tests()

    def test_idempotent_second_call_noop(self, monkeypatch):
        mod._reset_for_tests()
        with patch("faulthandler.enable"):
            with patch("signal.signal", return_value=signal.SIG_DFL):
                mod.install_signal_observability(observed_signals=(signal.SIGTERM,))
                first_handlers = dict(mod._get_installed_handlers())
                mod.install_signal_observability(observed_signals=(signal.SIGTERM,))
                # same handlers, no new install
                assert mod._get_installed_handlers() == first_handlers
        mod._reset_for_tests()

    def test_empty_signals_returns_false(self, monkeypatch):
        mod._reset_for_tests()
        with patch("faulthandler.enable"):
            result = mod.install_signal_observability(observed_signals=())
            assert result is False
        mod._reset_for_tests()

    def test_install_failure_continues(self, monkeypatch):
        mod._reset_for_tests()
        with patch("faulthandler.enable"):
            with patch("signal.signal", side_effect=OSError("no")):
                result = mod.install_signal_observability(
                    observed_signals=(signal.SIGTERM,)
                )
                assert result is False
        mod._reset_for_tests()


class TestResetForTests:
    def test_clears_handlers(self, monkeypatch):
        monkeypatch.setattr(mod, "_prior_handlers", {15: signal.SIG_DFL})
        with patch("signal.signal"):
            mod._reset_for_tests()
            assert mod._get_installed_handlers() == {}


class TestGetInstalledHandlers:
    def test_returns_copy(self, monkeypatch):
        monkeypatch.setattr(mod, "_prior_handlers", {15: signal.SIG_DFL})
        result = mod._get_installed_handlers()
        result[99] = None
        # mutating copy should not affect original
        assert 99 not in mod._prior_handlers
