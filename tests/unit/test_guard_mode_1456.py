# SPDX-License-Identifier: Apache-2.0
# Tests for #1456 guard mode: SIGTERM deferral while requests are in flight.

from __future__ import annotations

import signal
import threading

import pytest

from fusion_mlx import _parent_watchdog as pwd


@pytest.fixture(autouse=True)
def _reset_guard_state(monkeypatch):
    pwd._SHUTDOWN_EVENT.clear()
    pwd._SHUTDOWN_COMPLETE.clear()
    pwd._GUARD_DEFERRING.clear()
    pwd._busy_probes.clear()
    monkeypatch.setattr(pwd, "_GUARD_FORCE", False)
    yield
    pwd._SHUTDOWN_EVENT.clear()
    pwd._SHUTDOWN_COMPLETE.clear()
    pwd._GUARD_DEFERRING.clear()
    pwd._busy_probes.clear()


class TestGuardConfig:
    def test_guard_mode_env_values(self, monkeypatch):
        for raw, expected in [
            ("1", True),
            ("true", True),
            ("YES", True),
            ("on", True),
            ("0", False),
            ("", False),
            ("off", False),
        ]:
            monkeypatch.setenv(pwd.ENV_GUARD_MODE, raw)
            assert pwd._guard_mode_enabled() is expected, raw

    def test_guard_disabled_without_env(self, monkeypatch):
        monkeypatch.delenv(pwd.ENV_GUARD_MODE, raising=False)
        assert pwd._guard_mode_enabled() is False

    def test_max_defer_default_when_unset(self, monkeypatch):
        monkeypatch.delenv(pwd.ENV_GUARD_MAX_DEFER_SEC, raising=False)
        assert pwd._guard_max_defer_sec() == pwd._GUARD_DEFAULT_MAX_DEFER_SEC

    def test_max_defer_parses_valid_value(self, monkeypatch):
        monkeypatch.setenv(pwd.ENV_GUARD_MAX_DEFER_SEC, "60")
        assert pwd._guard_max_defer_sec() == 60.0

    def test_max_defer_invalid_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv(pwd.ENV_GUARD_MAX_DEFER_SEC, "not-a-number")
        assert pwd._guard_max_defer_sec() == pwd._GUARD_DEFAULT_MAX_DEFER_SEC

    def test_max_defer_negative_clamped_to_zero(self, monkeypatch):
        monkeypatch.setenv(pwd.ENV_GUARD_MAX_DEFER_SEC, "-5")
        assert pwd._guard_max_defer_sec() == 0.0


class TestBusyProbes:
    def test_no_probes_means_not_busy(self):
        assert pwd._jobs_in_flight() is False

    def test_true_probe_wins(self):
        pwd.register_busy_probe("a", lambda: False)
        pwd.register_busy_probe("b", lambda: True)
        assert pwd._jobs_in_flight() is True

    def test_false_probes_not_busy(self):
        pwd.register_busy_probe("a", lambda: False)
        pwd.register_busy_probe("b", lambda: None)
        assert pwd._jobs_in_flight() is False

    def test_raising_probe_counts_as_busy(self):
        def _boom():
            raise RuntimeError("probe broke")

        pwd.register_busy_probe("bad", _boom)
        assert pwd._jobs_in_flight() is True

    def test_unregister_removes_probe(self):
        pwd.register_busy_probe("a", lambda: True)
        pwd.unregister_busy_probe("a")
        assert pwd._jobs_in_flight() is False

    def test_unregister_missing_probe_is_noop(self):
        pwd.unregister_busy_probe("never-registered")

    def test_non_callable_rejected(self):
        with pytest.raises(TypeError):
            pwd.register_busy_probe("bad", "not-callable")


class TestDeferralLoop:
    def test_drain_then_self_sigterm(self, monkeypatch):
        calls = []
        monkeypatch.setattr(pwd.time, "sleep", lambda s: None)
        monkeypatch.setattr(pwd.os, "getpid", lambda: 12345)

        def fake_kill(pid, sig):
            calls.append((pid, sig))

        monkeypatch.setattr(pwd.os, "kill", fake_kill)
        flips = {"n": 0}

        def busy():
            flips["n"] += 1
            return flips["n"] <= 2

        pwd.register_busy_probe("job", busy)
        pwd._GUARD_DEFERRING.set()
        pwd._run_guard_deferral("SIGTERM", 900.0)
        assert calls == [(12345, signal.SIGTERM)]
        assert pwd._GUARD_DEFERRING.is_set() is False
        assert pwd._GUARD_FORCE is True
        assert flips["n"] == 3

    def test_max_defer_expiry_forces_shutdown(self, monkeypatch):
        calls = []
        monkeypatch.setattr(pwd.time, "sleep", lambda s: None)
        monkeypatch.setattr(pwd.os, "getpid", lambda: 12345)
        monkeypatch.setattr(pwd.os, "kill", lambda pid, sig: calls.append((pid, sig)))
        pwd.register_busy_probe("stuck", lambda: True)
        pwd._GUARD_DEFERRING.set()
        pwd._run_guard_deferral("SIGTERM", 0.0)
        assert calls == [(12345, signal.SIGTERM)]
        assert pwd._GUARD_FORCE is True


class TestSignalHandlerGuard:
    @pytest.fixture()
    def handler(self, monkeypatch):
        # install_signal_handlers chains the CURRENT handlers; arm harmless
        # no-ops first so the chained previous never raises (SIGINT's
        # default_int_handler raises KeyboardInterrupt, which the handler's
        # except-Exception does not catch), capture the installed handler,
        # then restore the originals.
        def _noop(signum, frame):
            pass

        prev_term = signal.getsignal(signal.SIGTERM)
        prev_int = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGTERM, _noop)
        signal.signal(signal.SIGINT, _noop)
        pwd.install_signal_handlers()
        h = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, prev_term or signal.SIG_DFL)
        signal.signal(signal.SIGINT, prev_int or signal.SIG_DFL)
        return h

    def test_shutdown_immediate_when_guard_off(self, handler, monkeypatch):
        monkeypatch.delenv(pwd.ENV_GUARD_MODE, raising=False)
        pwd.register_busy_probe("busy", lambda: True)
        handler(signal.SIGTERM, None)
        assert pwd._SHUTDOWN_EVENT.is_set() is True
        assert pwd._GUARD_DEFERRING.is_set() is False

    def test_shutdown_immediate_when_not_busy(self, handler, monkeypatch):
        monkeypatch.setenv(pwd.ENV_GUARD_MODE, "1")
        pwd.register_busy_probe("idle", lambda: False)
        handler(signal.SIGTERM, None)
        assert pwd._SHUTDOWN_EVENT.is_set() is True
        assert pwd._GUARD_DEFERRING.is_set() is False

    def test_sigterm_defers_when_busy(self, handler, monkeypatch):
        monkeypatch.setenv(pwd.ENV_GUARD_MODE, "1")
        started = threading.Event()
        monkeypatch.setattr(
            pwd,
            "_start_guard_deferral",
            lambda sig: started.set(),
        )
        pwd.register_busy_probe("busy", lambda: True)
        handler(signal.SIGTERM, None)
        assert started.wait(2) is True
        assert pwd._SHUTDOWN_EVENT.is_set() is False
        assert pwd._GUARD_DEFERRING.is_set() is False

    def test_second_sigterm_overrides_deferral(self, handler, monkeypatch):
        monkeypatch.setenv(pwd.ENV_GUARD_MODE, "1")
        monkeypatch.setattr(pwd, "_start_guard_deferral", lambda sig: None)
        pwd.register_busy_probe("busy", lambda: True)
        handler(signal.SIGTERM, None)
        assert pwd._SHUTDOWN_EVENT.is_set() is False
        # Simulate an active deferral, then the operator override.
        pwd._GUARD_DEFERRING.set()
        handler(signal.SIGTERM, None)
        assert pwd._SHUTDOWN_EVENT.is_set() is True

    def test_sigint_never_defers(self, handler, monkeypatch):
        monkeypatch.setenv(pwd.ENV_GUARD_MODE, "1")
        started = threading.Event()
        monkeypatch.setattr(
            pwd,
            "_start_guard_deferral",
            lambda sig: started.set(),
        )
        pwd.register_busy_probe("busy", lambda: True)

        def _noop(signum, frame):
            pass

        prev_int = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, _noop)
        pwd.install_signal_handlers()
        int_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, prev_int or signal.SIG_DFL)
        int_handler(signal.SIGINT, None)
        assert started.is_set() is False
        assert pwd._SHUTDOWN_EVENT.is_set() is True

    def test_force_flag_skips_deferral(self, handler, monkeypatch):
        monkeypatch.setenv(pwd.ENV_GUARD_MODE, "1")
        monkeypatch.setattr(pwd, "_GUARD_FORCE", True)
        started = threading.Event()
        monkeypatch.setattr(
            pwd,
            "_start_guard_deferral",
            lambda sig: started.set(),
        )
        pwd.register_busy_probe("busy", lambda: True)
        handler(signal.SIGTERM, None)
        assert started.is_set() is False
        assert pwd._SHUTDOWN_EVENT.is_set() is True


class TestEnginePoolProbe:
    def test_any_active_requests_false_when_idle(self):
        from fusion_mlx.pool.engine_pool import EnginePool

        pool = EnginePool.__new__(EnginePool)
        pool._entries = {}
        assert pool.any_active_requests() is False

    def test_any_active_requests_true_when_in_use(self):
        from fusion_mlx.pool.engine_pool import EnginePool

        pool = EnginePool.__new__(EnginePool)
        entry = type("Entry", (), {"in_use": 1, "engine": None})()
        pool._entries = {"m": entry}
        assert pool.any_active_requests() is True


class TestServeFlagWiring:
    def _cli_serve_source(self) -> str:
        from pathlib import Path

        path = (
            Path(__file__).resolve().parents[2]
            / "fusion_mlx"
            / "cli_serve"
            / "serve_command.py"
        )
        return path.read_text()

    def test_serve_command_stamps_guard_env(self):
        source = self._cli_serve_source()
        idx_serve = source.find("def serve_command(")
        assert idx_serve != -1
        assert (
            'os.environ.setdefault("FUSION_MLX_GUARD_MODE", "1")' in source[idx_serve:]
        ), (
            "#1456: serve_command no longer stamps FUSION_MLX_GUARD_MODE "
            "from --guard-mode. Restore the env stamp."
        )

    def test_cli_declares_guard_mode_flag(self):
        from pathlib import Path

        path = Path(__file__).resolve().parents[2] / "fusion_mlx" / "cli.py"
        source = path.read_text()
        assert (
            '"--guard-mode"' in source
        ), "#1456: --guard-mode serve flag missing from cli.py"
