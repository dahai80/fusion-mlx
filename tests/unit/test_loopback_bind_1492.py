# SPDX-License-Identifier: Apache-2.0
# #1492 iron law: fusion-mlx binds loopback forever. A non-loopback bind
# configuration must fail visibly at startup with an actionable error
# naming the gateway alternative; loopback / UDS binds are unaffected.

import types

import pytest

from fusion_mlx.config import (
    LoopbackBindError,
    ServerConfig,
    enforce_loopback_bind,
)


class TestServerConfigInvariant:
    def test_default_host_is_loopback(self):
        assert ServerConfig().host == "127.0.0.1"

    def test_non_loopback_rejected(self):
        with pytest.raises(LoopbackBindError) as ei:
            ServerConfig(host="0.0.0.0")
        # Actionable: names the gateway alternative.
        assert "fusion-gateway" in str(ei.value)

    def test_empty_host_wildcard_rejected(self):
        # Empty host = wildcard-bind alias; fail closed.
        with pytest.raises(LoopbackBindError):
            ServerConfig(host="")

    def test_lan_ip_rejected(self):
        with pytest.raises(LoopbackBindError):
            ServerConfig(host="192.168.1.5")

    def test_unresolvable_hostname_rejected(self):
        # Fail closed: only literal loopback names are accepted.
        with pytest.raises(LoopbackBindError):
            ServerConfig(host="my-mac.local")

    def test_loopback_hosts_allowed(self):
        for host in ("127.0.0.1", "::1", "localhost", "127.0.0.2"):
            assert ServerConfig(host=host).host == host

    def test_uds_mode_allowed(self):
        assert ServerConfig(host="unix:/tmp/fusion-mlx.sock").host.startswith("unix:")


class TestEnforceLoopbackBind:
    def test_none_passes(self):
        # None = "no host override" (e.g. listen-fd arm callers); the
        # caller either binds via fd (checked separately) or uses the
        # default loopback config.
        enforce_loopback_bind(None)

    def test_gateway_named_in_error(self):
        with pytest.raises(LoopbackBindError, match="gateway"):
            enforce_loopback_bind("0.0.0.0")


class TestServeCommandPreflight:
    # serve_command must fail BEFORE any model download / preflight work.
    # Build the namespace the same way test_serve_uds does (full argparse
    # via cli.main) so the prologue sees a real serve namespace.
    def test_serve_command_non_loopback_fails_early(self, monkeypatch, tmp_path):
        import sys
        from unittest.mock import patch

        from fusion_mlx import cli

        argv = ["fusion-mlx", "serve", "qwen3.5-4b-4bit", "--host", "0.0.0.0"]
        captured = []
        with (
            patch.object(sys, "argv", argv),
            patch.object(cli, "serve_command", side_effect=captured.append),
        ):
            cli.main()
        ns = captured[0]

        def _no_boot(*_a, **_k):
            raise AssertionError("uvicorn must not run for a refused bind")

        monkeypatch.setattr("uvicorn.run", _no_boot)
        with pytest.raises(LoopbackBindError):
            cli.serve_command(ns)


class TestRunUvicornChokepoint:
    # _run_uvicorn is the last chokepoint before uvicorn for every CLI
    # serve path; verify it refuses before uvicorn.run is reached.
    def test_non_loopback_refused_before_uvicorn(self, monkeypatch):
        from fusion_mlx import _cli_base

        def _boom(*_a, **_k):
            raise AssertionError("uvicorn.run must not be reached")

        monkeypatch.setattr("uvicorn.run", _boom)
        ns = types.SimpleNamespace(host="0.0.0.0", port=12345, listen_fd=None)
        with pytest.raises(LoopbackBindError):
            _cli_base._run_uvicorn(app=None, args=ns, log_level="INFO")

    def test_loopback_reaches_dispatch(self, monkeypatch):
        from fusion_mlx import _cli_base

        called = {}
        monkeypatch.setattr(
            "uvicorn.run", lambda *a, **k: called.setdefault("ran", True)
        )
        ns = types.SimpleNamespace(host="127.0.0.1", port=12345, listen_fd=None)
        _cli_base._run_uvicorn(app=None, args=ns, log_level="INFO")
        assert called.get("ran") is True
