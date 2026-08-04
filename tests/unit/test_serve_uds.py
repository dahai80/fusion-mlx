# SPDX-License-Identifier: Apache-2.0
# Tests for ``fusion-mlx serve --host unix:/path`` UDS listen mode (#351).
#
# UDS listen mode is physical (transport-layer) isolation on top of the
# #349/#350 auth chain: with ``--host unix:/run/fusion-mlx.sock`` MLX binds
# an AF_UNIX socket with owner-only (0600) permissions, so only a process
# with filesystem access to the socket can reach it. No TCP port is opened
# in UDS mode. TCP ``--host 127.0.0.1`` remains the default.
#
# These tests pin the contract:
#   * ``_uds_path_from_host`` parses the ``unix:`` prefix (valid -> path,
#     bare ``unix:`` -> ValueError, everything else -> None).
#   * ``_prepare_uds_socket`` creates a real AF_UNIX socket, chmods it 0600
#     BEFORE listen, unlinks a stale path first, and returns the fd.
#   * ``_cleanup_uds_socket`` unlinks the path and swallows ENOENT.
#   * ``_run_uvicorn`` dispatches ``uvicorn.run(app, fd=..., ...)`` in UDS
#     mode (no host/port) and cleans up the socket on every exit path.
#   * ``serve_command`` skips the TCP port preflight in UDS mode, stamps
#     ``ServerConfig.bind_uds``, and resets stale bind fields across calls.

from __future__ import annotations

import os
import shutil
import stat
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from fusion_mlx import cli
from fusion_mlx._cli_base import (
    _cleanup_uds_socket,
    _prepare_uds_socket,
    _run_uvicorn,
    _uds_path_from_host,
)

# ---------------------------------------------------------------------------
# Helpers - mirror test_serve_listen_fd.py: drive ``cli.main()`` and capture
# the resolved Namespace via a ``serve_command`` stub (no model boot).
# ---------------------------------------------------------------------------


def _capture_serve_args(argv):
    captured = []
    with (
        patch.object(sys, "argv", argv),
        patch.object(cli, "serve_command", side_effect=captured.append),
    ):
        cli.main()
    return captured


def _minimal_serve_ns(**overrides):
    captured = []
    argv = ["rapid-mlx", "serve", "qwen3.5-4b-4bit"]
    for k, v in overrides.items():
        if k == "listen_fd":
            argv += ["--listen-fd", str(v)]
        elif k == "port":
            argv += ["--port", str(v)]
        elif k == "host":
            argv += ["--host", v]
    with (
        patch.object(sys, "argv", argv),
        patch.object(cli, "serve_command", side_effect=captured.append),
    ):
        cli.main()
    return captured[0]


def _capture_uvicorn_run(monkeypatch):
    captured = {}

    def fake_run(app, **kwargs):
        captured["app"] = app
        captured.update(kwargs)

    import uvicorn

    monkeypatch.setattr(uvicorn, "run", fake_run)
    return captured


def _free_tcp_port(host="127.0.0.1"):
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]


@pytest.fixture
def short_sock_dir():
    # AF_UNIX socket paths are capped at ~104 chars (SUN_LEN). macOS
    # pytest tmp_path (/private/var/folders/...) exceeds that, so create
    # the socket under /tmp where the path stays well under the limit.
    d = Path(tempfile.mkdtemp(prefix="uds_", dir="/tmp"))
    yield d
    shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# _uds_path_from_host - prefix parsing
# ---------------------------------------------------------------------------


def test_uds_path_from_host_parses_unix_prefix():
    assert _uds_path_from_host("unix:/run/fusion-mlx.sock") == "/run/fusion-mlx.sock"
    assert _uds_path_from_host("unix:relative.sock") == "relative.sock"


def test_uds_path_from_host_none_for_tcp_hosts():
    assert _uds_path_from_host("127.0.0.1") is None
    assert _uds_path_from_host("0.0.0.0") is None
    assert _uds_path_from_host("localhost") is None


def test_uds_path_from_host_none_for_non_string():
    assert _uds_path_from_host(None) is None
    assert _uds_path_from_host(123) is None


def test_uds_path_from_host_none_for_empty():
    assert _uds_path_from_host("") is None


def test_uds_path_from_host_bare_unix_prefix_raises():
    # A bare ``unix:`` (no path) must raise so a typo does not silently
    # fall back to TCP - that would defeat the physical-isolation goal.
    with pytest.raises(ValueError, match="socket path"):
        _uds_path_from_host("unix:")


def test_serve_host_unix_parses_to_host_attr():
    captured = _capture_serve_args(
        ["rapid-mlx", "serve", "qwen3.5-4b-4bit", "--host", "unix:/run/x.sock"]
    )
    assert len(captured) == 1
    assert captured[0].host == "unix:/run/x.sock"


# ---------------------------------------------------------------------------
# _prepare_uds_socket - real AF_UNIX, 0600 before listen, stale unlink
# ---------------------------------------------------------------------------


def test_prepare_uds_socket_creates_0600_socket(short_sock_dir):
    sock_path = short_sock_dir / "uds.sock"
    fd = _prepare_uds_socket(str(sock_path))
    try:
        assert sock_path.exists()
        assert stat.S_IMODE(os.stat(sock_path).st_mode) == 0o600
        assert isinstance(fd, int)
        assert fd > 2
    finally:
        os.close(fd)


def test_prepare_uds_socket_unlinks_stale_path(short_sock_dir):
    # A leftover socket file from a crashed process must not block bind.
    sock_path = short_sock_dir / "stale.sock"
    sock_path.write_text("stale")
    assert sock_path.exists()
    fd = _prepare_uds_socket(str(sock_path))
    try:
        assert sock_path.exists()
        assert stat.S_IMODE(os.stat(sock_path).st_mode) == 0o600
    finally:
        os.close(fd)


def test_prepare_uds_socket_replaces_existing_socket(short_sock_dir):
    # Re-preparing the same path (e.g. restart) replaces cleanly.
    sock_path = short_sock_dir / "reuse.sock"
    fd1 = _prepare_uds_socket(str(sock_path))
    os.close(fd1)
    fd2 = _prepare_uds_socket(str(sock_path))
    try:
        assert sock_path.exists()
        assert stat.S_IMODE(os.stat(sock_path).st_mode) == 0o600
    finally:
        os.close(fd2)


# ---------------------------------------------------------------------------
# _cleanup_uds_socket - unlink + ENOENT-tolerant
# ---------------------------------------------------------------------------


def test_cleanup_uds_socket_removes_path(short_sock_dir):
    sock_path = short_sock_dir / "gone.sock"
    fd = _prepare_uds_socket(str(sock_path))
    _cleanup_uds_socket(str(sock_path), fd=fd)
    assert not sock_path.exists()


def test_cleanup_uds_socket_missing_path_no_raise(short_sock_dir):
    # ENOENT on cleanup (e.g. already removed) must be swallowed.
    _cleanup_uds_socket(str(short_sock_dir / "never.sock"))


def test_cleanup_uds_socket_closed_fd_no_raise(short_sock_dir):
    # Best-effort fd close: an already-closed fd must not raise.
    sock_path = short_sock_dir / "fd.sock"
    fd = _prepare_uds_socket(str(sock_path))
    os.close(fd)
    _cleanup_uds_socket(str(sock_path), fd=fd)
    assert not sock_path.exists()


# ---------------------------------------------------------------------------
# _run_uvicorn - UDS dispatch arm (direct helper exercise)
# ---------------------------------------------------------------------------


def test_run_uvicorn_dispatches_fd_in_uds_mode(monkeypatch, short_sock_dir):
    captured = _capture_uvicorn_run(monkeypatch)
    sock_path = short_sock_dir / "run.sock"
    ns = _minimal_serve_ns(host=f"unix:{sock_path}", port=9000)
    sentinel_app = object()
    _run_uvicorn(sentinel_app, ns, "info")

    assert captured.get("app") is sentinel_app
    assert isinstance(captured.get("fd"), int), f"expected fd=, got {captured!r}"
    assert "host" not in captured
    assert "port" not in captured
    assert captured.get("log_level") == "info"
    assert captured.get("timeout_keep_alive") == 30
    # finally-block cleanup must have removed the socket.
    assert not sock_path.exists()


def test_run_uvicorn_tcp_host_when_uds_unset(monkeypatch):
    # Regression guard: a normal TCP host must NOT take the UDS arm.
    captured = _capture_uvicorn_run(monkeypatch)
    ns = _minimal_serve_ns(host="127.0.0.1", port=9000)
    _run_uvicorn(object(), ns, "info")

    assert captured.get("host") == "127.0.0.1"
    assert captured.get("port") == 9000
    assert "fd" not in captured


# ---------------------------------------------------------------------------
# Server.run() - legacy ``python -m fusion_mlx.server`` entry UDS dispatch.
# Server.__init__ boots the full app/engine, so drive run() as an unbound
# method on a lightweight fake-self (run() only reads self.config + self.app).
# ---------------------------------------------------------------------------


def test_server_run_dispatches_fd_in_uds_mode(monkeypatch, short_sock_dir):
    from types import SimpleNamespace

    from fusion_mlx.server import Server

    uds_path = str(short_sock_dir / "srv.sock")
    fake_self = SimpleNamespace(
        config=SimpleNamespace(host=f"unix:{uds_path}", port=1234),
        app=object(),
    )
    captured = _capture_uvicorn_run(monkeypatch)

    Server.run(fake_self)

    assert isinstance(captured.get("fd"), int)
    assert "host" not in captured
    assert "port" not in captured
    assert not os.path.exists(uds_path)


def test_server_run_tcp_when_host_is_ip(monkeypatch):
    from types import SimpleNamespace

    from fusion_mlx.server import Server

    fake_self = SimpleNamespace(
        config=SimpleNamespace(host="127.0.0.1", port=8399),
        app=object(),
    )
    captured = _capture_uvicorn_run(monkeypatch)

    Server.run(fake_self)

    assert captured.get("host") == "127.0.0.1"
    assert captured.get("port") == 8399
    assert "fd" not in captured


# ---------------------------------------------------------------------------
# serve_command - preflight skip + config wiring in UDS mode
# ---------------------------------------------------------------------------


@pytest.fixture
def stub_heavy_serve_deps(monkeypatch):
    # Stub the heavyweight prologue of ``serve_command`` so behavioral
    # tests reach the ``uvicorn.run`` call site without downloading a
    # model, importing mlx, or booting an engine. Mirror the
    # test_serve_listen_fd.py fixture; extend here if a new heavy step
    # surfaces as an ImportError/AttributeError.
    # serve_command resolves these as bare names in fusion_mlx.cli_serve
    # (where they are defined), NOT in fusion_mlx.cli - stubbing cli_mod
    # was a stale target left by the cli->cli_serve extraction and never
    # intercepted the real call. CORS was renamed configure_cors ->
    # configure_cors_from_env (returns the origins list).
    from fusion_mlx import _version_check
    from fusion_mlx import cli_serve as cli_serve_mod
    from fusion_mlx import server as server_mod

    monkeypatch.setattr(_version_check, "prompt_upgrade_if_available", lambda: False)
    monkeypatch.setattr(_version_check, "print_staleness_warning_if_any", lambda: None)
    monkeypatch.setattr(cli_serve_mod, "_ensure_model_downloaded", lambda model: None)
    monkeypatch.setattr(cli_serve_mod, "_check_memory_capacity", lambda *a, **kw: None)
    monkeypatch.setattr(cli_serve_mod, "_check_disk_space", lambda *a, **kw: None)
    monkeypatch.setattr(server_mod, "configure_logging", lambda level: "info")
    monkeypatch.setattr(server_mod, "load_model", lambda *a, **kw: None)
    monkeypatch.setattr(server_mod, "configure_cors_from_env", lambda *a, **kw: [])
    from fusion_mlx.middleware import auth as auth_mod

    monkeypatch.setattr(auth_mod, "configure_rate_limiter", lambda *a, **kw: None)
    return monkeypatch


def test_serve_command_dispatches_fd_in_uds_mode(stub_heavy_serve_deps, short_sock_dir):
    captured = _capture_uvicorn_run(stub_heavy_serve_deps)
    sock_path = short_sock_dir / "serve.sock"
    ns = _minimal_serve_ns(host=f"unix:{sock_path}", port=_free_tcp_port())
    cli.serve_command(ns)

    assert isinstance(captured.get("fd"), int), f"expected fd=, got {captured!r}"
    assert "host" not in captured
    assert "port" not in captured
    from fusion_mlx.config import get_config

    cfg = get_config()
    assert cfg.bind_uds == str(sock_path)
    assert cfg.bind_host is None
    assert cfg.bind_port is None
    assert cfg.bind_listen_fd is None
    # cleanup after the uvicorn.run stub return.
    assert not sock_path.exists()


def test_serve_command_skips_port_preflight_in_uds_mode(
    stub_heavy_serve_deps, short_sock_dir
):
    # With ``--host unix:...`` set, ``serve_command`` MUST skip the TCP
    # ``host``/``port`` bind preflight. UDS mode has no TCP port to probe,
    # so running the preflight against a blocked port would sys.exit(1)
    # before reaching uvicorn.run. Passes only when correctly skipped.
    import socket

    captured = _capture_uvicorn_run(stub_heavy_serve_deps)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as blocker:
        blocker.bind(("127.0.0.1", 0))
        blocker.listen(1)
        blocked_port = blocker.getsockname()[1]

        sock_path = short_sock_dir / "skip.sock"
        ns = _minimal_serve_ns(host=f"unix:{sock_path}", port=blocked_port)
        cli.serve_command(ns)

    assert isinstance(captured.get("fd"), int)
    assert "host" not in captured
    assert "port" not in captured


def test_serve_command_resets_stale_bind_fields_with_uds(
    stub_heavy_serve_deps, short_sock_dir
):
    # The singleton ServerConfig persists across in-process serve_command
    # calls. A prior TCP host/port stash must NOT leak into a subsequent
    # UDS run (and vice-versa) - otherwise the lifespan banner reports a
    # phantom listener.
    _capture_uvicorn_run(stub_heavy_serve_deps)
    from fusion_mlx.config import get_config

    cfg = get_config()

    # First call: TCP host/port.
    port_a = _free_tcp_port()
    cli.serve_command(_minimal_serve_ns(host="127.0.0.1", port=port_a))
    assert (cfg.bind_host, cfg.bind_port, cfg.bind_listen_fd, cfg.bind_uds) == (
        "127.0.0.1",
        port_a,
        None,
        None,
    )

    # Second call: UDS. Prior host/port must clear; bind_uds set.
    sock_path = short_sock_dir / "reset.sock"
    cli.serve_command(_minimal_serve_ns(host=f"unix:{sock_path}", port=port_a))
    assert (cfg.bind_host, cfg.bind_port, cfg.bind_listen_fd, cfg.bind_uds) == (
        None,
        None,
        None,
        str(sock_path),
    )

    # Third call: back to TCP. Prior UDS must clear.
    port_b = _free_tcp_port("0.0.0.0")
    cli.serve_command(_minimal_serve_ns(host="0.0.0.0", port=port_b))
    assert (cfg.bind_host, cfg.bind_port, cfg.bind_listen_fd, cfg.bind_uds) == (
        "localhost",
        port_b,
        None,
        None,
    )
