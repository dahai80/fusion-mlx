# SPDX-License-Identifier: Apache-2.0
"""Unit tests for fusion_mlx.cli_lifecycle — control-socket path + protocol."""

from __future__ import annotations

import json
import os
import shutil
import socket
import tempfile
import threading
import types
from pathlib import Path

import pytest

from fusion_mlx import cli_lifecycle


@pytest.fixture
def short_sock_path():
    # macOS tmp_path (/var/folders/…/test_name0/) can exceed the 104-byte
    # AF_UNIX sun_path limit; use a short dir under /tmp so the real socket
    # protocol is actually exercised (not short-circuited by OSError).
    d = tempfile.mkdtemp(prefix="fmx_", dir="/tmp")
    yield os.path.join(d, "control.sock")
    shutil.rmtree(d, ignore_errors=True)


class TestControlSocketPath:
    """Regression guard: the socket dir must be ``Fusion-MLX`` (with hyphen).

    The Swift app's AppConfig.appSupportURL() uses ``Fusion-MLX`` (hyphen);
    an earlier version of this module used ``FusionMLX`` (no hyphen, matching
    the stale admin/logs.py fallback) and could never reach the real socket.
    """

    def test_sock_rel_uses_hyphenated_dir(self):
        parts = cli_lifecycle._CONTROL_SOCK_REL.parts
        assert "Fusion-MLX" in parts
        assert "FusionMLX" not in parts

    def test_app_control_socket_path_endswith_hyphenated_form(self):
        path = cli_lifecycle._app_control_socket_path()
        assert path == Path.home() / "Library" / "Application Support" / "Fusion-MLX" / "control.sock"

    def test_app_control_socket_path_under_home(self):
        path = cli_lifecycle._app_control_socket_path()
        assert path.is_absolute()
        assert str(path).startswith(str(Path.home()))


def _serve_one(sock_path: str, response: dict | None, ready: threading.Event | None = None) -> bytes:
    """Bind a Unix socket, accept one client, return the raw request bytes.

    If ``response`` is None, accept then close without writing (simulates an
    app that exits before flushing a JSON reply). Otherwise write the JSON
    response followed by a newline. ``ready`` is set once the socket is
    listening, so the client can connect without racing the bind.
    """
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(sock_path)
    srv.listen(1)
    if ready is not None:
        ready.set()
    conn, _ = srv.accept()
    data = b""
    while b"\n" not in data:
        chunk = conn.recv(4096)
        if not chunk:
            break
        data += chunk
    if response is not None:
        conn.sendall(json.dumps(response).encode("utf-8") + b"\n")
    conn.close()
    srv.close()
    return data


class TestSendAppControlProtocol:
    def test_sends_command_json_with_newline(self, monkeypatch, short_sock_path):
        # Swift Request is Decodable {command: String}; readRequest reads one
        # newline-terminated line. Verify we send exactly {"command": "status"}\n.
        monkeypatch.setattr(cli_lifecycle, "_app_control_socket_path", lambda: Path(short_sock_path))
        captured: dict = {}
        ready = threading.Event()

        def server():
            captured["raw"] = _serve_one(
                short_sock_path,
                {"ok": True, "status": "ok", "state": "running", "pid": 123,
                 "host": "127.0.0.1", "port": 11435, "message": None},
                ready,
            )

        t = threading.Thread(target=server)
        t.start()
        ready.wait(2.0)
        resp = cli_lifecycle._send_app_control("status")
        t.join()

        assert json.loads(captured["raw"]) == {"command": "status"}
        assert resp["ok"] is True
        assert resp["state"] == "running"
        assert resp["port"] == 11435

    def test_empty_response_raises_value_error(self, monkeypatch, short_sock_path):
        # App accepts then closes without JSON → json.loads("") → ValueError.
        # lifecycle_command(stop) must treat this as "stopped" (return 0).
        monkeypatch.setattr(cli_lifecycle, "_app_control_socket_path", lambda: Path(short_sock_path))
        ready = threading.Event()

        def server():
            _serve_one(short_sock_path, None, ready)

        t = threading.Thread(target=server)
        t.start()
        ready.wait(2.0)
        with pytest.raises(ValueError):
            cli_lifecycle._send_app_control("stop")
        t.join()


def _args(command: str, timeout: float = 0.5, no_wait: bool = False) -> types.SimpleNamespace:
    return types.SimpleNamespace(command=command, timeout=timeout, no_wait=no_wait)


class TestLifecycleCommandPipFallback:
    def test_start_returns_1_and_hints_foreground(self, monkeypatch, capsys):
        monkeypatch.setattr(cli_lifecycle, "is_app_bundle", lambda: False)
        monkeypatch.setattr(cli_lifecycle, "is_homebrew", lambda: False)
        rc = cli_lifecycle.lifecycle_command(_args("start"))
        out = capsys.readouterr().out
        assert rc == 1
        assert "fusion-mlx serve" in out

    def test_stop_returns_1_without_app_or_homebrew(self, monkeypatch, capsys):
        monkeypatch.setattr(cli_lifecycle, "is_app_bundle", lambda: False)
        monkeypatch.setattr(cli_lifecycle, "is_homebrew", lambda: False)
        rc = cli_lifecycle.lifecycle_command(_args("stop"))
        out = capsys.readouterr().out
        assert rc == 1
        assert "requires" in out

    def test_stop_app_bundle_empty_response_is_treated_as_stopped(self, monkeypatch, short_sock_path, capsys):
        # is_app_bundle True + app closes without JSON → ValueError caught → rc 0.
        # With a short path the connect succeeds, so rc 0 is attributable to the
        # ValueError path (not an OSError short-circuit).
        monkeypatch.setattr(cli_lifecycle, "is_app_bundle", lambda: True)
        monkeypatch.setattr(cli_lifecycle, "_app_control_socket_path", lambda: Path(short_sock_path))
        ready = threading.Event()

        def server():
            _serve_one(short_sock_path, None, ready)

        t = threading.Thread(target=server)
        t.start()
        ready.wait(2.0)
        rc = cli_lifecycle.lifecycle_command(_args("stop"))
        t.join()
        out = capsys.readouterr().out
        assert rc == 0
        assert "stopped" in out
