# SPDX-License-Identifier: Apache-2.0
"""Background server lifecycle commands for fusion-mlx (start/stop/restart)."""

import json
import shutil
import socket
import subprocess
import time
from pathlib import Path

from .utils.install import get_app_bundle_path, is_app_bundle, is_homebrew

# ``Fusion-MLX`` (with hyphen) — matches AppConfig.appSupportURL() in the
# Swift app (apps/fusion-mac/Sources/Config/AppConfig.swift). The
# admin/logs.py ``FusionMLX`` (no hyphen) fallback is stale; match the code.
_CONTROL_SOCK_REL = Path("Library") / "Application Support" / "Fusion-MLX" / "control.sock"
_BREW_SERVICE = "fusion-mlx"


def _app_control_socket_path() -> Path:
    return Path.home() / _CONTROL_SOCK_REL


def _open_macos_app() -> None:
    app_path = get_app_bundle_path()
    subprocess.run(
        ["/usr/bin/open", "-gj", str(app_path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


def _send_app_control(command: str, timeout: float = 2.0) -> dict:
    sock_path = _app_control_socket_path()
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        sock.connect(str(sock_path))
        sock.sendall(json.dumps({"command": command}).encode("utf-8") + b"\n")
        chunks: list[bytes] = []
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
            chunks.append(chunk)
            if b"\n" in chunk:
                break
    raw = b"".join(chunks).split(b"\n", 1)[0]
    return json.loads(raw.decode("utf-8"))


def _send_app_control_with_launch(command: str, timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    _open_macos_app()
    while time.monotonic() < deadline:
        try:
            return _send_app_control(command)
        except OSError as exc:
            last_error = exc
            time.sleep(0.2)
    raise RuntimeError(f"Could not reach FusionMLX.app control socket: {last_error}")


def _wait_app_control_state(states: set[str], timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    last: dict = {}
    while time.monotonic() < deadline:
        last = _send_app_control("status")
        if last.get("state") in states:
            return last
        time.sleep(0.5)
    return last


def _run_brew_services(command: str) -> int:
    brew = shutil.which("brew")
    if not brew:
        print("Homebrew is not available on PATH.")
        return 1
    result = subprocess.run([brew, "services", command, _BREW_SERVICE])
    return result.returncode


def lifecycle_command(args) -> int:
    """Run background lifecycle commands for the current installation."""
    command = args.command
    timeout = getattr(args, "timeout", 60.0)
    no_wait = getattr(args, "no_wait", False)

    if is_app_bundle():
        try:
            if command == "stop":
                try:
                    response = _send_app_control(command)
                except (OSError, ValueError):
                    # OSError: socket gone (app already stopped).
                    # ValueError: app exited before flushing a JSON response
                    # (it still stopped). Either way, the server is stopped.
                    print("fusion-mlx stopped")
                    return 0
            else:
                response = _send_app_control_with_launch(command, timeout=timeout)
            if not response.get("ok"):
                print(response.get("message") or f"fusion-mlx {command} failed")
                return 1

            if command in {"start", "restart"} and not no_wait:
                response = _wait_app_control_state({"running", "unresponsive"}, timeout)
                if response.get("state") not in {"running", "unresponsive"}:
                    print(
                        f"fusion-mlx server is {response.get('state', 'unknown')} "
                        f"after {int(timeout)}s."
                    )
                    return 1

            if command == "stop":
                print("fusion-mlx stopped")
            elif command == "start":
                print(
                    f"fusion-mlx server {response.get('state')} on port {response.get('port')}"
                )
            elif command == "restart":
                print(f"fusion-mlx server restarted on port {response.get('port')}")
            return 0
        except Exception as exc:
            print(f"Failed to control FusionMLX.app: {exc}")
            return 1

    if is_homebrew():
        mapping = {"start": "start", "stop": "stop", "restart": "restart"}
        return _run_brew_services(mapping[command])

    if command == "start":
        print("Background start is available for the macOS app and Homebrew installs.")
        print("For this install, run foreground server mode with: fusion-mlx serve")
    else:
        print("Background stop/restart requires the macOS app or Homebrew service.")
    return 1
