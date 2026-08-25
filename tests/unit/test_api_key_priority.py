# Regression tests for the HTTP-auth infra bug: CLI ``--api-key`` /
# ``FUSION_MLX_API_KEY`` silently overridden by ``settings.json``
# ``auth.api_key``.
#
# Data flow (pre-fix):
#   1. cli_serve / server entry resolves the CLI/env key into the
#      module-level ``server._api_key`` global and calls ``set_api_key``.
#   2. Server.__init__ loads ``settings.json`` and at line 663 runs
#      ``if self.settings.api_key: set_api_key(self.settings.api_key)``
#      -> the settings.json key OVERWRITES the CLI/env key.
#   3. The auth middleware ``_get_configured_api_key`` reads
#      ``_get_global_settings().auth.api_key`` (== self.settings.api_key,
#      the settings.json value), so it demands the settings.json key
#      even though the operator passed ``--api-key <other>``.
#
# Symptom: ``curl -H "Authorization: Bearer <cli-key>"`` gets 401
# "Invalid API key" because the server enforces the settings.json key.
#
# Contract after fix: effective key priority is
#   CLI --api-key  >  FUSION_MLX_API_KEY env  >  settings.json auth.api_key
# matching ``_resolve_api_key``'s documented order, and every read path
# (``set_api_key`` module global, ``self.settings.api_key``, the
# ServerConfig singleton) agrees on the SAME resolved value.

from __future__ import annotations

import pytest


def _reset_server_key_globals(monkeypatch):
    # Neutralize anything a prior test / conftest left in the module global
    # so each case starts from a known state (see test_server_api_key_env_
    # fallback for the same CI-settings.json-load caveat).
    monkeypatch.setattr("fusion_mlx.server._api_key", None)
    from fusion_mlx.admin import auth as admin_auth

    admin_auth.set_api_key("")
    return admin_auth


def test_cli_key_beats_settings_json_key(monkeypatch):
    # Operator passes --api-key fg-admin-key; settings.json has dahai168.
    # Effective key MUST be the CLI value, not settings.json.
    from fusion_mlx.server import _resolve_effective_api_key

    _reset_server_key_globals(monkeypatch)
    effective, source = _resolve_effective_api_key(
        argv_key="fg-admin-key",
        settings_key="dahai168",
    )
    assert effective == "fg-admin-key"
    assert source == "cli"


def test_env_key_beats_settings_json_key(monkeypatch):
    # No CLI flag; FUSION_MLX_API_KEY set; settings.json also set.
    # Env wins over settings.json (operator env is more explicit than a
    # stale persisted file).
    from fusion_mlx.server import _resolve_effective_api_key

    _reset_server_key_globals(monkeypatch)
    monkeypatch.setenv("FUSION_MLX_API_KEY", "env-secret")
    effective, source = _resolve_effective_api_key(
        argv_key=None,
        settings_key="dahai168",
    )
    assert effective == "env-secret"
    assert source == "env"


def test_settings_json_key_used_when_no_cli_no_env(monkeypatch):
    # No CLI, no env -> settings.json is the fallback (preserves the
    # admin-set persisted key for bare ``fusion-mlx serve`` launches).
    from fusion_mlx.server import _resolve_effective_api_key

    _reset_server_key_globals(monkeypatch)
    monkeypatch.delenv("FUSION_MLX_API_KEY", raising=False)
    effective, source = _resolve_effective_api_key(
        argv_key=None,
        settings_key="dahai168",
    )
    assert effective == "dahai168"
    assert source == "settings"


def test_none_when_nothing_configured(monkeypatch):
    # No CLI, no env, no settings.json key -> anonymous dev path stays
    # anonymous-OK (must not regress test_server_auth_ordering leg 4).
    from fusion_mlx.server import _resolve_effective_api_key

    _reset_server_key_globals(monkeypatch)
    monkeypatch.delenv("FUSION_MLX_API_KEY", raising=False)
    effective, source = _resolve_effective_api_key(
        argv_key=None,
        settings_key=None,
    )
    assert effective is None
    assert source == "none"


def test_cli_wins_over_both_env_and_settings(monkeypatch):
    # Full stack: CLI > env > settings. Pins the top of the order.
    from fusion_mlx.server import _resolve_effective_api_key

    _reset_server_key_globals(monkeypatch)
    monkeypatch.setenv("FUSION_MLX_API_KEY", "env-secret")
    effective, source = _resolve_effective_api_key(
        argv_key="cli-secret",
        settings_key="settings-secret",
    )
    assert effective == "cli-secret"
    assert source == "cli"


@pytest.mark.real_model
def test_live_server_enforces_cli_key_not_settings_key(tmp_path):
    # End-to-end: boot a real server with --api-key <CLI> where
    # settings.json holds a DIFFERENT key. The middleware must accept the
    # CLI key and reject the settings.json key (pre-fix it was inverted).
    import http.client
    import os
    import socket
    import subprocess
    import sys
    import time

    def _free_port():
        for p in range(11940, 12040):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(("127.0.0.1", p))
                    return p
                except OSError:
                    continue
        raise RuntimeError("no free port")

    settings_dir = tmp_path / "settings"
    settings_dir.mkdir()
    (settings_dir / "settings.json").write_text(
        '{"auth": {"api_key": "SETTINGS_KEY_MUST_BE_REJECTED"}}'
    )

    cli_key = "CLI_KEY_MUST_BE_ACCEPTED"
    port = _free_port()
    env = {
        **os.environ,
        # Ensure env does not interfere with the CLI-vs-settings test.
        "FUSION_MLX_API_KEY": "",
    }
    cmd = [
        sys.executable,
        "-m",
        "fusion_mlx.cli",
        "serve",
        "qwen3.5-4b-4bit",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--api-key",
        cli_key,
        "--settings-dir",
        str(settings_dir),
    ]

    proc = subprocess.Popen(  # noqa: S603
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 180.0
        healthy = False
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                break
            try:
                c = http.client.HTTPConnection("127.0.0.1", port, timeout=1.0)
                c.request("GET", "/healthz")
                r = c.getresponse()
                r.read()
                c.close()
                if r.status == 200:
                    healthy = True
                    break
            except (OSError, http.client.HTTPException):
                pass
            time.sleep(0.5)
        if not healthy:
            pytest.skip("server did not reach /healthz in 180s")
        # The settings.json key MUST be rejected.
        bad = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        bad.request(
            "GET",
            "/v1/models",
            headers={"Authorization": "Bearer SETTINGS_KEY_MUST_BE_REJECTED"},
        )
        bad_r = bad.getresponse()
        bad_r.read()
        bad.close()
        assert bad_r.status == 401, (
            f"settings.json key was accepted ({bad_r.status}); "
            f"CLI --api-key was overridden by settings.json (the bug)."
        )
        # The CLI key MUST be accepted.
        good = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        good.request(
            "GET", "/v1/models", headers={"Authorization": f"Bearer {cli_key}"}
        )
        good_r = good.getresponse()
        good_r.read()
        good.close()
        assert good_r.status == 200, (
            f"CLI --api-key rejected ({good_r.status}); "
            f"server enforced settings.json key instead of CLI key."
        )
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        subprocess.run(
            ["pkill", "-f", f"fusion_mlx.cli.*{port}"],
            check=False,
            capture_output=True,
        )
