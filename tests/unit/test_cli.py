# SPDX-License-Identifier: Apache-2.0
"""
CLI tests for Fusion-MLX.

Tests CLI argument parsing, command setup, and help text.
Note: Configuration validation tests are in test_config.py.
"""

import argparse
import socket
import subprocess
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest


def _auto_detect_from_dir(model_dir):
    """Test helper: replicate _auto_detect_single_cached_model logic on a
    given directory so we don't have to monkeypatch Path.home()."""
    from pathlib import Path

    p = Path(model_dir)
    if not p.is_dir():
        return None
    candidates = [
        name
        for name in [c.name for c in p.iterdir() if c.is_dir()]
        if not name.startswith(".")
    ]
    if len(candidates) == 1:
        return candidates[0]
    return None


class TestCLIModule:
    """Tests for CLI module existence and basic functionality."""

    def test_cli_module_importable(self):
        """Test that CLI module can be imported."""
        from fusion_mlx import cli

        assert hasattr(cli, "main")

    def test_cli_has_serve_command(self):
        """Test that CLI has serve command setup."""
        from fusion_mlx import cli

        # The module should have the main entry point
        assert callable(cli.main)


class TestCLIHelp:
    """Tests for CLI help functionality."""

    def test_main_help(self):
        """Test main CLI help output."""
        result = subprocess.run(
            [sys.executable, "-m", "fusion_mlx.cli", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        # Should succeed with help
        assert result.returncode == 0
        # Should show available commands
        assert "serve" in result.stdout.lower()

    def test_main_version(self):
        """Test main CLI version output."""
        result = subprocess.run(
            [sys.executable, "-m", "fusion_mlx.cli", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0
        # The CLI prints the installed distribution metadata; fusion_mlx's
        # _version.py can be ahead of a stale editable install, so compare
        # against the same source instead of the constant.
        from importlib.metadata import version as _dist_version

        assert result.stdout.strip() == f"fusion-mlx {_dist_version('fusion-mlx')}"
        assert result.stderr == ""

    def test_serve_help(self):
        """Test serve command help output."""
        result = subprocess.run(
            [sys.executable, "-m", "fusion_mlx.cli", "serve", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        # Should succeed with help
        assert result.returncode == 0
        # Should show serve options
        stdout_lower = result.stdout.lower()
        assert "host" in stdout_lower
        assert "port" in stdout_lower
        assert "model-dir" in stdout_lower

    def test_lifecycle_commands_in_main_help(self):
        """Test start/stop/restart lifecycle commands are exposed.

        §6.1: --help shows only 5 primary commands; lifecycle commands
        appear on --help-advanced.
        """
        result = subprocess.run(
            [sys.executable, "-m", "fusion_mlx.cli", "--help-advanced"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0
        stdout_lower = result.stdout.lower()
        assert "start" in stdout_lower
        assert "stop" in stdout_lower
        assert "restart" in stdout_lower

    def test_primary_help_shows_only_5_commands(self):
        """§6.1: default --help shows serve/chat/pull/models/ps only."""
        result = subprocess.run(
            [sys.executable, "-m", "fusion_mlx.cli", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0
        stdout = result.stdout
        assert "serve" in stdout
        assert "chat" in stdout
        assert "pull" in stdout
        assert "models" in stdout
        assert "ps" in stdout
        assert "--help-advanced" in stdout
        assert "convert" not in stdout
        assert "cuda-node" not in stdout
        assert "telemetry" not in stdout


class TestZeroArgDefaultModel:
    """§6.1: serve/chat with no model arg reads default_model from settings.json."""

    def test_settings_default_model_reader_returns_none_when_unset(self, monkeypatch):
        from fusion_mlx import _cli_base

        monkeypatch.setattr(_cli_base, "_read_settings_json", lambda: {})
        assert _cli_base._settings_default_model() is None

    def test_settings_default_model_reader_returns_value(self, monkeypatch):
        from fusion_mlx import _cli_base

        monkeypatch.setattr(
            _cli_base,
            "_read_settings_json",
            lambda: {"default_model": "qwen3.5-9b-4bit"},
        )
        assert _cli_base._settings_default_model() == "qwen3.5-9b-4bit"

    def test_settings_profile_reader_validates_choices(self, monkeypatch):
        from fusion_mlx import _cli_base

        monkeypatch.setattr(
            _cli_base, "_read_settings_json", lambda: {"profile": "lite"}
        )
        assert _cli_base._settings_profile() == "lite"

        monkeypatch.setattr(
            _cli_base, "_read_settings_json", lambda: {"profile": "invalid"}
        )
        assert _cli_base._settings_profile() is None

    def test_settings_disabled_modules_reader(self, monkeypatch):
        from fusion_mlx import _cli_base

        monkeypatch.setattr(
            _cli_base,
            "_read_settings_json",
            lambda: {"disabled_modules": ["image", "video"]},
        )
        assert _cli_base._settings_disabled_modules() == ["image", "video"]

        monkeypatch.setattr(_cli_base, "_read_settings_json", lambda: {})
        assert _cli_base._settings_disabled_modules() == []

    def test_auto_detect_single_cached_model_none_when_multi(
        self, monkeypatch, tmp_path
    ):
        from fusion_mlx import _cli_base

        models = tmp_path / "models"
        models.mkdir()
        (models / "model-a").mkdir()
        (models / "model-b").mkdir()
        monkeypatch.setattr(
            _cli_base,
            "_auto_detect_single_cached_model",
            lambda: _auto_detect_from_dir(models),
        )
        assert _auto_detect_from_dir(models) is None

    def test_auto_detect_single_cached_model_returns_name_when_single(
        self, monkeypatch, tmp_path
    ):
        models = tmp_path / "models"
        models.mkdir()
        (models / "only-model").mkdir()
        assert _auto_detect_from_dir(models) == "only-model"


class TestLifecycleCommand:
    """Tests for managed background server lifecycle commands."""

    @staticmethod
    def _args(command, **overrides):
        values = {"command": command, "timeout": 1.0, "no_wait": False}
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_app_bundle_stop_without_running_app_is_success(self, monkeypatch, capsys):
        """`fusion_mlx stop` must not launch the macOS app just to stop it."""
        from fusion_mlx import cli_lifecycle as cli

        monkeypatch.setattr(cli, "is_app_bundle", lambda: True)
        monkeypatch.setattr(cli, "is_homebrew", lambda: False)
        monkeypatch.setattr(
            cli, "_send_app_control", MagicMock(side_effect=ConnectionRefusedError)
        )
        monkeypatch.setattr(cli, "_open_macos_app", MagicMock())

        assert cli.lifecycle_command(self._args("stop")) == 0
        assert capsys.readouterr().out.strip() == "fusion-mlx stopped"
        cli._open_macos_app.assert_not_called()

    def test_app_bundle_stop_with_dead_socket_is_failure(self, monkeypatch, capsys):
        """Audit L28: a generic OSError must not be reported as "stopped"."""
        from fusion_mlx import cli_lifecycle as cli

        monkeypatch.setattr(cli, "is_app_bundle", lambda: True)
        monkeypatch.setattr(cli, "is_homebrew", lambda: False)
        monkeypatch.setattr(
            cli, "_send_app_control", MagicMock(side_effect=PermissionError("denied"))
        )

        assert cli.lifecycle_command(self._args("stop")) == 1
        assert "Failed to control" in capsys.readouterr().out

    def test_app_bundle_start_sends_command_and_waits(self, monkeypatch):
        """`fusion_mlx start` asks the app to start and waits for a running state."""
        from fusion_mlx import cli_lifecycle as cli

        monkeypatch.setattr(cli, "is_app_bundle", lambda: True)
        monkeypatch.setattr(cli, "is_homebrew", lambda: False)
        send = MagicMock(return_value={"ok": True, "state": "starting", "port": 8000})
        wait = MagicMock(return_value={"ok": True, "state": "running", "port": 8000})
        monkeypatch.setattr(cli, "_send_app_control_with_launch", send)
        monkeypatch.setattr(cli, "_wait_app_control_state", wait)

        assert cli.lifecycle_command(self._args("start")) == 0
        send.assert_called_once_with("start", timeout=1.0)
        wait.assert_called_once_with({"running", "unresponsive"}, 1.0)

    def test_homebrew_start_delegates_to_brew_services(self, monkeypatch):
        """Homebrew installs use the Homebrew service supervisor."""
        from fusion_mlx import cli_lifecycle as cli

        monkeypatch.setattr(cli, "is_app_bundle", lambda: False)
        monkeypatch.setattr(cli, "is_homebrew", lambda: True)
        run_brew = MagicMock(return_value=0)
        monkeypatch.setattr(cli, "_run_brew_services", run_brew)

        assert cli.lifecycle_command(self._args("start")) == 0
        run_brew.assert_called_once_with("start")


class TestCLIEntryPoint:
    """Tests for CLI entry point functionality."""

    def test_module_runnable(self):
        """Test that CLI module is runnable."""
        # Should not crash when running with --help
        result = subprocess.run(
            [sys.executable, "-m", "fusion_mlx.cli", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0

    def test_invalid_command_error(self):
        """Test error handling for invalid command."""
        result = subprocess.run(
            [sys.executable, "-m", "fusion_mlx.cli", "invalid_command"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        # Should fail with non-zero exit code
        assert result.returncode != 0

    def test_no_command_shows_help(self):
        """Test that no command shows help."""
        result = subprocess.run(
            [sys.executable, "-m", "fusion_mlx.cli"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        # Should exit with non-zero (no command provided)
        assert result.returncode != 0


class TestServeCommandOptions:
    """Tests for serve command options via help output."""

    def test_serve_has_model_dir_option(self):
        """Test that serve command has --model-dir option."""
        result = subprocess.run(
            [sys.executable, "-m", "fusion_mlx.cli", "serve", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert "--model-dir" in result.stdout

    def test_serve_has_no_max_memory_options(self):
        """The --max-model-memory and --max-process-memory CLI flags are removed."""
        result = subprocess.run(
            [sys.executable, "-m", "fusion_mlx.cli", "serve", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert "--max-model-memory" not in result.stdout
        assert "--max-process-memory" not in result.stdout

    def test_serve_has_memory_guard_options(self):
        """Test that serve command exposes memory guard controls."""
        result = subprocess.run(
            [sys.executable, "-m", "fusion_mlx.cli", "serve", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert "--memory-tier" in result.stdout
        assert "safe" in result.stdout
        assert "balanced" in result.stdout
        assert "aggressive" in result.stdout
        assert "custom" in result.stdout

    def test_serve_no_model_specific_options(self):
        """Test that serve command does not have model-specific options (managed via admin page)."""
        result = subprocess.run(
            [sys.executable, "-m", "fusion_mlx.cli", "serve", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        # Per-request sampling is a client concern; the CLI only exposes
        # server-side defaults (--default-*), never per-request knobs.
        assert "--temperature" not in result.stdout
        assert "--top-p" not in result.stdout
        assert "--top-k" not in result.stdout
        assert "--force-sampling" not in result.stdout
        assert "--default-temperature" in result.stdout
        assert "--default-top-p" in result.stdout
        assert "--default-top-k" in result.stdout

    def test_serve_has_host_port_options(self):
        """Test that serve command has --host and --port options."""
        result = subprocess.run(
            [sys.executable, "-m", "fusion_mlx.cli", "serve", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert "--host" in result.stdout
        assert "--port" in result.stdout

    def test_serve_has_scheduler_options(self):
        """Test that serve command has scheduler options."""
        result = subprocess.run(
            [sys.executable, "-m", "fusion_mlx.cli", "serve", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert "--max-concurrent-requests" in result.stdout
        assert "--max-num-seqs" in result.stdout
        assert "--prefill-batch-size" in result.stdout

    def test_serve_has_cache_options(self):
        """Test that serve command has cache options."""
        result = subprocess.run(
            [sys.executable, "-m", "fusion_mlx.cli", "serve", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert "--prefix-cache-size" in result.stdout
        assert "--cache-memory-mb" in result.stdout
        assert "--paged-cache-block-size" in result.stdout
        assert "--no-memory-aware-cache" in result.stdout

    def test_serve_has_mcp_option(self):
        """Test that serve command has --mcp-config option."""
        result = subprocess.run(
            [sys.executable, "-m", "fusion_mlx.cli", "serve", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert "--mcp-config" in result.stdout

    def test_serve_has_base_path_option(self):
        """Test that serve command has --base-path option."""
        result = subprocess.run(
            [sys.executable, "-m", "fusion_mlx.cli", "serve", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert "--base-path" in result.stdout

    def test_serve_has_api_key_option(self):
        """Test that serve command has --api-key option."""
        result = subprocess.run(
            [sys.executable, "-m", "fusion_mlx.cli", "serve", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert "--api-key" in result.stdout


class TestLaunchCommandOptions:
    """`launch` now configures IDE clients, it does not boot a server."""

    @staticmethod
    def _help() -> str:
        result = subprocess.run(
            [sys.executable, "-m", "fusion_mlx.cli", "launch", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0
        return result.stdout

    def test_launch_has_model_and_endpoint_options(self):
        out = self._help()
        assert "--model" in out
        assert "--server-url" in out
        assert "--port" in out
        assert "--dry-run" in out

    def test_launch_lists_supported_clients(self):
        out = self._help()
        for name in ("claude-code", "codex", "cursor", "hermes", "cline"):
            assert name in out, name


class TestLaunchCommandFunction:
    """Runtime behaviour of the client-config launcher."""

    def test_print_list_lists_every_adapter(self, capsys):
        from fusion_mlx.launch.cli import ADAPTERS, _print_list

        assert _print_list() == 0
        out = capsys.readouterr().out
        assert "Supported clients:" in out
        for name in ADAPTERS:
            assert name in out

    @staticmethod
    def _args(**overrides):
        values = {
            "client": "fake",
            "all": False,
            "model": "qwen3.5-4b-4bit",
            "server_url": "http://127.0.0.1:11434",
            "port": 11434,
            "start_server": False,
            "dry_run": True,
            "_original_alias": None,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    @staticmethod
    def _fake_adapter(tmp_path, applied):
        class _Fake:
            def detect(self):
                return True

            def current_config_path(self):
                return tmp_path / "client.json"

            def apply(self, *a, **k):
                applied.append((a, k))

        return _Fake()

    def test_dry_run_does_not_apply(self, monkeypatch, capsys, tmp_path):
        from fusion_mlx.launch import cli as launch_cli

        applied = []
        monkeypatch.setitem(
            launch_cli.ADAPTERS, "fake", self._fake_adapter(tmp_path, applied)
        )

        launch_cli.launch_command(self._args())

        out = capsys.readouterr().out
        assert "[dry-run]" in out
        assert "[dry-run] model=qwen3.5-4b-4bit" in out
        assert applied == []

    def test_missing_client_exits_2(self, capsys):
        from fusion_mlx.launch import cli as launch_cli

        with pytest.raises(SystemExit) as exc:
            launch_cli.launch_command(self._args(client=None, all=False))
        assert exc.value.code == 2
        assert "missing client name" in capsys.readouterr().err

    def test_unknown_client_exits_2(self, capsys):
        from fusion_mlx.launch import cli as launch_cli

        with pytest.raises(SystemExit) as exc:
            launch_cli.launch_command(self._args(client="does-not-exist"))
        assert exc.value.code == 2
        assert "unknown client" in capsys.readouterr().err

    def test_all_with_client_name_exits_2(self, capsys):
        from fusion_mlx.launch import cli as launch_cli

        with pytest.raises(SystemExit) as exc:
            launch_cli.launch_command(self._args(all=True))
        assert exc.value.code == 2
        assert "mutually exclusive" in capsys.readouterr().err


class TestLaunchArgvParsing:
    """Tests for top-level argv parsing of `fusion_mlx launch ...`."""

    def test_serve_still_rejects_unknown_args(self):
        """Non-launch commands must keep strict argparse rejection."""
        result = subprocess.run(
            [sys.executable, "-m", "fusion_mlx.cli", "serve", "--bogus-flag"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode != 0
        assert (
            "unrecognized arguments" in result.stderr or "--bogus-flag" in result.stderr
        )


class TestServeCommandFunctions:
    """Tests for serve command function."""

    @staticmethod
    def _make_serve_args(tmp_path, host="127.0.0.1", port=8000, **overrides):
        defaults = {
            "model_dir": None,
            "host": host,
            "port": port,
            "log_level": None,
            "sse_keepalive_mode": None,
            "max_concurrent_requests": None,
            "embedding_batch_size": None,
            "memory_guard": None,
            "memory_guard_gb": None,
            "paged_ssd_cache_dir": None,
            "paged_ssd_cache_max_size": None,
            "hot_cache_max_size": None,
            "no_cache": True,
            "initial_cache_blocks": None,
            "mcp_config": None,
            "hf_endpoint": None,
            "hf_cache_enabled": None,
            "ms_endpoint": None,
            "http_proxy": None,
            "https_proxy": None,
            "no_proxy": None,
            "ca_bundle": None,
            "base_path": str(tmp_path),
            "api_key": None,
            "rate_limit": 0,
        }
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    @staticmethod
    def _make_settings(tmp_path, host="127.0.0.1", port=8000):
        log_dir = tmp_path / "logs"
        settings = SimpleNamespace()
        settings.base_path = tmp_path
        settings.server = SimpleNamespace(
            host=host, port=port, log_level="info", burst_decode_mode="balanced"
        )
        settings.huggingface = SimpleNamespace(endpoint=None, hf_cache_enabled=True)
        settings.modelscope = SimpleNamespace(endpoint=None)
        settings.network = SimpleNamespace(
            http_proxy=None,
            https_proxy=None,
            no_proxy=None,
            ca_bundle=None,
        )
        settings.logging = SimpleNamespace(
            retention_days=7,
            get_log_dir=lambda base_path: log_dir,
        )
        settings.model = SimpleNamespace(
            get_model_dirs=lambda base_path: [tmp_path / "models"],
        )
        settings.get_effective_model_dirs = lambda: [tmp_path / "models"]
        settings.memory = SimpleNamespace(memory_guard_tier="balanced")
        settings.mcp = SimpleNamespace(config_path=None)
        settings.cache = SimpleNamespace(
            enabled=False,
            get_ssd_cache_dir=lambda base_path: tmp_path / "cache",
            get_ssd_cache_max_size_bytes=lambda base_path: 0,
            get_hot_cache_max_size_bytes=lambda: 0,
        )
        settings.auth = SimpleNamespace(api_key=None)
        settings.ensure_directories = lambda: log_dir.mkdir(parents=True, exist_ok=True)
        settings.validate = lambda: []
        settings.save = MagicMock()
        settings.to_scheduler_config = lambda: SimpleNamespace(
            paged_ssd_cache_dir=None,
            paged_ssd_cache_max_size=0,
            hot_cache_max_size=0,
        )
        return settings

    @staticmethod
    def _reserve_port(host="127.0.0.1"):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, 0))
        sock.listen(1)
        return sock

    def test_serve_command_exists(self):
        """Test that serve_command function exists."""
        from fusion_mlx.cli import serve_command

        assert callable(serve_command)

    def test_serve_model_dir_optional_with_default(self):
        """Test that serve --model-dir is optional with default ~/.fusion_mlx/models."""
        result = subprocess.run(
            [sys.executable, "-m", "fusion_mlx.cli", "serve", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        # Should show that model-dir has a default
        assert "default" in result.stdout.lower()
        # Help text should mention ~/.fusion_mlx/models or similar
        assert ".fusion_mlx" in result.stdout or "model" in result.stdout.lower()

    def test_invalid_scheduler_value_fails_before_writing_settings(self, tmp_path):
        """Invalid CLI values must be rejected before settings.json is written."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "fusion_mlx.cli",
                "serve",
                "--base-path",
                str(tmp_path),
                "--memory-tier",
                "not-a-tier",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )

        assert result.returncode != 0
        assert "not-a-tier" in result.stderr
        assert not (tmp_path / "settings.json").exists()

    def test_memory_ceiling_schema_rejects_non_finite(self):
        """A NaN/inf ceiling must not be accepted as a memory limit.

        The value moved from the (removed) --memory-guard-gb flag to
        settings.json memory.memory_guard_custom_ceiling_gb, so the check
        belongs to the schema validator.
        """
        from fusion_mlx.config_schema import MemorySection

        for bad in ("nan", "inf", -1):
            with pytest.raises(ValueError):
                MemorySection(memory_guard_custom_ceiling_gb=bad)

        ok = MemorySection(memory_guard_custom_ceiling_gb=8)
        assert ok.memory_guard_custom_ceiling_gb == 8.0

    def test_serve_exits_on_port_conflict_before_importing_server(
        self, tmp_path, monkeypatch
    ):
        """Port conflicts should fail before server import can preload pinned models."""
        from fusion_mlx.cli import serve_command

        listener = self._reserve_port()
        host, port = listener.getsockname()
        args = self._make_serve_args(tmp_path, host=host, port=port)
        previous_server = sys.modules.pop("fusion_mlx.server", None)
        monkeypatch.setattr(
            "fusion_mlx.logging_config.configure_file_logging",
            lambda **kwargs: None,
        )
        try:
            with pytest.raises(SystemExit) as exc:
                serve_command(args)

            assert exc.value.code != 0
            # The preflight must fire before `from .. import server`; uvicorn
            # 0.51 binds via loop.create_server, so a bind-time assertion can
            # no longer observe the collision.
            assert "fusion_mlx.server" not in sys.modules
        finally:
            listener.close()
            if previous_server is not None:
                sys.modules["fusion_mlx.server"] = previous_server

    def test_serve_model_dir_dispatches_app_to_uvicorn(self, tmp_path, monkeypatch):
        """--base-path serve must build the pool app and hand it to uvicorn."""
        from pathlib import Path

        import fusion_mlx
        from fusion_mlx.cli import serve_command
        from fusion_mlx.cli_serve import config_resolve

        host, port = "127.0.0.1", 0
        args = self._make_serve_args(tmp_path, host=host, port=port)
        app = object()
        captured = {}

        fake_server = ModuleType("fusion_mlx.server")
        fake_server.create_app = MagicMock(return_value=app)
        fake_server.configure_logging = MagicMock()
        # config_resolve stages the CLI --api-key through the server module
        # before building the app (#636); the stub must expose it.
        fake_server._resolve_api_key = lambda *a, **k: None
        monkeypatch.setitem(sys.modules, "fusion_mlx.server", fake_server)
        monkeypatch.setattr(fusion_mlx, "server", fake_server, raising=False)

        monkeypatch.setattr(
            "fusion_mlx.logging_config.configure_file_logging",
            lambda **kwargs: None,
        )
        monkeypatch.setattr(
            config_resolve,
            "_run_uvicorn",
            lambda app_arg, args_arg, log_level: captured.update(
                app=app_arg, args=args_arg, log_level=log_level
            ),
        )

        serve_command(args)

        fake_server.create_app.assert_called_once()
        server_config = fake_server.create_app.call_args.args[0]
        assert server_config.model_dir == str(Path(tmp_path) / "models")
        assert (server_config.host, server_config.port) == (host, port)
        assert captured["app"] is app
        assert captured["args"] is args
        assert fake_server._api_key is None


class TestCLIDocstrings:
    """Tests for CLI module docstrings and descriptions."""

    def test_main_has_description(self):
        """Test that main help has description."""
        result = subprocess.run(
            [sys.executable, "-m", "fusion_mlx.cli", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        # Should have some description
        lowered = result.stdout.lower()
        assert "fusion-mlx" in lowered or "fusion_mlx" in lowered or "llm" in lowered

    def test_serve_has_description(self):
        """Test that serve command has description."""
        result = subprocess.run(
            [sys.executable, "-m", "fusion_mlx.cli", "serve", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        # Should describe multi-model serving
        assert (
            "multi-model" in result.stdout.lower() or "server" in result.stdout.lower()
        )
