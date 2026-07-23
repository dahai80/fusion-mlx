# SPDX-License-Identifier: Apache-2.0
"""Unit tests for fusion_mlx.integrations.claude + copilot — env var integrations.

Covers get_command, _find_claude_binary (PATH/local/fallback), launch env setup
(ANTHROPIC_BASE_URL/AUTH_TOKEN/API_KEY/model/context_window), _scrubbed_env.
"""

from __future__ import annotations

from unittest.mock import patch

from fusion_mlx.integrations.claude import ClaudeCodeIntegration
from fusion_mlx.integrations.copilot import CopilotIntegration


class TestClaudeCodeIntegration:
    def test_init(self):
        cli = ClaudeCodeIntegration()
        assert cli.name == "claude"
        assert cli.display_name == "Claude Code"

    def test_get_command(self):
        cli = ClaudeCodeIntegration()
        with patch("fusion_mlx.utils.install.get_cli_prefix", return_value="fusion-mlx"):
            cmd = cli.get_command(8080, "key", "model")
            assert "claude" in cmd

    def test_find_claude_binary_in_path(self):
        cli = ClaudeCodeIntegration()
        with patch("shutil.which", return_value="/usr/bin/claude"):
            assert cli._find_claude_binary() == "claude"

    def test_find_claude_binary_local(self, tmp_path):
        cli = ClaudeCodeIntegration()
        with patch("shutil.which", return_value=None):
            with patch("pathlib.Path.home", return_value=tmp_path):
                local_dir = tmp_path / ".claude" / "local"
                local_dir.mkdir(parents=True)
                (local_dir / "claude").write_text("script")
                result = cli._find_claude_binary()
                assert result.endswith("claude")

    def test_find_claude_binary_fallback(self, tmp_path):
        cli = ClaudeCodeIntegration()
        with patch("shutil.which", return_value=None):
            with patch("pathlib.Path.home", return_value=tmp_path):
                # tmp_path has no .claude/local/claude → fallback to "claude"
                assert cli._find_claude_binary() == "claude"

    def test_launch_sets_env_vars(self):
        cli = ClaudeCodeIntegration()
        with patch.object(cli, "_scrubbed_env", return_value={}):
            with patch.object(cli, "_find_claude_binary", return_value="claude"):
                with patch("os.execvpe") as mock_exec:
                    cli.launch(8080, "key123", "qwen", context_window=8000)
                    args = mock_exec.call_args[0]
                    env = (
                        mock_exec.call_args[0][2]
                        if len(mock_exec.call_args[0]) > 2
                        else mock_exec.call_args[0]
                    )
                    # execvpe(binary, argv, env) — env is 3rd positional
                    env = (
                        mock_exec.call_args[0][2]
                        if len(mock_exec.call_args[0]) >= 3
                        else mock_exec.call_args.kwargs.get("env", {})
                    )
                    assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8080"
                    assert env["ANTHROPIC_AUTH_TOKEN"] == "key123"
                    assert env["ANTHROPIC_API_KEY"] == ""
                    assert env["ANTHROPIC_DEFAULT_OPUS_MODEL"] == "qwen"
                    assert env["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] == "8000"

    def test_launch_no_api_key_uses_fusion_mlx(self):
        cli = ClaudeCodeIntegration()
        with patch.object(cli, "_scrubbed_env", return_value={}):
            with patch.object(cli, "_find_claude_binary", return_value="claude"):
                with patch("os.execvpe") as mock_exec:
                    cli.launch(8080, "", "qwen")
                    env = mock_exec.call_args[0][2]
                    assert env["ANTHROPIC_AUTH_TOKEN"] == "fusion-mlx"

    def test_launch_no_context_window_skips(self):
        cli = ClaudeCodeIntegration()
        with patch.object(cli, "_scrubbed_env", return_value={}):
            with patch.object(cli, "_find_claude_binary", return_value="claude"):
                with patch("os.execvpe") as mock_exec:
                    cli.launch(8080, "k", "m")
                    env = mock_exec.call_args[0][2]
                    assert "CLAUDE_CODE_AUTO_COMPACT_WINDOW" not in env

    def test_launch_no_model_skips(self):
        cli = ClaudeCodeIntegration()
        with patch.object(cli, "_scrubbed_env", return_value={}):
            with patch.object(cli, "_find_claude_binary", return_value="claude"):
                with patch("os.execvpe") as mock_exec:
                    cli.launch(8080, "k", "")
                    env = mock_exec.call_args[0][2]
                    assert "ANTHROPIC_DEFAULT_OPUS_MODEL" not in env


class TestCopilotIntegration:
    def test_init(self):
        cli = CopilotIntegration()
        assert cli.name == "copilot"

    def test_get_command(self):
        cli = CopilotIntegration()
        with patch("fusion_mlx.utils.install.get_cli_prefix", return_value="fusion-mlx"):
            cmd = cli.get_command(8080, "key", "model")
            assert "copilot" in cmd
