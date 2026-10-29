# SPDX-License-Identifier: Apache-2.0
"""#1442: fusion-plugin-server must not be spawned unconditionally.

Before this fix the MCP layer auto-registered `fusion-plugin-server stdio` as
soon as the binary was on PATH, so `fusion-mlx-server` always carried the extra
child process with no way to turn it off from settings.json. These tests pin
the new contract:

- auto-discovery is off unless explicitly opted in (settings or env)
- `mcp.enabled=false` disables MCP wholesale, even when a config file exists
- env vars keep winning over settings.json (pre-existing override contract)
"""

import json

import pytest

from fusion_mlx.mcp import config as mcp_config
from fusion_mlx.settings import MCPSettings, Settings


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch, tmp_path):
    monkeypatch.delenv("FUSION_MLX_MCP_AUTO_DISCOVER", raising=False)
    monkeypatch.delenv("FUSION_MLX_MCP_ENABLED", raising=False)
    monkeypatch.delenv("FUSION_MLX_MCP_CONFIG", raising=False)
    monkeypatch.delenv("RAPID_MLX_MCP_CONFIG", raising=False)
    monkeypatch.delenv("VLLM_MLX_MCP_CONFIG", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))


@pytest.fixture
def fake_plugin_server(monkeypatch):
    monkeypatch.setattr(
        mcp_config.shutil, "which", lambda name: "/usr/local/bin/" + name
    )


def _write_settings(home, payload):
    path = home / ".fusion-mlx" / "settings.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return path


class TestMCPSettingsDefaults:
    def test_defaults_are_on_demand(self):
        s = MCPSettings()
        assert s.enabled is True
        assert s.auto_discover is False

    def test_round_trip(self):
        s = MCPSettings(enabled=False, auto_discover=True)
        assert MCPSettings.from_dict(s.to_dict()) == s

    def test_settings_section_round_trip(self, tmp_path):
        path = tmp_path / "settings.json"
        original = Settings(mcp=MCPSettings(enabled=False, auto_discover=True))
        original.save(path)
        raw = json.loads(path.read_text())
        assert raw["mcp"] == {"enabled": False, "auto_discover": True}
        loaded = Settings.load(path)
        assert loaded.mcp == original.mcp

    def test_missing_section_uses_defaults(self, tmp_path):
        path = tmp_path / "settings.json"
        path.write_text(json.dumps({"api_key": None}))
        assert Settings.load(path).mcp == MCPSettings()


class TestAutoDiscoverGate:
    def test_plugin_server_on_path_is_ignored_by_default(
        self, fake_plugin_server, tmp_path
    ):
        assert mcp_config._auto_discover_servers() == {}

    def test_settings_opt_in_registers_plugin_server(
        self, fake_plugin_server, tmp_path
    ):
        _write_settings(tmp_path, {"mcp": {"auto_discover": True}})
        servers = mcp_config._auto_discover_servers()
        assert list(servers) == ["fusion-plugins"]
        assert servers["fusion-plugins"].command == "fusion-plugin-server"

    def test_settings_passed_explicitly_opt_in(self, fake_plugin_server):
        settings = Settings(mcp=MCPSettings(auto_discover=True))
        servers = mcp_config._auto_discover_servers(settings)
        assert "fusion-plugins" in servers

    def test_env_opt_in_beats_settings_default(self, fake_plugin_server, monkeypatch):
        monkeypatch.setenv("FUSION_MLX_MCP_AUTO_DISCOVER", "1")
        assert "fusion-plugins" in mcp_config._auto_discover_servers()

    def test_env_opt_out_beats_settings_opt_in(self, fake_plugin_server, monkeypatch):
        monkeypatch.setenv("FUSION_MLX_MCP_AUTO_DISCOVER", "0")
        settings = Settings(mcp=MCPSettings(auto_discover=True))
        assert mcp_config._auto_discover_servers(settings) == {}


class TestMCPDisabled:
    def test_disabled_settings_skips_config_file_entirely(self, monkeypatch, tmp_path):
        cfg = tmp_path / "mcp.json"
        cfg.write_text(
            json.dumps(
                {
                    "servers": {
                        "filesystem": {
                            "transport": "stdio",
                            "command": "npx",
                            "args": ["-y", "server-filesystem"],
                        }
                    }
                }
            )
        )
        monkeypatch.setenv("FUSION_MLX_MCP_CONFIG", str(cfg))
        settings = Settings(mcp=MCPSettings(enabled=False))
        assert mcp_config.load_mcp_config(settings=settings).servers == {}

    def test_disabled_settings_skips_auto_discovery(
        self, fake_plugin_server, monkeypatch
    ):
        monkeypatch.setenv("FUSION_MLX_MCP_AUTO_DISCOVER", "1")
        settings = Settings(mcp=MCPSettings(enabled=False, auto_discover=True))
        assert mcp_config.load_mcp_config(settings=settings).servers == {}

    def test_enabled_settings_still_loads_config_file(self, monkeypatch, tmp_path):
        cfg = tmp_path / "mcp.json"
        cfg.write_text(
            json.dumps(
                {
                    "servers": {
                        "filesystem": {
                            "transport": "stdio",
                            "command": "npx",
                            "args": ["-y", "server-filesystem"],
                        }
                    }
                }
            )
        )
        monkeypatch.setenv("FUSION_MLX_MCP_CONFIG", str(cfg))
        servers = mcp_config.load_mcp_config(settings=Settings()).servers
        assert "filesystem" in servers
