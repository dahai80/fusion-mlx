# SPDX-License-Identifier: Apache-2.0
"""OPS-P4-6 / OPS-P4-7 (#0907 audit): config schema + hot-reload tests."""

import json
from unittest.mock import MagicMock, patch

import pytest

from fusion_mlx.config_schema import SettingsSchema


class TestSettingsSchema:
    def test_valid_full(self):
        s = SettingsSchema.validate_dict(
            {
                "server": {
                    "log_level": "debug",
                    "port": 11434,
                    "sse_keepalive_mode": "chunk",
                },
                "memory": {
                    "memory_guard_tier": "balanced",
                    "prefill_memory_guard": True,
                },
                "scheduler": {"chunked_prefill": False, "max_concurrent_requests": 4},
                "idle_timeout": {"idle_timeout_seconds": 1800},
                "route_guard": {"warn_only": True, "enforce": False},
            }
        )
        assert s.server.log_level == "debug"
        assert s.server.port == 11434
        assert s.memory.memory_guard_tier == "balanced"
        assert s.idle_timeout.idle_timeout_seconds == 1800
        assert s.route_guard.warn_only is True

    def test_log_level_case_normalized(self):
        s = SettingsSchema.validate_dict({"server": {"log_level": "WARNING"}})
        assert s.server.log_level == "warning"

    def test_bad_memory_tier_raises(self):
        with pytest.raises(Exception):
            SettingsSchema.validate_dict({"memory": {"memory_guard_tier": "STRICT"}})

    def test_bad_log_level_raises(self):
        with pytest.raises(Exception):
            SettingsSchema.validate_dict({"server": {"log_level": "verbose"}})

    def test_bad_port_raises(self):
        with pytest.raises(Exception):
            SettingsSchema.validate_dict({"server": {"port": 99999}})

    def test_string_ceiling_coerced(self):
        s = SettingsSchema.validate_dict(
            {"memory": {"memory_guard_custom_ceiling_gb": "8"}}
        )
        assert s.memory.memory_guard_custom_ceiling_gb == 8.0

    def test_string_idle_timeout_rejected(self):
        with pytest.raises(Exception):
            SettingsSchema.validate_dict(
                {"idle_timeout": {"idle_timeout_seconds": "soon"}}
            )

    def test_extra_fields_allowed(self):
        s = SettingsSchema.validate_dict(
            {"server": {"log_level": "info", "unknown_field": 1}}
        )
        assert s.server.log_level == "info"

    def test_empty_dict_ok(self):
        s = SettingsSchema.validate_dict({})
        assert s.server.log_level is None


class TestReloadConfig:
    @pytest.mark.asyncio
    async def test_log_level_applied(self, tmp_path):
        settings_file = tmp_path / "settings.json"
        settings_file.write_text(json.dumps({"server": {"log_level": "error"}}))
        with (
            patch(
                "fusion_mlx.routes_internal.config_reload._settings_json_path",
                return_value=settings_file,
            ),
            patch("fusion_mlx.admin.helpers._apply_log_level_runtime") as mock_ll,
            patch(
                "fusion_mlx.server._server_state",
                {"process_memory_enforcer": None, "engine_pool": None},
            ),
        ):
            from fusion_mlx.routes_internal.config_reload import reload_config

            result = await reload_config(source="test")
            mock_ll.assert_called_once_with("error")
            assert "server.log_level" in result["applied"]

    @pytest.mark.asyncio
    async def test_bad_schema_returns_errors(self, tmp_path):
        settings_file = tmp_path / "settings.json"
        settings_file.write_text(
            json.dumps({"memory": {"memory_guard_tier": "STRICT"}})
        )
        with patch(
            "fusion_mlx.routes_internal.config_reload._settings_json_path",
            return_value=settings_file,
        ):
            from fusion_mlx.routes_internal.config_reload import reload_config

            result = await reload_config(source="test")
            assert result["applied"] == []
            assert len(result["errors"]) > 0

    @pytest.mark.asyncio
    async def test_idle_timeout_pushed_to_enforcer(self, tmp_path):
        settings_file = tmp_path / "settings.json"
        settings_file.write_text(
            json.dumps({"idle_timeout": {"idle_timeout_seconds": 600}})
        )
        enforcer = MagicMock()
        with (
            patch(
                "fusion_mlx.routes_internal.config_reload._settings_json_path",
                return_value=settings_file,
            ),
            patch(
                "fusion_mlx.server._server_state",
                {"process_memory_enforcer": enforcer, "engine_pool": None},
            ),
        ):
            from fusion_mlx.routes_internal.config_reload import reload_config

            result = await reload_config(source="test")
            enforcer.set_reloaded_idle_timeout.assert_called_once_with(600)
            assert "idle_timeout.idle_timeout_seconds" in result["applied"]

    @pytest.mark.asyncio
    async def test_restart_needed_fields_reported(self, tmp_path):
        settings_file = tmp_path / "settings.json"
        settings_file.write_text(
            json.dumps({"server": {"host": "0.0.0.0", "port": 8080}})
        )
        with (
            patch(
                "fusion_mlx.routes_internal.config_reload._settings_json_path",
                return_value=settings_file,
            ),
            patch(
                "fusion_mlx.server._server_state",
                {"process_memory_enforcer": None, "engine_pool": None},
            ),
        ):
            from fusion_mlx.routes_internal.config_reload import reload_config

            result = await reload_config(source="test")
            joined = " ".join(result["not_applied"])
            assert "server.host" in joined
            assert "server.port" in joined

    @pytest.mark.asyncio
    async def test_route_guard_env_mutated(self, tmp_path):
        import os

        settings_file = tmp_path / "settings.json"
        settings_file.write_text(
            json.dumps({"route_guard": {"warn_only": True, "token": "secret123"}})
        )
        old_warn = os.environ.get("FUSION_ROUTE_WARN_ONLY")
        old_tok = os.environ.get("FUSION_ROUTE_TOKEN")
        try:
            with (
                patch(
                    "fusion_mlx.routes_internal.config_reload._settings_json_path",
                    return_value=settings_file,
                ),
                patch(
                    "fusion_mlx.server._server_state",
                    {"process_memory_enforcer": None, "engine_pool": None},
                ),
            ):
                from fusion_mlx.routes_internal.config_reload import reload_config

                result = await reload_config(source="test")
                assert os.environ.get("FUSION_ROUTE_WARN_ONLY") == "true"
                assert os.environ.get("FUSION_ROUTE_TOKEN") == "secret123"
                assert "route_guard.warn_only" in result["applied"]
        finally:
            for k, v in [
                ("FUSION_ROUTE_WARN_ONLY", old_warn),
                ("FUSION_ROUTE_TOKEN", old_tok),
            ]:
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v


class TestEnforcerIdleTimeoutReload:
    def test_set_reloaded_idle_timeout(self):
        from fusion_mlx.pool.memory_enforcer import ProcessMemoryEnforcer

        enforcer = ProcessMemoryEnforcer.__new__(ProcessMemoryEnforcer)
        enforcer._global_settings = None
        enforcer._reloaded_idle_timeout_seconds = None
        assert enforcer.get_global_idle_timeout_seconds() is None
        enforcer.set_reloaded_idle_timeout(900)
        assert enforcer.get_global_idle_timeout_seconds() == 900
        enforcer.set_reloaded_idle_timeout(None)
        assert enforcer.get_global_idle_timeout_seconds() is None
