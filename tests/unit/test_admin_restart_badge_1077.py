# SPDX-License-Identifier: Apache-2.0
"""#1077: unified "requires restart" metadata for admin settings.

Covers:
- REQUIRES_RESTART_FIELDS canonical set + get_requires_restart_metadata.
- restart_fields_in_request intersects request fields with the restart set.
- get_global_settings response carries requires_restart_fields.
- update_global_settings response carries requires_restart + restart_fields.
- _build_fallback_global_settings carries requires_restart_fields.
"""

import asyncio
from unittest.mock import MagicMock, patch

from fusion_mlx.routes_internal.config_reload import (
    REQUIRES_RESTART_FIELDS,
    get_requires_restart_metadata,
    restart_fields_in_request,
)


class TestRequiresRestartMetadata:
    def test_canonical_set_contains_expected_fields(self):
        for f in (
            "host",
            "port",
            "max_concurrent_requests",
            "mcp_config",
            "ssd_cache_dir",
            "initial_cache_blocks",
            "hot_cache_only",
        ):
            assert f in REQUIRES_RESTART_FIELDS

    def test_hot_applied_fields_not_in_restart_set(self):
        # These are runtime-applied (hot) — must NOT be in the restart set.
        for f in ("log_level", "chunked_prefill", "cache_enabled", "api_key"):
            assert f not in REQUIRES_RESTART_FIELDS

    def test_get_requires_restart_metadata_sorted_list(self):
        meta = get_requires_restart_metadata()
        assert isinstance(meta, list)
        assert meta == sorted(meta)
        assert "host" in meta
        assert "ssd_cache_dir" in meta

    def test_restart_fields_in_request_full_intersect(self):
        changed = {"host", "port", "log_level", "api_key"}
        result = restart_fields_in_request(changed)
        assert "host" in result
        assert "port" in result
        assert "log_level" not in result
        assert "api_key" not in result

    def test_restart_fields_in_request_empty(self):
        assert restart_fields_in_request(set()) == []

    def test_restart_fields_in_request_no_overlap(self):
        assert restart_fields_in_request({"log_level", "api_key"}) == []

    def test_restart_fields_in_request_all_restart(self):
        result = restart_fields_in_request({"ssd_cache_dir", "hot_cache_only"})
        assert result == ["hot_cache_only", "ssd_cache_dir"]


class TestGetGlobalSettingsRestartsFields:
    def test_rich_response_includes_restart_fields(self):
        from fusion_mlx.admin import settings as admin_settings

        _MEM_KEYS = (
            "total_bytes",
            "total_formatted",
            "auto_limit_formatted",
            "available_bytes",
            "fusionmlx_phys_footprint_bytes",
            "free_memory_bytes",
            "inactive_memory_bytes",
            "active_memory_bytes",
            "iogpu_wired_limit_bytes",
            "fusionmlx_wired_limit_request_bytes",
        )
        _DISK_KEYS = ("total_bytes", "total_formatted")
        mem_info = {k: 0 for k in _MEM_KEYS}
        disk_info = {k: 0 for k in _DISK_KEYS}
        rich = MagicMock()
        rich.base_path = MagicMock()
        rich.base_path.__str__ = lambda self: "/tmp"
        rich.server.host = "127.0.0.1"
        rich.server.port = 11434
        rich.server.log_level = "info"
        rich.server.server_aliases = []
        rich.server.sse_keepalive_mode = "chunk"
        rich.model.get_model_dirs = MagicMock(return_value=[])
        rich.model.get_model_dir = MagicMock(return_value="/models")
        rich.model.model_fallback = False
        rich.memory.prefill_memory_guard = False
        rich.memory.memory_guard_tier = "safe"
        rich.memory.memory_guard_custom_ceiling_gb = None
        rich.scheduler.max_concurrent_requests = 8
        rich.scheduler.embedding_batch_size = 32
        rich.scheduler.chunked_prefill = False
        rich.cache.enabled = False
        rich.cache.get_ssd_cache_dir = MagicMock(return_value="/cache")
        rich.cache.get_ssd_cache_max_size_bytes = MagicMock(return_value=0)
        rich.cache.hot_cache_only = False
        rich.cache.hot_cache_max_size = "0"
        rich.cache.initial_cache_blocks = 0
        rich.cache.ssd_cache_dir = "/cache"
        rich.mcp.config_path = None
        rich.huggingface.endpoint = ""
        rich.modelscope.endpoint = ""
        rich.network.http_proxy = ""
        rich.network.https_proxy = ""
        rich.network.no_proxy = ""
        rich.network.ca_bundle = ""
        rich.sampling.max_context_window = 4096
        rich.sampling.max_tokens = 512
        rich.sampling.temperature = 0.0
        rich.sampling.top_p = 1.0
        rich.sampling.top_k = 0
        rich.sampling.repetition_penalty = 1.0
        rich.auth.api_key = ""
        rich.auth.skip_api_key_verification = False
        rich.auth.sub_keys = []
        rich.claude_code.context_scaling_enabled = False
        rich.claude_code.target_context_size = 200000
        rich.claude_code.mode = None
        rich.claude_code.opus_model = None
        rich.claude_code.sonnet_model = None
        rich.claude_code.haiku_model = None
        rich.integrations.codex_model = None
        rich.integrations.opencode_model = None
        rich.integrations.openclaw_model = None
        rich.integrations.hermes_model = None
        rich.integrations.pi_model = None
        rich.integrations.copilot_model = None
        rich.integrations.openclaw_tools_profile = None
        rich.ui.language = "en"
        rich.idle_timeout.idle_timeout_seconds = None
        with patch.object(
            admin_settings, "_get_rich_global_settings", return_value=rich
        ):
            with patch.object(
                admin_settings, "get_system_memory_info", return_value=mem_info
            ):
                with patch.object(
                    admin_settings, "get_ssd_disk_info", return_value=disk_info
                ):
                    with patch.object(
                        admin_settings, "_format_cache_size", return_value="0"
                    ):
                        result = asyncio.run(
                            admin_settings.get_global_settings(is_admin=True)
                        )
        assert "requires_restart_fields" in result
        assert "ssd_cache_dir" in result["requires_restart_fields"]
        assert "host" in result["requires_restart_fields"]


class TestFallbackGlobalSettingsRestartsFields:
    def test_fallback_response_includes_restart_fields(self):
        from fusion_mlx.admin import settings as admin_settings

        _MEM_KEYS = (
            "total_bytes",
            "total_formatted",
            "auto_limit_formatted",
            "available_bytes",
            "fusionmlx_phys_footprint_bytes",
            "free_memory_bytes",
            "inactive_memory_bytes",
            "active_memory_bytes",
            "iogpu_wired_limit_bytes",
            "fusionmlx_wired_limit_request_bytes",
        )
        _DISK_KEYS = ("total_bytes", "total_formatted")
        mem_info = {k: 0 for k in _MEM_KEYS}
        disk_info = {k: 0 for k in _DISK_KEYS}
        with patch.object(
            admin_settings, "_get_rich_global_settings", return_value=None
        ):
            with patch.object(admin_settings, "_get_server_state", return_value={}):
                with patch.object(
                    admin_settings, "_get_engine_pool", return_value=None
                ):
                    with patch.object(
                        admin_settings,
                        "get_system_memory_info",
                        return_value=mem_info,
                    ):
                        with patch.object(
                            admin_settings,
                            "get_ssd_disk_info",
                            return_value=disk_info,
                        ):
                            result = admin_settings._build_fallback_global_settings()
        assert "requires_restart_fields" in result
        assert "host" in result["requires_restart_fields"]


class TestUpdateGlobalSettingsRestartResponse:
    def test_save_response_has_requires_restart_false_when_no_restart_fields(
        self,
    ):
        from fusion_mlx.admin import settings as admin_settings

        request = MagicMock()
        request.model_fields_set = {"log_level", "api_key"}
        request.host = None
        request.port = None
        request.log_level = "debug"
        request.sse_keepalive_mode = None
        request.server_aliases = None
        request.model_dirs = None
        request.model_dir = None
        request.model_fallback = None
        request.memory_guard_tier = None
        request.memory_guard_custom_ceiling_gb = None
        request.memory_prefill_memory_guard = None
        request.max_concurrent_requests = None
        request.embedding_batch_size = None
        request.chunked_prefill = None
        request.cache_enabled = None
        request.ssd_cache_dir = None
        request.ssd_cache_max_size = None
        request.hot_cache_only = None
        request.hot_cache_max_size = None
        request.initial_cache_blocks = None
        request.mcp_config = None
        request.hf_endpoint = None
        request.ms_endpoint = None
        request.network_http_proxy = None
        request.network_https_proxy = None
        request.network_no_proxy = None
        request.network_ca_bundle = None
        request.sampling_max_context_window = None
        request.sampling_max_tokens = None
        request.sampling_temperature = None
        request.sampling_top_p = None
        request.sampling_top_k = None
        request.sampling_repetition_penalty = None
        request.claude_code_context_scaling_enabled = None
        request.claude_code_target_context_size = None
        request.claude_code_mode = None
        request.claude_code_opus_model = None
        request.claude_code_sonnet_model = None
        request.claude_code_haiku_model = None
        request.integrations_copilot_model = None
        request.integrations_codex_model = None
        request.integrations_opencode_model = None
        request.integrations_openclaw_model = None
        request.integrations_hermes_model = None
        request.integrations_pi_model = None
        request.integrations_openclaw_tools_profile = None
        request.ui_language = None
        request.idle_timeout_seconds = None
        request.api_key = None
        request.skip_api_key_verification = None
        with patch.object(
            admin_settings, "_get_rich_global_settings", return_value=None
        ):
            result = admin_settings._save_global_settings_fallback(request)
        assert result["requires_restart"] is False
        assert result["restart_fields"] == []

    def test_save_response_has_requires_restart_true_when_host_changed(self):
        from fusion_mlx.admin import settings as admin_settings

        request = MagicMock()
        request.model_fields_set = {"host", "log_level"}
        request.host = "0.0.0.0"
        request.port = None
        request.log_level = "info"
        request.sse_keepalive_mode = None
        request.server_aliases = None
        request.model_dirs = None
        request.model_dir = None
        request.model_fallback = None
        request.memory_guard_tier = None
        request.memory_guard_custom_ceiling_gb = None
        request.memory_prefill_memory_guard = None
        request.max_concurrent_requests = None
        request.embedding_batch_size = None
        request.chunked_prefill = None
        request.cache_enabled = None
        request.ssd_cache_dir = None
        request.ssd_cache_max_size = None
        request.hot_cache_only = None
        request.hot_cache_max_size = None
        request.initial_cache_blocks = None
        request.mcp_config = None
        request.hf_endpoint = None
        request.ms_endpoint = None
        request.network_http_proxy = None
        request.network_https_proxy = None
        request.network_no_proxy = None
        request.network_ca_bundle = None
        request.sampling_max_context_window = None
        request.sampling_max_tokens = None
        request.sampling_temperature = None
        request.sampling_top_p = None
        request.sampling_top_k = None
        request.sampling_repetition_penalty = None
        request.claude_code_context_scaling_enabled = None
        request.claude_code_target_context_size = None
        request.claude_code_mode = None
        request.claude_code_opus_model = None
        request.claude_code_sonnet_model = None
        request.claude_code_haiku_model = None
        request.integrations_copilot_model = None
        request.integrations_codex_model = None
        request.integrations_opencode_model = None
        request.integrations_openclaw_model = None
        request.integrations_hermes_model = None
        request.integrations_pi_model = None
        request.integrations_openclaw_tools_profile = None
        request.ui_language = None
        request.idle_timeout_seconds = None
        request.api_key = None
        request.skip_api_key_verification = None
        with patch.object(
            admin_settings, "_get_rich_global_settings", return_value=None
        ):
            with patch.object(admin_settings, "_read_settings_json", return_value={}):
                with patch.object(admin_settings, "_write_settings_json"):
                    with patch.object(
                        admin_settings,
                        "_apply_log_level_runtime",
                    ):
                        result = admin_settings._save_global_settings_fallback(request)
        assert result["requires_restart"] is True
        assert "host" in result["restart_fields"]
        assert "log_level" not in result["restart_fields"]
