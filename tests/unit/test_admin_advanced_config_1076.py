# SPDX-License-Identifier: Apache-2.0
"""#1076: admin Scheduler/Memory/Cloud advanced config tab.

Covers:
- 22 new fields registered in REQUIRES_RESTART_FIELDS.
- restart_fields_in_request intersects the new advanced fields.
- _read_live_advanced_config returns scheduler_advanced/memory_advanced/cloud.
- GET global-settings response carries the 3 advanced sections.
- POST fallback persists advanced fields into settings.json.
"""

import asyncio
from unittest.mock import MagicMock, patch

from fusion_mlx.routes_internal.config_reload import (
    REQUIRES_RESTART_FIELDS,
    restart_fields_in_request,
)

_ADV_FIELDS = (
    "scheduler_max_num_seqs",
    "scheduler_max_num_batched_tokens",
    "scheduler_policy",
    "scheduler_prefill_batch_size",
    "scheduler_completion_batch_size",
    "scheduler_prefill_step_size",
    "scheduler_max_waiting",
    "scheduler_use_paged_cache",
    "scheduler_paged_cache_block_size",
    "scheduler_max_cache_blocks",
    "scheduler_enable_mtp",
    "scheduler_spec_decode",
    "scheduler_gpu_memory_utilization",
    "memory_per_engine_pct",
    "memory_soft_threshold",
    "memory_hard_threshold",
    "cloud_router_enabled",
    "cloud_router_model",
    "cloud_router_api_key",
    "cloud_router_api_base",
    "cloud_router_threshold",
    "cloud_fallback_consent",
)


class TestAdvancedRestartFields:
    def test_all_22_advanced_fields_in_restart_set(self):
        for f in _ADV_FIELDS:
            assert f in REQUIRES_RESTART_FIELDS, f"{f} missing from restart set"

    def test_restart_fields_in_request_scheduler_subset(self):
        changed = {"scheduler_max_num_seqs", "scheduler_policy", "log_level"}
        result = restart_fields_in_request(changed)
        assert "scheduler_max_num_seqs" in result
        assert "scheduler_policy" in result
        assert "log_level" not in result

    def test_restart_fields_in_request_memory_subset(self):
        changed = {"memory_per_engine_pct", "memory_soft_threshold"}
        result = restart_fields_in_request(changed)
        assert "memory_per_engine_pct" in result
        assert "memory_soft_threshold" in result

    def test_restart_fields_in_request_cloud_subset(self):
        changed = {"cloud_router_enabled", "cloud_router_api_key", "api_key"}
        result = restart_fields_in_request(changed)
        assert "cloud_router_enabled" in result
        assert "cloud_router_api_key" in result
        assert "api_key" not in result

    def test_restart_fields_in_request_empty(self):
        assert restart_fields_in_request(set()) == []


class TestReadLiveAdvancedConfig:
    def test_returns_three_sections_from_live_config(self):
        from fusion_mlx.admin import settings as admin_settings

        cfg = MagicMock()
        cfg.scheduler.max_num_seqs = 256
        cfg.scheduler.max_num_batched_tokens = 65536
        cfg.scheduler.policy.value = "FCFS"
        cfg.scheduler.prefill_batch_size = 8
        cfg.scheduler.completion_batch_size = 32
        cfg.scheduler.prefill_step_size = 2048
        cfg.scheduler.max_waiting = 0
        cfg.scheduler.use_paged_cache = False
        cfg.scheduler.paged_cache_block_size = 64
        cfg.scheduler.max_cache_blocks = 1000
        cfg.scheduler.enable_mtp = False
        cfg.scheduler.spec_decode = "none"
        cfg.scheduler.gpu_memory_utilization = 0.9
        cfg.memory.per_engine_pct = 0.7
        cfg.memory.soft_threshold = 0.85
        cfg.memory.hard_threshold = 0.95
        cfg.cloud_router_enabled = False
        cfg.cloud_router_model = None
        cfg.cloud_router_api_key = None
        cfg.cloud_router_api_base = None
        cfg.cloud_router_threshold = 32768
        cfg.cloud_fallback_consent = False
        with patch("fusion_mlx.config.get_config", return_value=cfg):
            result = admin_settings._read_live_advanced_config()
        assert "scheduler_advanced" in result
        assert "memory_advanced" in result
        assert "cloud" in result
        assert result["scheduler_advanced"]["max_num_seqs"] == 256
        assert result["scheduler_advanced"]["policy"] == "FCFS"
        assert result["memory_advanced"]["per_engine_pct"] == 0.7
        assert result["cloud"]["cloud_router_threshold"] == 32768

    def test_masks_cloud_api_key_when_present(self):
        from fusion_mlx.admin import settings as admin_settings

        cfg = MagicMock()
        cfg.scheduler.policy.value = "FCFS"
        cfg.cloud_router_api_key = "sk-secret-1234567890"
        cfg.cloud_router_enabled = True
        cfg.cloud_router_model = "gpt-4o"
        cfg.cloud_router_api_base = "https://api.openai.com/v1"
        cfg.cloud_router_threshold = 4096
        cfg.cloud_fallback_consent = True
        with patch("fusion_mlx.config.get_config", return_value=cfg):
            result = admin_settings._read_live_advanced_config()
        masked = result["cloud"]["cloud_router_api_key"]
        assert masked is not None
        assert "sk-secret" not in masked
        assert "*" in masked

    def test_falls_back_to_empty_when_config_unavailable(self):
        from fusion_mlx.admin import settings as admin_settings

        with patch("fusion_mlx.config.get_config", side_effect=RuntimeError("no cfg")):
            with patch.object(
                admin_settings, "_get_settings_manager", return_value=None
            ):
                result = admin_settings._read_live_advanced_config()
        assert result == {
            "scheduler_advanced": {},
            "memory_advanced": {},
            "cloud": {},
        }


class TestGetGlobalSettingsAdvancedSections:
    def test_rich_response_includes_advanced_sections(self):
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
        adv_result = {
            "scheduler_advanced": {"max_num_seqs": 256, "policy": "FCFS"},
            "memory_advanced": {"per_engine_pct": 0.7},
            "cloud": {"cloud_router_enabled": False},
        }
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
                        with patch.object(
                            admin_settings,
                            "_read_live_advanced_config",
                            return_value=adv_result,
                        ):
                            result = asyncio.run(
                                admin_settings.get_global_settings(is_admin=True)
                            )
        assert "scheduler_advanced" in result
        assert result["scheduler_advanced"]["max_num_seqs"] == 256
        assert "memory_advanced" in result
        assert "cloud" in result
        assert "scheduler_max_num_seqs" in result["requires_restart_fields"]


class TestPostFallbackPersistsAdvanced:
    def test_fallback_writes_advanced_fields_to_settings_json(self):
        from fusion_mlx.admin import settings as admin_settings

        request = MagicMock()
        request.model_fields_set = {
            "scheduler_max_num_seqs",
            "scheduler_policy",
            "memory_per_engine_pct",
            "cloud_router_enabled",
        }
        for f in _ADV_FIELDS:
            setattr(request, f, None)
        request.scheduler_max_num_seqs = 512
        request.scheduler_policy = "PRIORITY"
        request.memory_per_engine_pct = 0.8
        request.cloud_router_enabled = True
        # Non-advanced fields the fallback reads — MagicMock hasattr is always
        # True, so set them explicitly to None to bypass validation.
        for attr in (
            "host",
            "port",
            "log_level",
            "sse_keepalive_mode",
            "server_aliases",
            "model_dirs",
            "model_dir",
            "model_fallback",
            "memory_guard_tier",
            "memory_guard_custom_ceiling_gb",
            "memory_prefill_memory_guard",
            "max_concurrent_requests",
            "embedding_batch_size",
            "chunked_prefill",
            "cache_enabled",
            "ssd_cache_dir",
            "ssd_cache_max_size",
            "hot_cache_only",
            "hot_cache_max_size",
            "initial_cache_blocks",
            "mcp_config",
            "hf_endpoint",
            "ms_endpoint",
            "network_http_proxy",
            "network_https_proxy",
            "network_no_proxy",
            "network_ca_bundle",
            "sampling_max_context_window",
            "sampling_max_tokens",
            "sampling_temperature",
            "sampling_top_p",
            "sampling_top_k",
            "sampling_repetition_penalty",
            "claude_code_context_scaling_enabled",
            "claude_code_target_context_size",
            "claude_code_mode",
            "claude_code_opus_model",
            "claude_code_sonnet_model",
            "claude_code_haiku_model",
            "integrations_copilot_model",
            "integrations_codex_model",
            "integrations_opencode_model",
            "integrations_openclaw_model",
            "integrations_hermes_model",
            "integrations_pi_model",
            "integrations_openclaw_tools_profile",
            "ui_language",
            "idle_timeout_seconds",
            "api_key",
            "skip_api_key_verification",
        ):
            setattr(request, attr, None)
        captured = {}

        def fake_write(sj):
            captured["sj"] = sj

        with patch.object(
            admin_settings, "_get_rich_global_settings", return_value=None
        ):
            with patch.object(admin_settings, "_get_server_state", return_value={}):
                with patch.object(
                    admin_settings, "_get_engine_pool", return_value=None
                ):
                    with patch.object(
                        admin_settings, "_read_settings_json", return_value={}
                    ):
                        with patch.object(
                            admin_settings,
                            "_write_settings_json",
                            side_effect=fake_write,
                        ):
                            with patch.object(
                                admin_settings, "_apply_log_level_runtime"
                            ):
                                result = admin_settings._save_global_settings_fallback(
                                    request
                                )
        sj = captured["sj"]
        assert sj["scheduler"]["max_num_seqs"] == 512
        assert sj["scheduler"]["policy"] == "PRIORITY"
        assert sj["memory"]["per_engine_pct"] == 0.8
        assert sj["cloud"]["cloud_router_enabled"] is True
        assert result["requires_restart"] is True
        assert "scheduler_max_num_seqs" in result["restart_fields"]
        assert "cloud_router_enabled" in result["restart_fields"]
