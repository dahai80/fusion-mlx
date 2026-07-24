# SPDX-License-Identifier: Apache-2.0
"""Admin panel routes for Fusion-MLX server configuration.

This module provides HTTP routes for the admin panel including:
- Login/logout with API key authentication
- Dashboard for server monitoring
- Model settings management (per-model sampling parameters, pinning, default)
- Global settings management
"""

import json
import logging
import os
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

from .auth import (
    require_admin,
    validate_api_key,
)

logger = logging.getLogger(__name__)


def _mask_api_key(key: str | None) -> str:
    if not key:
        return ""
    if len(key) <= 4:
        return "****"
    return "*" * (len(key) - 4) + key[-4:]


PRESET_REMOTE_URL = "http://bench.dpdns.org/assets/fusionmlx_preset.json"


from .helpers import (
    _apply_cache_settings_runtime,
    _apply_log_level_runtime,
    _apply_memory_guard_tier_runtime,
    _apply_model_dirs_runtime,
    _apply_sampling_settings_runtime,
    _format_cache_size,
    _get_engine_pool,
    _get_rich_global_settings,
    _get_server_state,
    _schedule_self_terminate,
    get_ssd_disk_info,
    get_system_memory_info,
)
from .models import (
    GlobalSettingsRequest,
)


def _get_settings_json_path() -> Path:
    return Path.home() / ".fusion-mlx" / "settings.json"


def _read_settings_json() -> dict:
    path = _get_settings_json_path()
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _write_settings_json(data: dict) -> None:
    path = _get_settings_json_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _save_global_settings_fallback(request: GlobalSettingsRequest) -> dict:
    """Save global settings to settings.json when rich GlobalSettings is unavailable.

    Reads settings.json, applies the requested changes, writes back.
    Returns a dict with success/message/runtime_applied.
    """
    sj = _read_settings_json()
    runtime_applied: list[str] = []

    # Server settings
    if request.host is not None:
        sj.setdefault("server", {})["host"] = request.host
    if request.port is not None:
        sj.setdefault("server", {})["port"] = request.port
    if request.log_level is not None:
        sj.setdefault("server", {})["log_level"] = request.log_level
        _apply_log_level_runtime(request.log_level)
        runtime_applied.append("log_level")
    if request.sse_keepalive_mode is not None:
        valid_modes = {"chunk", "comment", "off"}
        if request.sse_keepalive_mode not in valid_modes:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid sse_keepalive_mode: {request.sse_keepalive_mode}",
            )
        sj.setdefault("server", {})["sse_keepalive_mode"] = request.sse_keepalive_mode
        runtime_applied.append("sse_keepalive_mode")

    # Model settings
    if request.model_dirs is not None:
        new_dirs = [d for d in request.model_dirs if d.strip()]
        sj.setdefault("model", {})["model_dirs"] = new_dirs
        if new_dirs:
            sj["model"]["model_dir"] = new_dirs[0]
        runtime_applied.append("model_dirs")
    elif request.model_dir is not None:
        sj.setdefault("model", {})["model_dir"] = request.model_dir
        sj["model"]["model_dirs"] = [request.model_dir]
        runtime_applied.append("model_dirs")

    if request.model_fallback is not None:
        sj.setdefault("model", {})["model_fallback"] = request.model_fallback
        runtime_applied.append("model_fallback")

    # Memory settings
    if request.memory_guard_tier is not None:
        sj.setdefault("memory", {})["memory_guard_tier"] = request.memory_guard_tier
        runtime_applied.append("memory_guard_tier")
    if request.memory_guard_custom_ceiling_gb is not None:
        sj.setdefault("memory", {})[
            "memory_guard_custom_ceiling_gb"
        ] = request.memory_guard_custom_ceiling_gb
        runtime_applied.append("memory_guard_custom_ceiling_gb")
    if request.memory_prefill_memory_guard is not None:
        sj.setdefault("memory", {})[
            "prefill_memory_guard"
        ] = request.memory_prefill_memory_guard
        runtime_applied.append("prefill_memory_guard")

    # Scheduler settings
    if request.max_concurrent_requests is not None:
        sj.setdefault("scheduler", {})[
            "max_concurrent_requests"
        ] = request.max_concurrent_requests
    if request.embedding_batch_size is not None:
        if request.embedding_batch_size <= 0:
            raise HTTPException(
                status_code=400, detail="Invalid embedding_batch_size: must be > 0"
            )
        sj.setdefault("scheduler", {})[
            "embedding_batch_size"
        ] = request.embedding_batch_size
        runtime_applied.append("embedding_batch_size")
    if request.chunked_prefill is not None:
        sj.setdefault("scheduler", {})["chunked_prefill"] = request.chunked_prefill
        runtime_applied.append("chunked_prefill")

    # Cache settings
    if request.cache_enabled is not None:
        sj.setdefault("cache", {})["enabled"] = request.cache_enabled
        runtime_applied.append("cache")
    if request.ssd_cache_dir is not None:
        sj.setdefault("cache", {})["ssd_cache_dir"] = request.ssd_cache_dir
    if request.ssd_cache_max_size is not None:
        sj.setdefault("cache", {})["ssd_cache_max_size"] = request.ssd_cache_max_size
    if request.hot_cache_only is not None:
        sj.setdefault("cache", {})["hot_cache_only"] = request.hot_cache_only
    if request.hot_cache_max_size is not None:
        sj.setdefault("cache", {})["hot_cache_max_size"] = request.hot_cache_max_size
    if request.initial_cache_blocks is not None:
        sj.setdefault("cache", {})[
            "initial_cache_blocks"
        ] = request.initial_cache_blocks

    # MCP settings
    if request.mcp_config is not None:
        sj.setdefault("mcp", {})["config_path"] = request.mcp_config or None

    # HuggingFace settings
    if request.hf_endpoint is not None:
        sj.setdefault("huggingface", {})["endpoint"] = request.hf_endpoint
        if request.hf_endpoint:
            os.environ["HF_ENDPOINT"] = request.hf_endpoint
        else:
            os.environ.pop("HF_ENDPOINT", None)
        runtime_applied.append("hf_endpoint")

    # ModelScope settings
    if request.ms_endpoint is not None:
        sj.setdefault("modelscope", {})["endpoint"] = request.ms_endpoint
        if request.ms_endpoint:
            os.environ["MODELSCOPE_DOMAIN"] = request.ms_endpoint
        else:
            os.environ.pop("MODELSCOPE_DOMAIN", None)
        runtime_applied.append("ms_endpoint")

    # Network settings
    network_changed = False
    if request.network_http_proxy is not None:
        sj.setdefault("network", {})["http_proxy"] = request.network_http_proxy
        if request.network_http_proxy:
            os.environ["HTTP_PROXY"] = request.network_http_proxy
            os.environ["http_proxy"] = request.network_http_proxy
        else:
            os.environ.pop("HTTP_PROXY", None)
            os.environ.pop("http_proxy", None)
        network_changed = True
    if request.network_https_proxy is not None:
        sj.setdefault("network", {})["https_proxy"] = request.network_https_proxy
        if request.network_https_proxy:
            os.environ["HTTPS_PROXY"] = request.network_https_proxy
            os.environ["https_proxy"] = request.network_https_proxy
        else:
            os.environ.pop("HTTPS_PROXY", None)
            os.environ.pop("https_proxy", None)
        network_changed = True
    if request.network_no_proxy is not None:
        sj.setdefault("network", {})["no_proxy"] = request.network_no_proxy
        if request.network_no_proxy:
            os.environ["NO_PROXY"] = request.network_no_proxy
            os.environ["no_proxy"] = request.network_no_proxy
        else:
            os.environ.pop("NO_PROXY", None)
            os.environ.pop("no_proxy", None)
        network_changed = True
    if request.network_ca_bundle is not None:
        sj.setdefault("network", {})["ca_bundle"] = request.network_ca_bundle
        if request.network_ca_bundle:
            os.environ["REQUESTS_CA_BUNDLE"] = request.network_ca_bundle
            os.environ["SSL_CERT_FILE"] = request.network_ca_bundle
        else:
            os.environ.pop("REQUESTS_CA_BUNDLE", None)
            os.environ.pop("SSL_CERT_FILE", None)
        network_changed = True
    if network_changed:
        runtime_applied.append("network")

    # Sampling settings
    sampling_changed = False
    if request.sampling_max_context_window is not None:
        sj.setdefault("sampling", {})[
            "max_context_window"
        ] = request.sampling_max_context_window
        sampling_changed = True
    if request.sampling_max_tokens is not None:
        sj.setdefault("sampling", {})["max_tokens"] = request.sampling_max_tokens
        sampling_changed = True
    if request.sampling_temperature is not None:
        sj.setdefault("sampling", {})["temperature"] = request.sampling_temperature
        sampling_changed = True
    if request.sampling_top_p is not None:
        sj.setdefault("sampling", {})["top_p"] = request.sampling_top_p
        sampling_changed = True
    if request.sampling_top_k is not None:
        sj.setdefault("sampling", {})["top_k"] = request.sampling_top_k
        sampling_changed = True
    if request.sampling_repetition_penalty is not None:
        sj.setdefault("sampling", {})[
            "repetition_penalty"
        ] = request.sampling_repetition_penalty
        sampling_changed = True
    if sampling_changed:
        runtime_applied.append("sampling")

    # Claude Code settings
    cc_changed = False
    if request.claude_code_context_scaling_enabled is not None:
        sj.setdefault("claude_code", {})[
            "context_scaling_enabled"
        ] = request.claude_code_context_scaling_enabled
        cc_changed = True
    if request.claude_code_target_context_size is not None:
        sj.setdefault("claude_code", {})[
            "target_context_size"
        ] = request.claude_code_target_context_size
        cc_changed = True
    if request.claude_code_mode is not None:
        sj.setdefault("claude_code", {})["mode"] = request.claude_code_mode
        cc_changed = True
    if "claude_code_opus_model" in request.model_fields_set:
        sj.setdefault("claude_code", {})["opus_model"] = request.claude_code_opus_model
        cc_changed = True
    if "claude_code_sonnet_model" in request.model_fields_set:
        sj.setdefault("claude_code", {})[
            "sonnet_model"
        ] = request.claude_code_sonnet_model
        cc_changed = True
    if "claude_code_haiku_model" in request.model_fields_set:
        sj.setdefault("claude_code", {})[
            "haiku_model"
        ] = request.claude_code_haiku_model
        cc_changed = True
    if cc_changed:
        runtime_applied.append("claude_code")

    # Integrations settings
    int_changed = False
    if "integrations_copilot_model" in request.model_fields_set:
        sj.setdefault("integrations", {})[
            "copilot_model"
        ] = request.integrations_copilot_model
        int_changed = True
    if "integrations_codex_model" in request.model_fields_set:
        sj.setdefault("integrations", {})[
            "codex_model"
        ] = request.integrations_codex_model
        int_changed = True
    if "integrations_opencode_model" in request.model_fields_set:
        sj.setdefault("integrations", {})[
            "opencode_model"
        ] = request.integrations_opencode_model
        int_changed = True
    if "integrations_openclaw_model" in request.model_fields_set:
        sj.setdefault("integrations", {})[
            "openclaw_model"
        ] = request.integrations_openclaw_model
        int_changed = True
    if "integrations_hermes_model" in request.model_fields_set:
        sj.setdefault("integrations", {})[
            "hermes_model"
        ] = request.integrations_hermes_model
        int_changed = True
    if "integrations_pi_model" in request.model_fields_set:
        sj.setdefault("integrations", {})["pi_model"] = request.integrations_pi_model
        int_changed = True
    if "integrations_openclaw_tools_profile" in request.model_fields_set:
        sj.setdefault("integrations", {})[
            "openclaw_tools_profile"
        ] = request.integrations_openclaw_tools_profile
        int_changed = True
    if int_changed:
        runtime_applied.append("integrations")

    # UI settings
    if request.ui_language is not None:
        sj.setdefault("ui", {})["language"] = request.ui_language
        runtime_applied.append("ui_language")

    # Idle timeout
    if "idle_timeout_seconds" in request.model_fields_set:
        sj.setdefault("idle_timeout", {})[
            "idle_timeout_seconds"
        ] = request.idle_timeout_seconds
        runtime_applied.append("idle_timeout_seconds")

    # Auth settings
    if request.api_key is not None:
        is_valid, error_msg = validate_api_key(request.api_key)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)
        sj.setdefault("auth", {})["api_key"] = request.api_key
        state = _get_server_state()
        if state is not None:
            state.api_key = request.api_key
        runtime_applied.append("api_key")

    if request.skip_api_key_verification is not None:
        sj.setdefault("auth", {})[
            "skip_api_key_verification"
        ] = request.skip_api_key_verification
        runtime_applied.append("skip_api_key_verification")

    # Persist
    try:
        _write_settings_json(sj)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to save settings")

    logger.info(f"Global settings saved (fallback mode): {runtime_applied}")
    return {
        "success": True,
        "message": "Settings saved.",
        "runtime_applied": runtime_applied,
    }


def _build_fallback_global_settings() -> dict:
    """Build a global-settings response from settings.json + runtime state
    when the rich GlobalSettings object is not available (flat Settings
    mode). This is the path used when the server runs with the released
    Settings class that lacks .cache/.server/.model nested attrs."""
    state = _get_server_state() or {}
    pool = _get_engine_pool()

    # Read settings.json directly for host/port/model_dirs/hf etc.
    settings_path = Path.home() / ".fusion-mlx" / "settings.json"
    sj = {}
    if settings_path.exists():
        try:
            sj = json.loads(settings_path.read_text())
        except Exception:
            pass

    server_sj = sj.get("server", {})
    model_sj = sj.get("model", {})
    hf_sj = sj.get("huggingface", {})
    auth_sj = sj.get("auth", {})
    scheduler_sj = sj.get("scheduler", {})
    memory_sj = sj.get("memory", {})
    cache_sj = sj.get("cache", {})
    mcp_sj = sj.get("mcp", {})

    host = server_sj.get("host", "127.0.0.1")
    port = server_sj.get("port", 11435)
    model_dir = model_sj.get("model_dir", str(Path.home() / ".fusion-mlx" / "models"))
    model_dirs = model_sj.get("model_dirs", [model_dir])
    hf_endpoint = hf_sj.get("endpoint", "")
    api_key = auth_sj.get("api_key", "")

    memory_info = get_system_memory_info()
    base_path = str(Path.home() / ".fusion-mlx")
    cache_dir = str(Path(base_path) / "cache")
    disk_info = get_ssd_disk_info(cache_dir)

    return {
        "base_path": base_path,
        "server": {
            "host": host,
            "port": port,
            "log_level": "info",
            "server_aliases": [],
            "sse_keepalive_mode": "chunk",
        },
        "model": {
            "model_dirs": model_dirs or [model_dir],
            "model_dir": (model_dirs or [model_dir])[0],
            "model_fallback": False,
        },
        "memory": {
            "prefill_memory_guard": memory_sj.get("prefill_memory_guard", False),
            "memory_guard_tier": memory_sj.get("memory_guard_tier", "safe"),
            "memory_guard_custom_ceiling_gb": memory_sj.get(
                "memory_guard_custom_ceiling_gb", None
            ),
        },
        "scheduler": {
            "max_concurrent_requests": scheduler_sj.get("max_concurrent_requests", 8),
            "embedding_batch_size": scheduler_sj.get("embedding_batch_size", 32),
            "chunked_prefill": scheduler_sj.get("chunked_prefill", False),
        },
        "cache": {
            "enabled": cache_sj.get("enabled", False),
            "ssd_cache_dir": cache_sj.get("ssd_cache_dir", cache_dir),
            "ssd_cache_max_size": cache_sj.get("ssd_cache_max_size", "10GB"),
            "hot_cache_only": cache_sj.get("hot_cache_only", False),
            "hot_cache_max_size": cache_sj.get("hot_cache_max_size", None),
            "initial_cache_blocks": cache_sj.get("initial_cache_blocks", 0),
        },
        "mcp": {"config_path": mcp_sj.get("config_path")},
        "huggingface": {"endpoint": hf_endpoint},
        "modelscope": {"endpoint": ""},
        "network": {
            "http_proxy": "",
            "https_proxy": "",
            "no_proxy": "",
            "ca_bundle": "",
        },
        "sampling": {
            "max_context_window": sj.get("sampling", {}).get(
                "max_context_window", 4096
            ),
            "max_tokens": sj.get("sampling", {}).get("max_tokens", 512),
            "temperature": sj.get("sampling", {}).get("temperature", 0.0),
            "top_p": sj.get("sampling", {}).get("top_p", 1.0),
            "top_k": sj.get("sampling", {}).get("top_k", 0),
            "repetition_penalty": sj.get("sampling", {}).get("repetition_penalty", 1.0),
        },
        "auth": {
            "api_key_set": bool(api_key),
            "api_key": _mask_api_key(api_key),
            "skip_api_key_verification": False,
            "sub_keys": [],
        },
        "claude_code": {
            "context_scaling_enabled": False,
            "target_context_size": 200000,
            "mode": None,
            "opus_model": None,
            "sonnet_model": None,
            "haiku_model": None,
        },
        "integrations": {
            "codex_model": None,
            "opencode_model": None,
            "openclaw_model": None,
            "hermes_model": None,
            "pi_model": None,
            "copilot_model": None,
            "openclaw_tools_profile": None,
        },
        "system": {
            "total_memory_bytes": memory_info["total_bytes"],
            "total_memory": memory_info["total_formatted"],
            "auto_model_memory": memory_info["auto_limit_formatted"],
            "available_memory_bytes": memory_info["available_bytes"],
            "fusionmlx_phys_footprint_bytes": memory_info[
                "fusionmlx_phys_footprint_bytes"
            ],
            "free_memory_bytes": memory_info["free_memory_bytes"],
            "inactive_memory_bytes": memory_info["inactive_memory_bytes"],
            "active_memory_bytes": memory_info["active_memory_bytes"],
            "iogpu_wired_limit_bytes": memory_info["iogpu_wired_limit_bytes"],
            "fusionmlx_wired_limit_request_bytes": memory_info[
                "fusionmlx_wired_limit_request_bytes"
            ],
            "ssd_total_bytes": disk_info["total_bytes"],
            "ssd_total": disk_info["total_formatted"],
        },
        "ui": {"language": ""},
        "idle_timeout": {"idle_timeout_seconds": None},
    }


_router = APIRouter()

# =============================================================================
# Global Settings API Routes
# =============================================================================


@_router.get("/api/server-info")
async def get_server_info(is_admin: bool = Depends(require_admin)):
    """Return server connectivity metadata for the dashboard.

    Provides the configured host, port, and the list of user-facing
    aliases (hostnames/IPs) that the dashboard can use to render
    selectable API URL hints.

    Returns:
        JSON object with ``host``, ``port``, and ``aliases``.

    Raises:
        HTTPException: 401 if not authenticated, 503 if server not initialized.
    """
    global_settings = _get_rich_global_settings()
    if global_settings is not None:
        configured = list(global_settings.server.server_aliases)
        if configured:
            aliases = configured
        else:
            try:
                from ..utils.network import detect_server_aliases

                aliases = detect_server_aliases(host=global_settings.server.host)
            except ImportError:
                aliases = []
        return {
            "host": global_settings.server.host,
            "port": global_settings.server.port,
            "aliases": aliases,
        }

    # Fallback: build from settings.json + runtime state
    fb = _build_fallback_global_settings()
    try:
        from ..utils.network import detect_server_aliases

        aliases = detect_server_aliases(host=fb["server"]["host"])
    except ImportError:
        aliases = []
    return {
        "host": fb["server"]["host"],
        "port": fb["server"]["port"],
        "aliases": aliases,
    }


@_router.post("/api/server/restart")
async def restart_server(is_admin: bool = Depends(require_admin)):
    """Trigger a server restart via the menubar supervisor.

    The handler does not perform the restart itself — it returns 202 and
    schedules ``os.kill(os.getpid(), SIGTERM)`` 500ms after the response
    is queued. The menubar app's ``ServerManager._health_check_loop``
    detects the process exit and respawns the server with a short
    backoff (~5s).

    Gated by the ``FUSIONMLX_SUPERVISED`` environment variable so plain
    ``fusion-mlx serve`` (no supervisor) returns 503 rather than killing the
    server with no respawn path.
    """
    supervisor = os.environ.get("FUSION_SUPERVISED")
    if not supervisor:
        raise HTTPException(
            status_code=503,
            detail=(
                "Server is not running under a supervisor that can "
                "respawn it. Restart unavailable — use the menu bar "
                "app's Restart, or restart from your shell."
            ),
        )

    _schedule_self_terminate(0.5)
    logger.warning("Server restart requested (supervisor=%s)", supervisor)

    # 5s backoff in ServerManager + ~1-2s startup = ~7s downtime budget.
    return JSONResponse(
        status_code=202,
        content={
            "status": "restarting",
            "supervisor": supervisor,
            "expected_downtime_seconds": 7,
        },
    )


@_router.get("/api/global-settings")
async def get_global_settings(is_admin: bool = Depends(require_admin)):
    """
    Get current global server settings.

    Returns the full global settings including server, model, scheduler,
    cache, and MCP configurations.

    Returns:
        JSON object with global settings.

    Raises:
        HTTPException: 401 if not authenticated, 503 if server not initialized.
    """
    global_settings = _get_rich_global_settings()

    if global_settings is None:
        # Flat Settings mode — build response from settings.json + runtime
        return _build_fallback_global_settings()

    # Get system memory info for auto calculation
    memory_info = get_system_memory_info()

    # Get SSD disk info for cache directory
    cache_dir = global_settings.cache.ssd_cache_dir or str(
        global_settings.cache.get_ssd_cache_dir(global_settings.base_path)
    )
    disk_info = get_ssd_disk_info(cache_dir)

    return {
        "base_path": str(global_settings.base_path),
        "server": {
            "host": global_settings.server.host,
            "port": global_settings.server.port,
            "log_level": global_settings.server.log_level,
            "server_aliases": list(global_settings.server.server_aliases),
            "sse_keepalive_mode": global_settings.server.sse_keepalive_mode,
        },
        "model": {
            "model_dirs": [
                str(d)
                for d in global_settings.model.get_model_dirs(global_settings.base_path)
            ],
            "model_dir": str(
                global_settings.model.get_model_dir(global_settings.base_path)
            ),
            "model_fallback": global_settings.model.model_fallback,
        },
        "memory": {
            "prefill_memory_guard": global_settings.memory.prefill_memory_guard,
            "memory_guard_tier": global_settings.memory.memory_guard_tier,
            "memory_guard_custom_ceiling_gb": global_settings.memory.memory_guard_custom_ceiling_gb,
        },
        "scheduler": {
            "max_concurrent_requests": global_settings.scheduler.max_concurrent_requests,
            "embedding_batch_size": global_settings.scheduler.embedding_batch_size,
            "chunked_prefill": global_settings.scheduler.chunked_prefill,
        },
        "cache": {
            "enabled": global_settings.cache.enabled,
            "ssd_cache_dir": cache_dir,
            # Resolve "auto" to actual value (10% of SSD capacity)
            "ssd_cache_max_size": _format_cache_size(
                global_settings.cache.get_ssd_cache_max_size_bytes(
                    global_settings.base_path
                )
            ),
            "hot_cache_only": global_settings.cache.hot_cache_only,
            "hot_cache_max_size": global_settings.cache.hot_cache_max_size,
            "initial_cache_blocks": global_settings.cache.initial_cache_blocks,
        },
        "mcp": {
            "config_path": global_settings.mcp.config_path,
        },
        "huggingface": {
            "endpoint": global_settings.huggingface.endpoint,
        },
        "modelscope": {
            "endpoint": global_settings.modelscope.endpoint,
        },
        "network": {
            "http_proxy": global_settings.network.http_proxy,
            "https_proxy": global_settings.network.https_proxy,
            "no_proxy": global_settings.network.no_proxy,
            "ca_bundle": global_settings.network.ca_bundle,
        },
        "sampling": {
            "max_context_window": global_settings.sampling.max_context_window,
            "max_tokens": global_settings.sampling.max_tokens,
            "temperature": global_settings.sampling.temperature,
            "top_p": global_settings.sampling.top_p,
            "top_k": global_settings.sampling.top_k,
            "repetition_penalty": global_settings.sampling.repetition_penalty,
        },
        "auth": {
            "api_key_set": bool(global_settings.auth.api_key),
            "api_key": _mask_api_key(global_settings.auth.api_key),
            "skip_api_key_verification": global_settings.auth.skip_api_key_verification,
            "sub_keys": [sk.to_dict() for sk in global_settings.auth.sub_keys],
        },
        "claude_code": {
            "context_scaling_enabled": global_settings.claude_code.context_scaling_enabled,
            "target_context_size": global_settings.claude_code.target_context_size,
            "mode": global_settings.claude_code.mode,
            "opus_model": global_settings.claude_code.opus_model,
            "sonnet_model": global_settings.claude_code.sonnet_model,
            "haiku_model": global_settings.claude_code.haiku_model,
        },
        "integrations": {
            "codex_model": global_settings.integrations.codex_model,
            "opencode_model": global_settings.integrations.opencode_model,
            "openclaw_model": global_settings.integrations.openclaw_model,
            "hermes_model": global_settings.integrations.hermes_model,
            "pi_model": global_settings.integrations.pi_model,
            "copilot_model": global_settings.integrations.copilot_model,
            "openclaw_tools_profile": global_settings.integrations.openclaw_tools_profile,
        },
        "system": {
            "total_memory_bytes": memory_info["total_bytes"],
            "total_memory": memory_info["total_formatted"],
            "auto_model_memory": memory_info["auto_limit_formatted"],
            "available_memory_bytes": memory_info["available_bytes"],
            "fusionmlx_phys_footprint_bytes": memory_info[
                "fusionmlx_phys_footprint_bytes"
            ],
            "free_memory_bytes": memory_info["free_memory_bytes"],
            "inactive_memory_bytes": memory_info["inactive_memory_bytes"],
            "active_memory_bytes": memory_info["active_memory_bytes"],
            "iogpu_wired_limit_bytes": memory_info["iogpu_wired_limit_bytes"],
            "fusionmlx_wired_limit_request_bytes": memory_info[
                "fusionmlx_wired_limit_request_bytes"
            ],
            "ssd_total_bytes": disk_info["total_bytes"],
            "ssd_total": disk_info["total_formatted"],
        },
        "ui": {
            "language": global_settings.ui.language,
        },
        "idle_timeout": {
            "idle_timeout_seconds": global_settings.idle_timeout.idle_timeout_seconds,
        },
    }


@_router.post("/api/global-settings")
async def update_global_settings(
    request: GlobalSettingsRequest,
    is_admin: bool = Depends(require_admin),
):
    """
    Update global server settings.

    Updates are persisted to the global settings file. Some settings
    (log_level, model_dir, memory_guard_tier, cache) are applied immediately,
    while others (host, port, scheduler, mcp) require server restart.

    Args:
        request: GlobalSettingsRequest with the new settings.

    Returns:
        JSON response with success status, message, and list of runtime-applied settings.

    Raises:
        HTTPException: 401 if not authenticated, 503 if server not initialized,
                        400 if validation fails.
    """

    global_settings = _get_rich_global_settings()

    if global_settings is None:
        # Flat Settings mode — save directly to settings.json
        return _save_global_settings_fallback(request)

    # Track which settings were applied at runtime
    runtime_applied: list[str] = []
    pending_embedding_batch_size: int | None = None
    previous_embedding_batch_size: int | None = None

    # Apply server settings
    if request.host is not None:
        global_settings.server.host = request.host
    if request.port is not None:
        global_settings.server.port = request.port
    if request.log_level is not None:
        global_settings.server.log_level = request.log_level
        # Apply log level at runtime
        _apply_log_level_runtime(request.log_level)
        runtime_applied.append("log_level")
    if request.sse_keepalive_mode is not None:
        valid_modes = {"chunk", "comment", "off"}
        if request.sse_keepalive_mode not in valid_modes:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid sse_keepalive_mode: {request.sse_keepalive_mode} "
                f"(must be one of {sorted(valid_modes)})",
            )
        global_settings.server.sse_keepalive_mode = request.sse_keepalive_mode
        runtime_applied.append("sse_keepalive_mode")

    if request.server_aliases is not None:
        from ..utils.network import is_valid_alias

        cleaned: list[str] = []
        seen: set[str] = set()
        for alias in request.server_aliases:
            if not isinstance(alias, str):
                raise HTTPException(
                    status_code=400,
                    detail="Invalid server alias: each alias must be a string",
                )
            value = alias.strip()
            if not value or value in seen:
                continue
            if not is_valid_alias(value):
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid server alias: {value!r} (must be a hostname or IP address)",
                )
            seen.add(value)
            cleaned.append(value)
        global_settings.server.server_aliases = cleaned
        runtime_applied.append("server_aliases")

    # Apply model settings
    new_dirs = None
    if request.model_dirs is not None:
        new_dirs = [d for d in request.model_dirs if d.strip()]
    elif request.model_dir is not None:
        new_dirs = [request.model_dir]

    if new_dirs is not None:
        old_dirs = global_settings.model.model_dirs
        if new_dirs != old_dirs:
            success, msg = await _apply_model_dirs_runtime(new_dirs)
            if success:
                global_settings.model.model_dirs = new_dirs
                global_settings.model.model_dir = new_dirs[0] if new_dirs else None
                runtime_applied.append("model_dirs")
                logger.info(msg)
            else:
                raise HTTPException(
                    status_code=400, detail=f"Failed to change model directories: {msg}"
                )

    if request.model_fallback is not None:
        global_settings.model.model_fallback = request.model_fallback
        runtime_applied.append("model_fallback")

    # Apply memory guard tier + custom ceiling change (Live)
    if (
        request.memory_guard_tier is not None
        or request.memory_guard_custom_ceiling_gb is not None
    ):
        if request.memory_guard_tier is not None:
            global_settings.memory.memory_guard_tier = request.memory_guard_tier
        if request.memory_guard_custom_ceiling_gb is not None:
            global_settings.memory.memory_guard_custom_ceiling_gb = float(
                request.memory_guard_custom_ceiling_gb
            )
        try:
            success, msg = await _apply_memory_guard_tier_runtime(
                tier=request.memory_guard_tier,
                custom_ceiling_gb=request.memory_guard_custom_ceiling_gb,
            )
            if success:
                runtime_applied.append("memory_guard_tier")
                logger.info(msg)
            else:
                logger.warning(f"Failed to apply memory_guard_tier: {msg}")
        except Exception as e:
            logger.warning(f"Error applying memory_guard_tier: {e}")

    # Apply prefill memory guard setting (Live)
    if request.memory_prefill_memory_guard is not None:
        global_settings.memory.prefill_memory_guard = (
            request.memory_prefill_memory_guard
        )
        from ..server import _server_state

        if _server_state.process_memory_enforcer is not None:
            _server_state.process_memory_enforcer.prefill_memory_guard = (
                request.memory_prefill_memory_guard
            )
        runtime_applied.append("prefill_memory_guard")
        logger.info(
            f"Prefill memory guard "
            f"{'enabled' if request.memory_prefill_memory_guard else 'disabled'}"
        )

    # Apply scheduler settings (restart required)
    if request.max_concurrent_requests is not None:
        global_settings.scheduler.max_concurrent_requests = (
            request.max_concurrent_requests
        )

    # Apply embedding batch size setting (Live for loaded embedding engines)
    if request.embedding_batch_size is not None:
        if request.embedding_batch_size <= 0:
            raise HTTPException(
                status_code=400,
                detail="Invalid embedding_batch_size: must be > 0",
            )
        pending_embedding_batch_size = request.embedding_batch_size

    # Apply chunked prefill setting (Live)
    if request.chunked_prefill is not None:
        global_settings.scheduler.chunked_prefill = request.chunked_prefill
        from ..server import _server_state

        pool = _server_state.engine_pool
        if pool is not None:
            for mid, entry in pool._entries.items():
                if entry is None or entry.engine is None:
                    continue
                async_core = getattr(entry.engine, "_engine", None)
                core = (
                    getattr(async_core, "engine", None)
                    if async_core is not None
                    else None
                )
                scheduler = (
                    getattr(core, "scheduler", None) if core is not None else None
                )
                if scheduler is not None and hasattr(scheduler, "config"):
                    scheduler.config.chunked_prefill = request.chunked_prefill
        runtime_applied.append("chunked_prefill")
        logger.info(
            f"Chunked prefill {'enabled' if request.chunked_prefill else 'disabled'}"
        )

    # Apply cache settings
    cache_changed = False
    if request.cache_enabled is not None:
        global_settings.cache.enabled = request.cache_enabled
        cache_changed = True
    if request.ssd_cache_dir is not None:
        global_settings.cache.ssd_cache_dir = request.ssd_cache_dir
        cache_changed = True
    if request.ssd_cache_max_size is not None:
        global_settings.cache.ssd_cache_max_size = request.ssd_cache_max_size
        cache_changed = True
    if request.hot_cache_only is not None:
        global_settings.cache.hot_cache_only = request.hot_cache_only
    if request.hot_cache_max_size is not None:
        global_settings.cache.hot_cache_max_size = request.hot_cache_max_size
        cache_changed = True
    if request.initial_cache_blocks is not None:
        global_settings.cache.initial_cache_blocks = request.initial_cache_blocks

    if cache_changed:
        success, msg = await _apply_cache_settings_runtime(
            request.cache_enabled,
            request.ssd_cache_dir,
            request.ssd_cache_max_size,
            global_settings,
            hot_cache_max_size=request.hot_cache_max_size,
        )
        if success:
            runtime_applied.append("cache")
            logger.info(msg)
        else:
            logger.warning(f"Failed to apply cache settings runtime: {msg}")

    # Apply MCP settings (restart required)
    if request.mcp_config is not None:
        global_settings.mcp.config_path = (
            request.mcp_config if request.mcp_config else None
        )

    # Apply HuggingFace settings (Live - immediately applied via env var)
    if request.hf_endpoint is not None:
        global_settings.huggingface.endpoint = request.hf_endpoint
        if request.hf_endpoint:
            os.environ["HF_ENDPOINT"] = request.hf_endpoint
        elif "HF_ENDPOINT" in os.environ:
            del os.environ["HF_ENDPOINT"]
        runtime_applied.append("hf_endpoint")
        logger.info(
            f"HuggingFace endpoint updated to: " f"{request.hf_endpoint or '(default)'}"
        )

    # Apply ModelScope settings (Live - immediately applied via env var)
    if request.ms_endpoint is not None:
        global_settings.modelscope.endpoint = request.ms_endpoint
        if request.ms_endpoint:
            os.environ["MODELSCOPE_DOMAIN"] = request.ms_endpoint
        elif "MODELSCOPE_DOMAIN" in os.environ:
            del os.environ["MODELSCOPE_DOMAIN"]
        runtime_applied.append("ms_endpoint")
        logger.info(
            f"ModelScope endpoint updated to: " f"{request.ms_endpoint or '(default)'}"
        )

    # Apply network settings (Live - immediately applied via env vars)
    network_changed = False
    if request.network_http_proxy is not None:
        global_settings.network.http_proxy = request.network_http_proxy
        if request.network_http_proxy:
            os.environ["HTTP_PROXY"] = request.network_http_proxy
            os.environ["http_proxy"] = request.network_http_proxy
        else:
            os.environ.pop("HTTP_PROXY", None)
            os.environ.pop("http_proxy", None)
        network_changed = True

    if request.network_https_proxy is not None:
        global_settings.network.https_proxy = request.network_https_proxy
        if request.network_https_proxy:
            os.environ["HTTPS_PROXY"] = request.network_https_proxy
            os.environ["https_proxy"] = request.network_https_proxy
        else:
            os.environ.pop("HTTPS_PROXY", None)
            os.environ.pop("https_proxy", None)
        network_changed = True

    if request.network_no_proxy is not None:
        global_settings.network.no_proxy = request.network_no_proxy
        if request.network_no_proxy:
            os.environ["NO_PROXY"] = request.network_no_proxy
            os.environ["no_proxy"] = request.network_no_proxy
        else:
            os.environ.pop("NO_PROXY", None)
            os.environ.pop("no_proxy", None)
        network_changed = True

    if request.network_ca_bundle is not None:
        global_settings.network.ca_bundle = request.network_ca_bundle
        if request.network_ca_bundle:
            os.environ["REQUESTS_CA_BUNDLE"] = request.network_ca_bundle
            os.environ["SSL_CERT_FILE"] = request.network_ca_bundle
        else:
            os.environ.pop("REQUESTS_CA_BUNDLE", None)
            os.environ.pop("SSL_CERT_FILE", None)
        network_changed = True

    if network_changed:
        runtime_applied.append("network")
        logger.info("Network settings updated")

    # Apply sampling settings (Live - immediately applied)
    sampling_changed = False
    if request.sampling_max_context_window is not None:
        global_settings.sampling.max_context_window = (
            request.sampling_max_context_window
        )
        sampling_changed = True
    if request.sampling_max_tokens is not None:
        global_settings.sampling.max_tokens = request.sampling_max_tokens
        sampling_changed = True
    if request.sampling_temperature is not None:
        global_settings.sampling.temperature = request.sampling_temperature
        sampling_changed = True
    if request.sampling_top_p is not None:
        global_settings.sampling.top_p = request.sampling_top_p
        sampling_changed = True
    if request.sampling_top_k is not None:
        global_settings.sampling.top_k = request.sampling_top_k
        sampling_changed = True
    if request.sampling_repetition_penalty is not None:
        global_settings.sampling.repetition_penalty = (
            request.sampling_repetition_penalty
        )
        sampling_changed = True

    if sampling_changed:
        success, msg = _apply_sampling_settings_runtime(
            request.sampling_max_context_window,
            request.sampling_max_tokens,
            request.sampling_temperature,
            request.sampling_top_p,
            request.sampling_top_k,
            request.sampling_repetition_penalty,
        )
        if success:
            runtime_applied.append("sampling")
            logger.info(msg)

    # Apply Claude Code settings (Live - immediately applied)
    claude_code_changed = False
    if request.claude_code_context_scaling_enabled is not None:
        global_settings.claude_code.context_scaling_enabled = (
            request.claude_code_context_scaling_enabled
        )
        claude_code_changed = True
    if request.claude_code_target_context_size is not None:
        global_settings.claude_code.target_context_size = (
            request.claude_code_target_context_size
        )
        claude_code_changed = True
    # mode: standard is-not-None check is correct — mode must never be null
    if request.claude_code_mode is not None:
        global_settings.claude_code.mode = request.claude_code_mode
        claude_code_changed = True
    # model fields: use model_fields_set to distinguish "field absent from POST body"
    # from "field explicitly sent as null" — null must clear the field to None.
    # DO NOT use `is not None` here: that would prevent clearing a model field to null.
    if "claude_code_opus_model" in request.model_fields_set:
        global_settings.claude_code.opus_model = request.claude_code_opus_model
        claude_code_changed = True
    if "claude_code_sonnet_model" in request.model_fields_set:
        global_settings.claude_code.sonnet_model = request.claude_code_sonnet_model
        claude_code_changed = True
    if "claude_code_haiku_model" in request.model_fields_set:
        global_settings.claude_code.haiku_model = request.claude_code_haiku_model
        claude_code_changed = True

    if claude_code_changed:
        runtime_applied.append("claude_code")
        logger.info(
            f"Claude Code settings updated: "
            f"scaling={'enabled' if global_settings.claude_code.context_scaling_enabled else 'disabled'}, "
            f"target={global_settings.claude_code.target_context_size}, "
            f"mode={global_settings.claude_code.mode}, "
            f"opus={global_settings.claude_code.opus_model}, "
            f"sonnet={global_settings.claude_code.sonnet_model}, "
            f"haiku={global_settings.claude_code.haiku_model}"
        )

    # Apply integrations settings (Live - immediately applied)
    integrations_changed = False
    if "integrations_copilot_model" in request.model_fields_set:
        global_settings.integrations.copilot_model = request.integrations_copilot_model
        integrations_changed = True
    if "integrations_codex_model" in request.model_fields_set:
        global_settings.integrations.codex_model = request.integrations_codex_model
        integrations_changed = True
    if "integrations_opencode_model" in request.model_fields_set:
        global_settings.integrations.opencode_model = (
            request.integrations_opencode_model
        )
        integrations_changed = True
    if "integrations_openclaw_model" in request.model_fields_set:
        global_settings.integrations.openclaw_model = (
            request.integrations_openclaw_model
        )
        integrations_changed = True
    if "integrations_hermes_model" in request.model_fields_set:
        global_settings.integrations.hermes_model = request.integrations_hermes_model
        integrations_changed = True
    if "integrations_pi_model" in request.model_fields_set:
        global_settings.integrations.pi_model = request.integrations_pi_model
        integrations_changed = True
    if "integrations_openclaw_tools_profile" in request.model_fields_set:
        global_settings.integrations.openclaw_tools_profile = (
            request.integrations_openclaw_tools_profile
        )
        integrations_changed = True

    if integrations_changed:
        runtime_applied.append("integrations")
        logger.info(
            f"Integration settings updated: "
            f"copilot={global_settings.integrations.copilot_model}, "
            f"codex={global_settings.integrations.codex_model}, "
            f"opencode={global_settings.integrations.opencode_model}, "
            f"openclaw={global_settings.integrations.openclaw_model}, "
            f"hermes={global_settings.integrations.hermes_model}, "
            f"pi={global_settings.integrations.pi_model}"
        )

    # Apply UI settings
    if request.ui_language is not None:
        global_settings.ui.language = request.ui_language
        runtime_applied.append("ui_language")
        logger.info(f"UI language changed to: {request.ui_language}")

    # Apply idle timeout settings (Live)
    # Use model_fields_set to distinguish "explicitly sent as null" (disable)
    # from "not sent" (don't touch).
    if "idle_timeout_seconds" in request.model_fields_set:
        global_settings.idle_timeout.idle_timeout_seconds = request.idle_timeout_seconds
        runtime_applied.append("idle_timeout_seconds")
        if request.idle_timeout_seconds:
            logger.info(f"Idle timeout set to: {request.idle_timeout_seconds}s")
        else:
            logger.info("Idle timeout disabled")

    # Apply auth settings (API key change)
    if request.api_key is not None:
        from ..server import _server_state

        is_valid, error_msg = validate_api_key(request.api_key)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)

        global_settings.auth.api_key = request.api_key
        _server_state.api_key = request.api_key
        runtime_applied.append("api_key")
        logger.info("API key updated via admin settings")

    if request.skip_api_key_verification is not None:
        global_settings.auth.skip_api_key_verification = (
            request.skip_api_key_verification
        )
        runtime_applied.append("skip_api_key_verification")

    if pending_embedding_batch_size is not None:
        previous_embedding_batch_size = global_settings.scheduler.embedding_batch_size
        global_settings.scheduler.embedding_batch_size = pending_embedding_batch_size

    # Validate settings
    errors = global_settings.validate()
    if errors:
        if previous_embedding_batch_size is not None:
            global_settings.scheduler.embedding_batch_size = (
                previous_embedding_batch_size
            )
        raise HTTPException(status_code=400, detail=errors)

    # Persist to file
    try:
        global_settings.save()
    except Exception as e:
        if previous_embedding_batch_size is not None:
            global_settings.scheduler.embedding_batch_size = (
                previous_embedding_batch_size
            )
        raise HTTPException(status_code=500, detail="Failed to save settings")

    if pending_embedding_batch_size is not None:
        from ..server import _server_state

        pool = _server_state.engine_pool
        if pool is not None:
            await pool.apply_embedding_batch_size(pending_embedding_batch_size)
        runtime_applied.append("embedding_batch_size")
        logger.info(f"Embedding batch size set to {pending_embedding_batch_size}")

    # Build response message
    message = "Settings saved successfully."

    return {
        "success": True,
        "message": message,
        "runtime_applied": runtime_applied,
    }


router = _router
