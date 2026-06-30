import asyncio
import logging
import os
import shutil
import signal
import subprocess
from pathlib import Path

from fastapi import HTTPException, Request

from .auth import verify_api_key, verify_session

# =============================================================================
# Runtime Settings Application Functions
# =============================================================================


def _format_cache_size(size_bytes: int) -> str:
    """Format cache size in bytes to human-readable string (e.g., '100GB')."""
    gb = size_bytes / (1024 ** 3)
    if gb >= 1:
        return f"{gb:.0f}GB"
    mb = size_bytes / (1024 ** 2)
    return f"{mb:.0f}MB"


_PAROQUANT_REASON = (
    "Not supported on paroquant models yet (compatibility not verified)"
)


def _paroquant_compat_for_model(model_info: dict) -> tuple[bool, str]:
    """Detect whether a model is paroquant-quantized.

    Returns ``(is_paroquant, reason)``. ``is_paroquant`` is True iff
    ``config.json`` declares ``quantization_config.quant_method == "paroquant"``.
    Reason is the user-facing string surfaced as a tooltip/banner on the
    admin model settings modal when paroquant gates an experimental toggle.
    """
    import json
    from pathlib import Path

    model_path = model_info.get("model_path") or ""
    if not model_path:
        return False, ""
    cfg_path = Path(model_path) / "config.json"
    if not cfg_path.exists():
        return False, ""
    try:
        cfg = json.loads(cfg_path.read_text())
    except Exception:
        return False, ""
    qcfg = cfg.get("quantization_config") or {}
    method = (qcfg.get("quant_method") or "").lower()
    if method == "paroquant":
        return True, _PAROQUANT_REASON
    return False, ""


def _dflash_compat_for_model(model_info: dict) -> tuple[bool, str]:
    """Resolve dflash compatibility for an engine_pool model dict.

    Returns ``(False, "")`` when dflash-mlx is not installed so the UI hides
    the compat hint instead of pointing the user at an unrelated reason.
    """
    is_paro, paro_reason = _paroquant_compat_for_model(model_info)
    if is_paro:
        return False, paro_reason
    try:
        from ..engine.dflash import is_dflash_compatible
    except ImportError:
        return False, ""
    model_path = model_info.get("model_path") or ""
    if not model_path:
        return False, "model_path missing"
    return is_dflash_compatible(model_path)


def _mtp_compat_for_model(model_info: dict) -> tuple[bool, str]:
    """Mirror of ``_dflash_compat_for_model`` for the native MTP toggle.

    Returns ``(compatible, reason)``. Reason is empty on success and
    suitable for surfacing to users (admin UI shows it under the toggle).

    The check is conservative: even when the config declares MTP layers
    we also peek at the safetensors weight index to verify that the
    converter actually preserved the ``mtp.*`` tensors. Default mlx-lm
    converters strip them; PR 990 ships a separate path that keeps them.
    """
    import json
    from pathlib import Path

    from ..utils.model_loading import _has_mtp_heads, _is_mtp_compatible

    is_paro, paro_reason = _paroquant_compat_for_model(model_info)
    if is_paro:
        return False, paro_reason

    model_path = model_info.get("model_path") or ""
    if not model_path:
        return False, "model_path missing"
    cfg_path = Path(model_path) / "config.json"
    if not cfg_path.exists():
        return False, "config.json not found"
    try:
        cfg = json.loads(cfg_path.read_text())
    except Exception as e:
        return False, f"failed to read config: {e}"
    model_type = cfg.get("model_type")
    if not _has_mtp_heads(cfg):
        return False, "model has no MTP heads in config"
    if not _is_mtp_compatible(cfg, model_type):
        return False, (
            f"model_type={model_type!r} is not on the MTP whitelist "
            "(supported: qwen3_5*, qwen3_6*, deepseek_v4*)"
        )
    if not _model_has_mtp_weight_tensors(Path(model_path)):
        return False, (
            "Config declares MTP layers but the converted weights are missing "
            "mtp.* tensors. Re-convert from HF with a converter that preserves "
            "MTP weights."
        )
    return True, ""


def _model_has_mtp_weight_tensors(model_dir) -> bool:
    """Return True iff the model directory's weight files contain ``mtp.*`` keys.

    Uses ``model.safetensors.index.json`` when present (cheap — only reads
    the weight_map). Falls back to opening each ``*.safetensors`` and
    checking its keys when no index is present (single-shard models).
    Returns False on any error (we treat the model as incompatible rather
    than risking a confusing load failure mid-inference).
    """
    import json
    from pathlib import Path

    try:
        from safetensors import safe_open
    except ImportError:
        # Library should be installed via mlx-lm deps; if it's not we can't
        # peek the weights. Stay conservative and assume incompatible.
        return False

    model_dir = Path(model_dir)

    # Preferred path: read the index file's weight_map (no tensor data loaded).
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        try:
            index = json.loads(index_path.read_text())
            weight_map = index.get("weight_map", {})
            return any("mtp." in key for key in weight_map.keys())
        except Exception:
            return False

    # Single-shard fallback: enumerate keys via safe_open metadata. We
    # short-circuit on the first ``mtp.*`` key.
    for path in model_dir.glob("*.safetensors"):
        try:
            with safe_open(str(path), framework="numpy") as f:  # type: ignore[arg-type]
                for key in f.keys():
                    if "mtp." in key:
                        return True
        except Exception:
            continue
    return False


def _apply_log_level_runtime(level: str) -> None:
    """Apply log level change at runtime to all oMLX loggers and handlers."""
    level_name = level.upper()
    log_level = 5 if level_name == "TRACE" else getattr(logging, level_name, logging.INFO)

    # Update root logger level and all its handlers
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    for handler in root_logger.handlers:
        handler.setLevel(log_level)

    # Update omlx-related loggers
    omlx_loggers = [
        "omlx",
        "fusion_mlx.scheduler",
        "fusion_mlx.paged_ssd_cache",
        "fusion_mlx.memory_monitor",
        "fusion_mlx.paged_cache",
        "fusion_mlx.prefix_cache",
        "fusion_mlx.engine_pool",
        "fusion_mlx.model_discovery",
        "fusion_mlx.engine_core",
        "fusion_mlx.engine",
        "fusion_mlx.server",
        "fusion_mlx.admin",
    ]

    for logger_name in omlx_loggers:
        logging.getLogger(logger_name).setLevel(log_level)

    # Also update uvicorn logger
    logging.getLogger("uvicorn").setLevel(log_level)
    logging.getLogger("uvicorn.access").setLevel(log_level)


async def _apply_model_dirs_runtime(model_dirs: list[str]) -> tuple[bool, str]:
    """
    Apply model directories change at runtime by re-scanning models.

    This will:
    1. Validate all directories
    2. Unload all currently loaded models
    3. Clear the entries dictionary
    4. Re-discover models from the new directories

    Returns:
        Tuple of (success, message)
    """
    from pathlib import Path

    from ..server import _server_state

    if _server_state.engine_pool is None:
        return False, "Engine pool not initialized"

    # Validate all model directories
    for model_dir in model_dirs:
        model_path = Path(model_dir).expanduser().resolve()
        if not model_path.exists():
            return False, f"Model directory does not exist: {model_dir}"
        if not model_path.is_dir():
            return False, f"Path is not a directory: {model_dir}"

    pool = _server_state.engine_pool

    # Get pinned models from settings_manager
    pinned_models = []
    if _server_state.settings_manager is not None:
        pinned_models = _server_state.settings_manager.get_pinned_model_ids()

    # Unload all loaded models
    loaded_models = pool.get_loaded_model_ids()
    for model_id in loaded_models:
        try:
            await pool._unload_engine(model_id)
        except Exception as e:
            logger.warning(f"Error unloading {model_id}: {e}")

    # Clear entries
    pool._entries.clear()
    pool._current_model_memory = 0

    # Update downloader model directories
    global _hf_downloader, _ms_downloader, _oq_manager, _hf_uploader
    if model_dirs:
        primary_dir = model_dirs[0]
        if _hf_downloader is not None:
            _hf_downloader.update_model_dir(primary_dir)
        if _ms_downloader is not None:
            _ms_downloader.update_model_dir(primary_dir)

    # Update components that scan all model directories
    if _oq_manager is not None:
        _oq_manager.update_model_dirs(model_dirs)
    if _hf_uploader is not None:
        _hf_uploader.update_model_dirs(model_dirs)

    # Re-discover models from new directories
    try:
        pool.discover_models(model_dirs, pinned_models)
        if _server_state.settings_manager is not None:
            pool.apply_settings_overrides(_server_state.settings_manager)
    except Exception as e:
        return False, f"Failed to discover models: {e}"

    dir_count = len(model_dirs)
    return True, (
        f"Re-discovered {pool.model_count} models "
        f"from {dir_count} director{'ies' if dir_count > 1 else 'y'}"
    )


async def _reload_models() -> tuple[bool, str]:
    """
    Reload models: re-read model_settings.json, re-scan dirs, re-apply overrides,
    and preload pinned models.

    This does NOT re-read settings.json (global settings). It only refreshes
    the model inventory and per-model settings.

    Returns:
        Tuple of (success, message)
    """
    from ..server import _server_state

    if _server_state.engine_pool is None:
        return False, "Engine pool not initialized"

    global_settings = _get_global_settings()
    if global_settings is None:
        return False, "Global settings not initialized"

    # Re-read model_settings.json from disk
    settings_manager = _get_settings_manager()
    if settings_manager is not None:
        settings_manager._load()

    # Get current model_dirs from global settings
    model_dirs = global_settings.model.model_dirs or []
    if not model_dirs and global_settings.model.model_dir:
        model_dirs = [global_settings.model.model_dir]

    # Unload all, re-discover, re-apply overrides
    success, msg = await _apply_model_dirs_runtime(model_dirs)
    if not success:
        return False, msg

    # Preload pinned models
    pool = _server_state.engine_pool
    if pool is not None:
        await pool.preload_pinned_models()

    return True, msg


async def _apply_memory_guard_tier_runtime(
    tier: str | None = None,
    custom_ceiling_gb: float | None = None,
) -> tuple[bool, str]:
    """
    Apply memory_guard_tier (and optionally custom ceiling) at runtime.

    Pushes both values into the running ProcessMemoryEnforcer, which
    recomputes static + dynamic ceilings on its next propagation tick.
    `tier` and `custom_ceiling_gb` can be passed together (Custom tier
    save) or independently.

    Returns:
        Tuple of (success, message)
    """
    from ..server import _server_state
    from ..settings import VALID_MEMORY_GUARD_TIERS

    enforcer = _server_state.process_memory_enforcer
    if enforcer is None:
        return False, "Process memory enforcer not initialized"

    changes = []
    if tier is not None:
        value = tier.strip().lower()
        if value not in VALID_MEMORY_GUARD_TIERS:
            return False, (
                f"Invalid memory_guard_tier: '{tier}' "
                f"(must be one of {sorted(VALID_MEMORY_GUARD_TIERS)})"
            )
        old_tier = enforcer.memory_guard_tier
        enforcer.memory_guard_tier = value
        changes.append(f"tier: {old_tier} -> {value}")
    if custom_ceiling_gb is not None:
        new_bytes = max(0, int(float(custom_ceiling_gb) * 1024**3))
        enforcer.memory_guard_custom_ceiling_bytes = new_bytes
        changes.append(f"custom_ceiling: {custom_ceiling_gb} GB")
    if not changes:
        return True, "(no change)"
    return True, "Memory guard updated — " + ", ".join(changes)


async def _apply_cache_settings_runtime(
    enabled: bool | None,
    ssd_cache_dir: str | None,
    ssd_cache_max_size: str | None,
    global_settings,
    hot_cache_max_size: str | None = None,
) -> tuple[bool, str]:
    """
    Apply cache settings at runtime.

    Updates the scheduler_config and unloads all models so they
    will use the new cache settings when reloaded.

    Returns:
        Tuple of (success, message)
    """
    from ..config import parse_size
    from ..server import _server_state

    if _server_state.engine_pool is None:
        return False, "Engine pool not initialized"

    pool = _server_state.engine_pool

    # Update scheduler config based on cache settings
    if enabled is False or (enabled is None and not global_settings.cache.enabled):
        pool._scheduler_config.paged_ssd_cache_dir = None
        pool._scheduler_config.paged_ssd_cache_max_size = 0
    else:
        # Cache is enabled
        if ssd_cache_dir is not None:
            pool._scheduler_config.paged_ssd_cache_dir = ssd_cache_dir
        elif global_settings.cache.ssd_cache_dir:
            pool._scheduler_config.paged_ssd_cache_dir = global_settings.cache.ssd_cache_dir
        else:
            # Use default cache dir
            pool._scheduler_config.paged_ssd_cache_dir = str(
                global_settings.cache.get_ssd_cache_dir(global_settings.base_path)
            )

        if ssd_cache_max_size is not None:
            # Handle "auto" value
            if ssd_cache_max_size.lower() == "auto":
                pool._scheduler_config.paged_ssd_cache_max_size = (
                    global_settings.cache.get_ssd_cache_max_size_bytes(global_settings.base_path)
                )
            else:
                pool._scheduler_config.paged_ssd_cache_max_size = parse_size(ssd_cache_max_size)
        elif global_settings.cache.ssd_cache_max_size:
            # Use settings value (handles "auto")
            pool._scheduler_config.paged_ssd_cache_max_size = (
                global_settings.cache.get_ssd_cache_max_size_bytes(global_settings.base_path)
            )
        elif global_settings.cache.ssd_cache_max_size:
            pool._scheduler_config.paged_ssd_cache_max_size = parse_size(
                global_settings.cache.ssd_cache_max_size
            )

    # Apply hot cache max size
    if hot_cache_max_size is not None:
        hot_bytes = 0 if hot_cache_max_size == "0" else parse_size(hot_cache_max_size)
        old_hot = pool._scheduler_config.hot_cache_max_size
        pool._scheduler_config.hot_cache_max_size = hot_bytes
        if hot_bytes != old_hot:
            from ..utils.formatting import format_bytes
            old_str = "Off" if old_hot == 0 else format_bytes(old_hot)
            new_str = "Off" if hot_bytes == 0 else format_bytes(hot_bytes)
            logger.info(f"Hot cache max size changed: {old_str} -> {new_str}")
    elif global_settings.cache.hot_cache_max_size:
        pool._scheduler_config.hot_cache_max_size = (
            global_settings.cache.get_hot_cache_max_size_bytes()
        )

    # Unload all loaded models so they use new config when reloaded
    loaded_models = pool.get_loaded_model_ids()
    for model_id in loaded_models:
        try:
            await pool._unload_engine(model_id)
        except Exception as e:
            logger.warning(f"Error unloading {model_id}: {e}")

    return True, f"Cache settings updated. Unloaded {len(loaded_models)} models."


def _apply_sampling_settings_runtime(
    max_context_window: int | None,
    max_tokens: int | None,
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
    repetition_penalty: float | None = None,
) -> tuple[bool, str]:
    """
    Apply sampling default settings at runtime.

    Updates _server_state.sampling which is used for all new API requests.

    Returns:
        Tuple of (success, message)
    """
    from ..server import _server_state

    changes = []

    if max_context_window is not None:
        _server_state.sampling.max_context_window = max_context_window
        changes.append(f"max_context_window={max_context_window}")

    if max_tokens is not None:
        _server_state.sampling.max_tokens = max_tokens
        changes.append(f"max_tokens={max_tokens}")

    if temperature is not None:
        _server_state.sampling.temperature = temperature
        changes.append(f"temperature={temperature}")

    if top_p is not None:
        _server_state.sampling.top_p = top_p
        changes.append(f"top_p={top_p}")

    if top_k is not None:
        _server_state.sampling.top_k = top_k
        changes.append(f"top_k={top_k}")

    if repetition_penalty is not None:
        _server_state.sampling.repetition_penalty = repetition_penalty
        changes.append(f"repetition_penalty={repetition_penalty}")

    if changes:
        return True, f"Sampling defaults updated: {', '.join(changes)}"
    return True, "No sampling changes"



# =============================================================================
# State Getters (set by server.py)
# =============================================================================

_get_server_state = None
_get_engine_pool = None
_get_settings_manager = None
_get_global_settings = None

def get_engine_pool():
    """Lazy accessor that always returns current engine pool getter result."""
    if _get_engine_pool is None:
        return None
    return _get_engine_pool()


def get_server_state():
    """Lazy accessor that always returns current server state."""
    if _get_server_state is None:
        return None
    return _get_server_state()


def get_global_settings():
    """Lazy accessor that always returns current global settings."""
    if _get_global_settings is None:
        return None
    return _get_global_settings()


def get_settings_manager():
    """Lazy accessor that always returns current settings manager."""
    if _get_settings_manager is None:
        return None
    return _get_settings_manager()

_hf_downloader = None
_ms_downloader = None
_oq_manager = None
_hf_uploader = None


def set_admin_getters(
    state_getter,
    pool_getter,
    settings_manager_getter,
    global_settings_getter,
):
    """
    Set the getter functions for accessing server state.

    This function must be called during server initialization to provide
    access to the server state objects.

    Args:
        state_getter: Function that returns the ServerState instance.
        pool_getter: Function that returns the EnginePool instance.
        settings_manager_getter: Function that returns the ModelSettingsManager.
        global_settings_getter: Function that returns the GlobalSettings.
    """
    global _get_server_state, _get_engine_pool, _get_settings_manager, _get_global_settings
    _get_server_state = state_getter
    _get_engine_pool = pool_getter
    _get_settings_manager = settings_manager_getter
    _get_global_settings = global_settings_getter


def set_hf_downloader(downloader):
    """Set the HFDownloader instance for admin routes.

    Args:
        downloader: HFDownloader instance created during server initialization.
    """
    global _hf_downloader
    _hf_downloader = downloader


def set_ms_downloader(downloader):
    """Set the MSDownloader instance for admin routes.

    Args:
        downloader: MSDownloader instance created during server initialization.
    """
    global _ms_downloader
    _ms_downloader = downloader


def set_oq_manager(manager):
    """Set the OQManager instance for admin routes.

    Args:
        manager: OQManager instance created during server initialization.
    """
    global _oq_manager
    _oq_manager = manager


def set_hf_uploader(uploader):
    """Set the HFUploader instance for admin routes.

    Args:
        uploader: HFUploader instance created during server initialization.
    """
    global _hf_uploader
    _hf_uploader = uploader



# =============================================================================
# Helper Functions
# =============================================================================


def format_size(size_bytes: int) -> str:
    """
    Format a byte size as a human-readable string.

    Args:
        size_bytes: Size in bytes.

    Returns:
        Human-readable string (e.g., "1.5 GB").
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024**2:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024**3:
        return f"{size_bytes / 1024**2:.1f} MB"
    elif size_bytes < 1024**4:
        return f"{size_bytes / 1024**3:.2f} GB"
    else:
        return f"{size_bytes / 1024**4:.2f} TB"


def get_ssd_disk_info(cache_dir: str) -> dict:
    """
    Get disk information for the SSD cache directory.

    Returns:
        Dictionary with total_bytes, total_formatted.
    """
    try:
        check_path = Path(cache_dir).expanduser().resolve()
        while not check_path.exists() and check_path.parent != check_path:
            check_path = check_path.parent
        stat = shutil.disk_usage(check_path)
        return {
            "total_bytes": stat.total,
            "total_formatted": format_size(stat.total),
        }
    except Exception as e:
        logger.warning(f"Failed to get disk info for {cache_dir}: {e}")
        return {
            "total_bytes": 0,
            "total_formatted": "Unknown",
        }


def get_system_memory_info() -> dict:
    """
    Get system memory information.

    Returns:
        Dictionary with total_bytes, total_formatted, auto_limit_bytes,
        and auto_limit_formatted (80% of total).
    """
    try:
        # macOS: use sysctl to get physical memory. Invoke by absolute path —
        # sysctl lives in /usr/sbin, which isn't on PATH in some headless
        # launchd contexts (brew services). See issue #1322.
        result = subprocess.run(
            ["/usr/sbin/sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        total_bytes = int(result.stdout.strip())
    except Exception:
        # Fallback: try os.sysconf (works on some Unix systems)
        try:
            total_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        except Exception:
            total_bytes = 0

    auto_limit_bytes = int(total_bytes * 0.8)

    # Live values so the admin UI can preview the actual hard ceiling for any
    # tier (static_ceiling + dynamic_ceiling depend on these). Read on each
    # call — never cached.
    try:
        import psutil

        available_bytes = int(psutil.virtual_memory().available)
    except Exception:
        available_bytes = 0
    try:
        from ..utils.proc_memory import get_phys_footprint

        omlx_phys_footprint_bytes = int(get_phys_footprint())
    except Exception:
        omlx_phys_footprint_bytes = 0

    # Effective Metal cap = sysctl iogpu.wired_limit_mb when set, else
    # Apple's max_recommended_working_set_size (~75% of RAM). The admin UI
    # compares this against the value oMLX wanted at start (static
    # ceiling) and warns when the cap is below the request.
    try:
        from ..process_memory_enforcer import get_effective_metal_cap_bytes

        iogpu_wired_limit_bytes = int(get_effective_metal_cap_bytes())
    except Exception:
        iogpu_wired_limit_bytes = 0
    omlx_wired_limit_request_bytes = 0
    try:
        from ..server import _server_state

        enforcer = getattr(_server_state, "process_memory_enforcer", None)
        if enforcer is not None:
            omlx_wired_limit_request_bytes = int(
                getattr(enforcer, "_metal_wired_limit_request", 0) or 0
            )
    except Exception:
        pass

    # Live macOS vm_stat layers so the admin dashboard can preview the
    # tier-aware ceiling (free + inactive + active * ratio). Zero on
    # non-macOS / call failure — JS falls back to available_bytes.
    free_memory_bytes = 0
    inactive_memory_bytes = 0
    active_memory_bytes = 0
    try:
        from ..process_memory_enforcer import get_macos_vm_stats

        vm = get_macos_vm_stats()
        if vm is not None:
            free_memory_bytes = int(vm.get("free", 0))
            inactive_memory_bytes = int(vm.get("inactive", 0))
            active_memory_bytes = int(vm.get("active", 0))
    except Exception:
        pass

    return {
        "total_bytes": total_bytes,
        "total_formatted": format_size(total_bytes),
        "auto_limit_bytes": auto_limit_bytes,
        "auto_limit_formatted": format_size(auto_limit_bytes),
        "available_bytes": available_bytes,
        "omlx_phys_footprint_bytes": omlx_phys_footprint_bytes,
        "iogpu_wired_limit_bytes": iogpu_wired_limit_bytes,
        "omlx_wired_limit_request_bytes": omlx_wired_limit_request_bytes,
        "free_memory_bytes": free_memory_bytes,
        "inactive_memory_bytes": inactive_memory_bytes,
        "active_memory_bytes": active_memory_bytes,
    }



async def _require_admin_or_bearer(request: Request) -> bool:
    """Allow admin session OR a valid Bearer API key (for CLI use)."""
    gs = _get_global_settings() if _get_global_settings else None

    # No-auth mode: always allow
    if gs is not None and gs.auth.skip_api_key_verification:
        return True

    # Valid admin session cookie
    if verify_session(request):
        return True

    # Bearer token matching the configured API key
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer ") and gs is not None:
        token = auth_header[7:]
        server_key = gs.auth.api_key or ""
        sub_keys = gs.auth.sub_keys or []
        if verify_api_key(token, server_key):
            return True
        for sk in sub_keys:
            if verify_api_key(token, getattr(sk, "key", "")):
                return True

    raise HTTPException(
        status_code=401,
        detail="Admin authentication required",
        headers={"WWW-Authenticate": "Bearer"},
    )

def _require_settings_manager():
    mgr = _get_settings_manager()
    if mgr is None:
        raise HTTPException(status_code=503, detail="Server not initialized")
    return mgr


def _require_model(model_id: str):
    pool = _get_engine_pool()
    if pool is None:
        raise HTTPException(status_code=503, detail="Engine pool not initialized")
    entry = pool.get_entry(model_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")
    return entry

def _schedule_self_terminate(delay: float = 0.5) -> None:
    """Schedule ``os.kill(getpid(), SIGTERM)`` on the running loop.

    Extracted from the restart handler so tests can patch this seam
    instead of mocking ``asyncio.get_running_loop`` globally (which
    interferes with FastAPI's TestClient portal).
    """
    pid = os.getpid()

    def _kill() -> None:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            # Already exited (e.g. concurrent SIGTERM) — nothing to do.
            pass
        except Exception:  # pragma: no cover — best-effort signal.
            logger.exception("Failed to self-terminate for restart")

    asyncio.get_running_loop().call_later(delay, _kill)


def _get_hf_downloader():
    return _hf_downloader


def _get_ms_downloader():
    return _ms_downloader


def _get_oq_manager():
    return _oq_manager


def _get_hf_uploader():
    return _hf_uploader
