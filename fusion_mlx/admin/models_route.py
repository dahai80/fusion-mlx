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
from dataclasses import asdict
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException

from ..model_profiles import EXCLUDED_FROM_PROFILES
from .auth import (
    require_admin,
)

logger = logging.getLogger(__name__)

PRESET_REMOTE_URL = "http://bench.dpdns.org/assets/fusionmlx_preset.json"


from .helpers import (
    _dflash_compat_for_model,
    _get_engine_pool,
    _get_server_state,
    _get_settings_manager,
    _mtp_compat_for_model,
    _paroquant_compat_for_model,
    _reload_models,
    _require_admin_or_bearer,
    format_size,
)
from .models import (
    ModelSettingsRequest,
)

_router = APIRouter()

# =============================================================================
# Models API Routes
# =============================================================================


@_router.get("/api/models")
async def list_models(is_admin: bool = Depends(require_admin)):
    """
    List all models with their settings.

    Returns model information from the engine pool combined with
    per-model settings from the settings manager.

    Returns:
        JSON list of models with their status and settings.

    Raises:
        HTTPException: 401 if not authenticated, 503 if server not initialized.
    """
    engine_pool = _get_engine_pool()
    settings_manager = _get_settings_manager()
    server_state = _get_server_state()

    if engine_pool is None:
        return {"models": []}

    # Get engine pool status
    status = engine_pool.get_status()
    models_status = status.get("models", [])

    # Get all model settings
    all_settings = settings_manager.get_all_settings() if settings_manager else {}

    # SSD cache dir is set on the scheduler_config when the user enables paged
    # SSD caching; admin UI consumes it to gate the dflash SSD toggle.
    ssd_cache_dir = getattr(
        getattr(engine_pool, "_scheduler_config", None),
        "paged_ssd_cache_dir",
        None,
    )
    dflash_ssd_cache_available = bool(ssd_cache_dir)

    # Combine model info with settings
    models = []
    for model_info in models_status:
        model_id = model_info["id"]
        settings = all_settings.get(model_id)

        is_paroquant, paroquant_reason = _paroquant_compat_for_model(model_info)
        compat_ok, compat_reason = _dflash_compat_for_model(model_info)
        mtp_compat_ok, mtp_compat_reason = _mtp_compat_for_model(model_info)

        model_data = {
            "id": model_id,
            "model_path": model_info.get("model_path", ""),
            "loaded": model_info.get("loaded", False),
            "is_loading": model_info.get("is_loading", False),
            "estimated_size": model_info.get("estimated_size", 0),
            "estimated_size_formatted": format_size(
                model_info.get("estimated_size", 0)
            ),
            "actual_size": model_info.get("actual_size") or 0,
            "actual_size_formatted": (
                format_size(model_info.get("actual_size", 0))
                if model_info.get("actual_size")
                else None
            ),
            "pinned": model_info.get("pinned", False),
            "is_default": (
                server_state.get("default_model") == model_id if server_state else False
            ),
            "engine_type": model_info.get("engine_type", "batched"),
            "model_type": model_info.get("model_type", "llm"),
            "config_model_type": model_info.get("config_model_type", ""),
            "thinking_default": model_info.get("thinking_default"),
            "preserve_thinking_default": model_info.get("preserve_thinking_default"),
            "last_access": model_info.get("last_access"),
            "dflash_compatible": compat_ok,
            "dflash_compatibility_reason": compat_reason,
            "dflash_ssd_cache_available": dflash_ssd_cache_available,
            "mtp_compatible": mtp_compat_ok,
            "mtp_compatibility_reason": mtp_compat_reason,
            "is_paroquant": is_paroquant,
            "paroquant_reason": paroquant_reason,
        }

        # Add settings if available
        if settings:
            model_data["settings"] = asdict(settings)

        models.append(model_data)

    return {"models": models}


@_router.post("/api/models/{model_id}/unload")
async def unload_model(
    model_id: str,
    is_admin: bool = Depends(require_admin),
):
    """Manually unload a model from memory."""
    engine_pool = _get_engine_pool()
    if engine_pool is None:
        raise HTTPException(status_code=503, detail="Engine pool not initialized")

    entry = engine_pool.get_entry(model_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")
    if entry.engine is None:
        raise HTTPException(status_code=400, detail=f"Model not loaded: {model_id}")

    await engine_pool._unload_engine(model_id)
    logger.info(f"Manually unloaded model: {model_id}")
    return {"status": "ok", "model_id": model_id, "message": f"Unloaded {model_id}"}


@_router.post("/api/models/{model_id}/load")
async def load_model(
    model_id: str,
    is_admin: bool = Depends(_require_admin_or_bearer),
):
    """Manually load a model into memory."""
    engine_pool = _get_engine_pool()
    if engine_pool is None:
        raise HTTPException(status_code=503, detail="Engine pool not initialized")

    entry = engine_pool.get_entry(model_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")
    if entry.engine is not None:
        return {
            "status": "ok",
            "model_id": model_id,
            "message": f"Already loaded: {model_id}",
        }
    if entry.is_loading:
        raise HTTPException(
            status_code=409, detail=f"Model is already loading: {model_id}"
        )

    try:
        await engine_pool.get_engine(model_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    logger.info(f"Manually loaded model: {model_id}")
    return {"status": "ok", "model_id": model_id, "message": f"Loaded {model_id}"}


@_router.post("/api/reload")
async def reload_models(is_admin: bool = Depends(require_admin)):
    """Reload models: re-read model settings, re-discover models, preload pinned."""
    success, message = await _reload_models()
    if success:
        return {"status": "ok", "message": message}
    raise HTTPException(status_code=500, detail=message)


@_router.put("/api/models/{model_id}/settings")
async def update_model_settings(
    model_id: str,
    request: ModelSettingsRequest,
    is_admin: bool = Depends(require_admin),
):
    """
    Update settings for a specific model.

    Updates are persisted to the settings file and applied immediately
    to the engine pool where applicable (e.g., pinned status).

    Args:
        model_id: The model identifier.
        request: ModelSettingsRequest with the new settings.

    Returns:
        JSON response with success status and updated settings.

    Raises:
        HTTPException: 401 if not authenticated, 404 if model not found.
    """
    engine_pool = _get_engine_pool()
    settings_manager = _get_settings_manager()
    server_state = _get_server_state()

    if engine_pool is None or settings_manager is None:
        # Flat Settings mode — save directly to settings.json
        return _save_model_settings_fallback(model_id, request)

    # Check if model exists
    entry = engine_pool.get_entry(model_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")

    # Get current settings
    current_settings = settings_manager.get_settings(model_id)

    # Apply updates — use model_fields_set to distinguish "sent as null"
    # (clear to default) from "not sent" (don't touch).
    sent = request.model_fields_set
    prev_engine_type = entry.engine_type  # Track for requires_reload check
    if "model_alias" in sent:
        alias_value = request.model_alias.strip() if request.model_alias else None
        if alias_value == "":
            alias_value = None
        if alias_value is not None:
            all_settings = settings_manager.get_all_settings()
            for mid, ms in all_settings.items():
                if mid != model_id and ms.model_alias == alias_value:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Alias '{alias_value}' is already used by model '{mid}'",
                    )
            for mid in engine_pool._entries:
                if mid != model_id and mid == alias_value:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Alias '{alias_value}' conflicts with model directory name '{mid}'",
                    )
        current_settings.model_alias = alias_value
    if "model_type_override" in sent:
        valid_types = {
            "llm",
            "vlm",
            "embedding",
            "reranker",
            "audio_stt",
            "audio_tts",
            "audio_sts",
            "image",
            "video",
        }
        # Treat empty string as None (auto-detect)
        override_value = request.model_type_override or None
        if override_value is not None and override_value not in valid_types:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model_type_override: {request.model_type_override}",
            )
        current_settings.model_type_override = override_value
        # Update engine pool entry type immediately
        type_to_engine = {
            "llm": "batched",
            "vlm": "vlm",
            "embedding": "embedding",
            "reranker": "reranker",
            "audio_stt": "audio_stt",
            "audio_tts": "audio_tts",
            "audio_sts": "audio_sts",
            "image": "image_gen",
            "video": "video_gen",
        }
        if override_value:
            entry.model_type = override_value
            entry.engine_type = type_to_engine.get(override_value, "batched")
        else:
            # Reset to auto-detected type
            from pathlib import Path

            try:
                from ..model_discovery import detect_model_type

                detected_type = detect_model_type(Path(entry.model_path))
            except ImportError:
                detected_type = entry.engine_type or "batched"
                logger.warning(
                    "detect_model_type not available, using %s", detected_type
                )
            entry.model_type = detected_type
            entry.engine_type = type_to_engine.get(detected_type, "batched")
    if "max_context_window" in sent:
        current_settings.max_context_window = request.max_context_window
    if "max_tokens" in sent:
        current_settings.max_tokens = request.max_tokens
    if "temperature" in sent:
        current_settings.temperature = request.temperature
    if "top_p" in sent:
        current_settings.top_p = request.top_p
    if "top_k" in sent:
        current_settings.top_k = request.top_k
    if "repetition_penalty" in sent:
        current_settings.repetition_penalty = request.repetition_penalty
    if "min_p" in sent:
        current_settings.min_p = request.min_p
    if "presence_penalty" in sent:
        current_settings.presence_penalty = request.presence_penalty
    if "force_sampling" in sent:
        current_settings.force_sampling = request.force_sampling
    if "max_tool_result_tokens" in sent:
        # 0 means disable (reset to None)
        current_settings.max_tool_result_tokens = (
            request.max_tool_result_tokens
            if request.max_tool_result_tokens and request.max_tool_result_tokens > 0
            else None
        )
    if "enable_thinking" in sent:
        current_settings.enable_thinking = request.enable_thinking
    if "thinking_budget_enabled" in sent:
        current_settings.thinking_budget_enabled = (
            request.thinking_budget_enabled or False
        )
    if "thinking_budget_tokens" in sent:
        current_settings.thinking_budget_tokens = (
            request.thinking_budget_tokens
            if request.thinking_budget_tokens and request.thinking_budget_tokens > 0
            else None
        )
    if "chat_template_kwargs" in sent:
        current_settings.chat_template_kwargs = request.chat_template_kwargs
    if "forced_ct_kwargs" in sent:
        current_settings.forced_ct_kwargs = request.forced_ct_kwargs
    if "ttl_seconds" in sent:
        current_settings.ttl_seconds = request.ttl_seconds
    if "index_cache_freq" in sent:
        # 0 means disable (reset to None)
        current_settings.index_cache_freq = (
            request.index_cache_freq
            if request.index_cache_freq and request.index_cache_freq >= 2
            else None
        )
    # TurboQuant KV cache settings
    if "turboquant_kv_enabled" in sent:
        current_settings.turboquant_kv_enabled = request.turboquant_kv_enabled or False
    if "turboquant_kv_bits" in sent:
        current_settings.turboquant_kv_bits = request.turboquant_kv_bits or 4
    # SpecPrefill settings
    if "specprefill_enabled" in sent:
        current_settings.specprefill_enabled = request.specprefill_enabled or False
    if "specprefill_draft_model" in sent:
        current_settings.specprefill_draft_model = (
            request.specprefill_draft_model or None
        )
    if "specprefill_keep_pct" in sent:
        current_settings.specprefill_keep_pct = request.specprefill_keep_pct or None
    if "specprefill_threshold" in sent:
        current_settings.specprefill_threshold = request.specprefill_threshold or None
    # DFlash settings
    if "dflash_enabled" in sent:
        new_dflash_enabled = bool(request.dflash_enabled)
        if new_dflash_enabled:
            try:
                from ..engine.dflash import is_dflash_compatible

                compat_ok, compat_reason = is_dflash_compatible(entry.model_path)
            except ImportError:
                compat_ok, compat_reason = False, "dflash module not available"
                logger.warning("is_dflash_compatible not available")
            if not compat_ok:
                raise HTTPException(status_code=400, detail=compat_reason)
        current_settings.dflash_enabled = new_dflash_enabled
    if "dflash_draft_model" in sent:
        current_settings.dflash_draft_model = request.dflash_draft_model or None
    if "dflash_draft_quant_enabled" in sent:
        current_settings.dflash_draft_quant_enabled = (
            bool(request.dflash_draft_quant_enabled)
            if request.dflash_draft_quant_enabled is not None
            else None
        )
    if "dflash_draft_quant_weight_bits" in sent:
        current_settings.dflash_draft_quant_weight_bits = (
            int(request.dflash_draft_quant_weight_bits)
            if request.dflash_draft_quant_weight_bits is not None
            else None
        )
    if "dflash_draft_quant_activation_bits" in sent:
        current_settings.dflash_draft_quant_activation_bits = (
            int(request.dflash_draft_quant_activation_bits)
            if request.dflash_draft_quant_activation_bits is not None
            else None
        )
    if "dflash_draft_quant_group_size" in sent:
        current_settings.dflash_draft_quant_group_size = (
            int(request.dflash_draft_quant_group_size)
            if request.dflash_draft_quant_group_size is not None
            else None
        )
    if "dflash_max_ctx" in sent:
        # 0/None means "unlimited" — the engine treats None as no fallback threshold
        value = request.dflash_max_ctx
        current_settings.dflash_max_ctx = value if value and value > 0 else None
    if "dflash_in_memory_cache" in sent:
        current_settings.dflash_in_memory_cache = bool(request.dflash_in_memory_cache)
    if "dflash_in_memory_cache_max_entries" in sent:
        value = request.dflash_in_memory_cache_max_entries
        current_settings.dflash_in_memory_cache_max_entries = (
            int(value) if value and value > 0 else 4
        )
    if (
        "dflash_in_memory_cache_max_bytes" in sent
        and request.dflash_in_memory_cache_max_bytes
    ):
        current_settings.dflash_in_memory_cache_max_bytes = int(
            request.dflash_in_memory_cache_max_bytes
        )
    if "dflash_ssd_cache" in sent:
        ssd_requested = bool(request.dflash_ssd_cache)
        if ssd_requested:
            in_mem_after = (
                bool(request.dflash_in_memory_cache)
                if "dflash_in_memory_cache" in sent
                else current_settings.dflash_in_memory_cache
            )
            if not in_mem_after:
                raise HTTPException(
                    status_code=400,
                    detail="DFlash SSD cache requires the in-memory cache to be enabled.",
                )
            ssd_dir = getattr(
                getattr(_get_engine_pool(), "_scheduler_config", None),
                "paged_ssd_cache_dir",
                None,
            )
            if not ssd_dir:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "DFlash SSD cache requires Fusion-MLX paged SSD cache to be enabled "
                        "(set --paged-ssd-cache-dir or configure it in settings)."
                    ),
                )
        current_settings.dflash_ssd_cache = ssd_requested
    if "dflash_ssd_cache_max_bytes" in sent and request.dflash_ssd_cache_max_bytes:
        current_settings.dflash_ssd_cache_max_bytes = int(
            request.dflash_ssd_cache_max_bytes
        )
    if "dflash_draft_window_size" in sent:
        # 0 / None / negative → fall back to dflash-mlx internal default (1024).
        value = request.dflash_draft_window_size
        current_settings.dflash_draft_window_size = (
            int(value) if value and value > 0 else None
        )
    if "dflash_draft_sink_size" in sent:
        # Negative is invalid; 0 is a legal sink-size (no sink tokens).
        value = request.dflash_draft_sink_size
        current_settings.dflash_draft_sink_size = (
            int(value) if value is not None and value >= 0 else None
        )
    if "dflash_verify_mode" in sent:
        value = request.dflash_verify_mode
        # dflash-mlx accepts: dflash | adaptive | ddtree | off.
        # Anything else (including empty string) → revert to dflash default.
        current_settings.dflash_verify_mode = (
            value if value in ("dflash", "adaptive", "ddtree", "off") else None
        )
    # N-gram self-speculative decode (per-model override of env defaults).
    # Sent-as-null clears to None so the FUSION_NGRAM_SPEC_* env default
    # takes over; a real value overrides it per-model.
    if "ngram_spec_enabled" in sent:
        current_settings.ngram_spec_enabled = (
            bool(request.ngram_spec_enabled)
            if request.ngram_spec_enabled is not None
            else None
        )
    if "ngram_spec_order" in sent:
        current_settings.ngram_spec_order = (
            int(request.ngram_spec_order)
            if request.ngram_spec_order is not None and request.ngram_spec_order >= 1
            else None
        )
    if "ngram_spec_num_draft" in sent:
        current_settings.ngram_spec_num_draft = (
            int(request.ngram_spec_num_draft)
            if request.ngram_spec_num_draft is not None
            and request.ngram_spec_num_draft >= 1
            else None
        )
    if "ngram_spec_break_even" in sent:
        current_settings.ngram_spec_break_even = (
            float(request.ngram_spec_break_even)
            if request.ngram_spec_break_even is not None
            and 0.0 <= request.ngram_spec_break_even <= 1.0
            else None
        )

    # Native MTP (mlx-lm PR 990 / PR 15 monkey-patch)
    if "mtp_enabled" in sent:
        new_mtp_enabled = bool(request.mtp_enabled)
        if new_mtp_enabled:
            # Compatibility check: the model needs MTP heads in config.json AND
            # the model_type must be one PR 990 / PR 15 covers AND the weight
            # files must actually contain mtp.* tensors. The last check is
            # the one that catches mlx-community converted weights where the
            # default sanitize path stripped the MTP heads.
            from pathlib import Path

            try:
                from .helpers import _mtp_compat_for_model

                model_info = {"model_path": entry.model_path}
                compat_ok, compat_reason = _mtp_compat_for_model(model_info)
                if not compat_ok:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Model is not MTP-compatible: {compat_reason}",
                    )
            except ImportError:
                logger.warning(
                    "_mtp_compat_for_model not available, skipping MTP compat check"
                )
            # Mutual exclusion with DFlash / TurboQuant — ModelSettings.__post_init__
            # also enforces this, but we surface a clearer error here.
            dflash_after = (
                bool(request.dflash_enabled)
                if "dflash_enabled" in sent
                else current_settings.dflash_enabled
            )
            if dflash_after:
                raise HTTPException(
                    status_code=400,
                    detail="MTP and DFlash cannot both be enabled; choose one speculative-decoding path.",
                )
            tq_after = (
                bool(request.turboquant_kv_enabled)
                if "turboquant_kv_enabled" in sent
                else current_settings.turboquant_kv_enabled
            )
            if tq_after:
                raise HTTPException(
                    status_code=400,
                    detail="MTP and TurboQuant KV cannot both be enabled; TurboQuant patches the attention path MTP relies on.",
                )
        current_settings.mtp_enabled = new_mtp_enabled

    # VLM MTP (mlx-vlm f96138e+, gemma4_assistant drafter)
    if "vlm_mtp_enabled" in sent:
        new_vlm_mtp = bool(request.vlm_mtp_enabled)
        if new_vlm_mtp:
            drafter_after = (
                request.vlm_mtp_draft_model
                if "vlm_mtp_draft_model" in sent
                else current_settings.vlm_mtp_draft_model
            )
            if not drafter_after:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "vlm_mtp_enabled requires vlm_mtp_draft_model "
                        "(path to a gemma4_assistant drafter, "
                        "e.g. 'gemma-4-26B-A4B-it-assistant')."
                    ),
                )
            # Mutex enforced again at ModelSettings.__post_init__ for
            # last-mile safety, but surface a clearer error here.
            for other_field, other_label in (
                ("dflash_enabled", "DFlash"),
                ("specprefill_enabled", "SpecPrefill"),
                ("mtp_enabled", "MTP"),
                ("turboquant_kv_enabled", "TurboQuant KV"),
            ):
                other_after = (
                    bool(getattr(request, other_field))
                    if other_field in sent
                    else getattr(current_settings, other_field)
                )
                if other_after:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            f"vlm_mtp_enabled and {other_label} cannot both be "
                            "enabled; choose one speculative-decoding path."
                        ),
                    )
        current_settings.vlm_mtp_enabled = new_vlm_mtp
    if "vlm_mtp_draft_model" in sent:
        current_settings.vlm_mtp_draft_model = request.vlm_mtp_draft_model or None
    if "vlm_mtp_draft_block_size" in sent:
        current_settings.vlm_mtp_draft_block_size = request.vlm_mtp_draft_block_size

    if "reasoning_parser" in sent:
        current_settings.reasoning_parser = request.reasoning_parser or None
    if request.is_pinned is not None:
        current_settings.is_pinned = request.is_pinned
        # Also update the engine pool entry
        entry.is_pinned = request.is_pinned
    if request.is_default is not None:
        current_settings.is_default = request.is_default
        # Update server_state.default_model if setting as default
        if request.is_default and server_state:
            server_state.default_model = model_id
    if "trust_remote_code" in sent:
        current_settings.trust_remote_code = bool(request.trust_remote_code)

    # If an active profile was set, clear it when the user's save diverges
    # from the profile's stored values.  Only compare fields present in
    # both the profile and the current settings — new fields in the model
    # settings that the profile doesn't have are silently merged in, and
    # removed fields (no longer in the profile) are skipped.
    if current_settings.active_profile_name:
        profile = settings_manager.get_profile(
            model_id, current_settings.active_profile_name
        )
        if profile is None:
            current_settings.active_profile_name = None
        else:
            profile_settings = profile.get("settings", {}) or {}
            candidate = current_settings.to_dict()
            diverged = False
            for key, expected in profile_settings.items():
                # Profile None means "unconstrained" — candidate.to_dict()
                # drops None, so treat profile None as no constraint to
                # keep the comparison symmetric.
                if expected is None:
                    continue
                if key not in candidate:
                    diverged = True
                    break
                if candidate[key] != expected:
                    diverged = True
                    break
            if diverged:
                current_settings.active_profile_name = None
            else:
                new_fields = {
                    k: v
                    for k, v in candidate.items()
                    if k not in profile_settings and k not in EXCLUDED_FROM_PROFILES
                }
                if new_fields:
                    profile_settings.update(new_fields)
                    profile["settings"] = profile_settings
                    settings_manager.update_profile(
                        model_id,
                        current_settings.active_profile_name,
                        settings=profile_settings,
                    )

    # Persist settings
    settings_manager.set_settings(model_id, current_settings)

    # Auto-unload (and re-load if pinned) when a setting that only takes
    # effect at engine construction time is changed on a loaded model.
    requires_reload = entry.engine is not None and (
        ("model_type_override" in sent and entry.engine_type != prev_engine_type)
        or "index_cache_freq" in sent
        or "dflash_enabled" in sent
        or "dflash_draft_model" in sent
        or "dflash_draft_quant_enabled" in sent
        or "dflash_draft_quant_weight_bits" in sent
        or "dflash_draft_quant_activation_bits" in sent
        or "dflash_draft_quant_group_size" in sent
        or "dflash_max_ctx" in sent
        or "dflash_in_memory_cache" in sent
        or "dflash_in_memory_cache_max_entries" in sent
        or "dflash_in_memory_cache_max_bytes" in sent
        or "dflash_ssd_cache" in sent
        or "dflash_ssd_cache_max_bytes" in sent
        # trust_remote_code is plumbed at model load time; toggling it on
        # an already-loaded engine has no effect until reload.
        or "trust_remote_code" in sent
    )
    auto_unloaded = False
    auto_reloaded = False
    if requires_reload:
        was_pinned = entry.is_pinned
        try:
            logger.info(
                f"Settings changed for loaded model {model_id}, auto-unloading."
            )
            await engine_pool._unload_engine(model_id)
            auto_unloaded = True
        except Exception as e:
            logger.warning(f"Auto-unload failed for {model_id}: {e}")
        if auto_unloaded and was_pinned:
            try:
                await engine_pool._load_engine(model_id)
                auto_reloaded = True
                logger.info(f"Auto-reloaded pinned model {model_id} with new settings.")
            except Exception as e:
                logger.warning(f"Auto-reload failed for pinned model {model_id}: {e}")

    return {
        "success": True,
        "model_id": model_id,
        "settings": current_settings.to_dict(),
        "model_type": entry.model_type,
        "engine_type": entry.engine_type,
        "requires_reload": requires_reload,
        "auto_unloaded": auto_unloaded,
        "auto_reloaded": auto_reloaded,
    }


def _read_settings_json() -> dict:
    path = Path.home() / ".fusion-mlx" / "settings.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _write_settings_json(data: dict) -> None:
    path = Path.home() / ".fusion-mlx" / "settings.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _save_model_settings_fallback(model_id: str, request: ModelSettingsRequest) -> dict:
    """Save per-model settings to settings.json when settings_manager is unavailable.

    Reads settings.json's model_settings dict, applies changes, writes back.
    Returns a success response dict.
    """
    sj = _read_settings_json()
    model_settings = sj.setdefault("model_settings", {})
    entry = model_settings.setdefault(model_id, {})
    sent = request.model_fields_set
    requires_reload = False

    # Simple field mapping: request field -> JSON key
    FIELD_MAP = {
        "model_alias": "model_alias",
        "model_type_override": "model_type_override",
        "max_context_window": "max_context_window",
        "max_tokens": "max_tokens",
        "temperature": "temperature",
        "top_p": "top_p",
        "top_k": "top_k",
        "repetition_penalty": "repetition_penalty",
        "min_p": "min_p",
        "presence_penalty": "presence_penalty",
        "force_sampling": "force_sampling",
        "max_tool_result_tokens": "max_tool_result_tokens",
        "ttl_seconds": "ttl_seconds",
        "index_cache_freq": "index_cache_freq",
        "enable_thinking": "enable_thinking",
        "thinking_budget_enabled": "thinking_budget_enabled",
        "thinking_budget_tokens": "thinking_budget_tokens",
        "turboquant_kv_enabled": "turboquant_kv_enabled",
        "turboquant_kv_bits": "turboquant_kv_bits",
        "specprefill_enabled": "specprefill_enabled",
        "specprefill_draft_model": "specprefill_draft_model",
        "specprefill_keep_pct": "specprefill_keep_pct",
        "specprefill_threshold": "specprefill_threshold",
        "dflash_enabled": "dflash_enabled",
        "dflash_draft_model": "dflash_draft_model",
        "dflash_draft_quant_enabled": "dflash_draft_quant_enabled",
        "dflash_draft_quant_weight_bits": "dflash_draft_quant_weight_bits",
        "dflash_draft_quant_activation_bits": "dflash_draft_quant_activation_bits",
        "dflash_draft_quant_group_size": "dflash_draft_quant_group_size",
        "dflash_max_ctx": "dflash_max_ctx",
        "dflash_in_memory_cache": "dflash_in_memory_cache",
        "dflash_in_memory_cache_max_entries": "dflash_in_memory_cache_max_entries",
        "dflash_in_memory_cache_max_bytes": "dflash_in_memory_cache_max_bytes",
        "dflash_ssd_cache": "dflash_ssd_cache",
        "dflash_ssd_cache_max_bytes": "dflash_ssd_cache_max_bytes",
        "dflash_draft_window_size": "dflash_draft_window_size",
        "dflash_draft_sink_size": "dflash_draft_sink_size",
        "dflash_verify_mode": "dflash_verify_mode",
        "mtp_enabled": "mtp_enabled",
        "vlm_mtp_enabled": "vlm_mtp_enabled",
        "vlm_mtp_draft_model": "vlm_mtp_draft_model",
        "vlm_mtp_draft_block_size": "vlm_mtp_draft_block_size",
        "ngram_spec_enabled": "ngram_spec_enabled",
        "ngram_spec_order": "ngram_spec_order",
        "ngram_spec_num_draft": "ngram_spec_num_draft",
        "ngram_spec_break_even": "ngram_spec_break_even",
        "reasoning_parser": "reasoning_parser",
        "is_pinned": "is_pinned",
        "is_default": "is_default",
        "trust_remote_code": "trust_remote_code",
    }

    # Complex fields that need special handling
    if "chat_template_kwargs" in sent:
        entry["chat_template_kwargs"] = request.chat_template_kwargs
    if "forced_ct_kwargs" in sent:
        entry["forced_ct_kwargs"] = request.forced_ct_kwargs

    # Validate model_type_override
    if "model_type_override" in sent:
        valid_types = {
            "llm",
            "vlm",
            "embedding",
            "reranker",
            "audio_stt",
            "audio_tts",
            "audio_sts",
            "image",
            "video",
        }
        override_value = request.model_type_override or None
        if override_value is not None and override_value not in valid_types:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model_type_override: {request.model_type_override}",
            )

    # Validate dflash_verify_mode
    if "dflash_verify_mode" in sent and request.dflash_verify_mode:
        valid_modes = {"soft", "hard", "off"}
        if request.dflash_verify_mode not in valid_modes:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid dflash_verify_mode: {request.dflash_verify_mode}",
            )

    # Validate reasoning_parser
    if "reasoning_parser" in sent and request.reasoning_parser:
        valid_parsers = {"disabled", "deepseek", "anthropic"}
        if request.reasoning_parser not in valid_parsers:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid reasoning_parser: {request.reasoning_parser}",
            )

    # Check which fields require reload
    RELOAD_FIELDS = {
        "model_type_override",
        "max_context_window",
        "chat_template_kwargs",
        "forced_ct_kwargs",
        "dflash_enabled",
        "dflash_draft_model",
        "dflash_draft_quant_enabled",
        "dflash_draft_quant_weight_bits",
        "dflash_draft_quant_activation_bits",
        "dflash_draft_quant_group_size",
        "dflash_max_ctx",
        "dflash_draft_window_size",
        "dflash_draft_sink_size",
        "dflash_verify_mode",
        "mtp_enabled",
        "vlm_mtp_enabled",
        "vlm_mtp_draft_model",
        "vlm_mtp_draft_block_size",
        "ngram_spec_enabled",
        "ngram_spec_order",
        "ngram_spec_num_draft",
        "specprefill_enabled",
        "specprefill_draft_model",
        "turboquant_kv_enabled",
        "turboquant_kv_bits",
        "enable_thinking",
        "thinking_budget_enabled",
        "thinking_budget_tokens",
        "dflash_in_memory_cache",
        "dflash_in_memory_cache_max_entries",
        "dflash_in_memory_cache_max_bytes",
        "dflash_ssd_cache",
        "dflash_ssd_cache_max_bytes",
        "trust_remote_code",
    }
    requires_reload = bool(sent & RELOAD_FIELDS)

    # Apply simple fields
    for req_field, json_key in FIELD_MAP.items():
        if req_field in sent:
            val = getattr(request, req_field)
            entry[json_key] = val

    # Handle is_default: if setting a new default, clear old default
    if "is_default" in sent and request.is_default:
        for mid, ms in model_settings.items():
            if mid != model_id and ms.get("is_default"):
                ms["is_default"] = False

    try:
        _write_settings_json(sj)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save settings: {e}")

    logger.info(
        f"Model settings saved (fallback mode): {model_id}, fields={list(sent)}"
    )
    return {
        "success": True,
        "model_id": model_id,
        "settings": entry,
        "requires_reload": requires_reload,
        "auto_unloaded": False,
        "auto_reloaded": False,
    }


router = _router
