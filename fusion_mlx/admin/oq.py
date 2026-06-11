# SPDX-License-Identifier: Apache-2.0
"""Admin panel routes for oMLX server configuration.

This module provides HTTP routes for the admin panel including:
- Login/logout with API key authentication
- Dashboard for server monitoring
- Model settings management (per-model sampling parameters, pinning, default)
- Global settings management
"""

import asyncio
import inspect
import json
import logging
import os
import re
import secrets
import shutil
import signal
import subprocess
import sys
import time
from collections import deque
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Optional

import requests
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from ..model_profiles import EXCLUDED_FROM_PROFILES
from ..settings import SubKeyEntry
from ..utils.release_check import select_latest_stable_release
from .auth import (
    REMEMBER_ME_MAX_AGE,
    SESSION_MAX_AGE,
    create_session_token,
    require_admin,
    validate_api_key,
    verify_api_key,
    verify_session,
)

logger = logging.getLogger(__name__)

PRESET_REMOTE_URL = "https://fusion_mlx.ai/assets/omlx_preset.json"



from .models import (
    LoginRequest, SetupApiKeyRequest, CreateSubKeyRequest, DeleteSubKeyRequest,
    CacheProbeRequest, ModelSettingsRequest,
    CreateProfileRequest, UpdateProfileRequest,
    CreateTemplateRequest, UpdateTemplateRequest,
    GlobalSettingsRequest,
    OQStartRequest,
)
from .helpers import (
    _format_cache_size, _paroquant_compat_for_model,
    _dflash_compat_for_model, _mtp_compat_for_model,
    _model_has_mtp_weight_tensors,
    _apply_log_level_runtime, _apply_model_dirs_runtime, _reload_models,
    _apply_memory_guard_tier_runtime, _apply_cache_settings_runtime,
    _apply_sampling_settings_runtime,
    format_size, get_ssd_disk_info, get_system_memory_info,
    _schedule_self_terminate,
    _require_settings_manager, _require_admin_or_bearer, _require_model,
    set_admin_getters, set_hf_downloader, set_ms_downloader,
    set_oq_manager, set_hf_uploader,
    _get_engine_pool, _get_global_settings, _get_server_state,
    _get_hf_downloader, _get_ms_downloader, _get_oq_manager,
    _get_hf_uploader, _get_settings_manager,
)

_router = APIRouter()

# =============================================================================
# oQ Quantization API Routes
# =============================================================================


@_router.get("/api/oq/models")
async def list_oq_models(is_admin: bool = Depends(require_admin)):
    """List non-quantized models available for oQ quantization."""
    if _oq_manager is None:
        raise HTTPException(
            status_code=503, detail="oQ quantizer not initialized"
        )
    source_models, all_models = await _oq_manager.list_quantizable_models()
    return {"models": source_models, "all_models": all_models}


@_router.get("/api/oq/estimate")
async def estimate_oq(
    model_path: str,
    oq_level: float,
    preserve_mtp: bool = False,
    is_admin: bool = Depends(require_admin),
):
    """Estimate effective bpw and output size for a model at given oQ level."""
    from ..oq import estimate_bpw_and_size

    try:
        result = await asyncio.to_thread(
            estimate_bpw_and_size,
            model_path,
            oq_level,
            64,  # group_size (default)
            preserve_mtp,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@_router.post("/api/oq/start")
async def start_oq_quantization(
    request: OQStartRequest,
    is_admin: bool = Depends(require_admin),
):
    """Start an oQ quantization task."""
    if _oq_manager is None:
        raise HTTPException(
            status_code=503, detail="oQ quantizer not initialized"
        )
    if request.oq_level not in (2, 3, 3.5, 4, 5, 6, 8):
        raise HTTPException(
            status_code=400,
            detail="Invalid oQ level. Must be 2, 3, 4, 5, 6, or 8",
        )
    if request.dtype not in ("bfloat16", "float16"):
        raise HTTPException(
            status_code=400,
            detail="Invalid dtype. Must be 'bfloat16' or 'float16'",
        )
    is_paro, _ = _paroquant_compat_for_model({"model_path": request.model_path})
    if is_paro:
        raise HTTPException(
            status_code=400,
            detail=(
                "Model is already quantized with paroquant; "
                "oQ re-quantization is not supported"
            ),
        )
    try:
        task = await _oq_manager.start_quantization(
            model_path=request.model_path,
            oq_level=request.oq_level,
            group_size=request.group_size,
            sensitivity_model_path=request.sensitivity_model_path,
            text_only=request.text_only,
            dtype=request.dtype,
            preserve_mtp=request.preserve_mtp,
            auto_proxy_sensitivity=request.auto_proxy_sensitivity,
        )
        return {"success": True, "task": task.to_dict()}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@_router.get("/api/oq/tasks")
async def list_oq_tasks(is_admin: bool = Depends(require_admin)):
    """List all quantization tasks."""
    if _oq_manager is None:
        raise HTTPException(
            status_code=503, detail="oQ quantizer not initialized"
        )
    return {"tasks": _oq_manager.get_tasks()}


@_router.post("/api/oq/cancel/{task_id}")
async def cancel_oq_task(
    task_id: str, is_admin: bool = Depends(require_admin)
):
    """Cancel an active quantization task."""
    if _oq_manager is None:
        raise HTTPException(
            status_code=503, detail="oQ quantizer not initialized"
        )
    success = await _oq_manager.cancel_quantization(task_id)
    if not success:
        raise HTTPException(
            status_code=404, detail="Task not found or not cancellable"
        )
    return {"success": True}


@_router.delete("/api/oq/task/{task_id}")
async def remove_oq_task(
    task_id: str, is_admin: bool = Depends(require_admin)
):
    """Remove a completed/failed/cancelled task."""
    if _oq_manager is None:
        raise HTTPException(
            status_code=503, detail="oQ quantizer not initialized"
        )
    success = _oq_manager.remove_task(task_id)
    if not success:
        raise HTTPException(
            status_code=404, detail="Task not found or still active"
        )
    return {"success": True}



router = _router
