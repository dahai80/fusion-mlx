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
# HuggingFace Upload Endpoints
# =============================================================================


@_router.post("/api/upload/validate-token")
async def validate_upload_token(
    request: HFValidateTokenRequest,
    is_admin: bool = Depends(require_admin),
):
    """Validate a HuggingFace token and return user info."""
    if _hf_uploader is None:
        raise HTTPException(
            status_code=503, detail="HF Uploader not initialized"
        )
    try:
        result = await _hf_uploader.validate_token(request.hf_token)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@_router.get("/api/upload/oq-models")
async def list_upload_oq_models(is_admin: bool = Depends(require_admin)):
    """List local oQ models available for upload."""
    if _hf_uploader is None:
        raise HTTPException(
            status_code=503, detail="HF Uploader not initialized"
        )
    oq_models = await _hf_uploader.list_oq_models()
    all_models = await _hf_uploader.list_all_models()
    return {"oq_models": oq_models, "all_models": all_models}


@_router.post("/api/upload/start")
async def start_upload(
    request: HFUploadRequest,
    is_admin: bool = Depends(require_admin),
):
    """Start an upload task to HuggingFace Hub."""
    if _hf_uploader is None:
        raise HTTPException(
            status_code=503, detail="HF Uploader not initialized"
        )
    try:
        task = await _hf_uploader.start_upload(
            model_path=request.model_path,
            repo_id=request.repo_id,
            token=request.hf_token,
            readme_source_path=request.readme_source_path,
            auto_readme=request.auto_readme,
            redownload_notice=request.redownload_notice,
            private=request.private,
        )
        return {"success": True, "task": task.to_dict()}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@_router.get("/api/upload/tasks")
async def list_upload_tasks(is_admin: bool = Depends(require_admin)):
    """List all upload tasks."""
    if _hf_uploader is None:
        raise HTTPException(
            status_code=503, detail="HF Uploader not initialized"
        )
    return {"tasks": _hf_uploader.get_tasks()}


@_router.post("/api/upload/cancel/{task_id}")
async def cancel_upload_task(
    task_id: str, is_admin: bool = Depends(require_admin)
):
    """Cancel an active or pending upload task."""
    if _hf_uploader is None:
        raise HTTPException(
            status_code=503, detail="HF Uploader not initialized"
        )
    success = await _hf_uploader.cancel_upload(task_id)
    if not success:
        raise HTTPException(
            status_code=404, detail="Task not found or not cancellable"
        )
    return {"success": True}


@_router.delete("/api/upload/task/{task_id}")
async def remove_upload_task(
    task_id: str, is_admin: bool = Depends(require_admin)
):
    """Remove a completed/failed/cancelled upload task."""
    if _hf_uploader is None:
        raise HTTPException(
            status_code=503, detail="HF Uploader not initialized"
        )
    success = _hf_uploader.remove_task(task_id)
    if not success:
        raise HTTPException(
            status_code=404, detail="Task not found or still active"
        )
    return {"success": True}

router = _router
