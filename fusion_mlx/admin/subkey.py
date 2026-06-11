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
# Sub Key Management Routes
# =============================================================================


@_router.post("/api/sub-keys")
async def create_sub_key(
    request: CreateSubKeyRequest, is_admin: bool = Depends(require_admin)
):
    """Create a new sub API key.

    Sub keys can only be used for API authentication, not admin login.

    Args:
        request: CreateSubKeyRequest with key and optional name.

    Returns:
        JSON with the created sub key entry.

    Raises:
        HTTPException: 400 if validation fails or key already exists.
    """
    global_settings = _get_global_settings()
    if global_settings is None:
        raise HTTPException(status_code=503, detail="Server not initialized")

    # Validate key format
    is_valid, error_msg = validate_api_key(request.key)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_msg)

    # Check for duplicate (against main key and existing sub keys)
    if global_settings.auth.api_key and secrets.compare_digest(
        request.key, global_settings.auth.api_key
    ):
        raise HTTPException(
            status_code=400, detail="Sub key cannot be the same as the main key"
        )

    for sk in global_settings.auth.sub_keys:
        if sk.key and secrets.compare_digest(request.key, sk.key):
            raise HTTPException(
                status_code=400, detail="This key already exists"
            )

    entry = SubKeyEntry(
        key=request.key,
        name=request.name or "",
        created_at=datetime.now(UTC).isoformat(),
    )
    global_settings.auth.sub_keys.append(entry)

    try:
        global_settings.save()
    except Exception as e:
        # Rollback
        global_settings.auth.sub_keys.pop()
        raise HTTPException(
            status_code=500, detail=f"Failed to save settings: {e}"
        )

    logger.info(f"Sub key created: {request.name or '(unnamed)'}")
    return {"success": True, "sub_key": entry.to_dict()}


@_router.delete("/api/sub-keys")
async def delete_sub_key(
    request: DeleteSubKeyRequest, is_admin: bool = Depends(require_admin)
):
    """Delete a sub API key.

    Args:
        request: DeleteSubKeyRequest with the key to delete.

    Returns:
        JSON with success status.

    Raises:
        HTTPException: 404 if key not found.
    """
    global_settings = _get_global_settings()
    if global_settings is None:
        raise HTTPException(status_code=503, detail="Server not initialized")

    # Find and remove the key
    for i, sk in enumerate(global_settings.auth.sub_keys):
        if sk.key and secrets.compare_digest(request.key, sk.key):
            removed = global_settings.auth.sub_keys.pop(i)
            try:
                global_settings.save()
            except Exception as e:
                global_settings.auth.sub_keys.insert(i, removed)
                raise HTTPException(
                    status_code=500, detail=f"Failed to save settings: {e}"
                )
            logger.info(f"Sub key deleted: {sk.name or '(unnamed)'}")
            return {"success": True}

    raise HTTPException(status_code=404, detail="Sub key not found")



router = _router
