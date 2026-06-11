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
# Authentication API Routes
# =============================================================================


@_router.post("/api/login")
async def login(request: LoginRequest, response: Response):
    """
    Authenticate with API key and create session.

    Requires an API key to be configured on the server. If no API key
    is configured, returns 400 directing the user to set one up first.

    Args:
        request: LoginRequest containing the API key.
        response: FastAPI response object for setting cookies.

    Returns:
        JSON response with success status.

    Raises:
        HTTPException: 400 if no API key configured, 401 if invalid.
    """
    global_settings = _get_global_settings()
    server_api_key = global_settings.auth.api_key if global_settings else None

    # Reject login if no API key is configured (must use setup first)
    if not server_api_key:
        raise HTTPException(
            status_code=400,
            detail="No API key configured. Please set up an API key first.",
        )

    # Main key only — sub keys must not grant admin login
    if not verify_api_key(request.api_key, server_api_key):
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
        )

    # Create session token and set cookie
    token = create_session_token(remember=request.remember)
    cookie_max_age = REMEMBER_ME_MAX_AGE if request.remember else SESSION_MAX_AGE
    response.set_cookie(
        key="omlx_admin_session",
        value=token,
        httponly=True,
        samesite="lax",
        max_age=cookie_max_age,
    )

    return {"success": True}


@_router.post("/api/setup-api-key")
async def setup_api_key(request: SetupApiKeyRequest, response: Response):
    """
    Set up the initial API key when none is configured.

    This endpoint is only available when no API key is currently set.
    After successful setup, a session is created so the user is
    immediately logged in.

    Args:
        request: SetupApiKeyRequest with api_key and api_key_confirm.
        response: FastAPI response object for setting cookies.

    Returns:
        JSON response with success status.

    Raises:
        HTTPException: 400 if key already configured, validation fails,
                        or keys don't match.
    """
    from ..server import _server_state

    global_settings = _get_global_settings()

    # Only allow setup if no API key is currently configured
    if global_settings and global_settings.auth.api_key:
        raise HTTPException(
            status_code=400,
            detail="API key is already configured. Use settings to change it.",
        )

    # Validate confirmation match
    if request.api_key != request.api_key_confirm:
        raise HTTPException(status_code=400, detail="API keys do not match")

    # Validate key format
    is_valid, error_msg = validate_api_key(request.api_key)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_msg)

    # Apply to settings and runtime
    global_settings.auth.api_key = request.api_key
    _server_state.api_key = request.api_key

    # Persist to file
    try:
        global_settings.save()
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to save settings: {e}"
        )

    logger.info("API key configured via initial setup")

    # Create session token and set cookie (auto-login after setup)
    token = create_session_token()
    response.set_cookie(
        key="omlx_admin_session",
        value=token,
        httponly=True,
        samesite="lax",
        max_age=86400,  # 24 hours
    )

    return {"success": True, "message": "API key configured successfully"}


@_router.post("/api/logout")
async def logout(response: Response):
    """
    Clear session cookie and logout.

    Args:
        response: FastAPI response object for clearing cookies.

    Returns:
        JSON response with success status.
    """
    response.delete_cookie(key="omlx_admin_session")
    return {"success": True}


@_router.get("/auto-login")
async def auto_login(key: str = "", redirect: str = "/admin/dashboard"):
    """
    Auto-login using API key and redirect to the target admin page.

    Used by the macOS menubar app to open admin pages with automatic
    authentication, bypassing the manual login form.

    Args:
        key: The API key for authentication.
        redirect: The path to redirect to after login. Must start with /admin.

    Returns:
        HTTP 302 redirect with session cookie set.
    """
    if not redirect.startswith("/admin"):
        raise HTTPException(status_code=400, detail="Invalid redirect path")

    global_settings = _get_global_settings()
    server_api_key = global_settings.auth.api_key if global_settings else None

    # Main key only — sub keys must not grant admin login
    if not key or not server_api_key or not verify_api_key(key, server_api_key):
        return RedirectResponse(url="/admin", status_code=302)

    token = create_session_token()
    response = RedirectResponse(url=redirect, status_code=302)
    response.set_cookie(
        key="omlx_admin_session",
        value=token,
        httponly=True,
        samesite="lax",
        max_age=86400,
    )
    return response



router = _router
