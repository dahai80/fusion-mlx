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
# HTML Page Routes
# =============================================================================


@_router.get("/", response_class=HTMLResponse)
async def login_page(request: Request):
    """
    Render the admin login page or setup page.

    If no API key is configured, the page will show the initial setup form.
    Otherwise, it shows the standard login form.

    Returns:
        HTML login/setup page.
    """
    # Redirect to dashboard if already authenticated
    from .auth import verify_session

    if verify_session(request):
        return RedirectResponse(url="/admin/dashboard", status_code=302)

    global_settings = _get_global_settings()

    # Skip login page when skip_api_key_verification is enabled
    if (
        global_settings is not None
        and global_settings.auth.skip_api_key_verification
    ):
        return RedirectResponse(url="/admin/dashboard", status_code=302)

    api_key_configured = bool(global_settings and global_settings.auth.api_key)
    return templates.TemplateResponse(
        request,
        "login.html",
        {"api_key_configured": api_key_configured},
    )


@_router.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page(request: Request, is_admin: bool = Depends(require_admin)):
    """
    Render the admin dashboard page.

    Requires admin authentication via session cookie.

    Returns:
        HTML dashboard page with server status and model list.
    """
    return templates.TemplateResponse(request, "dashboard.html", {})


@_router.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request, is_admin: bool = Depends(require_admin)):
    """
    Render the chat page for interacting with models.

    Requires admin authentication via session cookie.
    The API key is injected into the template context so that
    the chat page can auto-set it in localStorage, bypassing
    the manual API key entry modal.

    Returns:
        HTML chat page.
    """
    global_settings = _get_global_settings()
    api_key = global_settings.auth.api_key if global_settings else ""
    return templates.TemplateResponse(
        request, "chat.html", {"api_key": api_key or ""}
    )


@_router.get("/static/{path:path}")
async def admin_static(path: str):
    """Serve static files for admin panel (CSS, JS, fonts, logos, etc.)."""
    file_path = static_dir / path
    if not file_path.is_file() or not file_path.resolve().is_relative_to(
        static_dir.resolve()
    ):
        raise HTTPException(status_code=404, detail="File not found")
    media_types = {
        ".svg": "image/svg+xml",
        ".png": "image/png",
        ".ico": "image/x-icon",
        ".css": "text/css",
        ".js": "application/javascript",
        ".woff2": "font/woff2",
        ".woff": "font/woff",
        ".ttf": "font/ttf",
    }
    media_type = media_types.get(file_path.suffix, "application/octet-stream")
    return FileResponse(file_path, media_type=media_type)



router = _router
