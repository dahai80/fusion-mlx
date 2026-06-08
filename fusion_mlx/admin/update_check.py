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
# Update Check
# =============================================================================

_update_cache: dict[str, Any] | None = None
_update_cache_time: float = 0.0
_UPDATE_CACHE_TTL = 3600  # 1 hour


@_router.get("/api/update-check")
async def check_update(
    is_admin: bool = Depends(require_admin),
):
    """Check GitHub Releases for newer oMLX version (cached 24h)."""
    global _update_cache, _update_cache_time

    now = time.time()
    if _update_cache is not None and now - _update_cache_time < _UPDATE_CACHE_TTL:
        return _update_cache

    no_update = {
        "update_available": False,
        "latest_version": None,
        "release_url": None,
    }

    try:
        # Use the releases list (not /releases/latest) and pick the highest
        # stable PEP 440 tag. Dev/rc tags here have historically been
        # published with the GitHub prerelease flag unset, which makes
        # /releases/latest return them as if they were stable.
        resp = await asyncio.to_thread(
            requests.get,
            "https://api.github.com/repos/jundot/omlx/releases",
            params={"per_page": 20},
            timeout=5,
        )
        if resp.status_code != 200:
            _update_cache = no_update
            _update_cache_time = now
            return _update_cache

        data = select_latest_stable_release(resp.json())
        if data is None:
            _update_cache = no_update
            _update_cache_time = now
            return _update_cache

        latest = data["tag_name"].lstrip("v")

        try:
            from packaging.version import Version

            update_available = Version(latest) > Version(_omlx_version)
        except Exception:
            update_available = False

        if update_available:
            _update_cache = {
                "update_available": True,
                "latest_version": latest,
                "release_url": data.get("html_url"),
            }
        else:
            _update_cache = no_update

        _update_cache_time = now
    except Exception:
        _update_cache = no_update
        _update_cache_time = now

    return _update_cache



router = _router
