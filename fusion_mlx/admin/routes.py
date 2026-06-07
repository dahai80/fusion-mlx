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



# =============================================================================
# Router and Templates
# =============================================================================

router = APIRouter(prefix="/admin", tags=["admin"])
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")
static_dir = Path(__file__).parent / "static"


def _static_version(path: str) -> str:
    """Append file mtime as query string for cache busting."""
    file_path = static_dir / path
    if file_path.is_file():
        mtime = int(file_path.stat().st_mtime)
        return f"/admin/static/{path}?v={mtime}"
    return f"/admin/static/{path}"


templates.env.globals["static"] = _static_version

from fusion_mlx._version import __version__ as _omlx_version

templates.env.globals["version"] = _omlx_version

# Sub-routers (split by function for maintainability)
from .html_routes import router as html_router
from .auth_routes import router as auth_routes_router
from .subkey import router as subkey_router
from .grammar_route import router as grammar_router
from .models_route import router as models_router
from .profile import router as profile_router
from .preset import router as preset_router
from .settings import router as settings_router
from .logs import router as logs_router
from .stats import router as stats_router
from .hf_download import router as hf_download_router
from .ms_download import router as ms_download_router
from .accuracy_bench import router as accuracy_bench_router
from .bench import router as bench_router
from .update_check import router as update_check_router
from .oq import router as oq_router
from .hf_upload import router as hf_upload_router

# Register all sub-routers
router.include_router(html_router)
router.include_router(auth_routes_router)
router.include_router(subkey_router)
router.include_router(grammar_router)
router.include_router(models_router)
router.include_router(profile_router)
router.include_router(preset_router)
router.include_router(settings_router)
router.include_router(logs_router)
router.include_router(stats_router)
router.include_router(hf_download_router)
router.include_router(ms_download_router)
router.include_router(accuracy_bench_router)
router.include_router(bench_router)
router.include_router(update_check_router)
router.include_router(oq_router)
router.include_router(hf_upload_router)
