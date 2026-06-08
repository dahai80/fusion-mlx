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
# Logs API Routes
# =============================================================================


def _tail_file(file_path: Path, num_lines: int) -> tuple[str, int]:
    """
    Read the last N lines of a file efficiently.

    Uses a deque to efficiently keep only the last N lines in memory.

    Args:
        file_path: Path to the log file.
        num_lines: Number of lines to return.

    Returns:
        Tuple of (content_string, total_line_count)
    """
    if not file_path.exists():
        return "", 0

    # Use deque for efficient tail operation
    lines = deque(maxlen=num_lines)
    total_lines = 0

    with open(file_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            lines.append(line)
            total_lines += 1

    return "".join(lines), total_lines


def _get_available_log_files(log_dir: Path) -> list[str]:
    """
    Get list of available log files sorted by modification time.

    Args:
        log_dir: Directory containing log files.

    Returns:
        List of log file names, newest first.
    """
    if not log_dir.exists():
        return []

    files = []
    for f in log_dir.iterdir():
        # Match server.log and server.log.YYYY-MM-DD patterns
        if f.name.startswith("server") and (f.suffix == ".log" or ".log." in f.name):
            files.append(f.name)

    # Sort by modification time (newest first)
    files.sort(key=lambda x: (log_dir / x).stat().st_mtime, reverse=True)
    return files


@_router.get("/api/logs")
async def get_logs(
    lines: int = 100,
    file: str | None = None,
    is_admin: bool = Depends(require_admin),
):
    """
    Get server logs.

    Returns the last N lines of the specified log file (or current log).
    Supports viewing historical rotated log files.

    Args:
        lines: Number of lines to return (default: 100, max: 10000).
        file: Optional specific log file name. If not specified, uses current log.

    Returns:
        JSON response with log content and metadata:
        - logs: The log content string
        - total_lines: Total number of lines in the file
        - log_file: Name of the log file being read
        - available_files: List of available log files

    Raises:
        HTTPException: 401 if not authenticated, 503 if server not initialized,
                        400 if invalid file name, 404 if log file not found.
    """
    global_settings = _get_global_settings()

    if global_settings is None:
        raise HTTPException(status_code=503, detail="Server not initialized")

    # Limit lines to prevent memory issues
    lines = min(max(1, lines), 10000)

    log_dir = global_settings.logging.get_log_dir(global_settings.base_path)

    # Get available log files
    available_files = _get_available_log_files(log_dir)

    # Determine which file to read
    if file:
        # Validate file name (prevent path traversal)
        if "/" in file or "\\" in file or ".." in file:
            raise HTTPException(status_code=400, detail="Invalid file name")
        log_file = log_dir / file
        if not log_file.exists():
            raise HTTPException(status_code=404, detail=f"Log file not found: {file}")
    else:
        # Default to current log file
        log_file = log_dir / "server.log"

    # Read log content
    if log_file.exists():
        content, total_lines = _tail_file(log_file, lines)
    else:
        content = ""
        total_lines = 0

    return {
        "logs": content,
        "total_lines": total_lines,
        "log_file": log_file.name,
        "available_files": available_files,
    }



router = _router
