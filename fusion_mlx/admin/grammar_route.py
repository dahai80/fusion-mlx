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
# Grammar API Routes
# =============================================================================


_SUPPORTED_MODELS_DOC_RE = re.compile(
    r"Supported models:\s*\n((?:\s*-\s*\S.*\n?)+)",
)


def _models_from_docstring(fn) -> list[str]:
    """Extract the ``Supported models:`` bullet list from an xgrammar 0.1.34+
    structural-tag function's docstring. Returns ``[]`` if the section is
    absent or unparseable."""
    doc = inspect.getdoc(fn) or ""
    match = _SUPPORTED_MODELS_DOC_RE.search(doc)
    if not match:
        return []
    return [
        line.strip().lstrip("-").strip()
        for line in match.group(1).splitlines()
        if line.strip().startswith("-")
    ]


@_router.get("/api/grammar/parsers")
async def list_grammar_parsers(is_admin: bool = Depends(require_admin)):
    """Return available reasoning parser names from xgrammar.

    Supports both API generations:

    - **xgrammar 0.1.34+** exposes a per-model registry at
        ``xgrammar.builtin_structural_tag._structural_tag_registry``; supported
        model names are pulled from each function's docstring.
    - **xgrammar 0.1.32–0.1.33** exposes the now-removed helper
        ``get_builtin_structural_tag_supported_models()``.

    Returns ``[]`` if xgrammar is missing, fails to load (e.g. broken native
    binding on macOS arm64), or has neither API available.
    """
    # Install the torch stub BEFORE any xgrammar import. If this lives
    # inside the first try-block, a failure on the 0.1.34+ path can leave
    # the fallback try-block importing xgrammar without the stub, which
    # is guaranteed ImportError on stub-only (DMG) deployments.
    try:
        from fusion_mlx._torch_stub import install as _install_torch_stub

        _install_torch_stub()
    except Exception as e:  # pragma: no cover — defensive
        logger.debug("torch stub install failed: %s", e)

    # Prefer the 0.1.34+ registry so newer parsers (qwen3_6, gemma4,
    # deepseek_v4, ...) are exposed.
    try:
        from xgrammar.builtin_structural_tag import _structural_tag_registry

        return [
            {"value": style, "label": style, "models": _models_from_docstring(fn)}
            for style, fn in _structural_tag_registry.items()
        ]
    except Exception as e:
        logger.debug("xgrammar 0.1.34+ registry unavailable: %s", e)

    # Fall back to the pre-0.1.34 helper.
    try:
        from xgrammar import get_builtin_structural_tag_supported_models

        supported = get_builtin_structural_tag_supported_models()
        return [
            {"value": style, "label": style, "models": models}
            for style, models in supported.items()
        ]
    except Exception as e:
        logger.warning("xgrammar parser discovery unavailable: %s", e)
        return []



router = _router
