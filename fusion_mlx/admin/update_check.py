# SPDX-License-Identifier: Apache-2.0
"""Admin panel routes for oMLX server configuration.

This module provides HTTP routes for the admin panel including:
- Login/logout with API key authentication
- Dashboard for server monitoring
- Model settings management (per-model sampling parameters, pinning, default)
- Global settings management
"""

import asyncio
import logging
import time
from typing import Any

import requests
from fastapi import APIRouter, Depends

from ..utils.release_check import select_latest_stable_release
from .auth import (
    require_admin,
)

logger = logging.getLogger(__name__)

PRESET_REMOTE_URL = "http://bench.dpdns.org/assets/omlx_preset.json"




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
