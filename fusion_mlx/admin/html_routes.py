# SPDX-License-Identifier: Apache-2.0
"""Admin panel routes for oMLX server configuration.

This module provides HTTP routes for the admin panel including:
- Login/logout with API key authentication
- Dashboard for server monitoring
- Model settings management (per-model sampling parameters, pinning, default)
- Global settings management
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse

from .auth import (
    require_admin,
)

logger = logging.getLogger(__name__)

PRESET_REMOTE_URL = "https://fusion_mlx.ai/assets/omlx_preset.json"



from .helpers import (
    _get_global_settings,
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
