# SPDX-License-Identifier: Apache-2.0
"""Admin panel routes for oMLX server configuration.

This module provides HTTP routes for the admin panel including:
- Login/logout with API key authentication
- Dashboard for server monitoring
- Model settings management (per-model sampling parameters, pinning, default)
- Global settings management
"""

import logging

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import RedirectResponse

from .auth import (
    REMEMBER_ME_MAX_AGE,
    SESSION_MAX_AGE,
    create_session_token,
    validate_api_key,
    verify_api_key,
)

logger = logging.getLogger(__name__)

PRESET_REMOTE_URL = "http://bench.dpdns.org/assets/omlx_preset.json"


from .helpers import (
    _get_global_settings,
)
from .models import (
    LoginRequest,
    SetupApiKeyRequest,
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
async def setup_api_key(
    request: SetupApiKeyRequest, response: Response, fastapi_request: Request
):
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

    # Only allow from localhost to prevent remote takeover
    client_host = fastapi_request.client.host if fastapi_request.client else ""
    if client_host not in ("127.0.0.1", "::1", "localhost"):
        raise HTTPException(
            status_code=403,
            detail="Initial API key setup is only allowed from localhost",
        )

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
    _server_state["api_key"] = request.api_key

    # Persist to file
    try:
        global_settings.save()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save settings: {e}")

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


@_router.post("/auto-login")
async def auto_login(fastapi_request: Request, redirect: str = "/admin/dashboard"):
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

    # Read API key from POST body instead of query param to avoid leaking in logs/history
    try:
        body = await fastapi_request.json()
        key = body.get("key", "")
    except Exception:
        key = ""

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


@_router.get("/auto-login")
async def auto_login_get(
    fastapi_request: Request,
    redirect: str = "/admin/dashboard",
    key: str = "",
):
    """
    GET variant of auto-login for browser bookmarks and menubar URLs.

    The macOS menubar constructs a URL like
      /admin/auto-login?redirect=/admin/dashboard&key=<api_key>
    which the browser opens as GET. This handler reads the key from the
    query string, validates it, sets the session cookie, and redirects.
    """
    if not redirect.startswith("/admin"):
        raise HTTPException(status_code=400, detail="Invalid redirect path")

    global_settings = _get_global_settings()
    server_api_key = global_settings.auth.api_key if global_settings else None

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
