# SPDX-License-Identifier: Apache-2.0
"""Admin panel routes for Fusion-MLX server configuration.

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

PRESET_REMOTE_URL = "https://bench.dpdns.org/assets/fusionmlx_preset.json"


from ..middleware.auth import check_rate_limit
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
async def login(
    request: LoginRequest,
    response: Response,
    fastapi_request: Request,
):
    """
    Authenticate with API key and create session.

    Requires an API key to be configured on the server. If no API key
    is configured, returns 400 directing the user to set one up first.

    Args:
        request: LoginRequest containing the API key.
        response: FastAPI response object for setting cookies.
        fastapi_request: FastAPI Request for rate-limit bucketing,
            injected by the router.

    Returns:
        JSON response with success status.

    Raises:
        HTTPException: 400 if no API key configured, 401 if invalid.
    """
    # E-32 (#811): /api/login had no rate limit. A brute-force attacker
    # could try API keys unbounded. check_rate_limit buckets by HMAC of
    # the supplied bearer key when present, else client /24 subnet.
    await check_rate_limit(fastapi_request)
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
        key="fusionmlx_admin_session",
        value=token,
        httponly=True,
        secure=True,
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
    from ..middleware.auth import _is_loopback_client
    from ..server import _server_state, _server_state_lock

    # R-14 (#811): only allow from localhost to prevent remote takeover.
    # The prior inline `client.host in ("127.0.0.1", "::1", "localhost")`
    # check ignored forwarded headers, so a loopback reverse proxy
    # (nginx -> 127.0.0.1:11434) let an external attacker hit this endpoint
    # (the proxy connection originates from 127.0.0.1) and set the initial
    # API key before the operator. Use the hardened _is_loopback_client,
    # which rejects any X-Forwarded-For / Forwarded / Via / cf-connecting-ip
    # header so a proxied request is never mistaken for a local one.
    if not _is_loopback_client(fastapi_request):
        raise HTTPException(
            status_code=403,
            detail="Initial API key setup is only allowed from a direct "
            "localhost connection (no forwarded/proxy headers)",
        )

    # AS-7 (#0907 audit): hold the server-state lock across the
    # read-modify-write so two concurrent setup requests cannot both pass
    # the "not yet configured" check and clobber each other's key.
    async with _server_state_lock:
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
            raise HTTPException(status_code=500, detail="Failed to save settings")

        logger.info("API key configured via initial setup")
        # OP-9 (#0907 audit): record the admin write to the audit log.
        from .helpers import _audit_admin_action

        _audit_admin_action(
            "api_key_setup",
            actor=fastapi_request.client.host if fastapi_request.client else "unknown",
            detail={"source": "initial_setup"},
        )

    # Create session token and set cookie (auto-login after setup)
    token = create_session_token()
    response.set_cookie(
        key="fusionmlx_admin_session",
        value=token,
        httponly=True,
        secure=True,
        samesite="lax",
        max_age=86400,  # 24 hours
    )

    return {"success": True, "message": "API key configured successfully"}


@_router.post("/api/logout")
async def logout(request: Request, response: Response):
    from .auth import SESSION_COOKIE_NAME, _active_sessions, _sessions_lock

    token = request.cookies.get(SESSION_COOKIE_NAME)
    if token:
        with _sessions_lock:
            _active_sessions.pop(token, None)
    response.delete_cookie(key="fusionmlx_admin_session")
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
        key="fusionmlx_admin_session",
        value=token,
        httponly=True,
        secure=True,
        samesite="lax",
        max_age=86400,
    )
    return response


@_router.get("/auto-login")
async def auto_login_get(
    fastapi_request: Request,
    redirect: str = "/admin/dashboard",
):
    """
    GET variant of auto-login for browser bookmarks and menubar URLs.

    The API key is read from the ``Authorization: Bearer <key>`` header,
    NOT from the query string. A query-string key leaks into access logs,
    browser history, Referer headers, and proxy logs (E-31 #811). The
    macOS menubar client must set the Authorization header on the GET.
    No header or wrong key → redirect to /admin login form (no error
    surfaced, to avoid leaking whether a key is valid).
    """
    if not redirect.startswith("/admin"):
        raise HTTPException(status_code=400, detail="Invalid redirect path")

    # E-31 (#811): never accept the API key from the query string.
    raw_auth = fastapi_request.headers.get("authorization", "") or ""
    key = ""
    if raw_auth.lower().startswith("bearer "):
        key = raw_auth[7:].strip()

    global_settings = _get_global_settings()
    server_api_key = global_settings.auth.api_key if global_settings else None

    if not key or not server_api_key or not verify_api_key(key, server_api_key):
        return RedirectResponse(url="/admin", status_code=302)

    token = create_session_token()
    response = RedirectResponse(url=redirect, status_code=302)
    response.set_cookie(
        key="fusionmlx_admin_session",
        value=token,
        httponly=True,
        secure=True,
        samesite="lax",
        max_age=86400,
    )
    return response


router = _router
