# SPDX-License-Identifier: Apache-2.0
"""Admin authentication helpers."""

import hashlib
import logging
import secrets
import time
from typing import Any

from fastapi import HTTPException, Request

logger = logging.getLogger(__name__)

SESSION_COOKIE_NAME = "fusionmlx_admin_session"
SESSION_MAX_AGE = 3600  # 1 hour
REMEMBER_ME_MAX_AGE = 86400  # 24 hours

_active_sessions: dict = {}
_api_key: str = ""
_global_settings_getter: Any = None


def set_api_key(key: str):
    """Set the admin API key."""
    global _api_key
    _api_key = key


def set_global_settings_getter(getter):
    """Set the global settings getter for auth checks."""
    global _global_settings_getter
    _global_settings_getter = getter


def create_session_token(user: str = "admin", remember: bool = False) -> str:
    """Create a new session token."""
    token = secrets.token_hex(32)
    _active_sessions[token] = {
        "user": user,
        "expires": time.time() + (REMEMBER_ME_MAX_AGE if remember else SESSION_MAX_AGE),
    }
    return token


def verify_session_from_request(request: Request) -> bool:
    """Check if a request has a valid session cookie."""
    token = request.cookies.get(SESSION_COOKIE_NAME)
    if not token:
        return False
    if token not in _active_sessions:
        return False
    if time.time() > _active_sessions[token]["expires"]:
        del _active_sessions[token]
        return False
    return True


def verify_session(token: str) -> bool:
    """Check if a session token is valid."""
    if token not in _active_sessions:
        return False
    if time.time() > _active_sessions[token]["expires"]:
        del _active_sessions[token]
        return False
    return True


def validate_api_key(key: str) -> tuple[bool, str]:
    """Validate API key format. Returns (is_valid, error_message)."""
    if len(key) < 4:
        return (False, "API key must be at least 4 characters")
    if not key.isascii():
        return (False, "API key must contain only ASCII characters")
    return (True, "")


def verify_api_key(input_key: str | None, expected_key: str) -> bool:
    """Verify two API keys match using constant-time comparison."""
    if not input_key or not expected_key:
        return False
    a = hashlib.sha256(input_key.encode()).hexdigest()
    b = hashlib.sha256(expected_key.encode()).hexdigest()
    return secrets.compare_digest(a, b)


def _get_settings_api_key(gs) -> str:
    """Extract API key from settings object (handles both flat and nested auth)."""
    if gs is None:
        return ""
    if hasattr(gs, "auth") and gs.auth:
        return getattr(gs.auth, "api_key", "") or ""
    return getattr(gs, "api_key", "") or ""


def _get_settings_sub_keys(gs) -> list:
    """Extract sub keys from settings object (handles both flat and nested auth)."""
    if gs is None:
        return []
    if hasattr(gs, "auth") and gs.auth:
        return getattr(gs.auth, "sub_keys", []) or []
    return getattr(gs, "sub_keys", []) or []


def _is_skip_api_key_verification(gs) -> bool:
    """Check if API key verification is skipped."""
    if gs is None:
        return False
    if hasattr(gs, "auth") and gs.auth:
        return getattr(gs.auth, "skip_api_key_verification", False)
    gs_dict = getattr(gs, "global_settings", {}) or {}
    return gs_dict.get("skip_api_key_verification", False)


async def require_admin(request: Request) -> bool:
    """FastAPI dependency for admin authentication.

    Checks session cookie or Bearer token against configured API key.
    When skip_api_key_verification is enabled, always returns True.
    """
    from .helpers import _get_global_settings as _helpers_get_gs

    gs = _helpers_get_gs() if _helpers_get_gs else None

    if _is_skip_api_key_verification(gs):
        return True

    if verify_session_from_request(request):
        return True

    auth_header = request.headers.get("authorization", "")
    if auth_header.startswith("Bearer "):
        bearer_key = auth_header[7:]
        if _api_key and verify_api_key(bearer_key, _api_key):
            return True
        server_key = _get_settings_api_key(gs)
        if verify_api_key(bearer_key, server_key):
            return True
        for sk in _get_settings_sub_keys(gs):
            sk_key = getattr(sk, "key", "") if hasattr(sk, "key") else ""
            if verify_api_key(bearer_key, sk_key):
                return True

    # Allow browser-based access via ?key= or ?api_key= query param
    query_key = request.query_params.get("key") or request.query_params.get("api_key")
    if query_key:
        if _api_key and verify_api_key(query_key, _api_key):
            return True
        server_key = _get_settings_api_key(gs)
        if verify_api_key(query_key, server_key):
            return True

    raise HTTPException(
        status_code=401,
        detail="Admin authentication required",
        headers={"WWW-Authenticate": "Bearer"},
    )


def extract_session_token(request) -> str | None:
    """Extract session token from request cookies."""
    return request.cookies.get(SESSION_COOKIE_NAME)
