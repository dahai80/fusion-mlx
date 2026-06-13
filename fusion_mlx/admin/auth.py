# SPDX-License-Identifier: Apache-2.0
"""Admin authentication helpers."""

import hashlib
import secrets
import time
from collections.abc import Callable
from functools import wraps
from typing import Any

SESSION_MAX_AGE = 3600          # 1 hour
REMEMBER_ME_MAX_AGE = 86400     # 24 hours

_active_sessions: dict = {}
_api_key: str = ""


def set_api_key(key: str):
    """Set the admin API key."""
    global _api_key
    _api_key = key


def create_session_token(user: str = "admin", remember: bool = False) -> str:
    """Create a new session token."""
    token = secrets.token_hex(32)
    _active_sessions[token] = {
        "user": user,
        "expires": time.time() + (REMEMBER_ME_MAX_AGE if remember else SESSION_MAX_AGE),
    }
    return token


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
    if len(key) < 8:
        return (False, "Must be at least 8 characters")
    if not any(c.isupper() for c in key):
        return (False, "Must contain an uppercase letter")
    if not any(c.islower() for c in key):
        return (False, "Must contain a lowercase letter")
    if not any(c.isdigit() for c in key):
        return (False, "Must contain a digit")
    return (True, "")


def verify_api_key(input_key: str | None, expected_key: str) -> bool:
    """Verify two API keys match."""
    if not input_key or not expected_key:
        return False
    a = hashlib.sha256(input_key.encode()).hexdigest()
    b = hashlib.sha256(expected_key.encode()).hexdigest()
    return a == b


def require_admin(f: Callable) -> Callable:
    """Decorator that requires admin authentication via session cookie or Bearer token."""
    @wraps(f)
    def wrapper(request: Any, *args: Any, **kwargs: Any):
        from fastapi import HTTPException
        token = extract_session_token(request)
        if token and verify_session(token):
            return f(request, *args, **kwargs)
        auth = request.headers.get("authorization", "")
        if auth.startswith("Bearer "):
            bearer_key = auth[7:]
            if _api_key:
                a = hashlib.sha256(bearer_key.encode()).hexdigest()
                b = hashlib.sha256(_api_key.encode()).hexdigest()
                if a == b:
                    return f(request, *args, **kwargs)
        raise HTTPException(status_code=401, detail="Unauthorized")
    return wrapper


def extract_session_token(request) -> str | None:
    """Extract session token from request cookies."""
    return request.cookies.get("session_token")
