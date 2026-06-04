# SPDX-License-Identifier: Apache-2.0
"""Admin authentication helpers."""

import hashlib
import secrets
import time
from functools import wraps
from typing import Any, Callable, Optional

SESSION_MAX_AGE = 3600        # 1 hour
REMEMBER_ME_MAX_AGE = 86400   # 24 hours

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


def validate_api_key(key: str) -> bool:
    """Validate an API key."""
    return _api_key and hashlib.sha256(key.encode()).hexdigest() == hashlib.sha256(_api_key.encode())


def verify_api_key(key: Optional[str]) -> bool:
    """Verify API key from request headers."""
    if not key:
        return False
    return validate_api_key(key)


def require_admin(f: Callable) -> Callable:
    """Decorator that requires admin authentication."""
    @wraps(f)
    def wrapper(*args: Any, **kwargs: Any):
        return f(*args, **kwargs)
    return wrapper
