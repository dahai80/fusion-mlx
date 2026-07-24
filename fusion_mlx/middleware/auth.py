# SPDX-License-Identifier: Apache-2.0
"""Authentication and rate limiting middleware for fusion-mlx."""

import hashlib
import hmac
import ipaddress
import logging
import secrets
import threading
import time
from collections import defaultdict, deque

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

logger = logging.getLogger(__name__)

security = HTTPBearer(auto_error=False)

_RATE_LIMIT_HMAC_KEY = secrets.token_bytes(32)


class RateLimiter:
    """In-memory sliding-window rate limiter with amortized O(1) cleanup."""

    _CLEANUP_DICT_THRESHOLD = 100

    def __init__(self, requests_per_minute: int = 60, enabled: bool = False):
        self.requests_per_minute = requests_per_minute
        self.enabled = enabled
        self.window_size = 60.0
        self._requests: dict[str, deque[float]] = defaultdict(deque)
        self._last_seen: dict[str, float] = {}
        self._last_cleanup_mono: float = float("-inf")
        self._lock = threading.Lock()

    def _maybe_cleanup(self, window_start: float, client_id: str) -> None:
        if len(self._requests) <= self._CLEANUP_DICT_THRESHOLD:
            return
        now_mono = time.monotonic()
        if now_mono - self._last_cleanup_mono < self.window_size:
            return
        self._last_cleanup_mono = now_mono
        for k in list(self._requests.keys()):
            if k == client_id:
                continue
            last = self._last_seen.get(k)
            if last is None or last <= window_start:
                self._requests.pop(k, None)
                self._last_seen.pop(k, None)

    def is_allowed(self, client_id: str) -> tuple[bool, int]:
        if not self.enabled:
            return True, 0
        current_time = time.time()
        window_start = current_time - self.window_size
        with self._lock:
            self._maybe_cleanup(window_start, client_id)
            # #57: deque popleft until in-window -> amortized O(1) per hit
            # instead of O(n) list rebuild + min() scan under the lock.
            # Timestamps are append-right and monotonic, so deque[0] is oldest.
            bucket = self._requests[client_id]
            while bucket and bucket[0] <= window_start:
                bucket.popleft()
            if len(bucket) >= self.requests_per_minute:
                oldest = bucket[0]
                retry_after = int(oldest + self.window_size - current_time) + 1
                self._last_seen[client_id] = current_time
                return False, max(1, retry_after)
            bucket.append(current_time)
            self._last_seen[client_id] = current_time
            return True, 0


rate_limiter = RateLimiter(requests_per_minute=60, enabled=True)


def configure_rate_limiter(
    requests_per_minute: int,
    *,
    enabled: bool = True,
) -> RateLimiter:
    logger.info(
        "Configuring rate limiter: rpm=%d enabled=%s", requests_per_minute, enabled
    )
    with rate_limiter._lock:
        rate_limiter.requests_per_minute = requests_per_minute
        rate_limiter.enabled = enabled
        rate_limiter._requests.clear()
        rate_limiter._last_seen.clear()
        rate_limiter._last_cleanup_mono = float("-inf")
    return rate_limiter


def _extract_bearer_token(authorization: str | None) -> str | None:
    if not authorization:
        return None
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        return None
    return token


def _bucket_id(raw: str) -> str:
    return hmac.new(_RATE_LIMIT_HMAC_KEY, raw.encode(), hashlib.sha256).hexdigest()[:16]


def _subnet_bucket(host: str) -> str:
    try:
        addr = ipaddress.ip_address(host)
        if isinstance(addr, ipaddress.IPv4Address):
            network = ipaddress.ip_network(f"{addr}/24", strict=False)
        else:
            network = ipaddress.ip_network(f"{addr}/64", strict=False)
        return str(network.network_address)
    except ValueError:
        return host


def _rate_limit_client_id(request: Request) -> str:
    authorization = request.headers.get("Authorization")
    if authorization:
        bearer_key = _extract_bearer_token(authorization)
        raw = bearer_key or authorization
        return _bucket_id(raw)
    if request.client and request.client.host:
        return _subnet_bucket(request.client.host)
    return "unknown"


def _anthropic_rate_limit_client_id(request: Request) -> str:
    bearer_key = _extract_bearer_token(request.headers.get("Authorization"))
    if bearer_key:
        return _bucket_id(bearer_key)
    x_api_key = request.headers.get("x-api-key")
    if x_api_key:
        return _bucket_id(x_api_key)
    if request.client and request.client.host:
        return _subnet_bucket(request.client.host)
    return "unknown"


async def check_rate_limit(request: Request):
    client_id = _rate_limit_client_id(request)
    allowed, retry_after = rate_limiter.is_allowed(client_id)
    if not allowed:
        logger.warning(
            "Rate limit exceeded for client=%s retry_after=%d",
            client_id[:8],
            retry_after,
        )
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Retry after {retry_after} seconds.",
            headers={"Retry-After": str(retry_after)},
        )


async def check_rate_limit_or_x_api_key(request: Request):
    client_id = _anthropic_rate_limit_client_id(request)
    allowed, retry_after = rate_limiter.is_allowed(client_id)
    if not allowed:
        logger.warning(
            "Rate limit exceeded for client=%s retry_after=%d",
            client_id[:8],
            retry_after,
        )
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Retry after {retry_after} seconds.",
            headers={"Retry-After": str(retry_after)},
        )


def _get_configured_api_key() -> str | None:
    try:
        from ..admin.helpers import _get_global_settings

        settings = _get_global_settings()
        if settings is not None:
            auth = getattr(settings, "auth", None)
            if auth is not None:
                key = getattr(auth, "api_key", None)
                if key:
                    return key
    except Exception:
        logger.debug("Failed to read configured API key from global settings", exc_info=True)
    try:
        from ..config import get_config

        cfg = get_config()
        key = getattr(cfg, "api_key", None)
        if key:
            return key
    except Exception:
        logger.debug("Failed to read configured API key from config", exc_info=True)
    return None


def _verify_api_key_values(*api_keys: str | None) -> bool:
    configured_key = _get_configured_api_key()
    if configured_key is None:
        logger.debug("No API key configured — anonymous access allowed (dev mode)")
        return True
    provided_keys = [api_key for api_key in api_keys if api_key]
    if not provided_keys:
        raise HTTPException(status_code=401, detail="API key required")
    if not all(
        secrets.compare_digest(api_key, configured_key) for api_key in provided_keys
    ):
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True


async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    bearer_key = credentials.credentials if credentials is not None else None
    return _verify_api_key_values(bearer_key)


async def verify_api_key_or_x_api_key(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    bearer_key = credentials.credentials if credentials is not None else None
    return _verify_api_key_values(bearer_key, request.headers.get("x-api-key"))


_LOOPBACK_LITERALS = frozenset({"127.0.0.1", "::1", "localhost"})


def _is_loopback_client(request: Request) -> bool:
    client = request.client
    if client is None or not client.host:
        return False
    forwarded_headers = (
        "x-forwarded-for",
        "x-forwarded-host",
        "x-forwarded-proto",
        "forwarded",
        "via",
        "cf-connecting-ip",
        "true-client-ip",
    )
    for h in forwarded_headers:
        if request.headers.get(h):
            return False
    host = client.host
    if host in _LOOPBACK_LITERALS:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False
