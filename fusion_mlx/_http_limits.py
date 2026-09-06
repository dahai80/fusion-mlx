import logging
import os

import httpx

logger = logging.getLogger(__name__)

# RC-5 (#811 audit 0906): all outbound httpx clients previously used the
# default httpx.Limits (max_connections=100, max_keepalive=20) but created a
# NEW client per call with no shared cap, so concurrent request bursts could
# open hundreds of sockets. This central helper gives every call site a
# bounded, env-overridable limit so fan-out is capped at the process level.
#
# Env:
#   FUSION_HTTP_MAX_CONNECTIONS         (default 64)
#   FUSION_HTTP_MAX_KEEPALIVE           (default 16)
#   FUSION_HTTP_KEEPALIVE_EXPIRY        (default 5.0 seconds)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        value = int(raw)
        if value <= 0:
            logger.warning(
                "RC-5: env %s=%r invalid (must be >0), falling back to %d",
                name,
                raw,
                default,
            )
            return default
        return value
    except ValueError:
        logger.warning(
            "RC-5: env %s=%r not an int, falling back to %d", name, raw, default
        )
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        value = float(raw)
        if value <= 0:
            logger.warning(
                "RC-5: env %s=%r invalid (must be >0), falling back to %s",
                name,
                raw,
                default,
            )
            return default
    except ValueError:
        logger.warning(
            "RC-5: env %s=%r not a float, falling back to %s", name, raw, default
        )
        return default
    return value


def bounded_limits() -> httpx.Limits:
    max_connections = _env_int("FUSION_HTTP_MAX_CONNECTIONS", 64)
    max_keepalive = _env_int("FUSION_HTTP_MAX_KEEPALIVE", 16)
    keepalive_expiry = _env_float("FUSION_HTTP_KEEPALIVE_EXPIRY", 5.0)
    return httpx.Limits(
        max_connections=max_connections,
        max_keepalive_connections=max_keepalive,
        keepalive_expiry=keepalive_expiry,
    )
