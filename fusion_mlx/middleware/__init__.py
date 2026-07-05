# SPDX-License-Identifier: Apache-2.0
"""Fusion-MLX server middleware — auth, rate limiting, body guards, error handlers."""

from .auth import (
    RateLimiter,
    check_rate_limit,
    check_rate_limit_or_x_api_key,
    configure_rate_limiter,
    rate_limiter,
    verify_api_key,
    verify_api_key_or_x_api_key,
)
from .body_depth import (
    RequestBodyDepthMiddleware,
    install_request_body_depth_middleware,
)
from .body_size import RequestBodyLimitMiddleware, install_request_body_limit_middleware
from .exception_handlers import install_exception_handlers

__all__ = [
    "RateLimiter",
    "RequestBodyDepthMiddleware",
    "RequestBodyLimitMiddleware",
    "check_rate_limit",
    "check_rate_limit_or_x_api_key",
    "configure_rate_limiter",
    "install_exception_handlers",
    "install_request_body_depth_middleware",
    "install_request_body_limit_middleware",
    "rate_limiter",
    "verify_api_key",
    "verify_api_key_or_x_api_key",
]
