# SPDX-License-Identifier: Apache-2.0
#
# Per-request spec-decode method selection. Bridges SpecAutoRouter (a pure
# decision function designed for request-time signals) to the scheduler's
# dispatch point: given which methods are actually loaded at boot + the
# previous request's acceptance rate + prompt length, pick the active method
# for THIS request. Returns None when no method is loaded (spec disabled).
#
# This is a SETUP-time per-request choice, not mid-decode hot-swap: the active
# method is fixed for the lifetime of one request (decided when the request
# first enters pure-decode), so no KV cache rebuild or draft reload happens
# mid-generation. Methods not loaded at boot are never recommended - per-request
# cross-method routing to an UNLOADED method would need lazy draft load (seconds
# of latency, unacceptable); resident multi-method is the operator's choice via
# --enable-dflash --suffix-decoding etc.
from __future__ import annotations

import logging

from .auto_router import (
    METHOD_DFLASH,
    METHOD_DSPARK,
    METHOD_MTP,
    METHOD_NGRAM,
    RouteSignals,
    SpecAutoRouter,
)

logger = logging.getLogger(__name__)

_DEFAULT_ROUTER = SpecAutoRouter()


def loaded_methods(
    *,
    suffix: bool = False,
    dflash: bool = False,
    dspark: bool = False,
    mtp: bool = False,
) -> dict[str, bool]:
    return {
        METHOD_NGRAM: suffix,
        METHOD_DFLASH: dflash,
        METHOD_DSPARK: dspark,
        METHOD_MTP: mtp,
    }


def select_active_method(
    prompt_token_count: int,
    loaded: dict[str, bool],
    *,
    recent_accept_rate: float | None = None,
    current_method: str | None = None,
    has_mtp: bool = False,
    router: SpecAutoRouter | None = None,
) -> str | None:
    active_router = _DEFAULT_ROUTER if router is None else router
    available = frozenset(m for m, ok in loaded.items() if ok)
    if not available:
        logger.debug(
            "spec-route: no spec method loaded -> spec disabled for this request"
        )
        return None
    signals = RouteSignals(
        prompt_token_count=prompt_token_count,
        has_mtp=has_mtp and METHOD_MTP in available,
        recent_accept_rate=recent_accept_rate,
        current_method=current_method if current_method in available else None,
        available=available,
    )
    method = active_router.decide(signals)
    # Degenerate fallback (e.g. the only loaded method was abandoned and
    # excluded) can yield a method not in available - disable spec rather
    # than dispatch an unloaded method.
    if method not in available:
        logger.info(
            "spec-route: decided %s not in loaded %s -> spec disabled",
            method,
            sorted(available),
        )
        return None
    logger.info(
        "spec-route: prompt_tokens=%d loaded=%s recent_accept=%s current=%s -> %s",
        prompt_token_count,
        sorted(available),
        f"{recent_accept_rate:.2f}" if recent_accept_rate is not None else "None",
        current_method,
        method,
    )
    return method
