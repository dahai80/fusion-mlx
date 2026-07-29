# SPDX-License-Identifier: Apache-2.0
"""MTP dispatch gate helpers for BatchedEngine.

Resolved model_type + sidecar path are fed to ``dispatch_mtp_inject`` after the
LLM weights load. The dispatch table is keyed by TARGET model_type (qwen3_5,
qwen3_5_moe, gemma4_unified); sidecar-only config types are NOT dispatch keys.
This module isolates the synchronous decision surface; the async
``_apply_mtp_dispatch`` wrapper lives alongside BatchedEngine._start_llm.
"""

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_DISPATCH_ATTACHED = "attached"
_DISPATCH_UNRESOLVED = "unresolved"
_DISPATCH_REJECTED = "rejected"
_DISPATCH_NO_INJECT = "no_inject"

_DEFAULT_MTP_DISPATCH_TIMEOUT_SEC = 30.0
_TIMEOUT_ENV = "FUSION_MLX_MTP_DISPATCH_TIMEOUT_SEC"


def _resolve_hf_model_type(model: Any) -> str | None:
    cfg = getattr(model, "config", None)
    if cfg is not None:
        mt = getattr(cfg, "model_type", None)
        if isinstance(mt, str) and mt:
            return mt
        if isinstance(cfg, dict):
            mt = cfg.get("model_type")
            if isinstance(mt, str) and mt:
                return mt
    args = getattr(model, "args", None)
    if args is not None:
        mt = getattr(args, "model_type", None)
        if isinstance(mt, str) and mt:
            return mt
    tc = getattr(model, "text_config", None)
    if isinstance(tc, dict):
        mt = tc.get("model_type")
        if isinstance(mt, str) and mt:
            return mt
    mt = getattr(model, "model_type", None)
    if isinstance(mt, str) and mt:
        return mt
    logger.debug("[MTP-dispatch] could not resolve model_type from model object")
    return None


def _run_dispatch_mtp_inject(
    model: Any,
    model_type: str | None = None,
    mtp_sidecar: Any = None,
    *,
    cli_vetted_model_type: str | None = None,
    allow_random_init: bool = False,
) -> str:
    from ...speculative.mtp import dispatch as _disp

    effective = cli_vetted_model_type or model_type
    if effective is None:
        effective = _resolve_hf_model_type(model)
    if effective is None:
        logger.debug("[MTP-dispatch] no model_type resolved -> UNRESOLVED")
        return _DISPATCH_UNRESOLVED
    table = getattr(_disp, "_MTP_INJECT_DISPATCH", {})
    if effective not in table:
        logger.debug(
            "[MTP-dispatch] model_type=%s not registered -> NO_INJECT", effective
        )
        return _DISPATCH_NO_INJECT
    ok = _disp.dispatch_mtp_inject(
        model,
        effective,
        mtp_sidecar=mtp_sidecar,
        allow_random_init=allow_random_init,
    )
    if ok:
        logger.debug(
            "[MTP-dispatch] attached model_type=%s sidecar=%s",
            effective,
            mtp_sidecar,
        )
        return _DISPATCH_ATTACHED
    logger.debug(
        "[MTP-dispatch] injector refused model_type=%s -> REJECTED", effective
    )
    return _DISPATCH_REJECTED


def _get_mtp_dispatch_timeout_sec() -> float:
    raw = os.environ.get(_TIMEOUT_ENV)
    if raw is None or raw == "":
        return _DEFAULT_MTP_DISPATCH_TIMEOUT_SEC
    try:
        val = float(raw)
    except (TypeError, ValueError):
        logger.warning(
            "[MTP-dispatch] malformed %s=%r, falling back to default %.1fs",
            _TIMEOUT_ENV,
            raw,
            _DEFAULT_MTP_DISPATCH_TIMEOUT_SEC,
        )
        return _DEFAULT_MTP_DISPATCH_TIMEOUT_SEC
    return val


def _log_mtp_dispatch_timeout(
    model_type: str | None,
    mtp_sidecar: Any,
    timeout_sec: float,
) -> None:
    logger.critical(
        "[MTP-dispatch] TIMEOUT after %.1fs model_type=%s sidecar=%s - "
        "dispatch did not complete; refusing to proceed with MTP",
        timeout_sec,
        model_type,
        mtp_sidecar,
    )
