# SPDX-License-Identifier: Apache-2.0
import importlib
import logging
from typing import Any

logger = logging.getLogger(__name__)

_MTP_INJECT_DISPATCH: dict[str, tuple[str, str]] = {
    "qwen3_5": ("fusion_mlx.speculative.mtp.qwen3_5_inject", "inject_mtp_qwen3_5"),
    "qwen3_5_moe": ("fusion_mlx.speculative.mtp.qwen3_5_inject", "inject_mtp_qwen3_5"),
    "gemma4_unified": ("fusion_mlx.speculative.mtp.gemma4_inject", "inject_mtp_gemma4"),
}

_MTP_VALIDATE_DISPATCH: dict[str, tuple[str, str]] = {
    "qwen3_5": ("fusion_mlx.speculative.mtp.qwen3_5_inject", "validate_mtp_qwen3_5"),
    "qwen3_5_moe": (
        "fusion_mlx.speculative.mtp.qwen3_5_inject",
        "validate_mtp_qwen3_5",
    ),
    "gemma4_unified": (
        "fusion_mlx.speculative.mtp.gemma4_inject",
        "validate_mtp_gemma4",
    ),
}


def dispatch_mtp_inject(
    model: Any,
    model_type: str,
    mtp_sidecar: Any = None,
    allow_random_init: bool = False,
) -> bool:
    entry = _MTP_INJECT_DISPATCH.get(model_type)
    if entry is None:
        logger.debug("mtp/dispatch: no inject for model_type=%s", model_type)
        return False
    module_path, func_name = entry
    try:
        mod = importlib.import_module(module_path)
    except ImportError as e:
        logger.warning("mtp/dispatch: cannot import %s: %s", module_path, e)
        return False
    func = getattr(mod, func_name, None)
    if func is None:
        logger.warning("mtp/dispatch: %s has no %s", module_path, func_name)
        return False
    try:
        result = func(
            model, mtp_sidecar=mtp_sidecar, allow_random_init=allow_random_init
        )
        if result:
            logger.info("mtp/dispatch: injected MTP for %s", model_type)
        return bool(result)
    except Exception as e:
        logger.error("mtp/dispatch: inject failed for %s: %s", model_type, e)
        return False


def dispatch_mtp_validate(model: Any, model_type: str) -> bool:
    entry = _MTP_VALIDATE_DISPATCH.get(model_type)
    if entry is None:
        return False
    module_path, func_name = entry
    try:
        mod = importlib.import_module(module_path)
    except ImportError:
        return False
    func = getattr(mod, func_name, None)
    if func is None:
        return False
    try:
        return bool(func(model))
    except Exception as e:
        logger.debug("mtp/dispatch: validate failed for %s: %s", model_type, e)
        return False
