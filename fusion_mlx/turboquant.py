# SPDX-License-Identifier: Apache-2.0
"""TurboQuant mode resolution and compatibility checks.

Adapted from Rapid-MLX vllm_mlx/turboquant.py (Apache-2.0).
Fusion-mlx branding: env-vars use FUSION_MLX_ prefix.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

TURBOQUANT_MODES: tuple[str, ...] = ("v4", "k8v4")
DEFAULT_TURBOQUANT_MODE = "v4"

SKIP_REASON_SLIDING = "sliding-window"
SKIP_REASON_MLA = "mla"
SKIP_REASON_OTHER = "other"

MODELS_INCOMPATIBLE_WITH_TURBOQUANT: dict[str, str] = {
    r"gemma[-_]?3": SKIP_REASON_SLIDING,
    r"gpt[-_]?oss": SKIP_REASON_SLIDING,
    r"deepseek[-_]?v3": SKIP_REASON_MLA,
    r"deepseek[-_]?v4": SKIP_REASON_MLA,
    r"kimi[-_]?k2\.?5": SKIP_REASON_MLA,
    r"kimi[-_]?k2\.?6": SKIP_REASON_MLA,
}

_COMPILED_SKIP_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(pat, re.IGNORECASE), reason)
    for pat, reason in MODELS_INCOMPATIBLE_WITH_TURBOQUANT.items()
)


def resolve_turboquant_mode_default(args: Any, *, model_name: str) -> str | None:
    raw = getattr(args, "kv_cache_turboquant", None)
    if raw == "none":
        logger.info(
            "TurboQuant off-switch: --kv-cache-turboquant=none — "
            "disabling TurboQuant even though alias %r may carry turboquant_tier.",
            model_name,
        )
        return None
    if raw is not None:
        return raw
    if getattr(args, "kv_cache_quantization", False):
        return None
    try:
        from .model_auto_config import detect_model_config
    except ImportError:
        return None
    cfg = detect_model_config(model_name)
    if cfg is not None and cfg.turboquant_tier == "k8v4_verified":
        logger.info(
            "TurboQuant default: alias %r is turboquant_tier=k8v4_verified "
            "— engine defaults to --kv-cache-turboquant k8v4.",
            model_name,
        )
        return "k8v4"
    return None


def is_incompatible_with_turboquant(
    *,
    model_name: str | None = None,
    hf_path: str | None = None,
    hf_config: dict[str, Any] | None = None,
    alias_metadata: dict[str, Any] | None = None,
) -> tuple[bool, str | None]:
    if alias_metadata:
        if alias_metadata.get("sliding_window"):
            return True, SKIP_REASON_SLIDING
        if alias_metadata.get("is_mla"):
            return True, SKIP_REASON_MLA
    if hf_config:
        sw = hf_config.get("sliding_window")
        if isinstance(sw, int) and sw > 0:
            return True, SKIP_REASON_SLIDING
        model_type = hf_config.get("model_type")
        if isinstance(model_type, str):
            mt = model_type.lower()
            if mt in {"gemma3", "gemma3_text", "gpt_oss"}:
                return True, SKIP_REASON_SLIDING
            if mt in {"deepseek_v3", "deepseek_v4"}:
                return True, SKIP_REASON_MLA
        needle = f"{model_name or ''} {hf_path or ''}".lower()
        if any(
            pat in needle
            for pat in ("deepseek-v3", "deepseek_v3", "deepseek-v4", "deepseek_v4")
        ):
            q_rank = hf_config.get("q_lora_rank")
            kv_rank = hf_config.get("kv_lora_rank")
            if (
                isinstance(q_rank, int)
                and q_rank > 0
                and isinstance(kv_rank, int)
                and kv_rank > 0
            ):
                return True, SKIP_REASON_MLA
    needle = f"{model_name or ''} {hf_path or ''}"
    for pattern, reason in _COMPILED_SKIP_PATTERNS:
        if pattern.search(needle):
            return True, reason
    return False, None
