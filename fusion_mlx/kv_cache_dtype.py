# SPDX-License-Identifier: Apache-2.0
"""KV cache dtype utilities and safe-list for auto-downgrade."""

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

_BF16_ONLY_FAMILIES: frozenset[str] = frozenset()
_REASONING_INT8_FAMILIES: frozenset[str] = frozenset()
REASONING_KV_CACHE_DTYPE = "int8"


@dataclass
class KVCacheDtypeDecision:
    dtype: str
    original_request: str
    reason: str = ""


def resolve_kv_cache_dtype(
    requested_dtype: str,
    *,
    reasoning: bool = False,
    model_name: str = "",
    model_family: str = "",
    hf_path: str | None = None,
    hf_config: Any = None,
    alias_metadata: dict | None = None,
) -> KVCacheDtypeDecision:
    dtype = requested_dtype
    reason = "user-specified"
    if reasoning and model_family not in _BF16_ONLY_FAMILIES:
        dtype = "int8"
        reason = "reasoning-mode override"
    if model_family in _BF16_ONLY_FAMILIES and dtype in ("int4", "int8"):
        logger.info(
            "Auto-downgrading KV cache dtype from %s to bf16 for %s",
            dtype,
            model_family,
        )
        dtype = "bf16"
        reason = "auto-downgrade for model family"
    return KVCacheDtypeDecision(
        dtype=dtype,
        original_request=requested_dtype,
        reason=reason,
    )


def dtype_to_quantization_bits(dtype: str) -> tuple[bool, int | None]:
    mapping = {
        "bf16": (False, None),
        "int8": (True, 8),
        "int4": (True, 4),
    }
    return mapping.get(dtype, (False, None))


def log_kv_cache_decision(
    decision: KVCacheDtypeDecision, *, model_name: str = ""
) -> None:
    logger.info(
        "KV cache dtype: %s (requested=%s, reason=%s, model=%s)",
        decision.dtype,
        decision.original_request,
        decision.reason,
        model_name,
    )
