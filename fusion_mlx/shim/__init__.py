"""fusion-mlx C++ Shim layer (MLX enhancement layer borrowing from llama.cpp).

Implements the v2 technical plan
(/architecture/fusion-mlx-vs-llamacpp-doubao-gemini-v2-0917.md): a Shim /
enhancement extension on top of MLX, NOT a reimplementation of ggml. Every
enhancement has a degrade switch that falls back to native mlx_lm/mlx_vlm.

PR-A scope: C++ extension skeleton + Tier-1 safety base (hardware probe,
memory-pressure sentinel, C-ABI exception envelope, alignas(128) shared
structs). No Metal kernels yet — those land in PR-G (fused RoPE/RMSNorm)
onward.

Master switch: ``FUSION_SHIM_ENABLED`` (default ``0`` = OFF, all paths go
through native MLX). Individual ops get their own switches in later PRs
(``FUSION_SHIM_FUSED_ROPE``, ``FUSION_SHIM_FUSED_RMSNORM``, ...).
"""

from __future__ import annotations

import logging
import os

from .fast import (
    hardware_probe,
    import_error,
    is_native_available,
    last_error_code,
    last_error_message,
    last_memory_pressure,
    last_native_error,
    missing_symbols,
    native_symbols,
    start_memory_sentinel,
    stop_memory_sentinel,
)

logger = logging.getLogger(__name__)

__all__ = [
    "is_shim_enabled",
    "is_native_available",
    "import_error",
    "hardware_probe",
    "start_memory_sentinel",
    "stop_memory_sentinel",
    "last_memory_pressure",
    "last_error_code",
    "last_error_message",
    "last_native_error",
    "native_symbols",
    "missing_symbols",
]


def is_shim_enabled() -> bool:
    # Master switch. Default OFF in the prototype phase: the native shim
    # extension is opt-in so existing inference paths are untouched until
    # an operator explicitly enables a shim op.
    return os.environ.get("FUSION_SHIM_ENABLED", "0") == "1"


def status() -> dict[str, object]:
    # Observability hook for /metrics + admin dashboard.
    return {
        "shim_enabled": is_shim_enabled(),
        "native_available": is_native_available(),
        "import_error": str(import_error()) if import_error() is not None else None,
        "native_symbols": list(native_symbols()),
        "hardware_probe": hardware_probe() if is_native_available() else None,
    }
