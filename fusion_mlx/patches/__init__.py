# SPDX-License-Identifier: Apache-2.0
"""Post-load model patches for performance optimization and correctness.

Central registry at patches/registry.py is the single source of truth.
Each patch registers its id, target, reason, and apply function.
apply_all_for_model() dispatches to all matching registered patches.

Per-model subpackages (deepseek_v4, glm_moe_dsa, mlx_lm_mtp, mlx_vlm_*mtp,
step3p7) register their patches via registry.register() at import time.
"""

import importlib
import logging

from .registry import (
    PatchEntry,
    apply_all_for_model,
    list_patches,
    register,
    reset_for_tests,
)

logger = logging.getLogger(__name__)

_ALL_REGISTERED = False


def ensure_all_registered() -> None:
    """Import every patch subpackage so each self-registers into the registry.

    Called by doctor (list_patches) and maybe_apply_pre_load_patches so the
    registry is a complete audit surface — not just patches that matched the
    current model. Idempotent: the _ALL_REGISTERED guard skips re-import.
    """
    global _ALL_REGISTERED
    if _ALL_REGISTERED:
        return
    _ALL_REGISTERED = True
    for name in (
        "deepseek_v4",
        "glm_moe_dsa",
        "step3p7",
        "mlx_lm_mtp",
        "llama4_attention",
        "turboquant_attention",
        "mlx_vlm_diffusion",
        "qwen3_6_nested_visual",
        "minimax_m3_sparse_attention",
    ):
        try:
            importlib.import_module(f"fusion_mlx.patches.{name}")
        except Exception as e:
            logger.debug("Patch subpackage %s import skipped: %s", name, e)


__all__ = [
    "apply_all_for_model",
    "ensure_all_registered",
    "list_patches",
    "register",
    "reset_for_tests",
    "PatchEntry",
]
