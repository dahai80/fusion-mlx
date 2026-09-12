# SPDX-License-Identifier: Apache-2.0
"""Post-load model patches for performance optimization and correctness.

Central registry at patches/registry.py is the single source of truth.
Each patch registers its id, target, reason, and apply function.
apply_all_for_model() dispatches to all matching registered patches.

Per-model subpackages (deepseek_v4, glm_moe_dsa, mlx_lm_mtp, mlx_vlm_*mtp,
step3p7) register their patches via registry.register() at import time.
"""

import logging

from .registry import (
    PatchEntry,
    apply_all_for_model,
    list_patches,
    register,
    reset_for_tests,
)

logger = logging.getLogger(__name__)

__all__ = [
    "apply_all_for_model",
    "list_patches",
    "register",
    "reset_for_tests",
    "PatchEntry",
]
