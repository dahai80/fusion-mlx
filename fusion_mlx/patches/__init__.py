# SPDX-License-Identifier: Apache-2.0
"""Post-load model patches for performance optimization and correctness.

A-P3-4: This package is intentionally minimal. Patches live in per-model
subpackages (deepseek_v4, glm_moe_dsa, mlx_lm_mtp, mlx_vlm_*mtp, step3p7)
and are applied from model-specific load code with per-class idempotency
guards. There is no central registry — auditing coverage requires grepping
call sites. apply_all_for_model() below provides a no-op entry point for
forward compatibility; model load code that wants to register its patches
centrally can populate this in the future.
"""

import logging

logger = logging.getLogger(__name__)


def apply_all_for_model(model_type: str, config=None) -> None:
    """No-op entry point for centralised patch registration.

    Per-model patches are currently applied directly from load code in
    engine_pool.py / BatchedEngine.start(). This function exists so that
    future patch registration can be centralised here without changing
    call-site signatures. Calling it today logs the model type at DEBUG
    and returns.
    """
    logger.debug("apply_all_for_model: %s (no central patches registered)", model_type)
