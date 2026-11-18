# SPDX-License-Identifier: Apache-2.0
"""Idempotency guard for mlx-lm gradient checkpointing.

Upstream mlx_lm.tuner.trainer.grad_checkpoint() patches
type(layer).__call__ with an mx.checkpoint wrapper on every call.
Inside a long-lived serve process this nests on the second and later
training jobs (wrapper wraps wrapper): each extra nesting level doubles
backward-pass recompute cost without further activation-memory savings.

The guard marks already-wrapped classes so repeat calls are no-ops.
Works for both the SFT path (mlx-lm calls grad_checkpoint internally
per train()) and the DPO/ORPO path (we call it directly per job).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_GUARD_ATTR = "_fusion_mlx_gc_wrapped"
_installed = False


def ensure_grad_checkpoint_guard() -> None:
    global _installed
    if _installed:
        return
    from mlx_lm.tuner import trainer as tuner_trainer

    original = tuner_trainer.grad_checkpoint

    def guarded_grad_checkpoint(layer):
        if getattr(type(layer), _GUARD_ATTR, False):
            return
        original(layer)
        setattr(type(layer), _GUARD_ATTR, True)
        logger.debug(
            "gradient checkpointing wrapped class %s (first time)",
            type(layer).__name__,
        )

    tuner_trainer.grad_checkpoint = guarded_grad_checkpoint
    _installed = True
    logger.info("gradient checkpointing idempotency guard installed")
