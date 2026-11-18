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

#1647 follow-up: mlx_vlm.trainer.utils.grad_checkpoint() has the same
re-wrap-per-call behavior, so the VLM SFT path is covered too. mlx_vlm
is optional (text-only installs never import it) — pass include_vlm=True
from the VLM training body to patch its entry point on first use.
"""

from __future__ import annotations

import importlib
import logging
import sys
import weakref

logger = logging.getLogger(__name__)

_GUARD_ATTR = "_fusion_mlx_gc_wrapped"
_WRAPPER_ATTR = "_fusion_mlx_gc_guard"

# Weak registry of classes this guard has marked. Classes reject new
# attributes in exotic metaclass setups, and tests need to reset the
# marker state between cases — the registry covers both.
_MARKED_CLASSES = weakref.WeakSet()

# (module, attribute) pairs that own a grad_checkpoint entry point.
# mlx_lm is required; mlx_vlm is only patched on explicit request.
_MLM_TARGET = ("mlx_lm.tuner.trainer", "grad_checkpoint")
_VLM_TARGET = ("mlx_vlm.trainer.utils", "grad_checkpoint")

# Modules that bind grad_checkpoint by name at import time
# (`from .utils import grad_checkpoint`). If they are already in
# sys.modules the bound name must be re-patched in place; if not, they
# will re-bind from the already-patched utils module at import time.
_VLM_REFERENCING_MODULES = (
    "mlx_vlm.trainer.sft_trainer",
    "mlx_vlm.trainer.orpo_trainer",
)


def _make_guarded(upstream, source: str):
    def guarded_grad_checkpoint(layer):
        cls = type(layer)
        if getattr(cls, _GUARD_ATTR, False) or cls in _MARKED_CLASSES:
            logger.debug(
                "gc_guard: %s already gradient-checkpointed, skipping re-wrap",
                cls.__name__,
            )
            return
        upstream(layer)
        _MARKED_CLASSES.add(cls)
        try:
            setattr(cls, _GUARD_ATTR, True)
        except (AttributeError, TypeError):
            # Class disallows new attributes; the weak registry entry
            # above still guards repeat calls within this process.
            logger.debug("gc_guard: could not set marker on %r", cls)
        logger.debug(
            "gc_guard: gradient checkpointing wrapped class %s (first time, via %s)",
            type(layer).__name__,
            source,
        )

    guarded_grad_checkpoint._fusion_mlx_gc_guard = True
    return guarded_grad_checkpoint


def _patch_site(mod, source: str, attr: str) -> int:
    current = getattr(mod, attr, None)
    if current is None:
        logger.warning("gc_guard: %s has no %s entry point", source, attr)
        return 0
    if getattr(current, _WRAPPER_ATTR, False):
        return 0
    setattr(mod, attr, _make_guarded(current, source))
    return 1


def ensure_grad_checkpoint_guard(include_vlm: bool = False) -> None:
    # Idempotent: each call patches only sites not already guarded, so
    # repeat calls are cheap no-ops and a later include_vlm=True call
    # still installs the VLM side after a text-only first call.
    from mlx_lm.tuner import trainer as tuner_trainer

    patched = _patch_site(tuner_trainer, _MLM_TARGET[0], _MLM_TARGET[1])

    if include_vlm:
        try:
            vlm_utils = importlib.import_module(_VLM_TARGET[0])
        except ImportError:
            logger.warning(
                "gc_guard: include_vlm=True but %s is not importable; "
                "VLM gradient checkpointing stays unguarded",
                _VLM_TARGET[0],
            )
        else:
            patched += _patch_site(vlm_utils, _VLM_TARGET[0], _VLM_TARGET[1])
            for ref_name in _VLM_REFERENCING_MODULES:
                ref_mod = sys.modules.get(ref_name)
                if ref_mod is not None:
                    patched += _patch_site(ref_mod, ref_name, "grad_checkpoint")

    if patched:
        logger.info(
            "gc_guard: installed idempotent grad_checkpoint guard (%d site(s))",
            patched,
        )
