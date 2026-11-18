# SPDX-License-Identifier: Apache-2.0
# Idempotency guard for mlx-lm / mlx-vlm grad_checkpoint class patching (#1647).
#
# mlx_lm.tuner.trainer.grad_checkpoint() patches type(layer).__call__ with an
# mx.checkpoint wrapper on EVERY call, capturing the current __call__ as the
# wrapped fn. In a long-lived serve process (thread-mode training, or several
# sequential jobs that reuse the imported trainer classes) each repeat call
# nests another wrapper: no additional activation-memory savings (the class
# is already checkpointed), but each extra nesting level doubles the backward
# recompute cost (2 layers -> recompute twice, 3 layers -> 4x, ...).
#
# This module installs an idempotent wrapper over the upstream entry points
# that marks already-patched classes so repeat calls are no-ops. install()
# is itself idempotent and safe to call before every training job.

import importlib
import logging
import sys
import weakref

logger = logging.getLogger(__name__)

_MARKER = "_fusion_gc_wrapped"
_WRAPPED_CLASSES = weakref.WeakSet()

# (module, attribute) pairs that own a grad_checkpoint entry point. mlx_vlm
# is optional at guard-install time (text-only installs never import it).
_PATCH_TARGETS = (
    ("mlx_lm.tuner.trainer", "grad_checkpoint"),
    ("mlx_vlm.trainer.utils", "grad_checkpoint"),
)

# Modules that bind grad_checkpoint by name at import time
# (`from .utils import grad_checkpoint`). If they are already in
# sys.modules the name must be re-patched in place; if not, they will
# re-bind from the already-patched utils module at import time.
_REFERENCING_MODULES = (
    "mlx_vlm.trainer.sft_trainer",
    "mlx_vlm.trainer.orpo_trainer",
)


def _is_wrapped(cls) -> bool:
    return getattr(cls, _MARKER, False) or cls in _WRAPPED_CLASSES


def _mark_wrapped(cls) -> None:
    _WRAPPED_CLASSES.add(cls)
    try:
        setattr(cls, _MARKER, True)
    except Exception:
        # Class disallows new attributes; the WeakSet entry above still
        # guards repeat calls within this process.
        logger.debug("gc_guard: could not set marker on %r", cls, exc_info=True)


def _make_guarded(upstream, source: str):
    def guarded_grad_checkpoint(layer):
        cls = type(layer)
        if _is_wrapped(cls):
            logger.debug(
                "gc_guard: %s already gradient-checkpointed, skipping re-wrap",
                cls.__name__,
            )
            return
        upstream(layer)
        _mark_wrapped(cls)
        logger.info(
            "gc_guard: gradient checkpointing wrapped %s (via %s, first wrap)",
            cls.__name__,
            source,
        )

    guarded_grad_checkpoint._fusion_gc_guard = True
    return guarded_grad_checkpoint


def _patch_module(mod, mod_name: str, attr: str) -> int:
    current = getattr(mod, attr, None)
    if current is None:
        return 0
    if getattr(current, "_fusion_gc_guard", False):
        return 0
    setattr(mod, attr, _make_guarded(current, mod_name))
    return 1


def install() -> int:
    # Patch the upstream grad_checkpoint entry points with the idempotent
    # guard. Returns the number of sites patched in this call (0 when all
    # sites were already guarded). Call before every training job; repeat
    # calls are cheap no-ops.
    patched = 0
    for mod_name, attr in _PATCH_TARGETS:
        try:
            mod = importlib.import_module(mod_name)
        except ImportError:
            continue
        patched += _patch_module(mod, mod_name, attr)
    for ref_name in _REFERENCING_MODULES:
        mod = sys.modules.get(ref_name)
        if mod is not None:
            patched += _patch_module(mod, ref_name, "grad_checkpoint")
    if patched:
        logger.info(
            "gc_guard: installed idempotent grad_checkpoint guard (%d site(s))",
            patched,
        )
    return patched
