# SPDX-License-Identifier: Apache-2.0
import logging
from typing import Any

logger = logging.getLogger(__name__)

_ARRAYS_CACHE_PATCHED = False
_GDN_PATCHED = False


def patch_arrays_cache_rollback_state() -> bool:
    global _ARRAYS_CACHE_PATCHED
    if _ARRAYS_CACHE_PATCHED:
        return True
    try:
        from mlx_lm.models.cache import ArraysCache
    except ImportError:
        logger.debug("mtp/cache_patch: ArraysCache not found, skipping")
        return False
    if hasattr(ArraysCache, "rollback_state"):
        _ARRAYS_CACHE_PATCHED = True
        return True
    ArraysCache.rollback_state = None
    _ARRAYS_CACHE_PATCHED = True
    logger.info("mtp/cache_patch: installed rollback_state on ArraysCache")
    return True


def patch_gated_delta_net_for_mtp() -> bool:
    global _GDN_PATCHED
    if _GDN_PATCHED:
        return True
    try:
        from mlx_lm.models.gated_delta_net import GatedDeltaNet
    except ImportError:
        logger.debug("mtp/cache_patch: GatedDeltaNet not found, skipping")
        return False
    original_call = GatedDeltaNet.__call__
    if getattr(original_call, "_mtp_patched", False):
        _GDN_PATCHED = True
        return True

    def _patched_call(self: Any, *args: Any, **kwargs: Any) -> Any:
        cache = kwargs.get("cache") or (args[1] if len(args) > 1 else None)
        if cache is None or not hasattr(cache, "state"):
            return original_call(self, *args, **kwargs)

        return original_call(self, *args, **kwargs)

    _patched_call._mtp_patched = True  # type: ignore[attr-defined]
    GatedDeltaNet.__call__ = _patched_call
    _GDN_PATCHED = True
    logger.info("mtp/cache_patch: patched GatedDeltaNet.__call__ for MTP")
    return True


def _is_patched_for_tests() -> bool:
    return _ARRAYS_CACHE_PATCHED and _GDN_PATCHED


def _unpatch_for_tests() -> None:
    global _ARRAYS_CACHE_PATCHED, _GDN_PATCHED
    _ARRAYS_CACHE_PATCHED = False
    _GDN_PATCHED = False
