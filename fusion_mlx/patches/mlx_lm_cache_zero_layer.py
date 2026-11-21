# SPDX-License-Identifier: Apache-2.0
#
# #1671: mlx-lm (0d4d3d04f) ArraysCache.empty() indexes self.cache[0]
# without a zero-layer guard, so any ArraysCache constructed with
# size=0 (e.g. a cache whose state was fully trimmed/detached) crashes
# with IndexError inside CacheList-style merge() -> all(c.empty()).
# The crash surfaced as engine-loop death + restart loops on
# /v1/chat/completions (issue #1671: SizedArraysCache.merge ->
# mlx_lm generate._merge_caches).
#
# Fix: patch ArraysCache.empty to treat a zero-layer cache as empty
# (upstream merge() then early-returns a valid cls(0) cache) and patch
# ArraysCache.merge to reject a mixed zero-layer/stateful batch with a
# clear ValueError instead of silently dropping state via range(0).
#
# Mixed-batch note: silently returning an empty cache for a batch that
# still holds stateful rows would corrupt the batch (dropped KV), so a
# loud error is the correct behavior; callers must filter such caches
# before merge.

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_INSTALLED = False


def install_mlx_lm_cache_zero_layer_patch() -> bool:
    global _INSTALLED
    if _INSTALLED:
        return True
    try:
        from mlx_lm.models import cache as lm_cache
    except ImportError:
        return False

    orig_empty = lm_cache.ArraysCache.empty
    orig_merge = lm_cache.ArraysCache.merge

    def _empty(self):
        return len(self.cache) == 0 or orig_empty(self)

    def _merge(cls, caches):
        sizes = [len(c.cache) for c in caches]
        if 0 in sizes and any(s > 0 for s in sizes):
            raise ValueError(
                "cannot merge zero-layer ArraysCache with stateful "
                f"ArraysCache (layer counts: {sizes}); filter the batch "
                "before merge"
            )
        return orig_merge(caches)

    lm_cache.ArraysCache.empty = _empty
    lm_cache.ArraysCache.merge = classmethod(_merge)
    _INSTALLED = True
    logger.info("mlx-lm ArraysCache zero-layer guard installed (#1671)")
    return True
