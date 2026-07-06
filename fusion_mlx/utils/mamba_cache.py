# SPDX-License-Identifier: Apache-2.0
import logging

import mlx.core as mx

from .. import _mlx_compat as _mlx_compat

_mlx_compat.install()

try:
    from mlx_lm.models.cache import MambaCache
except ImportError:
    from mlx_lm.models.cache import ArraysCache as MambaCache

logger = logging.getLogger(__name__)


class BatchMambaCache(MambaCache):
    def __init__(self, left_padding: list[int] | None = None, size: int = 2):
        super().__init__(size=size, left_padding=left_padding)
        self._batch_size = len(left_padding) if left_padding else 0

    def extract(self, idx: int) -> MambaCache:
        size = len(self.cache)
        cache = MambaCache(size=size)
        cache.cache = [
            mx.contiguous(c[idx : idx + 1]) if c is not None else None
            for c in self.cache
        ]
        cache.left_padding = None
        return cache

    @classmethod
    def merge(cls, caches: list[MambaCache]) -> "BatchMambaCache":
        if not caches:
            return cls([])
        batch_size = len(caches)
        merged_cache = cls([0] * batch_size)
        num_arrays = len(caches[0].cache)
        merged_cache.cache = []
        for i in range(num_arrays):
            arrays = [c.cache[i] for c in caches if c.cache[i] is not None]
            if arrays:
                merged_cache.cache.append(mx.concatenate(arrays, axis=0))
            else:
                merged_cache.cache.append(None)
        return merged_cache


def patch_mlx_lm_for_mamba():
    import importlib

    from .. import _mlx_compat as _mlx_compat

    _mlx_compat.install()

    gen_module = importlib.import_module("mlx_lm.generate")
    from mlx_lm.models.cache import (
        ArraysCache,
        CacheList,
        KVCache,
        RotatingKVCache,
    )

    try:
        from mlx_lm.models.cache import MambaCache as OrigMambaCache
    except ImportError:
        OrigMambaCache = ArraysCache

    from mlx_lm.generate import BatchKVCache, BatchRotatingKVCache

    _original_make_cache = gen_module._make_cache

    def _patched_make_cache(model, left_padding, max_kv_size=None):
        def to_batch_cache(c):
            if isinstance(c, KVCache):
                return BatchKVCache(left_padding)
            elif isinstance(c, OrigMambaCache):
                return BatchMambaCache(left_padding)
            elif isinstance(c, ArraysCache):
                c.left_padding = mx.array(left_padding)
                return c
            elif isinstance(c, RotatingKVCache):
                if c.keep > 0:
                    raise ValueError(
                        "RotatingKVCache with keep tokens is not supported."
                    )
                return BatchRotatingKVCache(c.max_size, left_padding)
            elif isinstance(c, CacheList):
                return CacheList(*(to_batch_cache(sub_c) for sub_c in c.caches))
            else:
                raise ValueError(f"{type(c)} does not yet support batching")

        if hasattr(model, "make_cache"):
            cache = model.make_cache()
            return [to_batch_cache(c) for c in cache]
        elif max_kv_size is not None:
            return [
                BatchRotatingKVCache(max_kv_size, left_padding)
                for _ in model.layers
            ]
        else:
            return [BatchKVCache(left_padding) for _ in model.layers]

    gen_module._make_cache = _patched_make_cache

    _original_merge_caches = gen_module._merge_caches

    def _patched_merge_caches(caches):
        batch_cache = []
        for i in range(len(caches[0])):
            cache = None
            if isinstance(caches[0][i], KVCache):
                cache = BatchKVCache.merge([c[i] for c in caches])
            elif isinstance(caches[0][i], RotatingKVCache):
                cache = BatchRotatingKVCache.merge([c[i] for c in caches])
            elif isinstance(caches[0][i], (OrigMambaCache, BatchMambaCache)):
                cache = BatchMambaCache.merge([c[i] for c in caches])
            else:
                raise ValueError(
                    f"{type(caches[0][i])} does not yet support batching with history"
                )
            batch_cache.append(cache)
        return batch_cache

    gen_module._merge_caches = _patched_merge_caches
    logger.info("Patched mlx-lm for MambaCache batching support")


_patched = False


def ensure_mamba_support():
    global _patched
    if not _patched:
        logger.info(
            "[MambaCache] Skipping _make_cache patch — "
            "mlx-lm ArraysCache has native batching support"
        )
        _patched = True
