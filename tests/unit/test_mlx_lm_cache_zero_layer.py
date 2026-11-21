import mlx.core as mx
import pytest
from mlx_lm.models.cache import ArraysCache

from fusion_mlx.patches.mlx_lm_cache_zero_layer import (
    install_mlx_lm_cache_zero_layer_patch,
)


def test_zero_layer_cache_is_empty():
    install_mlx_lm_cache_zero_layer_patch()
    c = ArraysCache(0)
    assert c.empty() is True


def test_merge_all_zero_layer_returns_empty():
    install_mlx_lm_cache_zero_layer_patch()
    merged = ArraysCache.merge([ArraysCache(0), ArraysCache(0)])
    assert len(merged.cache) == 0
    assert merged.empty() is True


def test_merge_mixed_zero_layer_raises():
    install_mlx_lm_cache_zero_layer_patch()
    empty_cache = ArraysCache(0)
    stateful = ArraysCache(1)
    stateful[0] = mx.zeros((1, 2, 4, 8), mx.float16)
    with pytest.raises(ValueError, match="zero-layer"):
        ArraysCache.merge([empty_cache, stateful])


def test_merge_stateful_unaffected():
    install_mlx_lm_cache_zero_layer_patch()
    a = ArraysCache(1)
    b = ArraysCache(1)
    a[0] = mx.zeros((1, 2, 4, 8), mx.float16)
    b[0] = mx.zeros((1, 2, 4, 8), mx.float16)
    merged = ArraysCache.merge([a, b])
    assert merged[0].shape[0] == 2
    assert merged.empty() is False
