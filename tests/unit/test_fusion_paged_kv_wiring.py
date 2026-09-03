import os
import types

import mlx.core as mx
import pytest

from fusion_mlx.custom_kernels.fusion_paged_kv import (
    evict_request,
    install_paged_kv,
    register_cache,
)
from fusion_mlx.custom_kernels.paged_kv_pool import (
    FusionPagedKVPool,
    FusionPagedRequestCache,
)
from fusion_mlx.fusion_takeover.config import FusionConfig


class FakeModel:
    model_type = "llama"
    layers = [object(), object(), object()]

    def __init__(self):
        self.make_cache = lambda: [None for _ in self.layers]


def test_pool_mode_make_cache_returns_request_caches():
    model = FakeModel()
    cfg = FusionConfig(
        enabled=True,
        paged_kv_enabled=True,
        pool_enabled=True,
        pool_num_blocks=32,
        paged_kv_block_size=4,
    )
    install_paged_kv(model, cfg)
    assert hasattr(model, "_fusion_paged_pool")
    assert isinstance(model._fusion_paged_pool, FusionPagedKVPool)
    caches = model.make_cache()
    assert all(isinstance(c, FusionPagedRequestCache) for c in caches)
    assert all(c.pool is model._fusion_paged_pool for c in caches)
    caches2 = model.make_cache()
    assert caches[0].request_id != caches2[0].request_id


def test_pool_mode_non_pool_keeps_paged_cache():
    from fusion_mlx.custom_kernels.paged_kv_cache import FusionPagedKVCache

    model = FakeModel()
    cfg = FusionConfig(enabled=True, paged_kv_enabled=True, pool_enabled=False)
    install_paged_kv(model, cfg)
    caches = model.make_cache()
    assert all(isinstance(c, FusionPagedKVCache) for c in caches)
    assert not hasattr(model, "_fusion_paged_pool")


def test_pool_config_fields():
    cfg = FusionConfig(enabled=True, paged_kv_enabled=True)
    assert cfg.pool_enabled is False
    assert cfg.pool_num_blocks == 256
    settings = types.SimpleNamespace(
        fusion_takeover_enabled=True,
        fusion_paged_kv_enabled=True,
        fusion_paged_pool="on",
        fusion_paged_pool_num_blocks="512",
    )
    cfg2 = FusionConfig.from_model_settings(settings)
    assert cfg2.pool_enabled is True
    assert cfg2.pool_num_blocks == 512
