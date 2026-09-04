from __future__ import annotations

import logging
import types

import mlx.core as mx
import mlx.nn as nn

from fusion_mlx.custom_kernels.paged_kv_cache import FusionPagedKVCache
from fusion_mlx.fusion_takeover.config import FusionConfig
from fusion_mlx.fusion_takeover.patcher import FusionModulePatcher

logger = logging.getLogger(__name__)


def test_fused_decode_config_field_default_off():
    cfg = FusionConfig(enabled=True, paged_kv_enabled=True)
    assert cfg.fused_decode_enabled is False


def test_fused_decode_config_field_from_settings():
    ns = types.SimpleNamespace(
        fusion_takeover_enabled=True,
        fusion_paged_kv_enabled=True,
        fusion_paged_fused_kernel="on",
    )
    cfg = FusionConfig.from_model_settings(ns)
    assert cfg.fused_decode_enabled is True


def test_patcher_wraps_llama_attention(monkeypatch):
    monkeypatch.setenv("FUSION_PAGED_FUSED_KERNEL", "on")

    class FakeRope:
        def __call__(self, x, offset=None):
            return x

    class FakeAttention(nn.Module):
        n_heads = 8
        n_kv_heads = 2
        head_dim = 8
        scale = 1.0 / (8**0.5)

        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(64, 64)
            self.k_proj = nn.Linear(64, 16)
            self.v_proj = nn.Linear(64, 16)
            self.o_proj = nn.Linear(64, 64)
            self.rope = FakeRope()

        def __call__(self, x, mask=None, cache=None):
            B, L, D = x.shape
            queries = (
                self.q_proj(x).reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
            )
            keys = (
                self.k_proj(x).reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
            )
            values = (
                self.v_proj(x).reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
            )
            if cache is not None:
                queries = self.rope(queries, offset=cache.offset)
                keys = self.rope(keys, offset=cache.offset)
                keys, values = cache.update_and_fetch(keys, values)
            output = mx.fast.scaled_dot_product_attention(
                queries, keys, values, cache=cache, scale=self.scale, mask=mask
            )
            output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
            return self.o_proj(output)

    class FakeLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = FakeAttention()

    class FakeModel(nn.Module):
        model_type = "llama"

        def __init__(self):
            super().__init__()
            self.layers = [FakeLayer()]

        def make_cache(self):
            return [FusionPagedKVCache(block_size=4, num_blocks=8)]

    model = FakeModel()
    cfg = FusionConfig(enabled=True, paged_kv_enabled=True, fused_decode_enabled=True)
    patcher = FusionModulePatcher()
    patcher.patch_model(model, cfg)

    cache = model.make_cache()[0]
    assert isinstance(cache, FusionPagedKVCache)
    for _ in range(5):
        k = mx.random.normal(shape=(1, 2, 1, 8)) * 0.1
        v = mx.random.normal(shape=(1, 2, 1, 8)) * 0.1
        cache.update_and_fetch(k, v)
    assert cache.offset > 0

    attn = model.layers[0].attention
    x = mx.random.normal(shape=(1, 1, 64)) * 0.1
    out = attn(x, cache=cache)
    assert out.shape == (1, 1, 64)


def test_patcher_wraps_gemma_family():
    from fusion_mlx.fusion_takeover.patcher import _FUSED_DECODE_MODEL_FAMILIES

    assert "gemma3" in _FUSED_DECODE_MODEL_FAMILIES

    class FakeModel(nn.Module):
        model_type = "gemma3"
        layers = []

    model = FakeModel()
    cfg = FusionConfig(enabled=True, paged_kv_enabled=True, fused_decode_enabled=True)
    patcher = FusionModulePatcher()
    patcher.patch_model(model, cfg)
    assert getattr(model, "_fusion_takeover_applied", False)
