# SPDX-License-Identifier: Apache-2.0
"""#1201: slow-path attention families moved to mx.fast SDPA.

The manual (scores @ softmax) path materializes the full (B,H,L,L)
score matrix; mx.fast.scaled_dot_product_attention is fused and
memory-efficient. These tests verify numerical equivalence against the
manual reference plus real forward passes through the converted modules.
"""

import math

import mlx.core as mx
import mlx.nn as nn

from fusion_mlx.video.hunyuanvideo.vae import MidAttention
from fusion_mlx.video.latentsync_mlx.attention import Attention as LSAttention
from fusion_mlx.video.latentsync_mlx.vae import VAEAttention
from fusion_mlx.video.mage.layers import MageFlowAttention
from fusion_mlx.video.svd.unet import SpatialAttention


def _manual_sdpa(q, k, v, scale, mask=None):
    attn = (q * scale) @ k.transpose(0, 1, 3, 2)
    if mask is not None:
        attn = attn + mask
    attn = mx.softmax(attn.astype(mx.float32), axis=-1).astype(q.dtype)
    return attn @ v


def test_fast_sdpa_matches_manual_reference():
    mx.random.seed(7)
    b, h, l, d = 2, 4, 32, 64
    q = mx.random.normal((b, h, l, d)).astype(mx.bfloat16)
    k = mx.random.normal((b, h, l, d)).astype(mx.bfloat16)
    v = mx.random.normal((b, h, l, d)).astype(mx.bfloat16)
    scale = d**-0.5
    fast = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
    ref = _manual_sdpa(q, k, v, scale)
    mx.eval(fast, ref)
    assert mx.abs(fast.astype(mx.float32) - ref.astype(mx.float32)).max() < 2e-2


def test_fast_sdpa_matches_manual_with_additive_mask():
    mx.random.seed(11)
    b, h, l, s, d = 2, 4, 16, 24, 64
    q = mx.random.normal((b, h, l, d)).astype(mx.bfloat16)
    k = mx.random.normal((b, h, s, d)).astype(mx.bfloat16)
    v = mx.random.normal((b, h, s, d)).astype(mx.bfloat16)
    mask = mx.zeros((b, 1, 1, s)).astype(mx.bfloat16)
    mask[:, :, :, s - 4 :] = -1e9
    scale = d**-0.5
    fast = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
    ref = _manual_sdpa(q, k, v, scale, mask)
    mx.eval(fast, ref)
    assert mx.abs(fast.astype(mx.float32) - ref.astype(mx.float32)).max() < 2e-2


def test_hunyuanvideo_mid_attention_forward():
    m = MidAttention(channels=64)
    mx.random.seed(3)
    x = mx.random.normal((1, 64, 2, 4, 4)).astype(mx.bfloat16)
    out = m(x)
    mx.eval(out)
    assert out.shape == x.shape
    assert not mx.isnan(out).any()


def test_svd_spatial_attention_forward():
    m = SpatialAttention(dim=64, num_heads=4)
    mx.random.seed(5)
    x = mx.random.normal((1, 64, 2, 8, 8)).astype(mx.bfloat16)
    out = m(x)
    mx.eval(out)
    assert out.shape == x.shape
    assert not mx.isnan(out).any()


def test_latentsync_attention_forward():
    m = LSAttention(query_dim=128, heads=4, dim_head=32)
    mx.random.seed(9)
    x = mx.random.normal((1, 16, 128)).astype(mx.bfloat16)
    ctx = mx.random.normal((1, 8, 128)).astype(mx.bfloat16)
    out = m(x, ctx)
    mx.eval(out)
    assert out.shape == x.shape
    assert not mx.isnan(out).any()


def test_latentsync_vae_attention_headless():
    m = VAEAttention(dims=64)
    mx.random.seed(13)
    x = mx.random.normal((1, 8, 8, 64)).astype(mx.bfloat16)
    out = m(x)
    mx.eval(out)
    assert out.shape == x.shape
    assert not mx.isnan(out).any()


def test_mage_flow_attention_forward():
    m = MageFlowAttention(dim=128, num_heads=4)
    mx.random.seed(15)
    x = mx.random.normal((1, 16, 128)).astype(mx.bfloat16)
    out = m(x, context=None, rope=None)
    mx.eval(out)
    assert out.shape == x.shape
    assert not mx.isnan(out).any()
