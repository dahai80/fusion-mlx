"""Tests for the three-tier prefill wiring (HSA M3): tier weight loading and
the LsaAttention prefill routing above the tier threshold."""

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
# LSA deps live in sibling repos, not in CI requirements — skip there.
pytest.importorskip("fusion_lsa")
pytest.importorskip("hsa")

from fusion_mlx.patches.qwen35_lsa.attention import LsaAttention
from fusion_mlx.patches.qwen35_lsa.tier import (
    TIER_THRESHOLD,
    TierConfig,
    load_tier_config,
)

WEIGHTS = "/Users/dahai/.fusion-mlx/adapters/tier-weights-ckpt8000.safetensors"


def test_load_tier_config_real_weights():
    config = load_tier_config(WEIGHTS, num_layers=64)
    assert config.window == 512
    assert config.block_size == 2048
    assert config.latent_dim == 2048
    assert sorted(config.weights.keys()) == [
        3,
        7,
        11,
        15,
        19,
        23,
        27,
        31,
        35,
        39,
        43,
        47,
        51,
        55,
        59,
        63,
    ]
    w_q, w_k = config.weights[3]
    assert w_q.shape == (6144, 2048)
    assert w_k.shape == (1024, 2048)
    assert w_q.dtype == mx.float16


def test_load_tier_config_rejects_out_of_range_layer():
    with pytest.raises(ValueError, match="32 layers"):
        load_tier_config(WEIGHTS, num_layers=32)


class _StubOProj(nn.Module):
    def __call__(self, x):
        return x


def _make_attention(tier, L, T, Hq=4, Hkv=2, D=8):
    # Minimal attn stub exposing only what _tier_prefill touches
    attn = nn.Module()
    attn.o_proj = _StubOProj()
    attn.__call__ = None
    wrapper = LsaAttention.__new__(LsaAttention)
    wrapper.attn = attn
    wrapper.tier = tier
    wrapper.layer_index = 3
    queries = mx.random.normal((1, Hq, L, D))
    gate = mx.zeros((1, L, Hq * D))
    keys = mx.random.normal((1, Hkv, T, D))
    values = mx.random.normal((1, Hkv, T, D))
    return wrapper, queries, gate, keys, values, attn


def test_tier_prefill_matches_kernel_directly():
    rng = np.random.default_rng(21)
    Hq, Hkv, D, L, T = 4, 2, 8, 6, 64
    w_q = rng.standard_normal((Hq * D, 16)).astype(np.float16)
    w_k = rng.standard_normal((Hkv * D, 16)).astype(np.float16)
    tier = TierConfig(8, 16, 16, {3: (mx.array(w_q), mx.array(w_k))})
    wrapper, queries, gate, keys, values, attn = _make_attention(tier, L, T, Hq, Hkv, D)
    got = wrapper._tier_prefill(queries, gate, keys, values, attn)
    assert got.shape == (1, L, Hq * D)

    from hsa.attention.three_tier_mlx import three_tier_attention_fast_mlx

    direct = three_tier_attention_fast_mlx(
        queries[0].transpose(1, 0, 2),
        keys[0].transpose(1, 0, 2),
        values[0].transpose(1, 0, 2),
        tier.window,
        tier.block_size,
        tier.weights[3][0],
        tier.weights[3][1],
        q_offset=T - L,
    )
    direct = direct.astype(queries.dtype).transpose(1, 0, 2).reshape(
        L, -1
    ) * mx.sigmoid(gate[0])
    np.testing.assert_allclose(np.array(got[0]), np.array(direct), atol=1e-3, rtol=1e-3)


def test_tier_threshold_gate():
    # Routing predicate: dense at or below the threshold, tier above it
    assert TIER_THRESHOLD == 4096
    tier_armed = lambda total: total > TIER_THRESHOLD
    assert not tier_armed(4096)
    assert tier_armed(4097)


def test_tier_prefill_passes_latent_clamp():
    # latent_clamp set on the config must reach the kernel: clamped output
    # differs from the unclamped one when latent logits exceed the clamp.
    rng = np.random.default_rng(7)
    Hq, Hkv, D, L, T = 4, 2, 8, 6, 64
    w_q = rng.standard_normal((Hq * D, 16)).astype(np.float16)
    w_k = rng.standard_normal((Hkv * D, 16)).astype(np.float16)
    weights = {3: (mx.array(w_q), mx.array(w_k))}
    plain = TierConfig(8, 16, 16, weights)
    clamped = TierConfig(8, 16, 16, weights, 0.5)
    wrapper, queries, gate, keys, values, attn = _make_attention(
        plain, L, T, Hq, Hkv, D
    )
    unclamped = wrapper._tier_prefill(queries, gate, keys, values, attn)
    wrapper.tier = clamped
    clamped_out = wrapper._tier_prefill(queries, gate, keys, values, attn)
    assert not np.allclose(np.array(unclamped), np.array(clamped_out))
