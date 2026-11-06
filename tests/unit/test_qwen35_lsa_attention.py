# SPDX-License-Identifier: Apache-2.0
"""Tests for the qwen3_5 LSA attention wrapper: prefill build + decode gather."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
from fusion_lsa.core.index_space import LearnedProjection
from safetensors.numpy import save_file

LATENT = 48
ROTARY = 8
KV_HEADS = 2
Q_HEADS = 4
HEAD_DIM = 32
HIDDEN = 64


class _Args:
    def __init__(self):
        self.hidden_size = HIDDEN
        self.num_attention_heads = Q_HEADS
        self.num_key_value_heads = KV_HEADS
        self.head_dim = HEAD_DIM
        self.attention_bias = False
        self.rms_norm_eps = 1e-6
        self.partial_rotary_factor = ROTARY / HEAD_DIM
        self.rope_theta = 10000.0
        self.rope_scaling = None
        self.max_position_embeddings = 4096


def _make_attn():
    from mlx_lm.models.qwen3_next import Qwen3NextAttention

    mx.random.seed(11)
    attn = Qwen3NextAttention(_Args())
    return attn


def _make_selector(config=None):
    from fusion_mlx.patches.qwen35_lsa.attention import build_selector

    free = KV_HEADS * (HEAD_DIM - ROTARY)
    wq = mx.random.normal((free, LATENT), key=mx.random.key(1))
    wk = mx.random.normal((free, LATENT), key=mx.random.key(2))
    projection = LearnedProjection(
        query_projection=wq / mx.linalg.norm(wq, axis=-1, keepdims=True),
        key_projection=wk / mx.linalg.norm(wk, axis=-1, keepdims=True),
    )
    return build_selector(projection, config, rotary_dim=ROTARY, kv_heads=KV_HEADS)


def _write_map(tmp_path, layers=(0,)):
    tensors = {}
    for layer in layers:
        tensors[f"layer{layer}.query_projection"] = np.zeros(
            (LATENT, 8), dtype=np.float32
        )
        tensors[f"layer{layer}.key_projection"] = np.zeros(
            (LATENT, 8), dtype=np.float32
        )
    path = str(tmp_path / "map.safetensors")
    save_file(
        tensors,
        path,
        metadata={
            "model_arch": "qwen38_27b_test",
            "indexer_layers": ",".join(str(l) for l in layers),
            "latent_dim": str(LATENT),
            "index_dim": "8",
        },
    )
    return path


def test_prefill_then_decode_runs_and_builds_index():
    from fusion_mlx.patches.qwen35_lsa.attention import LsaAttention
    from fusion_mlx.patches.qwen35_lsa.selection import LsaSelectionConfig

    attn = _make_attn()
    small = LsaSelectionConfig(
        pool_size=2,
        bucket_size=4,
        bucket_cap=8,
        top_p=8,
        top_k=32,
        sink_size=2,
        window_size=16,
    )
    wrapper = LsaAttention(attn, _make_selector(small))
    cache = __import__("mlx_lm.models.cache", fromlist=["KVCache"]).KVCache()

    prefill = mx.random.normal((1, 48, HIDDEN), key=mx.random.key(5))
    out = wrapper(prefill, mask=None, cache=cache)
    assert out.shape == (1, 48, HIDDEN)
    assert cache.offset == 48
    assert wrapper.selector.index.key_count > 0

    for _ in range(3):
        step = mx.random.normal((1, 1, HIDDEN), key=mx.random.key(6))
        out = wrapper(step, mask=None, cache=cache)
        assert out.shape == (1, 1, HIDDEN)
    assert cache.offset == 51
    assert int(mx.abs(out).sum().item()) > 0


def test_decode_picked_keys_match_manual_gather():
    from mlx_lm.models.base import scaled_dot_product_attention

    from fusion_mlx.patches.qwen35_lsa.attention import LsaAttention

    attn = _make_attn()
    wrapper = LsaAttention(attn, _make_selector())
    cache = __import__("mlx_lm.models.cache", fromlist=["KVCache"]).KVCache()
    wrapper(
        mx.random.normal((1, 32, HIDDEN), key=mx.random.key(7)), mask=None, cache=cache
    )

    x = mx.random.normal((1, 1, HIDDEN), key=mx.random.key(8))
    got = wrapper(x, mask=None, cache=cache)

    picked = np.array(wrapper.last_picked)
    position = 32
    B, L = 1, 1
    q_proj_output = attn.q_proj(x)
    queries, gate = mx.split(q_proj_output.reshape(B, L, Q_HEADS, -1), 2, axis=-1)
    gate = gate.reshape(B, L, -1)
    keys, values = attn.k_proj(x), attn.v_proj(x)
    queries = attn.q_norm(queries).transpose(0, 2, 1, 3)
    keys = attn.k_norm(keys.reshape(B, L, KV_HEADS, -1)).transpose(0, 2, 1, 3)
    values = values.reshape(B, L, KV_HEADS, -1).transpose(0, 2, 1, 3)
    queries = attn.rope(queries, offset=position)
    keys = attn.rope(keys, offset=position)
    keys, values = cache.update_and_fetch(keys, values)
    idx = mx.array(picked).reshape(1, 1, -1, 1)
    keys_sel = mx.take_along_axis(keys, idx, axis=2)
    values_sel = mx.take_along_axis(values, idx, axis=2)
    output = scaled_dot_product_attention(
        queries, keys_sel, values_sel, None, attn.scale, None
    )
    output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
    want = attn.o_proj(output * mx.sigmoid(gate))
    assert float(mx.abs(got - want).max()) < 1e-4


def test_wrap_only_mapped_full_attention_layers(tmp_path):
    from fusion_mlx.patches.qwen35_lsa.attention import LsaAttention, wrap_lsa_layers

    map_path = _write_map(tmp_path, layers=(1,))
    from fusion_lsa.convert.map_converter import LsaMap

    lsa_map = LsaMap(map_path, "qwen38_27b_test", num_layers=0, latent_dim=LATENT)

    class _Layer:
        def __init__(self, full):
            if full:
                self.self_attn = _make_attn()

    layers = [_Layer(False), _Layer(True), _Layer(True)]
    wrapped = wrap_lsa_layers(layers, lsa_map, rotary_dim=ROTARY, kv_heads=KV_HEADS)
    assert len(wrapped) == 1
    assert isinstance(layers[1].self_attn, LsaAttention)
    assert not isinstance(layers[2].self_attn, LsaAttention)


def test_batch_decode_falls_back_to_dense():
    from fusion_mlx.patches.qwen35_lsa.attention import LsaAttention

    attn = _make_attn()
    wrapper = LsaAttention(attn, _make_selector())
    got = wrapper(
        mx.random.normal((2, 1, HIDDEN), key=mx.random.key(10)),
        mask=None,
        cache=__import__("mlx_lm.models.cache", fromlist=["KVCache"]).KVCache(),
    )
    assert got.shape == (2, 1, HIDDEN)
    assert wrapper.last_picked is None
