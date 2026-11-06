# SPDX-License-Identifier: Apache-2.0
"""Tests for the qwen3_5 LSA attention wrapper: prefill build + decode gather."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

# fusion-lsa is a sibling monorepo package (pip install -e ~/fusion/fusion-lsa),
# not a fusion-mlx dependency — CI runners don't have it. Skip instead of
# erroring at collection; these tests run on dev machines with it installed.
pytest.importorskip("fusion_lsa")

from fusion_lsa.core.index_space import LearnedProjection  # noqa: E402
from safetensors.numpy import save_file  # noqa: E402

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
            "model_arch": "qwen38_27b",
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
    # the index is deferred to the first single-stream decode step, which
    # rebuilds it from the cache instead of appending during prefill
    wrapper(
        mx.random.normal((1, 1, HIDDEN), key=mx.random.key(6)), mask=None, cache=cache
    )
    assert wrapper.selector.index.key_count == 48

    for i in range(3):
        step = mx.random.normal((1, 1, HIDDEN), key=mx.random.key(7 + i))
        out = wrapper(step, mask=None, cache=cache)
        assert out.shape == (1, 1, HIDDEN)
    assert cache.offset == 52
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

    lsa_map = LsaMap(map_path, "qwen38_27b", num_layers=0, latent_dim=LATENT)

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


def test_new_request_resets_selector_index():
    from mlx_lm.models.cache import KVCache

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

    cache_a = KVCache()
    wrapper(
        mx.random.normal((1, 32, HIDDEN), key=mx.random.key(9)),
        mask=None,
        cache=cache_a,
    )
    wrapper(
        mx.random.normal((1, 1, HIDDEN), key=mx.random.key(10)),
        mask=None,
        cache=cache_a,
    )
    assert wrapper.selector.index.key_count == 32

    cache_b = KVCache()
    wrapper(
        mx.random.normal((1, 16, HIDDEN), key=mx.random.key(11)),
        mask=None,
        cache=cache_b,
    )
    wrapper(
        mx.random.normal((1, 1, HIDDEN), key=mx.random.key(12)),
        mask=None,
        cache=cache_b,
    )
    assert wrapper.selector.index.key_count == 16


def test_radix_cache_hit_rebuilds_index_from_cache():
    from mlx_lm.models.cache import KVCache

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

    cache_a = KVCache()
    wrapper(
        mx.random.normal((1, 16, HIDDEN), key=mx.random.key(13)),
        mask=None,
        cache=cache_a,
    )

    cache_b = KVCache()
    restored_k = mx.random.normal((1, KV_HEADS, 32, HEAD_DIM), key=mx.random.key(14))
    restored_v = mx.random.normal((1, KV_HEADS, 32, HEAD_DIM), key=mx.random.key(15))
    cache_b.update_and_fetch(restored_k, restored_v)
    assert cache_b.offset == 32

    wrapper(
        mx.random.normal((1, 1, HIDDEN), key=mx.random.key(16)),
        mask=None,
        cache=cache_b,
    )
    assert wrapper.selector.index.key_count == 32


def test_batched_cache_offset_array_runs_dense():
    from mlx_lm.models.cache import BatchKVCache

    from fusion_mlx.patches.qwen35_lsa.attention import LsaAttention

    attn = _make_attn()
    wrapper = LsaAttention(attn, _make_selector())

    cache = BatchKVCache(left_padding=[0])
    k1 = mx.random.normal((1, KV_HEADS, 4, HEAD_DIM), key=mx.random.key(21))
    v1 = mx.random.normal((1, KV_HEADS, 4, HEAD_DIM), key=mx.random.key(22))
    cache.update_and_fetch(k1, v1)
    k2 = mx.random.normal((1, KV_HEADS, 6, HEAD_DIM), key=mx.random.key(23))
    v2 = mx.random.normal((1, KV_HEADS, 6, HEAD_DIM), key=mx.random.key(24))
    other = BatchKVCache(left_padding=[0])
    other.update_and_fetch(k2, v2)
    cache.extend(other)
    assert int(mx.reshape(cache.offset, (-1,)).shape[0]) == 2

    # two concurrent sequences: dense path must run, not crash on the
    # [B]-shaped offset array
    x = mx.random.normal((2, 1, HIDDEN), key=mx.random.key(25))
    out = wrapper(x, mask=None, cache=cache)
    assert out.shape == (2, 1, HIDDEN)


def test_in_place_extend_and_filter_arms_rebuild():
    from mlx_lm.models.cache import BatchKVCache

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

    cache = BatchKVCache(left_padding=[0])
    wrapper(
        mx.random.normal((1, 16, HIDDEN), key=mx.random.key(31)), mask=None, cache=cache
    )
    wrapper(
        mx.random.normal((1, 1, HIDDEN), key=mx.random.key(32)), mask=None, cache=cache
    )
    assert wrapper.selector.index.key_count == 16

    # a second request joins the batch in place: same object, width 1 -> 2
    other = BatchKVCache(left_padding=[0])
    other.update_and_fetch(
        mx.random.normal((1, KV_HEADS, 4, HEAD_DIM), key=mx.random.key(33)),
        mx.random.normal((1, KV_HEADS, 4, HEAD_DIM), key=mx.random.key(34)),
    )
    cache.extend(other)
    wrapper(
        mx.random.normal((2, 1, HIDDEN), key=mx.random.key(35)), mask=None, cache=cache
    )
    assert wrapper._pending_rebuild is True

    # batch filters back to one sequence: rows regress, the decode step
    # must re-arm and consume a rebuild over the filtered cache
    wrapper._pending_rebuild = False
    cache.filter(mx.array([0]))
    wrapper(
        mx.random.normal((1, 1, HIDDEN), key=mx.random.key(36)), mask=None, cache=cache
    )
    seq_rows = int(mx.reshape(cache.offset, (-1,))[0])
    assert wrapper.selector.index.key_count == (seq_rows // 16) * 16


def test_decode_runs_on_turboquant_cache():
    """Serving composition: the scheduler converts the prefilled KVCache to
    TurboQuant after prefill. The object swap must re-arm the selector, the
    index rebuild must read dequantized rows (the quantized state proxy is
    not a key array), and the decode gather must attend over dequantized
    keys/values. k8v4: 8-bit keys keep top-k agreement at ~0.99 (probe)."""
    from mlx_lm.models.cache import KVCache

    from fusion_mlx.patches.qwen35_lsa.attention import LsaAttention
    from fusion_mlx.patches.qwen35_lsa.selection import LsaSelectionConfig

    pytest.importorskip("mlx_vlm")
    from fusion_mlx.turboquant_kv import make_turboquant_kv_cache

    small = LsaSelectionConfig(
        pool_size=2,
        bucket_size=4,
        bucket_cap=8,
        top_p=8,
        top_k=32,
        sink_size=2,
        window_size=16,
    )
    attn = _make_attn()
    wrapper = LsaAttention(attn, _make_selector(small))
    cache = KVCache()
    wrapper(
        mx.random.normal((1, 48, HIDDEN), key=mx.random.key(5)), mask=None, cache=cache
    )

    tq = make_turboquant_kv_cache(4.0, "k8v4", source=cache)
    step = wrapper(
        mx.random.normal((1, 1, HIDDEN), key=mx.random.key(6)), mask=None, cache=tq
    )
    assert step.shape == (1, 1, HIDDEN)
    assert wrapper.selector.index.key_count == 48

    step = wrapper(
        mx.random.normal((1, 1, HIDDEN), key=mx.random.key(7)), mask=None, cache=tq
    )
    assert step.shape == (1, 1, HIDDEN)
    # 49 indexed rows close three 16-row pages; the fourth closes at 64
    assert wrapper.selector.index.key_count == 48
    assert int(mx.abs(step).sum().item()) > 0


def test_turboquant_decode_reuses_dequantized_rows():
    """The full per-step dequantize measured ~4x decode slowdown at 16k.
    After the rebuild-time dequantize, each step must grow the cached fp16
    rows with the step's own pre-quantization keys/values instead of
    dequantizing the whole cache again (one dequant call per request)."""
    from mlx_lm.models.cache import KVCache

    from fusion_mlx.patches.qwen35_lsa.attention import LsaAttention
    from fusion_mlx.patches.qwen35_lsa.selection import LsaSelectionConfig

    pytest.importorskip("mlx_vlm")
    from mlx_vlm.turboquant import TurboQuantKVCache

    from fusion_mlx.turboquant_kv import make_turboquant_kv_cache

    small = LsaSelectionConfig(
        pool_size=2, bucket_size=4, bucket_cap=8, top_p=8, top_k=32,
        sink_size=2, window_size=16,
    )
    attn = _make_attn()
    wrapper = LsaAttention(attn, _make_selector(small))
    cache = KVCache()
    wrapper(
        mx.random.normal((1, 48, HIDDEN), key=mx.random.key(5)), mask=None, cache=cache
    )
    tq = make_turboquant_kv_cache(4.0, "k8v4", source=cache)

    calls = 0
    original = tq.dequantize

    def counting():
        nonlocal calls
        calls += 1
        return original()

    tq.dequantize = counting
    for i in range(4):
        out = wrapper(
            mx.random.normal((1, 1, HIDDEN), key=mx.random.key(6 + i)),
            mask=None, cache=tq,
        )
        assert out.shape == (1, 1, HIDDEN)
    assert calls == 1
    assert wrapper._dequant_rows[2] == 52
    assert wrapper._dequant_rows[0].shape[2] >= 52
