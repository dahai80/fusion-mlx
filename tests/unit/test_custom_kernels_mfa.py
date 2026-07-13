# SPDX-License-Identifier: Apache-2.0
"""Tests for MFA (Metal Flash Attention) integration.

Exercises the absorbed custom_kernels/mfa package on the MLX SDPA fallback
path (mlx_mfa extension absent). On Linux CI this file is skipped via
conftest ``collect_ignore`` (mlx opt-dep list); on macOS (mlx present) it runs.
"""

from __future__ import annotations

import os
import tempfile

import mlx.core as mx
import mlx.nn as nn

from fusion_mlx.custom_kernels.mfa import (
    flash_attention,
    is_mfa_available,
)
from fusion_mlx.custom_kernels.mfa.dispatch_policy import (
    AttentionBackend,
    get_device_info,
    reset_device_cache,
    select_backend,
)
from fusion_mlx.custom_kernels.mfa.kv_cache import DenseKVCache, PagedKVCache
from fusion_mlx.custom_kernels.mfa.masks import (
    make_alibi_mask,
    make_causal_block_mask,
    make_causal_mask,
    make_sliding_window_mask,
    mask_from_attn_mask,
)
from fusion_mlx.custom_kernels.mfa.quantize import (
    QUANT_FP8_E4M3,
    QUANT_INT8,
    QUANT_NF4,
    dequantize,
    quantize_fp8,
    quantize_int8,
    quantize_nf4,
)
from fusion_mlx.custom_kernels.xfuser_attention import (
    FastAttnMethod,
    MLXFastAttention,
    calibrate_attention_strategy,
    compression_loss,
)


class TestMFABasic:
    def test_is_mfa_available(self):
        result = is_mfa_available()
        assert isinstance(result, bool)

    def test_flash_attention_basic(self):
        B, H, N, D = 1, 4, 64, 128
        q = mx.random.uniform(shape=(B, H, N, D)).astype(mx.float16)
        k = mx.random.uniform(shape=(B, H, N, D)).astype(mx.float16)
        v = mx.random.uniform(shape=(B, H, N, D)).astype(mx.float16)

        out = flash_attention(q, k, v, causal=True)
        assert out is not None
        assert out.shape == (B, H, N, D), f"Expected {(B, H, N, D)}, got {out.shape}"

    def test_flash_attention_non_causal(self):
        B, H, N, D = 2, 8, 32, 64
        q = mx.random.uniform(shape=(B, H, N, D)).astype(mx.float16)
        k = mx.random.uniform(shape=(B, H, N, D)).astype(mx.float16)
        v = mx.random.uniform(shape=(B, H, N, D)).astype(mx.float16)

        out = flash_attention(q, k, v, causal=False)
        assert out.shape == (B, H, N, D)

    def test_flash_attention_with_scale(self):
        B, H, N, D = 1, 4, 32, 64
        q = mx.random.uniform(shape=(B, H, N, D)).astype(mx.float16)
        k = mx.random.uniform(shape=(B, H, N, D)).astype(mx.float16)
        v = mx.random.uniform(shape=(B, H, N, D)).astype(mx.float16)

        scale = 0.125
        out = flash_attention(q, k, v, scale=scale, causal=True)
        assert out.shape == (B, H, N, D)

    def test_flash_attention_return_lse(self):
        B, H, N, D = 1, 4, 32, 64
        q = mx.random.uniform(shape=(B, H, N, D)).astype(mx.float16)
        k = mx.random.uniform(shape=(B, H, N, D)).astype(mx.float16)
        v = mx.random.uniform(shape=(B, H, N, D)).astype(mx.float16)

        result = flash_attention(q, k, v, causal=True, return_lse=True)
        assert isinstance(result, tuple)
        out, lse = result
        assert out.shape == (B, H, N, D)
        assert lse.shape == (B, H, N)

    def test_bf16_dtype(self):
        B, H, N, D = 1, 4, 32, 64
        q = mx.random.uniform(shape=(B, H, N, D)).astype(mx.bfloat16)
        k = mx.random.uniform(shape=(B, H, N, D)).astype(mx.bfloat16)
        v = mx.random.uniform(shape=(B, H, N, D)).astype(mx.bfloat16)

        out = flash_attention(q, k, v, causal=True)
        assert out.shape == (B, H, N, D)
        assert out.dtype == mx.bfloat16


class TestDispatchPolicy:
    def setup_method(self):
        reset_device_cache()

    def test_select_backend_decode(self):
        decision = select_backend(
            head_dim=128, seq_len_q=1, seq_len_kv=1024, is_decode=True
        )
        assert decision.backend == AttentionBackend.MLX_SDPA

    def test_select_backend_quantized(self):
        decision = select_backend(head_dim=128, quantized_kv=True)
        assert decision.backend == AttentionBackend.TURBOQUANT

    def test_select_backend_fallback(self):
        decision = select_backend(head_dim=1024)
        assert decision.backend == AttentionBackend.MLX_SDPA

    def test_get_device_info(self):
        info = get_device_info()
        assert isinstance(info.is_apple_silicon, bool)
        assert info.gpu_core_count >= 0


class TestMasks:
    def test_make_causal_mask(self):
        N = 8
        mask = make_causal_mask(N, N)
        assert mask.shape == (1, 1, N, N)
        assert mask[0, 0, 0, 0] == 0.0
        assert mask[0, 0, 0, 1] == -float("inf")

    def test_make_sliding_window_mask(self):
        N = 16
        ws = 4
        mask = make_sliding_window_mask(N, N, ws, causal=True)
        assert mask.shape == (1, 1, N, N)
        assert mask[0, 0, 4, 1] == 0.0
        assert mask[0, 0, 4, 5] == -float("inf")

    def test_mask_from_attn_mask_2d(self):
        N = 8
        mask_2d = mx.zeros((N, N), dtype=mx.bool_)
        mask_2d = mx.triu(mask_2d, k=1)
        mask = mask_from_attn_mask(mask_2d, N, N, 4)
        assert mask is not None
        assert mask.shape == (1, 1, N, N)

    def test_make_causal_block_mask(self):
        mask = make_causal_block_mask(256, block_size=64)
        assert mask.shape == (4, 4)
        assert mask[0, 0] == 1
        assert mask[0, 1] == 0


class TestQuantize:
    def test_quantize_int8_roundtrip(self):
        B, H, N, D = 1, 2, 16, 64
        t = mx.random.uniform(shape=(B, H, N, D)).astype(mx.float16)
        q, s = quantize_int8(t)
        assert q.dtype == mx.uint8
        assert s.shape == (B, H)

        deq = dequantize(q, s, QUANT_INT8, mx.float16)
        assert deq.shape == t.shape

    def test_quantize_fp8_roundtrip(self):
        B, H, N, D = 1, 2, 16, 64
        t = mx.random.uniform(shape=(B, H, N, D)).astype(mx.float16)
        q, s = quantize_fp8(t)
        assert q.dtype == mx.uint8
        deq = dequantize(q, s, QUANT_FP8_E4M3, mx.float16)
        assert deq.shape == t.shape

    def test_quantize_nf4_roundtrip(self):
        B, H, N, D = 1, 2, 16, 64
        t = mx.random.uniform(shape=(B, H, N, D)).astype(mx.float16)
        q, s = quantize_nf4(t)
        assert q.dtype == mx.uint8
        assert q.shape[-1] == D // 2
        deq = dequantize(q, s, QUANT_NF4, mx.float16)
        assert deq.shape == t.shape


class TestKVCache:
    def test_dense_kv_cache(self):
        B, H, D = 1, 4, 64
        max_seq = 1024
        cache = DenseKVCache(H, D, max_seq, batch_size=B)

        k = mx.random.uniform(shape=(B, H, 10, D)).astype(mx.float16)
        v = mx.random.uniform(shape=(B, H, 10, D)).astype(mx.float16)
        cache.append(k, v)
        assert cache.seqlen == 10

        cache.append(k, v)
        assert cache.seqlen == 20

        k_full = cache.k_for_attention()
        assert k_full.shape[-2] == 20

        cache.reset()
        assert cache.seqlen == 0

    def test_paged_kv_cache(self):
        B, H, D = 1, 4, 64
        block_size = 32
        cache = PagedKVCache(H, D, block_size=block_size, batch_size=B)

        k = mx.random.uniform(shape=(B, H, 65, D)).astype(mx.float16)
        v = mx.random.uniform(shape=(B, H, 65, D)).astype(mx.float16)
        cache.append(k, v)
        assert cache.seqlen == 65

        k_full = cache.k_for_attention()
        assert k_full.shape[-2] == 65

        bt = cache.get_block_table()
        assert bt.shape[0] >= 2

        cache.reset()
        assert cache.seqlen == 0


class TestXfuserAttention:
    def test_fast_attn_method_flags(self):
        m = FastAttnMethod.FULL_ATTN
        assert m.has(FastAttnMethod.FULL_ATTN)
        assert not m.has(FastAttnMethod.CFG_SHARE)

        m2 = FastAttnMethod.RESIDUAL_WINDOW_ATTN_CFG_SHARE
        assert m2.has(FastAttnMethod.RESIDUAL_WINDOW_ATTN)
        assert m2.has(FastAttnMethod.CFG_SHARE)

    def test_mlx_fast_attention_full(self):
        B, H, N, D = 1, 4, 32, 64
        q = mx.random.uniform(shape=(B, H, N, D)).astype(mx.float16)
        k = mx.random.uniform(shape=(B, H, N, D)).astype(mx.float16)
        v = mx.random.uniform(shape=(B, H, N, D)).astype(mx.float16)

        attn = MLXFastAttention(window_size=8)
        attn.set_methods([FastAttnMethod.FULL_ATTN] * 5)

        out = attn(q, k, v, step_idx=0, is_causal=False)
        assert out.shape == (B, H, N, D)

    def test_mlx_fast_attention_window_residual(self):
        B, H, N, D = 1, 4, 32, 64
        q = mx.random.uniform(shape=(B, H, N, D)).astype(mx.float16)
        k = mx.random.uniform(shape=(B, H, N, D)).astype(mx.float16)
        v = mx.random.uniform(shape=(B, H, N, D)).astype(mx.float16)

        attn = MLXFastAttention(window_size=8)
        methods = [FastAttnMethod.FULL_ATTN, FastAttnMethod.RESIDUAL_WINDOW_ATTN]
        attn.set_methods(methods)

        out0 = attn(q, k, v, step_idx=0, is_causal=False)
        assert out0.shape == (B, H, N, D)

        out1 = attn(q, k, v, step_idx=1, is_causal=False)
        assert out1.shape == (B, H, N, D)

    def test_compression_loss(self):
        a = mx.random.uniform(shape=(1, 4, 32, 64))
        b = a + mx.random.uniform(shape=(1, 4, 32, 64)) * 0.01
        loss = compression_loss(a, b)
        assert isinstance(loss, float)
        assert loss >= 0.0

    def test_mlx_fast_attention_output_share(self):
        B, H, N, D = 1, 4, 32, 64
        q = mx.random.uniform(shape=(B, H, N, D)).astype(mx.float16)
        k = mx.random.uniform(shape=(B, H, N, D)).astype(mx.float16)
        v = mx.random.uniform(shape=(B, H, N, D)).astype(mx.float16)

        attn = MLXFastAttention(window_size=8)
        methods = [FastAttnMethod.FULL_ATTN, FastAttnMethod.OUTPUT_SHARE]
        attn.set_methods(methods)

        out0 = attn(q, k, v, step_idx=0)
        out1 = attn(q, k, v, step_idx=1)
        assert out1.shape == (B, H, N, D)


class _ConstCalibModel:
    # Returns a constant output independent of attention method -> every
    # candidate is lossless, so calibration should pick the most aggressive.
    def __call__(self, prompts, n_steps):
        return mx.array(1.0)


class _RealisticCalibModel:
    # Actually drives each MLXFastAttention module, so different methods yield
    # different outputs (lossy vs FULL_ATTN baseline). seq=256 so SPARSE_ATTN
    # (block_size=64) and RESIDUAL_WINDOW_ATTN (window=8) genuinely compress
    # instead of degrading to full attention on a too-short sequence.
    def __init__(self, modules):
        self.modules = modules
        self.q = mx.random.normal((4, 4, 256, 64), dtype=mx.float32)
        self.k = mx.random.normal((4, 4, 256, 64), dtype=mx.float32)
        self.v = mx.random.normal((4, 4, 256, 64), dtype=mx.float32)

    def __call__(self, prompts, n_steps):
        outs = []
        for step in range(n_steps):
            for m in self.modules:
                o = m(self.q, self.k, self.v, step, batch_size=4)
                outs.append(mx.mean(o))
        return outs


class _ForwardAttrModel:
    # Exposes calibration_forward instead of __call__.
    def __init__(self, modules):
        self.modules = modules

    def calibration_forward(self, prompts, n_steps):
        return mx.array(2.0)


class TestCalibrateStrategy:
    def test_calibrate_no_longer_raises(self):
        mods = [MLXFastAttention(window_size=8) for _ in range(2)]
        strategies = calibrate_attention_strategy(_ConstCalibModel(), mods, 3, ["p"])
        assert len(strategies) == 2
        for s in strategies:
            assert len(s) == 3
            assert all(isinstance(m, FastAttnMethod) for m in s)

    def test_calibrate_empty_modules(self):
        strategies = calibrate_attention_strategy(_ConstCalibModel(), [], 3, ["p"])
        assert strategies == []

    def test_calibrate_invalid_n_steps(self):
        mods = [MLXFastAttention(window_size=8)]
        try:
            calibrate_attention_strategy(_ConstCalibModel(), mods, 0, ["p"])
            assert False, "expected ValueError"
        except ValueError:
            pass

    def test_calibrate_invalid_model(self):
        mods = [MLXFastAttention(window_size=8)]
        try:
            calibrate_attention_strategy(object(), mods, 2, ["p"])
            assert False, "expected TypeError"
        except TypeError:
            pass

    def test_calibrate_picks_most_aggressive_when_lossless(self):
        # Const model -> all candidates loss 0 <= threshold -> SPARSE_ATTN
        # (first/most aggressive in _CALIB_CANDIDATES) chosen for every module.
        mods = [MLXFastAttention(window_size=8) for _ in range(3)]
        strategies = calibrate_attention_strategy(
            _ConstCalibModel(), mods, 4, ["p"], threshold=0.1
        )
        for s in strategies:
            assert all(m == FastAttnMethod.SPARSE_ATTN for m in s)

    def test_calibrate_strict_threshold_keeps_full(self):
        # Realistic model -> candidates are lossy; threshold=0 rejects all.
        mods = [MLXFastAttention(window_size=8) for _ in range(2)]
        strategies = calibrate_attention_strategy(
            _RealisticCalibModel(mods), mods, 2, ["p"], threshold=0.0
        )
        for s in strategies:
            assert all(m == FastAttnMethod.FULL_ATTN for m in s)

    def test_calibrate_loose_threshold_picks_aggressive(self):
        mods = [MLXFastAttention(window_size=8) for _ in range(2)]
        strategies = calibrate_attention_strategy(
            _RealisticCalibModel(mods), mods, 2, ["p"], threshold=100.0
        )
        # At least one module should move off FULL_ATTN to an aggressive method.
        flat = {m for s in strategies for m in s}
        assert FastAttnMethod.SPARSE_ATTN in flat or (
            FastAttnMethod.RESIDUAL_WINDOW_ATTN in flat
            or FastAttnMethod.FULL_ATTN_CFG_SHARE in flat
        )

    def test_calibrate_uses_calibration_forward_attr(self):
        mods = [MLXFastAttention(window_size=8) for _ in range(2)]
        strategies = calibrate_attention_strategy(
            _ForwardAttrModel(mods), mods, 3, ["p"], threshold=0.1
        )
        # Const output -> all lossless -> SPARSE_ATTN.
        for s in strategies:
            assert all(m == FastAttnMethod.SPARSE_ATTN for m in s)

    def test_calibrate_verbose_runs(self):
        mods = [MLXFastAttention(window_size=8) for _ in range(2)]
        strategies = calibrate_attention_strategy(
            _ConstCalibModel(), mods, 2, ["p"], threshold=0.1, verbose=True
        )
        assert len(strategies) == 2

    def test_calibrate_sets_methods_on_modules(self):
        # After calibration, each module's steps_method must match the returned
        # strategy (so inference can proceed immediately).
        mods = [MLXFastAttention(window_size=8) for _ in range(2)]
        strategies = calibrate_attention_strategy(
            _ConstCalibModel(), mods, 3, ["p"], threshold=0.1
        )
        for m, s in zip(mods, strategies):
            assert m.steps_method == s


class TestMFABridge:
    def test_bridge_import(self):
        from fusion_mlx.custom_kernels import mfa_bridge

        assert mfa_bridge is not None

    def test_bridge_flash_attention(self):
        from fusion_mlx.custom_kernels.mfa_bridge import flash_attention

        B, H, N, D = 1, 4, 32, 64
        q = mx.random.uniform(shape=(B, H, N, D)).astype(mx.float16)
        k = mx.random.uniform(shape=(B, H, N, D)).astype(mx.float16)
        v = mx.random.uniform(shape=(B, H, N, D)).astype(mx.float16)

        out = flash_attention(q, k, v, causal=True)
        assert out.shape == (B, H, N, D)

    def test_bridge_make_causal_mask(self):
        from fusion_mlx.custom_kernels.mfa_bridge import make_causal_mask

        mask = make_causal_mask(32, 32)
        assert mask.shape == (1, 1, 32, 32)


class TestNumericalCorrectness:
    def test_sdpa_equivalence(self):
        B, H, N, D = 1, 4, 32, 64
        q = mx.random.uniform(shape=(B, H, N, D)).astype(mx.float16)
        k = mx.random.uniform(shape=(B, H, N, D)).astype(mx.float16)
        v = mx.random.uniform(shape=(B, H, N, D)).astype(mx.float16)

        scale = D**-0.5

        mask = mx.triu(mx.full((N, N), -float("inf"), dtype=mx.float16), k=1)
        ref = mx.fast.scaled_dot_product_attention(
            q,
            k,
            v,
            scale=scale,
            mask=mask.reshape(1, 1, N, N),
        )

        out = flash_attention(q, k, v, scale=scale, causal=True)

        assert out.shape == ref.shape
        assert out.dtype == ref.dtype
        assert mx.allclose(out, ref, rtol=1e-2, atol=1e-2)

    def test_causal_mask_correctness(self):
        N = 8
        mask = make_causal_mask(N, N)
        for i in range(N):
            for j in range(N):
                if j <= i:
                    assert mask[0, 0, i, j] == 0.0, f"Expected attend at ({i},{j})"
                else:
                    assert mask[0, 0, i, j] == -float(
                        "inf"
                    ), f"Expected mask at ({i},{j})"


class TestExtendedCoverage:
    def test_layout_normalization_bnhd(self):
        from fusion_mlx.custom_kernels.mfa.attention import _normalize_qkv_layout

        B, N, H, D = 1, 32, 4, 64
        q = mx.random.uniform(shape=(B, N, H, D))
        k = mx.random.uniform(shape=(B, N, H, D))
        v = mx.random.uniform(shape=(B, N, H, D))
        qn, kn, vn, transposed = _normalize_qkv_layout(q, k, v)
        assert transposed is True
        assert qn.shape == (B, H, N, D)

    def test_layout_normalization_bhnd(self):
        from fusion_mlx.custom_kernels.mfa.attention import _normalize_qkv_layout

        B, H, N, D = 1, 4, 64, 64
        q = mx.random.uniform(shape=(B, H, N, D))
        k = mx.random.uniform(shape=(B, H, N, D))
        v = mx.random.uniform(shape=(B, H, N, D))
        qn, kn, vn, transposed = _normalize_qkv_layout(q, k, v)
        assert transposed is False
        assert qn.shape == (B, H, N, D)

    def test_layout_normalization_ambiguous_square(self):
        from fusion_mlx.custom_kernels.mfa.attention import _normalize_qkv_layout

        B, H, N, D = 8, 8, 8, 8
        q = mx.random.uniform(shape=(B, H, N, D))
        k = mx.random.uniform(shape=(B, H, N, D))
        v = mx.random.uniform(shape=(B, H, N, D))
        qn, kn, vn, transposed = _normalize_qkv_layout(q, k, v)
        assert transposed is False

    def test_flash_attention_kvcache_with_append(self):
        from fusion_mlx.custom_kernels.mfa.kv_cache import DenseKVCache
        from fusion_mlx.custom_kernels.mfa_bridge import flash_attention_kvcache

        B, H, D = 1, 4, 64
        cache = DenseKVCache(H, D, 1024, batch_size=B)

        k1 = mx.random.uniform(shape=(B, H, 10, D)).astype(mx.float16)
        v1 = mx.random.uniform(shape=(B, H, 10, D)).astype(mx.float16)
        q1 = mx.random.uniform(shape=(B, H, 1, D)).astype(mx.float16)
        out1 = flash_attention_kvcache(q1, k1, v1, cache, causal=True)
        assert out1.shape == (B, H, 1, D)
        assert cache.seqlen == 10

        k2 = mx.random.uniform(shape=(B, H, 10, D)).astype(mx.float16)
        v2 = mx.random.uniform(shape=(B, H, 10, D)).astype(mx.float16)
        q2 = mx.random.uniform(shape=(B, H, 1, D)).astype(mx.float16)
        out2 = flash_attention_kvcache(q2, k2, v2, cache, causal=True)
        assert out2.shape == (B, H, 1, D)
        assert cache.seqlen == 20, f"Expected 20, got {cache.seqlen}"

    def test_paged_kv_cache_empty_batch2(self):
        from fusion_mlx.custom_kernels.mfa.kv_cache import PagedKVCache

        cache = PagedKVCache(num_heads=4, head_dim=64, batch_size=2)
        k = cache.k_for_attention()
        assert k.shape[0] == 2, f"Expected batch 2, got {k.shape[0]}"

    def test_flash_attention_varlen_fallback(self):
        from fusion_mlx.custom_kernels.mfa.attention import flash_attention_varlen_impl

        q = mx.random.uniform(shape=(12, 4, 64)).astype(mx.float16)
        k = mx.random.uniform(shape=(12, 4, 64)).astype(mx.float16)
        v = mx.random.uniform(shape=(12, 4, 64)).astype(mx.float16)
        cu_q = mx.array([0, 4, 12], dtype=mx.int32)
        cu_k = mx.array([0, 4, 12], dtype=mx.int32)
        out = flash_attention_varlen_impl(q, k, v, cu_q, cu_k, max_seq_len=12)
        assert out.shape == (12, 4, 64)

    def test_flash_attention_bnbd_layout(self):
        B, N, H, D = 1, 32, 4, 64
        q = mx.random.uniform(shape=(B, N, H, D)).astype(mx.float16)
        k = mx.random.uniform(shape=(B, N, H, D)).astype(mx.float16)
        v = mx.random.uniform(shape=(B, N, H, D)).astype(mx.float16)
        out = flash_attention(q, k, v, causal=True)
        assert out.shape == (B, N, H, D), f"Expected {(B, N, H, D)}, got {out.shape}"

        scale = D**-0.5
        q_bhnd = q.transpose(0, 2, 1, 3)
        k_bhnd = k.transpose(0, 2, 1, 3)
        v_bhnd = v.transpose(0, 2, 1, 3)
        mask = mx.triu(mx.full((N, N), -float("inf"), dtype=mx.float16), k=1).reshape(
            1, 1, N, N
        )
        ref = mx.fast.scaled_dot_product_attention(
            q_bhnd, k_bhnd, v_bhnd, scale=scale, mask=mask
        )
        ref = ref.transpose(0, 2, 1, 3)
        assert mx.allclose(out, ref, rtol=1e-2, atol=1e-2)

    def test_sage_attention_fallback(self):
        from fusion_mlx.custom_kernels.mfa.attention import sage_attention_impl

        B, H, N, D = 1, 4, 16, 64
        q = mx.random.uniform(shape=(B, H, N, D)).astype(mx.float16)
        k = mx.random.uniform(shape=(B, H, N, D)).astype(mx.float16)
        v = mx.random.uniform(shape=(B, H, N, D)).astype(mx.float16)
        block_mask = mx.array(
            [
                [1, 0, 0, 0],
                [1, 1, 0, 0],
                [1, 1, 1, 0],
                [1, 1, 1, 1],
            ],
            dtype=mx.uint8,
        )
        out = sage_attention_impl(q, k, v, block_mask)
        assert out.shape == (B, H, N, D)

    def test_alibi_mask(self):
        mask = make_alibi_mask(num_heads=4, seq_len_q=8, seq_len_kv=8)
        assert mask.shape == (1, 4, 8, 8)

    def test_make_sliding_window_mask_non_causal(self):
        mask = make_sliding_window_mask(8, 8, 3, causal=False)
        assert mask.shape == (1, 1, 8, 8)
        assert mask[0, 0, 0, 0] == 0.0
        assert mask[0, 0, 0, 3] == 0.0
        assert mask[0, 0, 0, 4] == -float("inf")

    def test_quantize_dequantize_consistency(self):
        B, H, N, D = 1, 2, 16, 64
        t = mx.random.uniform(shape=(B, H, N, D)).astype(mx.float16)
        q, s = quantize_int8(t)
        deq = dequantize(q, s, QUANT_INT8, mx.float16)
        rel_error = mx.mean(mx.abs(deq - t) / (mx.abs(t) + 1e-6))
        assert rel_error < 0.2, f"Relative error too large: {rel_error:.4f}"

    def test_paged_gather_pages(self):
        from fusion_mlx.custom_kernels.mfa.attention import _gather_pages

        num_blocks, H, block_size, head_dim = 10, 4, 64, 64
        pages = mx.random.uniform(shape=(num_blocks, H, block_size, head_dim)).astype(
            mx.float16
        )
        block_table = mx.array([0, 2, 1], dtype=mx.int32)
        gathered = _gather_pages(pages, block_table)
        assert gathered.shape == (
            1,
            H,
            3 * block_size,
            head_dim,
        ), f"Got {gathered.shape}"

    def test_dispatch_env_override(self):
        os.environ["MFA_FORCE_BACKEND"] = "MLX_SDPA"
        from fusion_mlx.custom_kernels.mfa.dispatch_policy import (
            AttentionBackend,
            reset_device_cache,
            select_backend,
        )

        reset_device_cache()
        d = select_backend(head_dim=128)
        assert d.backend == AttentionBackend.MLX_SDPA
        del os.environ["MFA_FORCE_BACKEND"]

    def test_turboquant_fallback(self):
        from fusion_mlx.custom_kernels.mfa.quantize import QUANT_INT8, quantize_int8
        from fusion_mlx.custom_kernels.mfa.turboquant import turboquant_attention

        B, H, N, D = 1, 2, 16, 64
        q = mx.random.uniform(shape=(B, H, N, D)).astype(mx.float16)
        k = mx.random.uniform(shape=(B, H, N, D)).astype(mx.float16)
        v = mx.random.uniform(shape=(B, H, N, D)).astype(mx.float16)
        kq, ks = quantize_int8(k)
        vq, vs = quantize_int8(v)
        out = turboquant_attention(q, kq, vq, ks, vs, QUANT_INT8)
        assert out.shape == (B, H, N, D)

    def test_save_load_strategy_config(self):
        from fusion_mlx.custom_kernels.xfuser_attention import (
            FastAttnMethod,
            load_strategy_config,
            save_strategy_config,
        )

        strategies = [
            [FastAttnMethod.FULL_ATTN, FastAttnMethod.RESIDUAL_WINDOW_ATTN],
            [FastAttnMethod.FULL_ATTN, FastAttnMethod.OUTPUT_SHARE],
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = f.name
        try:
            save_strategy_config(strategies, path)
            loaded = load_strategy_config(path)
            assert len(loaded) == 2
            assert loaded[0][0] == FastAttnMethod.FULL_ATTN
            assert loaded[0][1] == FastAttnMethod.RESIDUAL_WINDOW_ATTN
        finally:
            os.unlink(path)


class TestVideoAttention:
    def test_temporal_rearrange(self):
        from fusion_mlx.custom_kernels.mfa.video_attention import (
            _temporal_rearrange,
            _temporal_rearrange_back,
        )

        B, T, S, C = 2, 4, 16, 64
        x = mx.random.uniform(shape=(B * T, S, C))
        x_t, S_out = _temporal_rearrange(x, T)
        assert x_t.shape == (B * S, T, C), f"Got {x_t.shape}"
        assert S_out == S

        x_back = _temporal_rearrange_back(x_t, B, S, T, C)
        assert x_back.shape == (B * T, S, C)

    def test_temporal_attention_shape(self):
        from fusion_mlx.custom_kernels.mfa.video_attention import TemporalAttention

        B, T, S, C = 1, 4, 16, 64
        attn = TemporalAttention(dim=C, heads=4)
        x = mx.random.uniform(shape=(B * T, S, C)).astype(mx.float16)
        out = attn(x, timesteps=T)
        assert out.shape == (B * T, S, C), f"Expected {(B * T, S, C)}, got {out.shape}"

    def test_temporal_attention_non_causal(self):
        from fusion_mlx.custom_kernels.mfa.video_attention import TemporalAttention

        B, T, S, C = 1, 4, 16, 64
        attn = TemporalAttention(dim=C, heads=4, causal=False)
        x = mx.random.uniform(shape=(B * T, S, C)).astype(mx.float16)
        out = attn(x, timesteps=T)
        assert out.shape == x.shape

    def test_temporal_attention_causal(self):
        from fusion_mlx.custom_kernels.mfa.video_attention import TemporalAttention

        B, T, S, C = 1, 4, 16, 64
        attn = TemporalAttention(dim=C, heads=4, causal=True)
        x = mx.random.uniform(shape=(B * T, S, C)).astype(mx.float16)
        out = attn(x, timesteps=T)
        assert out.shape == x.shape

    def test_video_transformer_block_shape(self):
        from fusion_mlx.custom_kernels.mfa.video_attention import VideoTransformerBlock

        B, T, S, C = 1, 4, 16, 64
        context_dim = 128
        block = VideoTransformerBlock(dim=C, heads=4, context_dim=context_dim)
        x = mx.random.uniform(shape=(B * T, S, C)).astype(mx.float16)
        context = mx.random.uniform(shape=(B * T, 8, context_dim)).astype(mx.float16)
        out = block(x, context=context, timesteps=T)
        assert out.shape == (B * T, S, C)

    def test_spatial_video_transformer_shape(self):
        from fusion_mlx.custom_kernels.mfa.video_attention import (
            SpatialVideoTransformer,
        )

        B, T, C, H, W = 1, 4, 64, 8, 8
        transformer = SpatialVideoTransformer(
            in_channels=C,
            n_heads=4,
            d_head=16,
            depth=1,
            time_depth=1,
            timesteps=T,
        )
        x = mx.random.uniform(shape=(B * T, C, H, W)).astype(mx.float16)
        out = transformer(x, timesteps=T)
        assert out.shape == (
            B * T,
            C,
            H,
            W,
        ), f"Expected {(B * T, C, H, W)}, got {out.shape}"


class TestFP8Linear:
    def test_quantize_dequantize_block_fp8(self):
        from fusion_mlx.custom_kernels.mfa.fp8_linear import (
            _dequantize_block_fp8,
            _quantize_block_fp8,
        )

        x = mx.random.uniform(shape=(4, 256)).astype(mx.float16)
        fp8, scale = _quantize_block_fp8(x)
        assert fp8.dtype == mx.uint8
        deq = _dequantize_block_fp8(fp8, scale)
        assert deq.shape == x.shape
        rel_error = mx.mean(mx.abs(deq - x) / (mx.abs(x) + 1e-6))
        assert rel_error < 0.2, f"Relative error: {rel_error:.4f}"

    def test_fp8_linear_forward(self):
        from fusion_mlx.custom_kernels.mfa.fp8_linear import FP8Linear

        layer = FP8Linear(64, 128, bias=True)
        x = mx.random.uniform(shape=(2, 16, 64)).astype(mx.float16)
        out = layer(x)
        assert out.shape == (2, 16, 128)

    def test_fp8_linear_from_linear(self):
        from fusion_mlx.custom_kernels.mfa.fp8_linear import FP8Linear

        linear = nn.Linear(64, 128)
        fp8 = FP8Linear.from_linear(linear)
        x = mx.random.uniform(shape=(2, 16, 64)).astype(mx.float16)
        out = fp8(x)
        assert out.shape == (2, 16, 128)

    def test_fp8_linear_no_bias(self):
        from fusion_mlx.custom_kernels.mfa.fp8_linear import FP8Linear

        layer = FP8Linear(64, 128, bias=False)
        assert layer.bias is None
        x = mx.random.uniform(shape=(2, 16, 64)).astype(mx.float16)
        out = layer(x)
        assert out.shape == (2, 16, 128)
