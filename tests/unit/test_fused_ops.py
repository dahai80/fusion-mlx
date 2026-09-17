# SPDX-License-Identifier: Apache-2.0
"""Tests for fused RoPE + RMSNorm (PR-G, v2 doc §2.2/§5.6).

Validates the fused operators against stock MLX using the PR-F golden
reference harness (KL < 1e-6). Degrade switches default OFF — tests
enable them via monkeypatched env.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from fusion_mlx.eval.golden_reference import assert_logits_aligned
from fusion_mlx.shim.fused_ops import (
    _compute_yarn_freqs,
    fused_rmsnorm_residual,
    fused_rope,
    is_fused_rmsnorm_enabled,
    is_fused_rope_enabled,
    maybe_patch_model_rmsnorm,
)


def _enable(monkeypatch, name):
    # _env_on reads os.environ live, so just set the env var.
    monkeypatch.setenv(name, "1")


class TestFusedRMSNormResidual:
    def test_disabled_matches_stock(self):
        # When switch is OFF, fused_rmsnorm_residual == rms_norm + residual.
        x = mx.array(np.random.randn(2, 8, 16).astype(np.float32))
        w = mx.ones(16, dtype=mx.float32)
        eps = 1e-5
        stock = mx.fast.rms_norm(x, w, eps) + x
        out = fused_rmsnorm_residual(x, x, w, eps)
        assert_logits_aligned(stock, out, tol=1e-5, label="disabled matches stock")

    def test_enabled_matches_stock(self, monkeypatch):
        _enable(monkeypatch, "FUSION_SHIM_FUSED_RMSNORM")
        assert is_fused_rmsnorm_enabled() is True
        x = mx.array(np.random.randn(2, 8, 16).astype(np.float32))
        w = mx.array(np.random.randn(16).astype(np.float32))
        eps = 1e-5
        stock = mx.fast.rms_norm(x, w, eps) + x
        out = fused_rmsnorm_residual(x, x, w, eps)
        mx.eval([stock, out])
        assert_logits_aligned(stock, out, tol=1e-5, label="enabled matches stock")

    def test_f16_precision(self, monkeypatch):
        _enable(monkeypatch, "FUSION_SHIM_FUSED_RMSNORM")
        x = mx.array(np.random.randn(2, 4, 32).astype(np.float16))
        w = mx.ones(32, dtype=mx.float16)
        eps = 1e-5
        stock = mx.fast.rms_norm(x, w, eps) + x
        out = fused_rmsnorm_residual(x, x, w, eps)
        mx.eval([stock, out])
        # fp16 path — looser tolerance.
        assert_logits_aligned(stock, out, tol=1e-2, label="f16 precision")

    def test_residual_not_self(self, monkeypatch):
        # residual can differ from x (general case).
        _enable(monkeypatch, "FUSION_SHIM_FUSED_RMSNORM")
        x = mx.array(np.random.randn(2, 4, 16).astype(np.float32))
        residual = mx.array(np.random.randn(2, 4, 16).astype(np.float32))
        w = mx.ones(16, dtype=mx.float32)
        stock = mx.fast.rms_norm(x, w, 1e-5) + residual
        out = fused_rmsnorm_residual(x, residual, w, 1e-5)
        mx.eval([stock, out])
        assert_logits_aligned(stock, out, tol=1e-5, label="general residual")


class TestFusedRoPE:
    def test_disabled_matches_stock(self):
        x = mx.array(np.random.randn(1, 4, 2, 16).astype(np.float32))
        stock = mx.fast.rope(x, 16, traditional=True, base=10000.0, scale=1.0, offset=0)
        out = fused_rope(x, offset=0, dims=16)
        mx.eval([stock, out])
        assert_logits_aligned(stock, out, tol=1e-5, label="rope disabled matches stock")

    def test_enabled_standard_matches_stock(self, monkeypatch):
        _enable(monkeypatch, "FUSION_SHIM_FUSED_ROPE")
        assert is_fused_rope_enabled() is True
        x = mx.array(np.random.randn(1, 4, 2, 16).astype(np.float32))
        stock = mx.fast.rope(x, 16, traditional=True, base=10000.0, scale=1.0, offset=0)
        out = fused_rope(x, offset=0, dims=16, yarn_orig_ctx=0)
        mx.eval([stock, out])
        assert_logits_aligned(stock, out, tol=1e-5, label="rope enabled standard")

    def test_offset_advances(self, monkeypatch):
        _enable(monkeypatch, "FUSION_SHIM_FUSED_ROPE")
        x = mx.array(np.random.randn(1, 4, 2, 16).astype(np.float32))
        stock = mx.fast.rope(
            x, 16, traditional=True, base=10000.0, scale=1.0, offset=10
        )
        out = fused_rope(x, offset=10, dims=16, yarn_orig_ctx=0)
        mx.eval([stock, out])
        assert_logits_aligned(stock, out, tol=1e-5, label="rope offset advance")

    def test_yarn_freqs_shape(self):
        freqs = _compute_yarn_freqs(16, 10000.0, 1.0, 0, 0.1)
        assert freqs.shape == (8,)  # dims // 2

    def test_yarn_freqs_reduce_high_freq(self):
        # With context extension (scale > 1), low-freq dims should be scaled.
        base_freqs = _compute_yarn_freqs(32, 10000.0, 1.0, 0, 0.0)
        yarn_freqs = _compute_yarn_freqs(32, 10000.0, 2.0, 2048, 0.1)
        mx.eval([base_freqs, yarn_freqs])
        # The last (lowest-frequency) dim should differ.
        assert not mx.array_equal(base_freqs[-1], yarn_freqs[-1])

    def test_yarn_path_runs(self, monkeypatch):
        _enable(monkeypatch, "FUSION_SHIM_FUSED_ROPE")
        x = mx.array(np.random.randn(1, 8, 2, 16).astype(np.float32))
        # YaRN path should not crash.
        out = fused_rope(x, offset=0, dims=16, yarn_orig_ctx=2048, yarn_beta=0.1)
        mx.eval(out)
        assert out.shape == x.shape


class TestMaybePatchModel:
    def test_no_patch_when_disabled(self):
        model = nn.Module()
        model.layers = []
        assert maybe_patch_model_rmsnorm(model) == 0

    def test_no_patch_no_layers(self):
        model = nn.Module()
        assert maybe_patch_model_rmsnorm(model) == 0

    def test_patches_standard_layers(self, monkeypatch):
        _enable(monkeypatch, "FUSION_SHIM_FUSED_RMSNORM")

        class FakeLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.input_layernorm = nn.RMSNorm(16, eps=1e-5)
                self.post_attention_layernorm = nn.RMSNorm(16, eps=1e-5)
                self.self_attn = lambda x, mask, cache: x
                self.mlp = lambda x: x

            def __call__(self, x, mask=None, cache=None):
                r = self.self_attn(self.input_layernorm(x), mask, cache)
                h = x + r
                r = self.mlp(self.post_attention_layernorm(h))
                return h + r

        model = nn.Module()
        model.layers = [FakeLayer(), FakeLayer()]
        n = maybe_patch_model_rmsnorm(model)
        assert n == 2
        # Patched __call__ should run without error.
        x = mx.array(np.random.randn(1, 4, 16).astype(np.float32))
        out = model.layers[0](x)
        mx.eval(out)
        assert out.shape == x.shape
