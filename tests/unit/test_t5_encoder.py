# SPDX-License-Identifier: Apache-2.0
# Tests for the pure-MLX T5 encoder port. torch is unavailable in this env so
# there is no transformers.T5EncoderModel runtime oracle; instead each component
# is checked against a hand-written numpy reference of the exact transformers
# math (modeling_t5.py), plus a mock-weight from_pretrained + forward test.
import json
import math

import mlx.core as mx
import numpy as np
import pytest

from fusion_mlx.video.t5_encoder import (
    T5Attention,
    T5DenseGatedActDense,
    T5Encoder,
    T5EncoderConfig,
    T5LayerNorm,
)

_ERF = np.vectorize(math.erf)


def _gelu_np(x):
    return 0.5 * x * (1.0 + _ERF(x / np.sqrt(2.0)))


def _ref_bucket(rel_pos, bidirectional=True, num_buckets=32, max_distance=128):
    rel_pos = np.asarray(rel_pos)
    ret = np.zeros(rel_pos.shape, dtype=np.int64)
    n = num_buckets
    if bidirectional:
        n //= 2
        ret = ret + (rel_pos > 0).astype(np.int64) * n
        rel_pos = np.abs(rel_pos)
    else:
        rel_pos = -np.minimum(rel_pos, np.zeros_like(rel_pos))
    max_exact = n // 2
    is_small = rel_pos < max_exact
    rp = np.maximum(rel_pos.astype(np.float64), 1.0)
    large = max_exact + (
        np.log(rp / max_exact) / np.log(max_distance / max_exact) * (n - max_exact)
    ).astype(np.int64)
    large = np.minimum(large, n - 1)
    return ret + np.where(is_small, rel_pos.astype(np.int64), large)


def _small_cfg(**over):
    base = dict(
        d_model=16,
        num_layers=2,
        num_heads=4,
        d_kv=4,
        d_ff=32,
        vocab_size=64,
        rel_num_buckets=32,
        rel_max_distance=128,
    )
    base.update(over)
    return T5EncoderConfig(**base)


class TestRelativePositionBucket:
    def test_matches_numpy_reference(self):
        rel = np.arange(-12, 13)
        got = T5Attention._relative_position_bucket(
            mx.array(rel), bidirectional=True, num_buckets=32, max_distance=128
        )
        ref = _ref_bucket(rel, bidirectional=True, num_buckets=32, max_distance=128)
        np.testing.assert_array_equal(np.array(got), ref.astype(np.int32))

    def test_buckets_within_range(self):
        rel = np.arange(-40, 41)
        got = np.array(
            T5Attention._relative_position_bucket(
                mx.array(rel), bidirectional=True, num_buckets=32, max_distance=128
            )
        )
        assert got.min() >= 0
        assert got.max() <= 31  # bidirectional: 0..15 for rel<=0, 16..31 for rel>0

    def test_zero_relative_is_bucket_zero(self):
        got = int(
            np.array(
                T5Attention._relative_position_bucket(
                    mx.array([0]), bidirectional=True, num_buckets=32, max_distance=128
                )
            )[0]
        )
        assert got == 0


class TestLayerNorm:
    def test_matches_rmsnorm_reference(self):
        ln = T5LayerNorm(8, eps=1e-6)
        w = np.random.randn(8).astype(np.float32)
        ln.weight = mx.array(w)
        x = np.random.randn(2, 5, 8).astype(np.float32)
        out = np.array(ln(mx.array(x)))
        var = (x**2).mean(-1, keepdims=True)
        ref = w * (x / np.sqrt(var + 1e-6))
        np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-5)

    def test_no_mean_subtraction(self):
        # T5 norm uses variance only; a constant input should normalize to the
        # weight vector regardless of the constant's magnitude.
        ln = T5LayerNorm(4, eps=1e-6)
        w = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        ln.weight = mx.array(w)
        out = np.array(ln(mx.array(np.full((1, 1, 4), 5.0, dtype=np.float32))))
        np.testing.assert_allclose(out[0, 0], w, rtol=1e-4, atol=1e-5)


class TestGatedGeluFF:
    def test_matches_reference(self):
        d_model, d_ff = 8, 16
        ff = T5DenseGatedActDense(d_model, d_ff)
        # Xavier-scaled weights keep activations O(1) so the float32 port
        # matches the float64 numpy reference at tight tolerance (the check is
        # on the math, not on float accumulation noise).
        rng = np.random.RandomState(42)
        ff.gate_0.weight = mx.array(
            (rng.randn(d_ff, d_model) / np.sqrt(d_model)).astype(np.float32)
        )
        ff.fc1.weight = mx.array(
            (rng.randn(d_ff, d_model) / np.sqrt(d_model)).astype(np.float32)
        )
        ff.fc2.weight = mx.array(
            (rng.randn(d_model, d_ff) / np.sqrt(d_ff)).astype(np.float32)
        )
        x = (rng.randn(2, 3, d_model) * 0.5).astype(np.float32)
        out = np.array(ff(mx.array(x)))
        h_gelu = _gelu_np(x @ np.array(ff.gate_0.weight).T)
        h_lin = x @ np.array(ff.fc1.weight).T
        ref = (h_gelu * h_lin) @ np.array(ff.fc2.weight).T
        # float32 matmul accumulation differs ~1e-3 between mlx and numpy; a
        # wrong formula (e.g. missing gate) would error at O(0.1)+.
        np.testing.assert_allclose(out, ref, rtol=1e-2, atol=2e-3)


class TestAttentionMask:
    def test_padding_does_not_affect_valid_queries(self):
        cfg = _small_cfg(num_layers=1)
        attn = T5Attention(cfg, has_bias=True)
        for name in ("q", "k", "v", "o"):
            lin = getattr(attn, name)
            lin.weight = mx.array(
                np.random.randn(*np.array(lin.weight).shape).astype(np.float32)
            )
        attn.relative_attention_bias.weight = mx.array(
            np.random.randn(32, cfg.num_heads).astype(np.float32)
        )
        x = mx.array(np.random.randn(1, 6, cfg.d_model).astype(np.float32))
        mask = mx.array(
            (1.0 - np.array([[[[1, 1, 1, 0, 0, 0]]]], dtype=np.float32)) * -1e9
        )
        out_masked = np.array(attn(x, attn.compute_bias(6, 6), mask))
        out_sliced = np.array(attn(x[:, :3, :], attn.compute_bias(3, 3)))
        # First 3 queries attend only to first 3 keys once the last 3 are
        # masked, so they must equal the sliced (3-token) forward.
        np.testing.assert_allclose(
            out_masked[0, :3, :], out_sliced[0, :, :], rtol=1e-3, atol=1e-3
        )


class TestFromPretrained:
    def _write_mock(self, tmp_path, cfg):
        (tmp_path / "config.json").write_text(json.dumps(cfg))
        d_model = cfg["d_model"]
        d_ff = cfg["d_ff"]
        n_heads = cfg["num_heads"]
        inner = n_heads * cfg["d_kv"]
        vocab = cfg["vocab_size"]
        w = {}
        # HF 格式: (d_model, vocab_size), 加载时转置为 MLX (vocab_size, d_model)
        w["shared.weight"] = np.random.randn(d_model, vocab).astype(np.float32)
        for i in range(cfg["num_layers"]):
            p0 = f"encoder.block.{i}.layer.0."
            # HF T5 格式: 线性层 (d_model, inner), 加载时转置为 MLX (inner, d_model)
            w[p0 + "SelfAttention.q.weight"] = np.random.randn(d_model, inner).astype(
                np.float32
            )
            w[p0 + "SelfAttention.k.weight"] = np.random.randn(d_model, inner).astype(
                np.float32
            )
            w[p0 + "SelfAttention.v.weight"] = np.random.randn(d_model, inner).astype(
                np.float32
            )
            w[p0 + "SelfAttention.o.weight"] = np.random.randn(d_model, inner).astype(
                np.float32
            )
            if i == 0:
                # HF 格式: (num_heads, num_buckets), 加载时转置为 MLX (num_buckets, num_heads)
                w[p0 + "SelfAttention.relative_attention_bias.weight"] = (
                    np.random.randn(n_heads, 32).astype(np.float32)
                )
            w[p0 + "layer_norm.weight"] = np.random.randn(d_model).astype(np.float32)
            p1 = f"encoder.block.{i}.layer.1."
            # HF T5 格式: wi/wi_1 (d_model, d_ff), wo (d_ff, d_model)
            w[p1 + "DenseReluDense.wi_0.weight"] = np.random.randn(
                d_model, d_ff
            ).astype(np.float32)
            w[p1 + "DenseReluDense.wi_1.weight"] = np.random.randn(
                d_model, d_ff
            ).astype(np.float32)
            w[p1 + "DenseReluDense.wo.weight"] = np.random.randn(d_ff, d_model).astype(
                np.float32
            )
            w[p1 + "layer_norm.weight"] = np.random.randn(d_model).astype(np.float32)
        w["encoder.final_layer_norm.weight"] = np.random.randn(d_model).astype(
            np.float32
        )
        from safetensors.numpy import save_file

        save_file(w, str(tmp_path / "model.safetensors"))
        return w

    def test_loads_and_forwards(self, tmp_path):
        cfg = dict(
            d_model=16,
            num_layers=2,
            num_heads=4,
            d_kv=4,
            d_ff=32,
            vocab_size=64,
            relative_attention_num_buckets=32,
            relative_attention_max_distance=128,
            feed_forward_proj="gated-gelu",
            layer_norm_epsilon=1e-6,
        )
        w = self._write_mock(tmp_path, cfg)
        model = T5Encoder.from_pretrained(tmp_path, dtype=mx.float32)
        # token_embedding must come from shared.weight (HF transposed → MLX).
        np.testing.assert_allclose(
            np.array(model.token_embedding.weight), w["shared.weight"].T, rtol=1e-6
        )
        # relative_attention_bias must load on block 0 only (HF transposed → MLX).
        hf_bias = w[
            "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"
        ]
        np.testing.assert_allclose(
            np.array(
                model.blocks[0].layer0.attn.relative_attention_bias.weight
            ),
            hf_bias.T,
            rtol=1e-6,
        )
        assert model.blocks[1].layer0.attn.relative_attention_bias is None
        ids = mx.array([[1, 2, 3, 0, 0]])
        am = mx.array([[1, 1, 1, 0, 0]])
        out = model(ids, am)
        mx.eval(out)
        assert tuple(out.shape) == (1, 5, 16)
        assert bool(mx.all(mx.isfinite(out)))

    def test_rejects_non_gated_config(self, tmp_path):
        cfg = dict(
            d_model=16,
            num_layers=1,
            num_heads=4,
            d_kv=4,
            d_ff=32,
            vocab_size=64,
            feed_forward_proj="gelu",
        )
        (tmp_path / "config.json").write_text(json.dumps(cfg))
        with pytest.raises(ValueError, match="gated-gelu"):
            T5Encoder.from_pretrained(tmp_path)
