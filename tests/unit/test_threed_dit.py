# SPDX-License-Identifier: Apache-2.0
# #989 Hunyuan3D-2.1 MLX port — Session 3 tests: flow-match DiT denoiser.
# Real-weight smoke gated by real_model marker (needs dit.safetensors on disk).
from __future__ import annotations

import os
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from fusion_mlx.threed.config import ShapeConfig, load_shape_config
from fusion_mlx.threed.dit import (
    HunyuanDiT,
    build_sigmas,
    denoise,
    gelu_erf,
    timestep_embed,
)

_HUNYUAN3D_DIR = Path(
    os.path.expanduser("~/.fusion-mlx/models/Hunyuan3D-2.1-MLX-Serve-8bit")
)
_DIT_PATH = _HUNYUAN3D_DIR / "dit.safetensors"
_REAL = _DIT_PATH.exists()

real_model = pytest.mark.skipif(
    not _REAL, reason="needs dit.safetensors (download Hunyuan3D-2.1-MLX-Serve-8bit)"
)


def test_build_sigmas_reversed_flow_match():
    sig = build_sigmas(5)
    assert sig.shape == (6,)
    # Ascending 0..1 + appended trailing 1.0 (delta 0, skipped in loop).
    assert float(sig[0]) == 0.0
    assert abs(float(sig[4]) - 1.0) < 1e-6
    assert float(sig[5]) == 1.0
    assert all(sig[i + 1] >= sig[i] for i in range(5))


def test_build_sigmas_min_steps():
    with pytest.raises(ValueError):
        build_sigmas(1)


def test_timestep_embed_sin_first_f32():
    emb = timestep_embed(0.5, 2048)
    assert emb.shape == (1, 2048)
    assert emb.dtype == mx.float32
    # f_0 = exp(0) = 1, so sin(ang[0]) = sin(0.5), cos(ang[0]) = cos(0.5).
    assert abs(float(emb[0, 0]) - np.sin(0.5)) < 1e-5
    assert abs(float(emb[0, 1024]) - np.cos(0.5)) < 1e-5
    assert bool(mx.all(mx.isfinite(emb)).item())


def test_gelu_erf_finite():
    x = mx.array(np.linspace(-3, 3, 21, dtype=np.float32))
    y = gelu_erf(x)
    mx.eval(y)
    assert bool(mx.all(mx.isfinite(y)).item())
    # Zero input -> zero output.
    assert abs(float(y[10])) < 1e-6


def test_dit_block_count_and_moe_layout():
    cfg = ShapeConfig()
    dit = HunyuanDiT(cfg)
    assert len(dit.blocks) == cfg.depth  # 21
    moe_start = cfg.depth - cfg.num_moe_layers  # 15
    for i, blk in enumerate(dit.blocks):
        assert blk.use_moe == (i >= moe_start)
        # U-ViT skips: push on 0..half_push-1, pop from pop_from.
        assert blk.has_skip == (i >= cfg.depth // 2 + 1)
        if blk.use_moe:
            # gate has NO bias (checkpoint ships no moe.gate.bias).
            assert not hasattr(blk.moe.gate, "bias")
            assert blk.moe.n_experts == cfg.num_experts
            assert blk.moe.top_k == cfg.moe_top_k
            # shared always-on expert present.
            assert hasattr(blk.moe, "shared")
            # 8 expert stacks each for fc1/fc2.
            assert len(blk.moe.experts.fc1.linears) == cfg.num_experts
            assert len(blk.moe.experts.fc2.linears) == cfg.num_experts
        else:
            assert hasattr(blk, "mlp") and not blk.use_moe


def test_dit_skip_fuse_dim():
    # skip fuse: LayerNorm(Linear(cat([skip, h], -1))) — 2*dim -> dim.
    cfg = ShapeConfig()
    dit = HunyuanDiT(cfg)
    pop_blk = dit.blocks[cfg.depth // 2 + 1]  # first pop block
    assert pop_blk.has_skip
    assert pop_blk.skip.linear.weight.shape[0] == cfg.hidden_size  # 2048 out
    # QuantizedLinear: weight is packed [out, in//4]; in=2*dim.
    assert hasattr(pop_blk.skip, "norm")


def test_dit_attention_qkv_sources():
    # Self-attn: q from xq, k/v from xkv (same input). Cross-attn: q from h,
    # k/v from cond. Per-head RMSNorm(128) weight-only on q,k (not v).
    cfg = ShapeConfig()
    dit = HunyuanDiT(cfg)
    blk = dit.blocks[0]
    hd = cfg.hidden_size // cfg.num_heads  # 128
    assert blk.attn1.q_norm.weight.shape == (hd,)
    assert blk.attn1.k_norm.weight.shape == (hd,)
    # No v_norm.
    assert not hasattr(blk.attn1, "v_norm")
    # Cross-attn cond projection target is context_dim (DINOv2 1024).
    # Verified structurally by attn2 kv weight load path in real-weight test.


def test_dit_structural_forward():
    cfg = ShapeConfig()
    dit = HunyuanDiT(cfg)
    x = mx.random.normal((1, cfg.num_latents, cfg.embed_dim), dtype=mx.float16) * 0.3
    cond = mx.zeros((1, cfg.dino_num_tokens, cfg.context_dim), dtype=mx.float16)
    v = dit.forward(x, cond, 0.5)
    mx.eval(v)
    assert v.shape == (1, cfg.num_latents, cfg.embed_dim)
    # Unloaded nn.Linear init is f32; loaded weights are f16 — accept either.
    assert v.dtype in (mx.float16, mx.float32)
    assert bool(mx.all(mx.isfinite(v)).item())


def test_dit_structural_forward_batch2_cfg():
    # CFG batch=2: x=[x;x], cond=[cond;0] — batch dim doubles, not seq.
    cfg = ShapeConfig()
    dit = HunyuanDiT(cfg)
    x = mx.random.normal((2, cfg.num_latents, cfg.embed_dim), dtype=mx.float16) * 0.3
    cond = mx.zeros((2, cfg.dino_num_tokens, cfg.context_dim), dtype=mx.float16)
    v = dit.forward(x, cond, 0.5)
    mx.eval(v)
    assert v.shape == (2, cfg.num_latents, cfg.embed_dim)
    assert bool(mx.all(mx.isfinite(v)).item())


def test_dit_structural_forward_with_skip_path():
    # Full U-ViT path incl. skip push/pop must not crash + keep token0 dropped.
    cfg = ShapeConfig()
    dit = HunyuanDiT(cfg)
    x = mx.random.normal((1, cfg.num_latents, cfg.embed_dim), dtype=mx.float16) * 0.3
    cond = mx.zeros((1, cfg.dino_num_tokens, cfg.context_dim), dtype=mx.float16)
    v = dit.forward(x, cond, 0.0)
    mx.eval(v)
    # Token 0 (timestep) dropped — output seq == num_latents, not num_latents+1.
    assert v.shape == (1, cfg.num_latents, cfg.embed_dim)


@real_model
def test_dit_real_weights_load_clean():
    cfg = load_shape_config(_HUNYUAN3D_DIR)
    from fusion_mlx.threed.dit import load_dit

    dit = load_dit(str(_DIT_PATH), cfg)
    # Every block's MoE gate is bias-free (matches checkpoint).
    for blk in dit.blocks:
        if blk.use_moe:
            assert not hasattr(blk.moe.gate, "bias")


@real_model
def test_dit_real_forward_finite():
    cfg = load_shape_config(_HUNYUAN3D_DIR)
    from fusion_mlx.threed.dit import load_dit

    dit = load_dit(str(_DIT_PATH), cfg)
    x = mx.random.normal((1, cfg.num_latents, cfg.embed_dim), dtype=mx.float16) * 0.3
    cond = mx.zeros((1, cfg.dino_num_tokens, cfg.context_dim), dtype=mx.float16)
    v = dit.forward(x, cond, 0.5)
    mx.eval(v)
    assert v.shape == (1, cfg.num_latents, cfg.embed_dim)
    arr = np.asarray(v)
    assert bool(np.all(np.isfinite(arr)))
    # Real weights produce non-trivial spread (std > 0.05), not dead output.
    assert float(arr.std()) > 0.05


@real_model
def test_dit_real_denoise_few_steps():
    cfg = load_shape_config(_HUNYUAN3D_DIR)
    from fusion_mlx.threed.dit import load_dit

    dit = load_dit(str(_DIT_PATH), cfg)
    cond = mx.zeros((1, cfg.dino_num_tokens, cfg.context_dim), dtype=mx.float16)
    x = denoise(dit, cond, steps=4, guidance=5.0, seed=1)
    mx.eval(x)
    assert x.shape == (1, cfg.num_latents, cfg.embed_dim)
    assert x.dtype == mx.float32
    arr = np.asarray(x)
    assert bool(np.all(np.isfinite(arr)))
    assert float(arr.std()) > 0.05
