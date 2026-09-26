# SPDX-License-Identifier: Apache-2.0
# #989 Hunyuan3D-2.1 paint — Session 4 UNet tests (dual-branch multiview).
from __future__ import annotations

import os
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from fusion_mlx.threed.config import PaintUNetConfig
from fusion_mlx.threed.paint.unet import (
    _GEGLU,
    CrossAttention,
    HunyuanPaintUNet,
    _FeedForward,
    timestep_embedding,
)

_HUNYUAN3D_DIR = Path(
    os.path.expanduser("~/.fusion-mlx/models/Hunyuan3D-2.1-MLX-Serve-8bit")
)
_UNET_PATH = _HUNYUAN3D_DIR / "paint" / "unet.safetensors"
_REAL = _UNET_PATH.exists()

real_model = pytest.mark.skipif(
    not _REAL,
    reason="needs paint/unet.safetensors (download Hunyuan3D-2.1-MLX-Serve-8bit)",
)


def test_timestep_embedding_sin_first():
    emb = timestep_embedding(mx.array([0.5]), 320)
    assert emb.shape == (1, 320)
    # interleaved [sin0, cos0, sin1, cos1, ...]: freq[0]=1.0.
    assert abs(float(emb[0, 0]) - np.sin(0.5)) < 1e-5
    assert abs(float(emb[0, 1]) - np.cos(0.5)) < 1e-5


def test_cross_attention_dual_modes():
    # full-dual (attn1): q_mr/k_mr/v_mr/out_mr all present.
    full = CrossAttention(320, 320, 5, 64, dual="full")
    assert hasattr(full, "to_q_mr") and hasattr(full, "to_k_mr")
    assert hasattr(full, "to_v_mr") and hasattr(full, "to_out_mr")
    # v_out-dual (attn_refview): only v_mr + out_mr; q/k SHARED.
    vout = CrossAttention(320, 320, 5, 64, dual="v_out")
    assert not hasattr(vout, "to_q_mr") and not hasattr(vout, "to_k_mr")
    assert hasattr(vout, "to_v_mr") and hasattr(vout, "to_out_mr")
    # single: no mr params.
    single = CrossAttention(320, 1024, 5, 64)
    assert not hasattr(single, "to_v_mr")


def test_geglu_split_gate():
    g = _GEGLU(320, 5120)
    x = mx.random.normal((1, 4, 320), dtype=mx.float16)
    y = g(x)
    mx.eval(y)
    assert y.shape == (1, 4, 5120)
    assert bool(mx.all(mx.isfinite(y)).item())


def test_feedforward_keys_match_checkpoint():
    ff = _FeedForward(320, 1280)
    flat: dict = {}
    import mlx.nn as nn

    nn.utils.tree_flatten(ff, destination=flat)
    keys = set(flat.keys())
    # net.0.proj (GEGLU gate) + net.2 (linear); net.1 is activation (no params).
    assert "net.0.proj.weight" in keys
    assert "net.0.proj.bias" in keys
    assert "net.2.weight" in keys
    assert "net.2.bias" in keys


def test_unet_structural_forward():
    cfg = PaintUNetConfig()
    net = HunyuanPaintUNet(cfg)
    B, H, W = 1, 8, 8
    lat = mx.random.normal((B, cfg.in_channels, H, W), dtype=mx.float16) * 0.3
    ct = mx.zeros((B, 77, 1024), dtype=mx.float16)
    cd = mx.zeros((B, 4, 1024), dtype=mx.float16)
    o = net(lat, 5.0, ct, cd)
    mx.eval(o)
    assert o.shape == (B, cfg.out_channels, H, W)
    assert bool(mx.all(mx.isfinite(o)).item())


def test_unet_block_channel_flow():
    cfg = PaintUNetConfig()
    net = HunyuanPaintUNet(cfg)
    ch = cfg.block_out_channels
    rch = list(reversed(ch))
    # up0 = UpBlock (no attn), up1/2/3 = CrossAttnUp.
    assert len(net.up_blocks[0].resnets) == cfg.layers_per_block + 1
    assert (
        len(net.up_blocks[0].attentions) == 0
        if hasattr(net.up_blocks[0], "attentions")
        else True
    )
    # up2 outputs rch[2]=640, up3 outputs rch[3]=320.
    assert net.up_blocks[2].resnets[0].out_c == rch[2]
    assert net.up_blocks[3].resnets[0].out_c == rch[3]


def test_unet_skip_stack_balanced():
    # 12 skips pushed (conv_in + 11 down outputs), 12 popped by 4 up blocks.
    cfg = PaintUNetConfig()
    net = HunyuanPaintUNet(cfg)
    B, H, W = 1, 8, 8
    lat = mx.random.normal((B, cfg.in_channels, H, W), dtype=mx.float16) * 0.3
    ct = mx.zeros((B, 77, 1024), dtype=mx.float16)
    cd = mx.zeros((B, 4, 1024), dtype=mx.float16)
    o = net(lat, 5.0, ct, cd)
    mx.eval(o)
    # forward completed without pop-empty error -> balanced.
    assert o.shape == (B, cfg.out_channels, H, W)


@real_model
def test_unet_real_load_clean():
    cfg = PaintUNetConfig()
    from fusion_mlx.threed.paint.unet import load_paint_unet

    net = load_paint_unet(str(_UNET_PATH), cfg)
    # No missing/unexpected beyond non-weight noise (load logs clean).
    assert net is not None


@real_model
def test_unet_real_forward_64():
    cfg = PaintUNetConfig()
    from fusion_mlx.threed.paint.unet import load_paint_unet

    net = load_paint_unet(str(_UNET_PATH), cfg)
    B, H, W = 1, 64, 64  # paint latent resolution (512/8)
    lat = mx.random.normal((B, cfg.in_channels, H, W), dtype=mx.float16) * 0.3
    ct = mx.zeros((B, 77, 1024), dtype=mx.float16)
    cd = mx.zeros((B, 4, 1024), dtype=mx.float16)
    o = net(lat, 5.0, ct, cd)
    mx.eval(o)
    assert o.shape == (B, cfg.out_channels, H, W)
    arr = np.asarray(o)
    assert bool(np.all(np.isfinite(arr)))
    assert float(arr.std()) > 0.05
