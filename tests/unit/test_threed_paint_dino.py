# SPDX-License-Identifier: Apache-2.0
# #989 Hunyuan3D-2.1 paint — Session 4 DINOv2-Giant conditioner tests.
from __future__ import annotations

import os
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from fusion_mlx.threed.config import PaintDinoConfig
from fusion_mlx.threed.paint.dino import PaintDINOv2

_HUNYUAN3D_DIR = Path(
    os.path.expanduser("~/.fusion-mlx/models/Hunyuan3D-2.1-MLX-Serve-8bit")
)
_DINO_PATH = _HUNYUAN3D_DIR / "paint" / "dino.safetensors"
_REAL = _DINO_PATH.exists()

real_model = pytest.mark.skipif(
    not _REAL,
    reason="needs paint/dino.safetensors (download Hunyuan3D-2.1-MLX-Serve-8bit)",
)


def test_dino_config_defaults():
    cfg = PaintDinoConfig()
    assert cfg.hidden == 1536
    assert cfg.layers == 40
    assert cfg.heads == 24
    assert cfg.head_dim == 64
    assert cfg.mlp_hidden == 8192
    assert cfg.num_tokens == 1370  # 37*37 + 1


def test_dino_structural_forward():
    cfg = PaintDinoConfig()
    net = PaintDINOv2(cfg)
    x = mx.random.uniform(0, 1, (1, 3, 518, 518), dtype=mx.float32)
    o = net(x)
    mx.eval(o)
    assert o.shape == (1, 1370, 1536)
    assert bool(mx.all(mx.isfinite(o)).item())


def test_dino_block_structure():
    cfg = PaintDinoConfig()
    net = PaintDINOv2(cfg)
    assert len(net.blocks) == 40
    blk = net.blocks[0]
    # separate q/k/v/out (not fused qkv); mlp w_in/w_out (Giant naming).
    assert hasattr(blk.attn, "q") and hasattr(blk.attn, "k")
    assert hasattr(blk.attn, "v") and hasattr(blk.attn, "out")
    assert hasattr(blk.mlp, "w_in") and hasattr(blk.mlp, "w_out")
    assert hasattr(blk, "ls1") and hasattr(blk, "ls2")  # layer scale


def test_dino_mlp_hidden_dim():
    cfg = PaintDinoConfig()
    net = PaintDINOv2(cfg)
    # SwiGLU: w_in 1536 -> 8192 (fused gate+up), w_out 4096 -> 1536.
    assert net.blocks[0].mlp.w_in.weight.shape == (8192, 1536)
    assert net.blocks[0].mlp.w_out.weight.shape == (1536, 4096)
    assert net.blocks[0].mlp.intermediate == 4096


@real_model
def test_dino_real_load_clean():
    cfg = PaintDinoConfig()
    from fusion_mlx.threed.paint.dino import load_paint_dinov2

    net = load_paint_dinov2(str(_DINO_PATH), cfg)
    assert net is not None


@real_model
def test_dino_real_forward():
    cfg = PaintDinoConfig()
    from fusion_mlx.threed.paint.dino import load_paint_dinov2

    net = load_paint_dinov2(str(_DINO_PATH), cfg)
    x = mx.random.uniform(0, 1, (1, 3, 518, 518), dtype=mx.float32)
    o = net(x)
    mx.eval(o)
    assert o.shape == (1, 1370, 1536)
    arr = np.asarray(o)
    assert bool(np.all(np.isfinite(arr)))
    # real weights produce non-trivial spread (not zeros / not NaN).
    assert float(arr.std()) > 0.1
