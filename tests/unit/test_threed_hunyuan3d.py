# SPDX-License-Identifier: Apache-2.0
# #989 Hunyuan3D-2.1 MLX port — Session 1 tests: config parse + DINOv2
# conditioner module structure. Real-weight smoke load gated by real_model
# marker (needs 310MB conditioner.safetensors on disk).
from __future__ import annotations

import os
from pathlib import Path

import mlx.core as mx
import pytest

from fusion_mlx.threed.config import (
    ShapeConfig,
    load_shape_config,
)
from fusion_mlx.threed.dinov2 import DINOv2Block, DINOv2Conditioner

_HUNYUAN3D_DIR = Path(
    os.path.expanduser("~/.fusion-mlx/models/Hunyuan3D-2.1-MLX-Serve-8bit")
)


def test_shape_config_defaults():
    cfg = ShapeConfig()
    assert cfg.dino_hidden == 1024
    assert cfg.dino_layers == 24
    assert cfg.dino_image_size == 518
    assert cfg.dino_patch == 14
    assert cfg.dino_num_tokens == 1370  # 37*37 + 1 cls
    assert cfg.dino_group_size == 64
    assert cfg.num_moe_layers == 6
    assert cfg.num_experts == 8
    assert cfg.moe_top_k == 2


def test_dinov2_block_structure():
    cfg = ShapeConfig()
    blk = DINOv2Block(cfg.dino_hidden, cfg.dino_heads, 4, cfg.dino_group_size)
    # QuantizedLinear submodules expose weight/scales/biases/bias.
    assert hasattr(blk.attn.q, "weight")
    assert hasattr(blk.attn.q, "scales")
    assert hasattr(blk.attn.q, "biases")
    assert hasattr(blk.mlp.fc1, "weight")
    # LayerNorm + layerscale present.
    assert hasattr(blk, "norm1")
    assert hasattr(blk, "ls1")
    assert hasattr(blk, "ls2")


def test_dinov2_conditioner_forward_shape():
    # Structural forward with random init weights (no real load) — verifies
    # the ViT wiring produces the spec'd (B, 1370, 1024) token shape.
    cfg = ShapeConfig()
    model = DINOv2Conditioner(cfg)
    x = mx.random.normal((1, 3, cfg.dino_image_size, cfg.dino_image_size))
    out = model(x)
    mx.eval(out)
    assert out.shape == (1, cfg.dino_num_tokens, cfg.dino_hidden)
    # No NaN from random init + gelu + layernorm.
    assert mx.all(mx.isfinite(out)).item()


@pytest.mark.real_model
def test_dinov2_real_weight_smoke():
    # Real-weight smoke: load conditioner.safetensors, forward a dummy image,
    # verify output shape + finite values (Session 1 acceptance gate).
    if not _HUNYUAN3D_DIR.exists():
        pytest.skip("Hunyuan3D-2.1-MLX-Serve-8bit weights not downloaded")
    cfg = load_shape_config(_HUNYUAN3D_DIR)
    from fusion_mlx.threed.dinov2 import load_dinov2_conditioner

    model = load_dinov2_conditioner(
        str(_HUNYUAN3D_DIR / "conditioner.safetensors"), cfg
    )
    x = mx.random.normal((1, 3, 518, 518), dtype=mx.float32) * 0.5
    out = model(x)
    mx.eval(out)
    assert out.shape == (1, 1370, 1024)
    assert mx.all(mx.isfinite(out)).item()
    # Real DINOv2 output has nonzero std (loaded weights, not zeros).
    assert float(out.std()) > 0.1
