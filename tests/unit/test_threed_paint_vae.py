# SPDX-License-Identifier: Apache-2.0
# #989 Hunyuan3D-2.1 paint — Session 4 VAE decoder tests.
from __future__ import annotations

import os
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from fusion_mlx.threed.config import PaintVAEConfig
from fusion_mlx.threed.paint.vae import PaintVAEDecoder

_HUNYUAN3D_DIR = Path(
    os.path.expanduser("~/.fusion-mlx/models/Hunyuan3D-2.1-MLX-Serve-8bit")
)
_VAE_PATH = _HUNYUAN3D_DIR / "paint" / "vae.safetensors"
_REAL = _VAE_PATH.exists()

real_model = pytest.mark.skipif(
    not _REAL,
    reason="needs paint/vae.safetensors (download Hunyuan3D-2.1-MLX-Serve-8bit)",
)


def test_vae_config_defaults():
    cfg = PaintVAEConfig()
    assert cfg.latent_channels == 4
    assert cfg.out_channels == 3
    assert cfg.block_out_channels == [128, 256, 512, 512]
    assert abs(cfg.scaling_factor - 0.18215) < 1e-6


def test_vae_structural_decode_shapes():
    cfg = PaintVAEConfig()
    net = PaintVAEDecoder(cfg)
    # latent (1,4,64,64) -> RGB (1,3,512,512) (8x upscale, 3 upsamplers).
    lat = mx.random.normal((1, 4, 64, 64), dtype=mx.float16) * 0.3
    o = net(lat)
    mx.eval(o)
    assert o.shape == (1, 3, 512, 512)
    assert bool(mx.all(mx.isfinite(o)).item())


def test_vae_upscale_factor_varied_res():
    cfg = PaintVAEConfig()
    net = PaintVAEDecoder(cfg)
    for h in (8, 16, 32):
        lat = mx.zeros((1, 4, h, h), dtype=mx.float16)
        o = net(lat)
        mx.eval(o)
        assert o.shape == (1, 3, h * 8, h * 8)


def test_vae_mid_block_has_attention():
    cfg = PaintVAEConfig()
    net = PaintVAEDecoder(cfg)
    assert len(net.mid_block.attentions) == 1
    assert hasattr(net.mid_block.attentions[0], "query")
    assert hasattr(net.mid_block.attentions[0], "proj_attn")


def test_vae_up_block_structure():
    cfg = PaintVAEConfig()
    net = PaintVAEDecoder(cfg)
    # 3 resnets per up block; up0/1/2 have upsampler, up3 none.
    assert len(net.up_blocks[0].resnets) == cfg.layers_per_block + 1
    assert len(net.up_blocks[0].upsamplers) == 1
    assert len(net.up_blocks[3].upsamplers) == 0
    rch = list(reversed(cfg.block_out_channels))
    assert net.up_blocks[3].out_c == rch[3]  # 128 -> conv_out


@real_model
def test_vae_real_load_clean():
    cfg = PaintVAEConfig()
    from fusion_mlx.threed.paint.vae import load_paint_vae

    net = load_paint_vae(str(_VAE_PATH), cfg)
    assert net is not None


@real_model
def test_vae_real_decode_64():
    cfg = PaintVAEConfig()
    from fusion_mlx.threed.paint.vae import load_paint_vae

    net = load_paint_vae(str(_VAE_PATH), cfg)
    lat = mx.random.normal((1, 4, 64, 64), dtype=mx.float16) * 0.3
    o = net(lat)
    mx.eval(o)
    assert o.shape == (1, 3, 512, 512)
    arr = np.asarray(o)
    assert bool(np.all(np.isfinite(arr)))
    assert float(arr.std()) > 0.01  # real weights produce signal, not zeros
