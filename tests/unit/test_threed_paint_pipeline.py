# SPDX-License-Identifier: Apache-2.0
# #989 Hunyuan3D-2.1 paint — Session 4 pipeline orchestration tests.
from __future__ import annotations

import os
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

_HUNYUAN3D_DIR = Path(
    os.path.expanduser("~/.fusion-mlx/models/Hunyuan3D-2.1-MLX-Serve-8bit")
)
_PAINT_DIR = _HUNYUAN3D_DIR / "paint"
_REAL = all(
    (_PAINT_DIR / f).exists()
    for f in ("dino.safetensors", "unet.safetensors", "vae.safetensors")
)

real_model = pytest.mark.skipif(
    not _REAL, reason="needs paint/{dino,unet,vae}.safetensors"
)


def test_pipeline_import():
    from fusion_mlx.threed.paint.pipeline import PaintPipeline, load_paint_pipeline

    assert PaintPipeline is not None
    assert load_paint_pipeline is not None


@real_model
def test_pipeline_e2e_denoise():
    from fusion_mlx.threed.paint.pipeline import load_paint_pipeline

    p = load_paint_pipeline(str(_HUNYUAN3D_DIR))
    ref = mx.random.uniform(0, 1, (1, 3, 518, 518), dtype=mx.float32)
    img = p.denoise(ref, steps=4)
    mx.eval(img)
    assert img.shape == (1, 3, 512, 512)
    arr = np.asarray(img)
    assert bool(np.all(np.isfinite(arr)))
    assert float(arr.std()) > 0.05  # real denoise produces signal


@real_model
def test_pipeline_context_build():
    from fusion_mlx.threed.paint.pipeline import load_paint_pipeline

    p = load_paint_pipeline(str(_HUNYUAN3D_DIR))
    ref = mx.random.uniform(0, 1, (1, 3, 518, 518), dtype=mx.float32)
    ctx_text, ctx_dino = p._build_context(ref)
    mx.eval(ctx_text, ctx_dino)
    assert ctx_text.shape == (1, 77, 1024)
    assert ctx_dino.shape == (1, 4, 1024)  # image_proj_model_dino: 1536 -> 4x1024
