# SPDX-License-Identifier: Apache-2.0
# #989 Hunyuan3D-2.1 MLX port — Session 5 orchestrator + /v1/3d/generate tests.
from __future__ import annotations

import os
from pathlib import Path

import pytest

_HUNYUAN3D_DIR = Path(
    os.path.expanduser("~/.fusion-mlx/models/Hunyuan3D-2.1-MLX-Serve-8bit")
)
_REAL = (
    _HUNYUAN3D_DIR.exists()
    and (_HUNYUAN3D_DIR / "conditioner.safetensors").exists()
    and (_HUNYUAN3D_DIR / "dit.safetensors").exists()
    and (_HUNYUAN3D_DIR / "vae.safetensors").exists()
    and (_HUNYUAN3D_DIR / "paint").exists()
)

real_model = pytest.mark.skipif(not _REAL, reason="needs Hunyuan3D-2.1 full weights")


def test_orchestrator_import():
    from fusion_mlx.threed.orchestrator import (
        ThreeDOrchestrator,
        load_threed_orchestrator,
    )

    assert ThreeDOrchestrator is not None
    assert load_threed_orchestrator is not None


def test_orchestrator_lazy_props():
    # Construct is cheap — all stages lazily loaded on first access.
    from fusion_mlx.threed.orchestrator import ThreeDOrchestrator

    orch = ThreeDOrchestrator(str(_HUNYUAN3D_DIR))
    assert orch._shape_dino is None
    assert orch._dit is None
    assert orch._shape_vae is None
    assert orch._paint is None
    assert orch.shape_cfg is not None
    assert orch.paint_cfg is not None


def test_route_import():
    from fusion_mlx.api.threed_routes import (
        ThreeDGenerateRequest,
        router,
        set_threed_context,
    )

    assert router is not None
    assert set_threed_context is not None
    # pydantic models accept all fields.
    req = ThreeDGenerateRequest(image="data:,")
    assert req.shape_steps == 50
    assert req.output_format == "glb"


@real_model
def test_orchestrator_e2e_glb(tmp_path):
    # Full e2e: ref image -> textured GLB on disk. Slow (shape + paint).
    import mlx.core as mx

    from fusion_mlx.threed.orchestrator import load_threed_orchestrator

    orch = load_threed_orchestrator(str(_HUNYUAN3D_DIR))
    # Structured synthetic image (dark ellipse on white) — DINOv2 needs a
    # real-ish shape; randn noise decodes to an empty (all-negative) SDF grid.
    import numpy as np

    px = np.full((518, 518, 3), 255, np.uint8)
    yy, xx = np.mgrid[0:518, 0:518]
    mask = ((yy - 259) / 140) ** 2 + ((xx - 259) / 100) ** 2 < 1
    px[mask] = [60, 100, 160]
    arr = px.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)
    arr = (arr - mean) / std
    ref = mx.array(arr.transpose(2, 0, 1))[None]
    out = str(tmp_path / "out.glb")
    orch.generate_textured_glb(
        ref,
        out,
        shape_steps=20,
        paint_steps=8,
        grid_res=64,
        raster_res=64,
        atlas_res=128,
    )
    assert os.path.exists(out)
    assert os.path.getsize(out) > 1000
    trimesh = pytest.importorskip("trimesh")
    m = trimesh.load(out)
    assert m is not None
    # GLB may load as Scene (multi-geometry) or Trimesh; count total verts.
    if hasattr(m, "geometry"):
        nverts = sum(
            len(g.vertices) for g in m.geometry.values() if hasattr(g, "vertices")
        )
    else:
        nverts = len(m.vertices) if hasattr(m, "vertices") else 0
    assert nverts > 0
