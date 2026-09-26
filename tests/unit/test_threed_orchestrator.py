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
    and (_HUNYUAN3D_DIR / "diffusion.safetensors").exists()
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
    # small 318x318 random RGB normalized.
    import numpy as np

    arr = np.random.randn(1, 3, 518, 518).astype(np.float32) * 0.5
    ref = mx.array(arr)
    out = str(tmp_path / "out.glb")
    orch.generate_textured_glb(
        ref,
        out,
        shape_steps=4,
        paint_steps=4,
        grid_res=64,
        raster_res=64,
        atlas_res=128,
    )
    assert os.path.exists(out)
    assert os.path.getsize(out) > 1000
    trimesh = pytest.importorskip("trimesh")
    m = trimesh.load(out)
    assert m is not None
    assert len(m.vertices) > 0
