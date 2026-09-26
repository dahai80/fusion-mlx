# SPDX-License-Identifier: Apache-2.0
"""Hunyuan3D-2.1 3D generation API routes. POST /v1/3d/generate.

Accepts a reference image (base64 data URL or http(s) URL) + generation
params, runs the full ThreeDOrchestrator (shape DiT + ShapeVAE + marching
cubes + paint diffusion + UV bake), and returns the textured GLB as base64.
"""

from __future__ import annotations

import base64
import io
import logging
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..middleware.auth import verify_api_key

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["3d"])

_HUNYUAN3D_SUBDIR = "Hunyuan3D-2.1-MLX-Serve-8bit"
_ORCHESTRATOR: Any = None
_MODEL_DIR: str | None = None


class ThreeDGenerateRequest(BaseModel):
    image: str = Field(..., description="reference image: data URL or http(s) URL")
    prompt: str = Field(default="", description="text prompt (logged, not yet wired)")
    shape_steps: int = 50
    shape_guidance: float = 5.0
    grid_res: int = 128
    paint_steps: int = 30
    paint_guidance: float | None = None
    raster_res: int = 256
    atlas_res: int = 512
    seed: int = 0
    output_format: str = Field(default="glb", description="glb (only format supported)")


class ThreeDGenerateResponse(BaseModel):
    object: str = "threed.glb"
    created: int
    model: str
    format: str
    glb_base64: str
    vertices: int
    faces: int
    bytes: int


def set_threed_context(pool: Any, server_state: Any) -> None:
    global _MODEL_DIR
    cfg = getattr(server_state, "config", None)
    if cfg is not None and getattr(cfg, "model_dir", None):
        _MODEL_DIR = str(cfg.model_dir)
    if _MODEL_DIR is None:
        from ..config import get_config

        _MODEL_DIR = get_config().model_dir


def _resolve_model_dir() -> str:
    global _MODEL_DIR
    if _MODEL_DIR is None:
        from ..config import get_config

        _MODEL_DIR = get_config().model_dir
    p = Path(_MODEL_DIR) / _HUNYUAN3D_SUBDIR
    if not p.exists():
        raise HTTPException(
            status_code=404,
            detail={
                "error": {
                    "message": f"Hunyuan3D-2.1 model not found at {p}. "
                    f"Download via HF_MIRROR=https://hf-mirror.com fusion-mlx pull.",
                    "type": "not_found",
                    "code": "model_not_found",
                    "param": None,
                }
            },
        )
    return str(p)


def _get_orchestrator(model_dir: str):
    global _ORCHESTRATOR
    if _ORCHESTRATOR is None or getattr(_ORCHESTRATOR, "model_dir", None) != Path(
        model_dir
    ):
        from ..threed.orchestrator import load_threed_orchestrator

        _ORCHESTRATOR = load_threed_orchestrator(model_dir)
    return _ORCHESTRATOR


def _load_image(image: str) -> Any:
    import numpy as np

    if image.startswith("http://") or image.startswith("https://"):
        import httpx

        r = httpx.get(image, timeout=30.0, follow_redirects=True)
        r.raise_for_status()
        raw = r.content
    elif image.startswith("data:"):
        raw = base64.b64decode(image.split(",", 1)[-1])
    else:
        raw = base64.b64decode(image)
    from PIL import Image

    img = Image.open(io.BytesIO(raw)).convert("RGB").resize((518, 518))
    arr = np.asarray(img, dtype=np.float32) / 255.0  # [0,1]
    # standardize: ImageNet mean/std (DINOv2 convention).
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    import mlx.core as mx

    return mx.array(arr.transpose(2, 0, 1))[None]  # (1,3,518,518)


@router.post("/3d/generate", response_model=ThreeDGenerateResponse)
async def generate_3d(
    req: ThreeDGenerateRequest, _auth: Any = Depends(verify_api_key)
) -> ThreeDGenerateResponse:
    import mlx.core as mx

    model_dir = _resolve_model_dir()
    orch = _get_orchestrator(model_dir)
    t0 = time.time()
    try:
        ref = _load_image(req.image)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": f"image load failed: {e}",
                    "type": "invalid_image",
                    "code": "bad_image",
                    "param": None,
                }
            },
        )
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as tf:
        out_path = tf.name
    try:
        orch.generate_textured_glb(
            ref,
            out_path,
            shape_steps=req.shape_steps,
            shape_guidance=req.shape_guidance,
            grid_res=req.grid_res,
            paint_steps=req.paint_steps,
            paint_guidance=req.paint_guidance,
            raster_res=req.raster_res,
            atlas_res=req.atlas_res,
            seed=req.seed,
        )
    except Exception as e:
        logger.exception("3D generation failed")
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "message": f"3D generation failed: {e}",
                    "type": "generation_error",
                    "code": "gen_failed",
                    "param": None,
                }
            },
        )
    mx.eval()
    glb_bytes = Path(out_path).read_bytes()
    Path(out_path).unlink(missing_ok=True)
    import trimesh

    m = trimesh.load(io.BytesIO(glb_bytes), file_type="glb")
    # GLB may load as a Scene (multi-geometry) or a Trimesh; sum children.
    if hasattr(m, "geometry"):
        n_verts = sum(
            int(len(g.vertices)) for g in m.geometry.values() if hasattr(g, "vertices")
        )
        n_faces = sum(
            int(len(g.faces)) for g in m.geometry.values() if hasattr(g, "faces")
        )
    elif hasattr(m, "vertices"):
        n_verts = int(len(m.vertices))
        n_faces = int(len(m.faces)) if hasattr(m, "faces") else 0
    else:
        n_verts = n_faces = 0
    logger.info(
        "/v1/3d/generate done in %.1fs (%d verts, %d tris, %d bytes)",
        time.time() - t0,
        n_verts,
        n_faces,
        len(glb_bytes),
    )
    return ThreeDGenerateResponse(
        created=int(time.time()),
        model=_HUNYUAN3D_SUBDIR,
        format="glb",
        glb_base64=base64.b64encode(glb_bytes).decode(),
        vertices=n_verts,
        faces=n_faces,
        bytes=len(glb_bytes),
    )
