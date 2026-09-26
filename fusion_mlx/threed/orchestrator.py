# SPDX-License-Identifier: Apache-2.0
# Hunyuan3D-2.1 full 3D generation orchestrator (issue #989 Session 5).
# Chains the S1-S4 ports end-to-end: reference image -> DINOv2-Large shape
# conditioner -> flow-match MoE DiT denoiser -> ShapeVAE + marching cubes ->
# mesh -> paint multiview rasterizer + DINOv2-Giant/UNet/VAE texture diffusion
# -> xatlas UV unwrap + bake-back -> textured GLB.
#
# Heavy: loads ~8GB of dequanted weights (shape dino + DiT + shape VAE + paint
# dino + paint unet + paint vae). All components verified individually with
# real weights in S1-S4; this orchestrator wires them into one e2e path.
#
# KNOWN LIMITATIONS (paint multiview):
#   - The paint UNet's real multiview cross-view attention (ctx_mv/ctx_ref) and
#     the mr (metallic-roughness) branch use zeros fallback. Full multiview +
#     dual-branch PBR correctness needs the tencent paint reference.
#   - For bake-back, a single paint texture is applied across all 6 raster
#     views (real multiview diffusion generates 6 distinct view textures).
#     The textured GLB is structurally complete; texture quality is
#     view-averaged rather than per-view distinct.
from __future__ import annotations

import logging
from pathlib import Path

import mlx.core as mx
import numpy as np

from fusion_mlx.threed.config import (
    load_paint_config,
    load_shape_config,
)
from fusion_mlx.threed.dinov2 import load_dinov2_conditioner
from fusion_mlx.threed.dit import denoise as dit_denoise
from fusion_mlx.threed.dit import load_dit
from fusion_mlx.threed.marching_cubes import extract as mc_extract
from fusion_mlx.threed.paint.bake import (
    bake_to_atlas,
    export_glb,
    rasterize_views,
    uv_unwrap,
)
from fusion_mlx.threed.shape_vae import decode_volume, load_shape_vae

logger = logging.getLogger(__name__)

_DEFAULT_VIEWS = (
    [0, 90, 180, 270, 0, 180],
    [0, 0, 0, 0, 90, -90],
)


class ThreeDOrchestrator:
    # Full 3D generation: ref image -> textured GLB. Lazily loads each stage's
    # weights on first use so the orchestrator is cheap to construct.

    def __init__(self, model_dir: str):
        self.model_dir = Path(model_dir)
        self.shape_cfg = load_shape_config(self.model_dir)
        self.paint_cfg = load_paint_config(self.model_dir)
        self._shape_dino = None
        self._dit = None
        self._shape_vae = None
        self._paint = None

    # --- shape stages ---

    @property
    def shape_dino(self):
        if self._shape_dino is None:
            p = self.model_dir / "conditioner.safetensors"
            self._shape_dino = load_dinov2_conditioner(str(p), self.shape_cfg)
        return self._shape_dino

    @property
    def dit(self):
        if self._dit is None:
            p = self.model_dir / "dit.safetensors"
            self._dit = load_dit(str(p), self.shape_cfg)
        return self._dit

    @property
    def shape_vae(self):
        if self._shape_vae is None:
            p = self.model_dir / "vae.safetensors"
            self._shape_vae = load_shape_vae(str(p), self.shape_cfg)
        return self._shape_vae

    def _shape_context(self, ref_image: mx.array) -> mx.array:
        # DINOv2-Large -> (1, 1370, 1024) cross-attn context for the DiT.
        return self.shape_dino(ref_image)

    def generate_mesh(
        self,
        ref_image: mx.array,
        steps: int = 50,
        guidance: float = 5.0,
        grid_res: int = 128,
        seed: int = 0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # ref_image (1,3,518,518) -> mesh (verts, normals, faces).
        cond = self._shape_context(ref_image)
        logger.info("ThreeD shape: DINOv2 cond %s", tuple(cond.shape))
        latent = dit_denoise(self.dit, cond, steps=steps, guidance=guidance, seed=seed)
        logger.info("ThreeD shape: DiT latent %s", tuple(latent.shape))
        grid = decode_volume(self.shape_vae, latent, res=grid_res)
        logger.info("ThreeD shape: SDF grid %s", grid.shape)
        verts, normals, faces = mc_extract(grid)
        logger.info("ThreeD mesh: %d verts, %d tris", len(verts), len(faces))
        return verts, normals, faces

    # --- paint stage ---

    @property
    def paint(self):
        if self._paint is None:
            from fusion_mlx.threed.paint.pipeline import load_paint_pipeline

            self._paint = load_paint_pipeline(str(self.model_dir), self.paint_cfg)
        return self._paint

    def generate_textured_glb(
        self,
        ref_image: mx.array,
        out_path: str,
        shape_steps: int = 50,
        shape_guidance: float = 5.0,
        grid_res: int = 128,
        paint_steps: int = 30,
        paint_guidance: float | None = None,
        raster_res: int = 256,
        atlas_res: int = 512,
        seed: int = 0,
    ) -> str:
        # Full e2e: ref image -> mesh -> textured GLB at out_path.
        verts, normals, faces = self.generate_mesh(
            ref_image,
            steps=shape_steps,
            guidance=shape_guidance,
            grid_res=grid_res,
            seed=seed,
        )
        # rasterize 6 views for bake-back.
        azims, elevs = _DEFAULT_VIEWS
        raster = rasterize_views(verts, faces, normals, azims, elevs, res=raster_res)
        # paint diffusion -> single texture (applied across all 6 views;
        # real multiview generates 6 distinct view textures — TODO reference).
        texture = self.paint.denoise(
            ref_image, steps=paint_steps, guidance_scale=paint_guidance, seed=seed
        )
        tex_np = np.asarray(texture)
        # broadcast single texture to 6 view textures at raster_res.
        import mlx.nn as nn  # noqa: F401

        # downsample texture (512) -> raster_res via simple stride/mean.
        stride = 512 // raster_res
        if stride < 1:
            stride = 1
        tex_small = tex_np[0, ::stride, ::stride, :3]  # (raster_res,_,3)
        # ensure exact raster_res size
        h = min(tex_small.shape[0], raster_res)
        tex_small = tex_small[:raster_res, :raster_res]
        if tex_small.shape[0] < raster_res or tex_small.shape[1] < raster_res:
            pad_h = raster_res - tex_small.shape[0]
            pad_w = raster_res - tex_small.shape[1]
            tex_small = np.pad(tex_small, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
        view_textures = np.broadcast_to(
            tex_small[None], (6, raster_res, raster_res, 3)
        ).copy()
        # uv unwrap + bake + export.
        nv, nf, uv, vremap = uv_unwrap(verts, faces)
        atlas = bake_to_atlas(
            nv,
            nf,
            uv,
            vremap,
            verts,
            normals,
            view_textures,
            raster,
            atlas_res=atlas_res,
        )
        export_glb(nv, nf, uv, atlas, out_path, texture_in_unit=False)
        logger.info("ThreeD e2e complete: %s", out_path)
        return out_path


def load_threed_orchestrator(model_dir: str) -> ThreeDOrchestrator:
    return ThreeDOrchestrator(model_dir)
