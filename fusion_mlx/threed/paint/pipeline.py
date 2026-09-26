# SPDX-License-Identifier: Apache-2.0
# Hunyuan3D-2.1 paint pipeline orchestration (issue #989 Session 4).
# Wires DINOv2-Giant conditioner -> paint UNet (image_proj_model_dino +
# learned_text_clip embeddings) -> DDIM v_prediction scheduler denoise loop
# -> paint VAE decoder -> multiview texture images.
#
# Pipeline stages:
#   1. DINOv2 encodes the reference image -> (1, 1370, 1536). The patch-token
#      mean (drop cls) is pooled -> (1, 1536) -> unet.image_proj_model_dino ->
#      ctx_dino (1, 4, 1024).
#   2. ctx_text = learned_text_clip_albedo (77, 1024) broadcast to batch.
#   3. Latent (B, 12, 64, 64) noise; DDIM trailing timesteps (30 steps),
#      v_prediction, CFG (guidance_scale 3.0): cond = full context, uncond =
#      null dino/text. UNet predicts v; scheduler.step -> prev latent.
#   4. Final latent (B, 4, 64, 64) -> VAE decode -> RGB (B, 3, 512, 512).
#
# KNOWN LIMITATION: real multiview cross-view attention (ctx_mv / ctx_ref per
# block) and the mr (metallic-roughness) branch use zeros fallback — full
# multiview + dual-branch PBR correctness needs the tencent paint reference
# (not in the mlx-serve zig source). The diffusion mechanics (DINOv2 cond,
# DDIM v_pred loop, VAE decode) are verified end-to-end with real weights.
from __future__ import annotations

import logging

import mlx.core as mx
import numpy as np

from fusion_mlx.threed.config import PaintConfig
from fusion_mlx.threed.paint.dino import load_paint_dinov2
from fusion_mlx.threed.paint.scheduler import DDIMScheduler, make_scheduler_from_config
from fusion_mlx.threed.paint.unet import load_paint_unet

logger = logging.getLogger(__name__)


class PaintPipeline:
    # Holds all 4 paint components + scheduler. Loads weights lazily on first
    # use (heavy: ~5GB dequanted). Single reference-image -> texture image.

    def __init__(self, model_dir: str, cfg: PaintConfig | None = None):
        from pathlib import Path

        self.model_dir = Path(model_dir)
        self.paint_dir = self.model_dir / "paint"
        if cfg is None:
            from fusion_mlx.threed.config import load_paint_config

            cfg = load_paint_config(self.model_dir)
        self.cfg = cfg
        self._dino = None
        self._unet = None
        self._vae = None
        self._scheduler: DDIMScheduler | None = None

    @property
    def dino(self):
        if self._dino is None:
            self._dino = load_paint_dinov2(
                str(self.paint_dir / "dino.safetensors"), self.cfg.dino
            )
        return self._dino

    @property
    def unet(self):
        if self._unet is None:
            self._unet = load_paint_unet(
                str(self.paint_dir / "unet.safetensors"), self.cfg.unet
            )
        return self._unet

    @property
    def vae(self):
        if self._vae is None:
            from fusion_mlx.threed.paint.vae import load_paint_vae

            self._vae = load_paint_vae(
                str(self.paint_dir / "vae.safetensors"), self.cfg.vae
            )
        return self._vae

    @property
    def scheduler(self) -> DDIMScheduler:
        if self._scheduler is None:
            sched_cfg = {
                "beta_start": 0.00085,
                "beta_end": 0.012,
                "beta_schedule": "scaled_linear",
                "num_train_timesteps": 1000,
                "set_alpha_to_one": True,
                "steps_offset": 1,
                "timestep_spacing": "trailing",
                "rescale_betas_zero_snr": True,
                "clip_sample": False,
            }
            self._scheduler = make_scheduler_from_config(sched_cfg)
        return self._scheduler

    def _build_context(self, ref_image: mx.array) -> tuple[mx.array, mx.array]:
        # ref_image (B, 3, 518, 518) ImageNet-normalized [0,1].
        # DINOv2 -> (B, 1370, 1536); pool patch tokens (drop cls) -> (B, 1536).
        dino_out = self.dino(ref_image)
        dino_pool = dino_out[:, 1:, :].mean(axis=1)  # (B, 1536)
        ctx_dino = self.unet.image_proj_model_dino(dino_pool)  # (B, 4, 1024)
        B = ref_image.shape[0]
        ctx_text = mx.broadcast_to(
            self.unet.learned_text_clip_albedo.astype(mx.float16), (B, 77, 1024)
        )
        return ctx_text, ctx_dino

    def denoise(
        self,
        ref_image: mx.array,
        steps: int | None = None,
        guidance_scale: float | None = None,
        seed: int = 0,
    ) -> mx.array:
        # ref_image (B, 3, 518, 518). Returns texture RGB (B, 3, 512, 512) in [-1,1].
        steps = steps or self.cfg.num_inference_steps
        guidance_scale = (
            guidance_scale if guidance_scale is not None else self.cfg.guidance_scale
        )
        B = ref_image.shape[0]
        cfg_u = self.cfg.unet
        H = W = cfg_u.sample_size  # 64

        mx.random.seed(seed)
        # The diffusion target is the 4ch albedo latent (vae.latent_channels).
        # UNet in_channels=12 = [albedo_noisy(4) + mr_cond(4) + ref_cond(4)];
        # the 8 cond channels are the mr/ref conditioning (placeholder zeros —
        # real multiview/mr wiring needs the tencent paint reference).
        lat = cfg_u.out_channels  # 4
        latent = mx.random.normal((B, lat, H, W), dtype=mx.float16)
        cond_zero = mx.zeros((B, cfg_u.in_channels - lat, H, W), dtype=mx.float16)

        ctx_text, ctx_dino = self._build_context(ref_image)
        # uncond context for CFG: zero dino + zero text (null prompt).
        ctx_text_uncond = mx.zeros_like(ctx_text)
        ctx_dino_uncond = mx.zeros_like(ctx_dino)

        sched = self.scheduler
        timesteps = sched.set_timesteps(steps)
        logger.info(
            "PaintPipeline denoise: %d steps, cfg=%.2f, latent=%s",
            steps,
            guidance_scale,
            tuple(latent.shape),
        )
        for i, t in enumerate(timesteps):
            t_int = int(t)
            # UNet input = [noisy albedo(4) + cond zeros(8)] -> 12ch.
            unet_in = mx.concatenate([latent, cond_zero], axis=1)
            # CFG: run cond + uncond, blend.
            v_cond = self.unet(unet_in, t_int, ctx_text, ctx_dino)
            if guidance_scale != 1.0:
                v_uncond = self.unet(unet_in, t_int, ctx_text_uncond, ctx_dino_uncond)
                v_pred = v_uncond + guidance_scale * (v_cond - v_uncond)
            else:
                v_pred = v_cond
            latent_np = np.asarray(latent)
            v_np = np.asarray(v_pred).astype(np.float32)
            prev = sched.step(v_np, t_int, latent_np.astype(np.float32))
            latent = mx.array(prev).astype(mx.float16)
            if (i + 1) % max(1, steps // 6) == 0:
                logger.info("  step %d/%d t=%d", i + 1, steps, t_int)

        # Final 4ch albedo latent -> VAE decode -> RGB.
        img = self.vae(latent)
        logger.info("PaintPipeline done: texture %s", tuple(img.shape))
        return img


def load_paint_pipeline(
    model_dir: str, cfg: PaintConfig | None = None
) -> PaintPipeline:
    return PaintPipeline(model_dir, cfg)
