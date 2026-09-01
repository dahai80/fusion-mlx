# SPDX-License-Identifier: Apache-2.0
# LTX-2.5 distilled 去噪 (T2V-only path)。
# 与 ltx2 denoise_distilled 的差异: 使用 ltx2_5.Modality (跨模块 Modality 类不兼容,
# 见 GOTCHA: ltx2_5.Modality is not ltx2.Modality) 并调 LTX2_5Model。
# audio / I2V conditioning (state) 不在本轮 T2V 路径, 留空 fail visible。
from __future__ import annotations

import logging

import mlx.core as mx

from fusion_mlx.engines.video_backends._inpaint import apply_inpaint_mask

from .transformer import Modality

logger = logging.getLogger(__name__)


def denoise_distilled_t2v(
    latents: mx.array,
    positions: mx.array,
    text_embeddings: mx.array,
    transformer,
    sigmas: list,
    verbose: bool = True,
    controlnet_image=None,
    inpaint_mask=None,
    init_latent=None,
) -> mx.array:
    # 两阶段 distilled T2V 去噪。latents (b,c,f,h,w), sigmas 降序 -> 0。
    # 每步: 展平 latent -> Modality(context=text_embeddings) -> transformer ->
    # velocity -> x0 = latent - sigma*velocity -> 重新加噪到 sigma_next。
    # #735 Surface B: ControlNet not fabricatable for ltx2_5 (shared adapter is
    # Wan2-arch, no per-backend model). Fail visible — refuse silent T2V degrade.
    # #735 Surface C: DiT-agnostic latent-space inpaint re-composite after each
    # step's x0 prediction, so frozen regions stay frozen across re-noise steps.
    if controlnet_image is not None:
        raise RuntimeError(
            "ltx2_5: ControlNet (Surface B) not available for this backend — "
            "no per-backend ControlNet model (see issue #735 follow-up). "
            "Refusing to silently degrade to T2V (#735)."
        )
    dtype = latents.dtype
    latents = latents.astype(mx.float32)
    num_steps = len(sigmas) - 1
    if verbose:
        logger.info("Denoising T2V: %d steps", num_steps)
    logger.info(
        "ltx2_5 denoise: inpaint=%s controlnet=%s",
        inpaint_mask is not None,
        controlnet_image is not None,
    )

    for i in range(num_steps):
        sigma, sigma_next = sigmas[i], sigmas[i + 1]

        b, c, f, h, w = latents.shape
        num_tokens = f * h * w
        latents_flat = mx.transpose(mx.reshape(latents, (b, c, -1)), (0, 2, 1)).astype(
            dtype
        )

        timesteps = mx.full((b, num_tokens), sigma, dtype=dtype)

        video_modality = Modality(
            latent=latents_flat,
            timesteps=timesteps,
            positions=positions,
            context=text_embeddings,
            context_mask=None,
            enabled=True,
            sigma=mx.full((b,), sigma, dtype=dtype),
        )

        velocity, _audio_velocity = transformer(video=video_modality, audio=None)
        mx.eval(velocity)

        sigma_f32 = mx.array(sigma, dtype=mx.float32)
        latents_flat_f32 = mx.transpose(mx.reshape(latents, (b, c, -1)), (0, 2, 1))
        timesteps_f32 = mx.expand_dims(timesteps.astype(mx.float32), axis=-1)
        x0_f32 = latents_flat_f32 - timesteps_f32 * velocity.astype(mx.float32)
        denoised = mx.reshape(mx.transpose(x0_f32, (0, 2, 1)), (b, c, f, h, w))

        mx.eval(denoised)

        if sigma_next > 0:
            sigma_next_f32 = mx.array(sigma_next, dtype=mx.float32)
            latents = denoised + sigma_next_f32 * (latents - denoised) / sigma_f32
        else:
            latents = denoised

        mx.eval(latents)
        if inpaint_mask is not None and init_latent is not None:
            latents = apply_inpaint_mask(latents, init_latent, inpaint_mask)
            mx.eval(latents)
        if verbose:
            logger.info("step %d/%d", i + 1, num_steps)

    return latents.astype(mx.float32)


__all__ = ["denoise_distilled_t2v"]
