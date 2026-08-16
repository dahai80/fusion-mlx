# SPDX-License-Identifier: Apache-2.0
# LTX-2.5 distilled 去噪 (T2V-only path)。
# 与 ltx2 denoise_distilled 的差异: 使用 ltx2_5.Modality (跨模块 Modality 类不兼容,
# 见 GOTCHA: ltx2_5.Modality is not ltx2.Modality) 并调 LTX2_5Model。
# audio / I2V conditioning (state) 不在本轮 T2V 路径, 留空 fail visible。
from __future__ import annotations

import logging

import mlx.core as mx

from .transformer import Modality

logger = logging.getLogger(__name__)


def denoise_distilled_t2v(
    latents: mx.array,
    positions: mx.array,
    text_embeddings: mx.array,
    transformer,
    sigmas: list,
    verbose: bool = True,
) -> mx.array:
    # 两阶段 distilled T2V 去噪。latents (b,c,f,h,w), sigmas 降序 -> 0。
    # 每步: 展平 latent -> Modality(context=text_embeddings) -> transformer ->
    # velocity -> x0 = latent - sigma*velocity -> 重新加噪到 sigma_next。
    dtype = latents.dtype
    latents = latents.astype(mx.float32)
    num_steps = len(sigmas) - 1
    if verbose:
        logger.info("Denoising T2V: %d steps", num_steps)

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
        if verbose:
            logger.info("step %d/%d", i + 1, num_steps)

    return latents.astype(mx.float32)


__all__ = ["denoise_distilled_t2v"]
