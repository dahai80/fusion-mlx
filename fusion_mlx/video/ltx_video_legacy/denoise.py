# SPDX-License-Identifier: Apache-2.0
# Pure-MLX denoise loop for LTX-Video 0.9.x, ported from
# ltx_video/pipelines/pipeline_ltx_video.py (MIT). Simplified T2V path:
# classifier-free guidance + Euler step only. Dropped APG/STG/cfg_star_rescale,
# image conditioning, and the learned-sigma chunk (out_channels//2 != in_channels).

import logging

import mlx.core as mx

logger = logging.getLogger(__name__)


def denoise(
    transformer,
    scheduler,
    latents,
    pixel_coords,
    prompt_embeds,
    prompt_attn_mask,
    negative_embeds,
    negative_attn_mask,
    guidance_scale,
    num_inference_steps,
    frame_rate,
    latent_shape,
    dtype=mx.float32,
):
    # CFG batch order is [negative, prompt] (matches the reference cat); the
    # 3rd STG copy is intentionally dropped.
    do_cfg = float(guidance_scale) > 1.0
    if do_cfg:
        embeds = mx.concatenate([negative_embeds, prompt_embeds], axis=0)
        masks = mx.concatenate([negative_attn_mask, prompt_attn_mask], axis=0)
        num_conds = 2
    else:
        embeds = prompt_embeds
        masks = prompt_attn_mask
        num_conds = 1

    scheduler.set_timesteps(num_inference_steps, samples_shape=latent_shape)
    timesteps = scheduler.timesteps
    n_steps = timesteps.shape[0]

    # indices_grid: (num_conds, 3, n) float; temporal axis divided by frame_rate
    # before being fed to the transformer (reference: fractional_coords[:,0] *= 1/fps).
    frac = mx.concatenate([pixel_coords] * num_conds, axis=0).astype(mx.float32)
    temporal_scale = mx.array([1.0 / float(frame_rate), 1.0, 1.0], dtype=mx.float32)
    frac = frac * temporal_scale[None, :, None]

    logger.info(
        "denoise: start steps=%d cfg=%s guidance=%.2f tokens=%d",
        n_steps,
        do_cfg,
        float(guidance_scale),
        pixel_coords.shape[2],
    )

    for i, t in enumerate(timesteps.tolist()):
        t = float(t)
        latent_in = mx.concatenate([latents, latents], axis=0) if do_cfg else latents
        latent_in = scheduler.scale_model_input(latent_in, t)

        timestep = mx.full((num_conds, 1), t, dtype=dtype)

        noise_pred = transformer(
            latent_in.astype(dtype),
            indices_grid=frac,
            encoder_hidden_states=embeds.astype(dtype),
            encoder_attention_mask=masks,
            timestep=timestep,
        )

        if do_cfg:
            uncond, text = mx.split(noise_pred, 2, axis=0)
            noise_pred = uncond + float(guidance_scale) * (text - uncond)

        latents = scheduler.step(noise_pred, t, latents)
        logger.info("denoise: step=%d/%d t=%.4f", i + 1, n_steps, t)

    logger.info("denoise: done steps=%d", n_steps)
    return latents
