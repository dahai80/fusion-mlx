# SPDX-License-Identifier: Apache-2.0
# Pure-MLX denoise loop for LTX-Video 0.9.x, ported from
# ltx_video/pipelines/pipeline_ltx_video.py (MIT). Simplified path:
# classifier-free guidance + Euler step only. Dropped APG/STG/cfg_star_rescale
# and the learned-sigma chunk (out_channels//2 != in_channels).
# Image conditioning (I2V) supported via conditioning_latent + noise_mask.

import logging
from collections.abc import Callable

import mlx.core as mx
import numpy as np

from fusion_mlx.engines.video_backends._inpaint import apply_inpaint_mask

logger = logging.getLogger(__name__)


def _to_np(arr):
    return np.asarray(arr.astype(mx.float32))


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
    on_step_sync: Callable[[int, int], None] | None = None,
    conditioning_latent=None,
    noise_mask=None,
    inpaint_mask=None,
    init_latent=None,
):
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
    logger.info("ltx-legacy denoise: inpaint=%s", inpaint_mask is not None)

    mx.eval(prompt_embeds, negative_embeds, latents)
    pe_np = _to_np(prompt_embeds)
    ne_np = _to_np(negative_embeds)
    la_np = _to_np(latents)
    logger.info(
        "denoise: input check prompt_embeds range=[%.4f,%.4f] nan=%d/%d",
        float(np.nanmin(pe_np)),
        float(np.nanmax(pe_np)),
        int(np.isnan(pe_np).sum()),
        pe_np.size,
    )
    logger.info(
        "denoise: input check neg_embeds range=[%.4f,%.4f] nan=%d/%d",
        float(np.nanmin(ne_np)),
        float(np.nanmax(ne_np)),
        int(np.isnan(ne_np).sum()),
        ne_np.size,
    )
    logger.info(
        "denoise: input check latents range=[%.4f,%.4f] nan=%d/%d",
        float(np.nanmin(la_np)),
        float(np.nanmax(la_np)),
        int(np.isnan(la_np).sum()),
        la_np.size,
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

        if i == 0:
            mx.eval(noise_pred)
            np_pred = _to_np(noise_pred)
            logger.info(
                "denoise: step=1 noise_pred range=[%.4f,%.4f] nan=%d/%d",
                float(np.nanmin(np_pred)),
                float(np.nanmax(np_pred)),
                int(np.isnan(np_pred).sum()),
                np_pred.size,
            )

        if do_cfg:
            uncond, text = mx.split(noise_pred, 2, axis=0)
            noise_pred = uncond + float(guidance_scale) * (text - uncond)

        latents = scheduler.step(noise_pred, t, latents)

        if conditioning_latent is not None and noise_mask is not None:
            latents = conditioning_latent * (1.0 - noise_mask) + latents * noise_mask

        mx.eval(latents)
        if inpaint_mask is not None and init_latent is not None:
            latents = apply_inpaint_mask(latents, init_latent, inpaint_mask)
            mx.eval(latents)
        latents_np = _to_np(latents)
        nan_count = int(np.isnan(latents_np).sum())
        if nan_count > 0:
            logger.warning(
                "denoise: step=%d/%d t=%.4f NaN detected: %d/%d values",
                i + 1,
                n_steps,
                t,
                nan_count,
                latents_np.size,
            )

        logger.info(
            "denoise: step=%d/%d t=%.4f range=[%.4f,%.4f]",
            i + 1,
            n_steps,
            t,
            float(np.nanmin(latents_np)),
            float(np.nanmax(latents_np)),
        )
        if on_step_sync is not None:
            on_step_sync(i + 1, n_steps)

    logger.info("denoise: done steps=%d", n_steps)
    return latents
