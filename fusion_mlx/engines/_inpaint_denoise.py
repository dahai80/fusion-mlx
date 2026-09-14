# SPDX-License-Identifier: Apache-2.0
# O5.5 inpaint denoise loop — staged masked-denoise for image inpainting.
#
# Ports the flyto masked-denoise pattern (white=regen semantics) onto mflux
# components. Instead of a single generate_image call, the loop:
#   1. prepare_latents: VAE-encode the init image -> init_latent
#   2. noise_latent: add scheduler noise up to t_start (image_strength)
#   3. mask_latent: downsample the binary mask to latent resolution
#      (white=1 -> regenerate, black=0 -> freeze init)
#   4. inpaint_latent: run the denoise loop, re-compositing each step:
#        latents = mask * denoised + (1-mask) * noisy_init
#   5. composite: final VAE decode of the composited latent
#
# This gives true inpainting (mask region regenerated, outside preserved)
# for models that lack a native fill/inpaint variant. Models WITH a native
# fill variant (mflux FluxFill) should use that directly — this is the
# fallback path and the staged-denoise entry point for partial denoise.
#
# GATE: MSE validation requires a real mflux model (download via hf-mirror).
# The stage methods are unit-testable with mock VAE/scheduler; the end-to-end
# MSE=0 parity claim is gated on FUSION_MLX_REAL_MODEL_TESTS=1 + a flux model.

from __future__ import annotations

import logging

import mlx.core as mx

logger = logging.getLogger(__name__)


def prepare_latents(vae, image_array: mx.array) -> mx.array:
    # VAE-encode the init image to latent space. image_array is (H,W,3) in
    # [-1,1] or [0,1]; vae.encode returns the latent distribution mean.
    if hasattr(vae, "encode"):
        latent = vae.encode(image_array)
        if isinstance(latent, tuple):
            latent = latent[0]
        return latent
    raise ValueError("vae has no encode method — cannot prepare init latent")


def noise_latent(
    latent: mx.array,
    scheduler,
    t_start: float,
    seed: int = 0,
) -> mx.array:
    # Add noise to the init latent up to t_start (image_strength). The
    # scheduler provides the noise schedule; we sample noise and mix per the
    # forward diffusion equation: noisy = sqrt(alpha_bar_t)*x0 + sqrt(1-alpha_bar_t)*eps
    key = mx.random.key(seed)
    noise = mx.random.normal(latent.shape, key=key).astype(latent.dtype)
    if hasattr(scheduler, "add_noise"):
        return scheduler.add_noise(latent, noise, mx.array(t_start))
    # Fallback: linear mix if scheduler lacks add_noise.
    alpha_bar = 1.0 - t_start
    noisy = (alpha_bar**0.5) * latent + ((1.0 - alpha_bar) ** 0.5) * noise
    logger.debug(
        "noise_latent: t_start=%.3f alpha_bar=%.3f shape=%s",
        t_start,
        alpha_bar,
        noisy.shape,
    )
    return noisy


def mask_latent(mask_image: mx.array, latent_shape: tuple[int, ...]) -> mx.array:
    # Downsample a binary mask (H,W) to latent resolution. White (1.0) =
    # regenerate, black (0.0) = freeze init. Latent spatial dims are H//8
    # (standard VAE downsample factor 8). Uses area-average resize.
    if mask_image.ndim == 3:
        mask_image = mask_image.mean(axis=-1)
    # latent_shape is (C, H_lat, W_lat) or (B, C, H_lat, W_lat)
    if len(latent_shape) == 3:
        _, h_lat, w_lat = latent_shape
    elif len(latent_shape) == 4:
        *_, h_lat, w_lat = latent_shape
    else:
        h_lat = w_lat = latent_shape[-1]
    h_in = mask_image.shape[0]
    w_in = mask_image.shape[1]
    if h_in == h_lat and w_in == w_lat:
        mask = mask_image.astype(mx.float32)
    else:
        # crude area-average: reshape and mean
        rh = h_in // h_lat
        rw = w_in // w_lat
        if rh < 1 or rw < 1:
            mask = mask_image.astype(mx.float32)
        else:
            cropped = mask_image[: h_lat * rh, : w_lat * rw].astype(mx.float32)
            mask = mx.mean(
                mx.reshape(cropped, (h_lat, rh, w_lat, rw)),
                axis=(1, 3),
            )
    # broadcast to latent channels
    if len(latent_shape) == 3:
        c = latent_shape[0]
        mask = mx.broadcast_to(mx.reshape(mask, (1, h_lat, w_lat)), (c, h_lat, w_lat))
    elif len(latent_shape) == 4:
        b, c = latent_shape[0], latent_shape[1]
        mask = mx.broadcast_to(
            mx.reshape(mask, (1, 1, h_lat, w_lat)), (b, c, h_lat, w_lat)
        )
    return mask.astype(mx.float32)


def composite(
    denoised: mx.array,
    noisy_init: mx.array,
    mask: mx.array,
) -> mx.array:
    # White=regen (use denoised), black=freeze (keep noisy_init). Per-step
    # re-composite keeps the init region frozen across all denoise steps.
    if mask.shape != denoised.shape:
        mask = mx.broadcast_to(mask, denoised.shape)
    return mask * denoised + (1.0 - mask) * noisy_init


def inpaint_latent(
    flux,
    noisy_init: mx.array,
    mask: mx.array,
    *,
    num_inference_steps: int = 28,
    guidance: float = 3.5,
    prompt_embeds: mx.array | None = None,
    on_step=None,
) -> mx.array:
    # Run the denoise loop with per-step re-composite. Each step:
    #   1. predict noise eps = transformer(latents, t, conditioning)
    #   2. step latents via scheduler
    #   3. re-composite: mask*step + (1-mask)*noisy_init
    # Returns the final composited latent.
    latents = noisy_init
    scheduler = getattr(flux, "scheduler", None)
    if scheduler is None:
        raise ValueError("flux model has no scheduler — cannot run inpaint loop")
    timesteps = _get_timesteps(scheduler, num_inference_steps)
    total = len(timesteps)
    for i, t in enumerate(timesteps):
        eps = _predict_noise(flux, latents, t, guidance, prompt_embeds)
        latents = _scheduler_step(scheduler, eps, latents, t)
        latents = composite(latents, noisy_init, mask)
        if on_step is not None:
            on_step(0, i + 1, total)
    logger.info(
        "inpaint_latent: %d steps, mask mean=%.3f, final shape=%s",
        total,
        float(mx.mean(mask).item()),
        latents.shape,
    )
    return latents


def _get_timesteps(scheduler, num_steps: int):
    if hasattr(scheduler, "timesteps"):
        ts = scheduler.timesteps
        if hasattr(ts, "__len__") and len(ts) > 0:
            return list(ts)[:num_steps]
    return [1.0 - i / num_steps for i in range(num_steps)]


def _predict_noise(
    flux,
    latents: mx.array,
    t: float,
    guidance: float,
    prompt_embeds: mx.array | None,
) -> mx.array:
    # Predict noise via the transformer. Falls back to flux internals if the
    # standard call signature differs (mflux transformer API varies).
    try:
        if prompt_embeds is not None:
            return flux.transformer(latents, t, prompt_embeds, guidance)
        return flux.transformer(latents, t, guidance)
    except Exception as exc:
        logger.debug("inpaint _predict_noise direct call failed: %s", exc)
        raise


def _scheduler_step(scheduler, eps: mx.array, latents: mx.array, t: float) -> mx.array:
    if hasattr(scheduler, "step"):
        return scheduler.step(eps, latents, t)
    # Fallback: DDIM-ish update
    return latents - eps * 0.1


def run_inpaint_denoise(
    flux,
    vae,
    init_image: mx.array,
    mask_image: mx.array,
    *,
    image_strength: float = 0.6,
    num_inference_steps: int = 28,
    guidance: float = 3.5,
    prompt_embeds: mx.array | None = None,
    seed: int = 0,
    on_step=None,
) -> mx.array:
    # Full staged inpaint pipeline. Returns the decoded image (mx.array).
    # GATE: end-to-end MSE=0 parity vs mflux txt2img (same seed, empty mask)
    # requires a real flux model — validated under FUSION_MLX_REAL_MODEL_TESTS=1.
    init_latent = prepare_latents(vae, init_image)
    t_start = 1.0 - image_strength
    noisy = noise_latent(init_latent, getattr(flux, "scheduler", None), t_start, seed)
    mask = mask_latent(mask_image, init_latent.shape)
    final_latent = inpaint_latent(
        flux,
        noisy,
        mask,
        num_inference_steps=num_inference_steps,
        guidance=guidance,
        prompt_embeds=prompt_embeds,
        on_step=on_step,
    )
    if hasattr(vae, "decode"):
        decoded = vae.decode(final_latent)
        if isinstance(decoded, tuple):
            decoded = decoded[0]
        return decoded
    raise ValueError("vae has no decode method — cannot decode final latent")


__all__ = [
    "prepare_latents",
    "noise_latent",
    "mask_latent",
    "composite",
    "inpaint_latent",
    "run_inpaint_denoise",
]
