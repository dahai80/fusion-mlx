# SPDX-License-Identifier: Apache-2.0
# SVD (Stable Video Diffusion) image-to-video generation.
# Encodes input image via CLIP vision + VAE, then denoises with temporal UNet.

import logging
import math

import mlx.core as mx
import numpy as np

from .scheduler import SVDEulerScheduler

logger = logging.getLogger(__name__)

_DEFAULT_STEPS = 25
_DEFAULT_CFG = 3.0
_DEFAULT_NUM_FRAMES = 25
_MIN_FPS = 6


def generate_video(
    model_path,
    prompt=None,
    image=None,
    num_frames=_DEFAULT_NUM_FRAMES,
    width=576,
    height=1024,
    fps=7,
    seed=0,
    num_inference_steps=_DEFAULT_STEPS,
    cfg_scale=_DEFAULT_CFG,
    negative_prompt=None,
    output_path=None,
    dtype=mx.float16,
    on_step_sync=None,
):
    from .clip_vision import SVDCLIPVisionEncoder
    from .unet import SVDTemporalUNet
    from .vae import SVDVideoVAE

    logger.info(
        "svd generate: image=%s frames=%d %dx%d@%dfps seed=%d steps=%d cfg=%.2f",
        image,
        num_frames,
        width,
        height,
        fps,
        seed,
        num_inference_steps,
        cfg_scale,
    )

    mx.random.seed(seed)

    # Load components
    clip_encoder = SVDCLIPVisionEncoder(model_path, dtype=dtype)
    vae = SVDVideoVAE.from_pretrained(model_path, dtype=dtype)
    unet = SVDTemporalUNet.from_pretrained(model_path, dtype=dtype)
    scheduler = SVDEulerScheduler()

    # Encode image with CLIP vision
    clip_pooled, clip_seq = clip_encoder.encode_image(image)
    logger.info(
        "svd: clip_pooled shape=%s clip_seq shape=%s",
        clip_pooled.shape,
        clip_seq.shape,
    )

    # Encode image with VAE
    from PIL import Image as PILImage

    pil_img = PILImage.open(image).convert("RGB")
    pil_img = pil_img.resize((width, height), PILImage.LANCZOS)
    img_np = np.asarray(pil_img, dtype=np.float32) / 255.0
    # (H, W, 3) -> (1, 3, 1, H, W)
    img_np = np.transpose(img_np, (2, 0, 1))[None, :, None, :, :]
    img_mx = mx.array(img_np, dtype=dtype)
    img_latent = vae.encode(img_mx)
    mx.eval(img_latent)
    logger.info("svd: img_latent shape=%s", img_latent.shape)

    # Build conditioning latent: replicate image across all frames
    # SVD concatenates image latent along channel axis
    latent_ch = img_latent.shape[1]
    temporal_ratio = 4
    spatial_ratio = 8
    lf = num_frames // temporal_ratio
    lh = height // spatial_ratio
    lw = width // spatial_ratio

    # Replicate image latent for all frames
    img_latent_rep = mx.broadcast_to(
        img_latent[:, :, :, :, :],
        (1, latent_ch, lf, lh, lw),
    )
    # Concat image latent along channel axis for I2V conditioning
    latent_input_ch = latent_ch * 2  # noisy latent + image conditioning
    noise = mx.random.normal(shape=(1, latent_ch, lf, lh, lw), dtype=mx.float32)

    # Prepare scheduler
    scheduler.set_timesteps(num_inference_steps)
    timesteps = scheduler.timesteps

    # Build CLIP context (pooled + sequence for cross-attention)
    context = clip_seq  # (1, seq_len, 1024)
    if float(cfg_scale) > 1.0:
        # For CFG, duplicate context (no negative text for SVD)
        neg_context = mx.zeros_like(context)
        context = mx.concatenate([neg_context, context], axis=0)

    # Denoise
    latents = noise * scheduler.init_noise_sigma
    do_cfg = float(cfg_scale) > 1.0

    logger.info("svd: denoise start steps=%d cfg=%s", num_inference_steps, do_cfg)

    for i, t in enumerate(timesteps.tolist()):
        t_val = float(t)

        if do_cfg:
            latent_in = mx.concatenate([latents, latents], axis=0)
            img_cond = mx.concatenate([img_latent_rep, img_latent_rep], axis=0)
        else:
            latent_in = latents
            img_cond = img_latent_rep

        # Concat image conditioning along channel axis
        unet_input = mx.concatenate([latent_in, img_cond], axis=1)
        timestep = mx.array([t_val, t_val] if do_cfg else [t_val], dtype=dtype)

        noise_pred = unet(unet_input.astype(dtype), timestep=timestep, context=context)

        if do_cfg:
            uncond, cond = mx.split(noise_pred, 2, axis=0)
            noise_pred = uncond + float(cfg_scale) * (cond - uncond)

        latents = scheduler.step(noise_pred, t_val, latents)
        mx.eval(latents)

        logger.info("svd: step=%d/%d t=%.4f", i + 1, num_inference_steps, t_val)
        if on_step_sync is not None:
            on_step_sync(i + 1, num_inference_steps)

    logger.info("svd: denoise done, decoding latents shape=%s", latents.shape)

    # Decode
    decoded = vae.decode(latents.astype(dtype))
    mx.eval(decoded)

    # Convert to frames
    frames = _to_frames(decoded, num_frames, height, width)
    logger.info("svd: generated %d frames", len(frames))

    if output_path is not None:
        _write_mp4(frames, max(1, fps), output_path)
        logger.info("svd: wrote mp4 to %s", output_path)
    return frames


def _to_frames(decoded, num_frames, height, width):
    frames = []
    f_out = min(num_frames, decoded.shape[2])
    for fi in range(f_out):
        frame = decoded[0, :, fi, :height, :width]
        frame = np.clip(np.asarray(frame.astype(mx.float32)), 0.0, 1.0)
        frame = np.transpose(frame, (1, 2, 0))
        frame = (frame * 255.0).astype(np.uint8)
        frames.append(frame)
    return frames


def _write_mp4(frames, fps, path):
    import imageio

    imageio.mimwrite(path, frames, fps=fps, codec="libx264", quality=8)
