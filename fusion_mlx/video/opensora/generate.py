# SPDX-License-Identifier: Apache-2.0
# Open-Sora V2 generation pipeline: T2V and I2V with 3-branch CFG.

import logging
import os
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from .config import OpenSoraConfig
from .scheduler import (
    get_image_ids,
    get_noise,
    get_schedule,
    get_txt_ids,
    pack,
    unpack,
)
from .transformer import MMDiTModel

logger = logging.getLogger(__name__)


class I2VDenoiser:
    # 3-branch CFG with oscillating guidance for I2V.

    def __init__(
        self,
        model: MMDiTModel,
        guidance: float = 4.0,
        text_osci: float = 0.5,
        image_osci: float = 0.25,
    ):
        self.model = model
        self.guidance = guidance
        self.text_osci = text_osci
        self.image_osci = image_osci

    def __call__(
        self, img, img_ids, txt, txt_ids, timesteps, y_vec, cond, guidance_val
    ):
        B = img.shape[0]
        img_3 = mx.concatenate([img, img, img], axis=0)
        img_ids_3 = mx.concatenate([img_ids, img_ids, img_ids], axis=0)
        txt_3 = mx.concatenate([txt, txt, txt], axis=0)
        txt_ids_3 = mx.concatenate([txt_ids, txt_ids, txt_ids], axis=0)
        t_3 = mx.concatenate([timesteps, timesteps, timesteps], axis=0)
        y_3 = mx.concatenate([y_vec, y_vec, y_vec], axis=0)

        # Branch 0: cond (text + image), Branch 1: uncond (neg text), Branch 2: uncond2 (neg text + no image)
        if cond is not None:
            cond_3 = mx.concatenate([cond, cond, mx.zeros_like(cond)], axis=0)
        else:
            cond_3 = None

        g_val = guidance_val if guidance_val is not None else self.guidance
        g_3 = mx.concatenate(
            [
                mx.full((B,), g_val),
                mx.full((B,), g_val * self.text_osci),
                mx.full((B,), g_val * self.image_osci),
            ]
        )

        model_out = self.model(
            img=img_3,
            img_ids=img_ids_3,
            txt=txt_3,
            txt_ids=txt_ids_3,
            timesteps=t_3,
            y_vec=y_3,
            cond=cond_3,
            guidance=g_3,
        )

        out_cond, out_uncond, out_uncond2 = mx.split(model_out, 3, axis=0)

        out = (
            out_cond
            + self.guidance * (out_cond - out_uncond)
            + self.guidance * self.image_osci * (out_uncond - out_uncond2)
        )
        return out


class DistilledDenoiser:
    # Single-branch denoiser (no CFG).

    def __init__(self, model: MMDiTModel):
        self.model = model

    def __call__(
        self, img, img_ids, txt, txt_ids, timesteps, y_vec, cond, guidance_val
    ):
        return self.model(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            timesteps=timesteps,
            y_vec=y_vec,
            cond=cond,
            guidance=guidance_val,
        )


def generate_video(
    model_dir: str,
    prompt: str,
    negative_prompt: str = "",
    image: mx.array | None = None,
    num_frames: int = 51,
    height: int = 480,
    width: int = 854,
    num_steps: int = 30,
    guidance: float = 4.0,
    seed: int | None = None,
    shift_alpha: float = 1.0,
    text_encoder: object | None = None,
    vae: object | None = None,
    output_format: str = "mp4",
    config: OpenSoraConfig | None = None,
) -> mx.array:
    logger.info(
        f"Open-Sora V2 generate: {num_frames}f {height}x{width} {num_steps}steps"
    )

    if config is None:
        config_path = Path(model_dir) / "config.json"
        if config_path.exists():
            import json

            with open(config_path) as f:
                cfg_dict = json.load(f)
            config = OpenSoraConfig.from_dict(cfg_dict)
        else:
            config = OpenSoraConfig()
            logger.info("Using default config (no config.json found)")

    model = MMDiTModel(config)
    weights_path = Path(model_dir) / "weights.npz"
    if weights_path.exists():
        weights = mx.load(str(weights_path))
        model.load_weights(list(weights.items()))
        logger.info(f"Loaded {len(weights)} weights from {weights_path}")
    else:
        logger.warning(f"No weights found at {weights_path}")

    if text_encoder is not None:
        context, y_vec = text_encoder.encode([prompt, negative_prompt])
        if image is None:
            txt = context[:1]
            y = y_vec[:1]
        else:
            txt = context
            y = y_vec
    else:
        logger.warning("No text encoder provided, using random embeddings")
        txt = mx.random.normal((1, 512, config.context_in_dim))
        y = mx.random.normal((1, config.vec_in_dim))

    # in_channels is the packed dim (C_latent * patch_size^2)
    # Raw latent channels before packing
    latent_channels = config.in_channels // (config.patch_size**2)
    in_channels = config.in_channels
    latent_num_frames = (num_frames - 1) // 4 + 1
    latent_h = height // 8
    latent_w = width // 8

    noise = get_noise(latent_num_frames, latent_h, latent_w, 1, latent_channels, seed)
    x = noise

    x_packed = pack(x, config.patch_size)

    img_ids = get_image_ids(latent_num_frames, latent_h, latent_w, config.patch_size)
    txt_ids = get_txt_ids(txt.shape[1])

    cond = None
    if image is not None and config.cond_embed:
        if vae is not None:
            image_latent = vae.encode(image)
        else:
            image_latent = mx.random.normal((1, latent_channels, 1, latent_h, latent_w))
        if image_latent.shape[2] < latent_num_frames:
            pad_frames = latent_num_frames - image_latent.shape[2]
            image_latent = mx.pad(
                image_latent, [(0, 0), (0, 0), (0, pad_frames), (0, 0), (0, 0)]
            )
        cond = pack(image_latent, config.patch_size)

    image_seq_len = x_packed.shape[1]
    schedule = get_schedule(
        num_steps, image_seq_len, latent_num_frames, shift_alpha=shift_alpha
    )

    if image is not None and guidance > 0:
        denoiser = I2VDenoiser(model, guidance=guidance)
    else:
        denoiser = DistilledDenoiser(model)

    for step_idx, (t_cur, t_next) in enumerate(schedule):
        t = mx.array([t_cur])
        dt = t_cur - t_next

        model_out = denoiser(
            img=x_packed,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            timesteps=t,
            y_vec=y,
            cond=cond,
            guidance_val=mx.array([guidance]),
        )

        x_packed = x_packed + dt * model_out

        if step_idx % 5 == 0 or step_idx == len(schedule) - 1:
            logger.info(f"Step {step_idx + 1}/{num_steps}: t={t_cur:.4f}")

    x = unpack(x_packed, latent_h, latent_w, latent_num_frames, config.patch_size)
    # x shape: (B, latent_channels, T, H, W) ready for VAE decode

    if vae is not None:
        video = vae.decode(x)
        logger.info(f"Decoded video shape: {video.shape}")
    else:
        video = x
        logger.warning("No VAE provided, returning latent")

    if output_format == "raw":
        logger.info("Raw output: returning decoded frames")
        return video

    return video
