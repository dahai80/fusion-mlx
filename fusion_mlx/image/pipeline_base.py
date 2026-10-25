# SPDX-License-Identifier: Apache-2.0
"""Shared pieces of the SD1.5 / SD2 / SDXL MLX pipelines.

Audit 1019 L11: the three pipelines were ~300-line copies of each other, so
every fix (the #482 tokenizer subfolder resolution, img2img step slicing,
guidance batching) had to be applied three times. The text encoders, UNet
call signature and scheduler class are the only real differences; they stay
in the per-model modules.
"""

import logging
import os

import mlx.core as mx
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class GenResult:
    def __init__(self, image: Image.Image):
        self.image = image


def local_path(base_env: str, subfolder: str, filename: str, label: str):
    base = os.environ.get(base_env)
    if not base:
        return None
    cand = (
        os.path.join(base, subfolder, filename)
        if subfolder
        else os.path.join(base, filename)
    )
    if os.path.exists(cand):
        logger.info("%s local resolve %s", label, cand)
        return cand
    return None


def resolve_path(
    base_env: str, repo: str, subfolder: str, filename: str, label: str
) -> str:
    local = local_path(base_env, subfolder, filename, label)
    if local:
        return local
    from huggingface_hub import hf_hub_download

    path_in_repo = f"{subfolder}/{filename}" if subfolder else filename
    return hf_hub_download(repo, path_in_repo)


def resolve_dir(base_env: str, repo: str, subfolder: str, label: str) -> str:
    base = os.environ.get(base_env)
    if base and os.path.isdir(os.path.join(base, subfolder)):
        logger.info("%s local resolve_dir %s/%s", label, base, subfolder)
        return os.path.join(base, subfolder)
    from huggingface_hub import snapshot_download

    # snapshot_download returns the repo snapshot root (contains all allowed
    # subfolders), NOT the subfolder itself. Append subfolder so callers get
    # the actual directory (e.g. <root>/tokenizer with vocab.json +
    # merges.txt). Without this, CLIPTokenizer.from_pretrained(root) finds no
    # tokenizer_config and falls back to a degenerate tokenizer that emits
    # only BOS+pad tokens, making every prompt encode identically (#482).
    root = snapshot_download(
        repo, allow_patterns=[f"{subfolder}/*"], endpoint=os.environ.get("HF_ENDPOINT")
    )
    return os.path.join(root, subfolder)


def load_safetensors(path: str) -> dict:
    from safetensors import safe_open

    raw = {}
    with safe_open(path, "pt") as f:
        # safe_open objects expose .keys() but are NOT iterable in
        # safetensors >= 0.8 (ruff SIM118 is a false positive here).
        for k in f.keys():  # noqa: SIM118
            raw[k] = mx.array(f.get_tensor(k).numpy())
    return raw


def to_pil(image: mx.array) -> Image.Image:
    arr = np.array(image[0].astype(mx.float32))
    arr = np.transpose(arr, (1, 2, 0))
    arr = (arr / 2 + 0.5).clip(0, 1)
    arr = (arr * 255).astype(np.uint8)
    return Image.fromarray(arr)


def pil_to_nchw(image: Image.Image, height: int, width: int) -> mx.array:
    # Resize to target then normalize to [-1, 1] NCHW for VAE encode.
    if image.size != (width, height):
        image = image.resize((width, height), Image.LANCZOS)
    arr = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    arr = (arr * 2.0) - 1.0
    arr = np.transpose(arr, (2, 0, 1))
    return mx.array(arr)[None]


def prepare_schedule(
    label: str,
    prompt: str,
    num_inference_steps: int,
    image_path: str | None,
    image_strength: float | None,
    width: int,
    height: int,
    guidance: float,
    seed: int,
):
    # img2img: image_strength (a.k.a. denoise fraction) controls how many
    # steps run. strength=1.0 -> full txt2img from noise; strength=0.5 -> run
    # half the steps from a noised init image (#480). image_path None ->
    # pure txt2img.
    is_img2img = image_path is not None
    strength = float(image_strength) if image_strength is not None else 1.0
    if is_img2img:
        if strength <= 0.0 or strength > 1.0:
            logger.warning(
                "%s image_strength=%.3f out of (0,1], clamping", label, strength
            )
            strength = max(min(strength, 1.0), 1e-3)
        eff_steps = max(1, int(round(num_inference_steps * strength)))
        logger.info(
            "%s img2img prompt=%r steps=%d eff_steps=%d strength=%.3f %dx%d",
            label,
            prompt[:60],
            num_inference_steps,
            eff_steps,
            strength,
            width,
            height,
        )
    else:
        eff_steps = num_inference_steps
        logger.info(
            "%s txt2img prompt=%r steps=%d %dx%d guidance=%.2f seed=%d",
            label,
            prompt[:60],
            num_inference_steps,
            width,
            height,
            guidance,
            seed,
        )
    return is_img2img, eff_steps, strength


def slice_timesteps(scheduler, timesteps, eff_steps):
    # Take the last eff_steps timesteps (strongest denoise near the end) and
    # realign sigmas so step_index walks the sliced window, matching
    # diffusers img2img init_noise_sigma + skip.
    if eff_steps < len(timesteps):
        timesteps = timesteps[len(timesteps) - eff_steps :]
        scheduler.sigmas = scheduler.sigmas[len(scheduler.sigmas) - 1 - eff_steps :]
        scheduler._step_index = 0
    return timesteps


def init_latent_from_image(label, vae, scheduler, image_path, height, width):
    init_img = Image.open(image_path).convert("RGB")
    init_lat = vae.encode(pil_to_nchw(init_img, height, width)).astype(mx.float32)
    # Add noise at the starting timestep's sigma (epsilon noise convention).
    sigma_start = float(scheduler.sigmas[0])
    noise = mx.random.normal(init_lat.shape, dtype=mx.float32)
    latent = init_lat + sigma_start * noise
    mx.eval(latent)
    logger.info("%s img2img init latent encoded, sigma_start=%.4f", label, sigma_start)
    return latent


def init_noise_latent(cfg, h_lat: int, w_lat: int, scheduler):
    latent = mx.random.normal((1, cfg.in_channels, h_lat, w_lat), dtype=mx.float32)
    return latent * scheduler.init_noise_sigma


def denoise_loop(
    label,
    scheduler,
    timesteps,
    latent,
    guidance: float,
    cond: dict,
    uncond: dict | None,
    call_unet,
):
    # call_unet(latent, t_arr, kwargs) mirrors the per-model UNet signature:
    # SD1.5/SD2 take cross-attention context only, SDXL adds pooled text
    # embeddings and time ids. Classifier-free guidance batches the two rows.
    for i, t in enumerate(timesteps):
        t_arr = mx.array([float(t)])
        if guidance > 1 and uncond is not None:
            latent_in = mx.concatenate([latent, latent], axis=0)
            t_in = mx.concatenate([t_arr, t_arr], axis=0)
            merged = {
                key: mx.concatenate([cond[key], uncond[key]], axis=0) for key in cond
            }
            noise = call_unet(scheduler.scale_model_input(latent_in), t_in, merged)
            noise_cond, noise_un = mx.split(noise, 2, axis=0)
            noise = noise_un + guidance * (noise_cond - noise_un)
        else:
            noise = call_unet(scheduler.scale_model_input(latent), t_arr, cond)
        latent = scheduler.step(noise, latent)
        mx.eval(latent)
        logger.info("%s step %d/%d done", label, i + 1, len(timesteps))
    return latent
