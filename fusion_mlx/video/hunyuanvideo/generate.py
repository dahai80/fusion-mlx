# SPDX-License-Identifier: Apache-2.0
# HunyuanVideo generate: T2V + I2V.

import gc
import logging
import os

import mlx.core as mx
import numpy as np

from .dit import HUNYUAN_VIDEO_CONFIG, HunyuanVideoDiT
from .scheduler import HunyuanVideoScheduler
from .text_encoder import HunyuanDualTextEncoder
from .vae import HunyuanVideoVAE

logger = logging.getLogger(__name__)

_DEFAULT_FPS = 24
_DEFAULT_NUM_FRAMES = 33
_DEFAULT_WIDTH = 720
_DEFAULT_HEIGHT = 480


def _tokenize_prompt(prompt, max_length=77):
    tokens = [49406]
    for ch in prompt[: max_length - 2]:
        tokens.append(ord(ch) % 49408)
    tokens.append(49407)
    while len(tokens) < max_length:
        tokens.append(0)
    return mx.array(tokens[:max_length], dtype=mx.int32)[None]


def generate_video(
    model_path,
    prompt,
    num_frames=_DEFAULT_NUM_FRAMES,
    width=_DEFAULT_WIDTH,
    height=_DEFAULT_HEIGHT,
    fps=_DEFAULT_FPS,
    seed=None,
    image=None,
    cfg_scale=6.0,
    num_inference_steps=50,
    on_step=None,
    output_path=None,
):
    logger.info(
        "hunyuan generate: prompt='%s' frames=%d %dx%d fps=%d seed=%s i2v=%s steps=%d",
        prompt[:60],
        num_frames,
        width,
        height,
        fps,
        seed,
        image is not None,
        num_inference_steps,
    )
    if seed is not None:
        mx.random.seed(seed)
    else:
        seed = int(mx.random.key(0).item())
        mx.random.seed(seed)
        logger.info("hunyuan: auto seed=%d", seed)

    config = HUNYUAN_VIDEO_CONFIG
    scheduler = HunyuanVideoScheduler()

    dit_path = (
        os.path.join(model_path, "dit")
        if os.path.isdir(os.path.join(model_path, "dit"))
        else model_path
    )
    dit = HunyuanVideoDiT.from_pretrained(dit_path, config=config)
    vae_path = (
        os.path.join(model_path, "vae")
        if os.path.isdir(os.path.join(model_path, "vae"))
        else model_path
    )
    vae = HunyuanVideoVAE.from_pretrained(vae_path)
    text_path = (
        os.path.join(model_path, "text_encoder")
        if os.path.isdir(os.path.join(model_path, "text_encoder"))
        else model_path
    )
    text_encoder = HunyuanDualTextEncoder.from_pretrained(text_path)

    mx.eval(dit.parameters())
    mx.eval(text_encoder.parameters())
    mx.clear_cache()

    input_ids = _tokenize_prompt(prompt)
    text_emb, text_pooled = text_encoder(input_ids, input_ids)
    text_emb_null = mx.zeros_like(text_emb)

    latent_ch = config["in_channels"]
    pt, ph, pw = config["patch_size"]
    t_latent = num_frames // 4
    h_latent = height // 8
    w_latent = width // 8
    t_latent = max(1, t_latent)
    h_latent = (h_latent // ph) * ph
    w_latent = (w_latent // pw) * pw

    logger.info(
        "hunyuan: latent shape=(1,%d,%d,%d,%d)", latent_ch, t_latent, h_latent, w_latent
    )

    noise = mx.random.normal(
        (1, latent_ch, t_latent, h_latent, w_latent), dtype=mx.float32
    )

    image_cond = None
    if image is not None:
        import PIL.Image as PILImage

        if isinstance(image, str):
            img = PILImage.open(image).convert("RGB")
        else:
            img = image
        img_np = np.array(img.resize((width, height))).astype(np.float32) / 255.0
        img_arr = mx.array(img_np, dtype=mx.float32)
        img_arr = img_arr.transpose(2, 0, 1)[None]
        img_arr = img_arr[:, :, None, :, :]
        img_latent = vae.encode(img_arr)
        img_cond = mx.broadcast_to(
            img_latent, (1, latent_ch, t_latent, h_latent, w_latent)
        )
        noise = noise + img_cond * 0.1
        image_cond = img_latent

    scheduler.set_timesteps(num_inference_steps)

    latents = noise
    total_steps = len(scheduler.timesteps)
    cfg = float(cfg_scale) if cfg_scale is not None else 6.0
    for i, t in enumerate(scheduler.timesteps):
        timestep = mx.array([float(t)] * 1, dtype=mx.float32)
        noise_pred_uncond = dit(latents, timestep, text_emb_null, image_cond=image_cond)
        mx.eval(noise_pred_uncond)
        noise_pred_cond = dit(latents, timestep, text_emb, image_cond=image_cond)
        mx.eval(noise_pred_cond)
        noise_pred = noise_pred_uncond + cfg * (noise_pred_cond - noise_pred_uncond)
        del noise_pred_uncond, noise_pred_cond
        latents = scheduler.step(noise_pred, t, latents)
        del noise_pred
        mx.eval(latents)
        mx.clear_cache()
        if on_step is not None:
            on_step(i + 1, total_steps)
        logger.debug("hunyuan denoise step %d/%d", i + 1, total_steps)

    # Free DiT and text encoder before VAE decode to reduce peak memory
    del dit, text_encoder, text_emb, text_emb_null
    if image_cond is not None:
        del image_cond
    gc.collect()
    mx.clear_cache()
    mx.synchronize()
    logger.info(
        "hunyuan: after del dit+text, active_mem=%.1fGB", mx.get_active_memory() / 1e9
    )

    # VAE decode — use tiled for large latents to stay within Metal memory limits
    mx.eval(vae.parameters())
    B, C_l, T_l, H_l, W_l = latents.shape
    need_tiled = H_l > 64 or W_l > 64 or T_l > 16
    if need_tiled:
        logger.info(
            "hunyuan: using tiled VAE decode for large latent %s", latents.shape
        )
        video = vae.decode_tiled(
            latents,
            tile_t=8,
            tile_h=32,
            tile_w=32,
            overlap_t=2,
            overlap_h=4,
            overlap_w=4,
        )
    else:
        video = vae.decode(latents)
        mx.eval(video)
    logger.info("hunyuan: VAE decode done, shape=%s", video.shape)

    # Convert frames
    video_np = np.array(video)
    del video, latents, vae
    gc.collect()
    mx.clear_cache()

    frames_np_raw = video_np[0]
    frames_np_raw = frames_np_raw.transpose(1, 2, 3, 0)
    frames_np = np.clip(frames_np_raw * 255.0, 0, 255).astype(np.uint8)
    del video_np, frames_np_raw

    if output_path is None:
        import tempfile

        tmpdir = tempfile.TemporaryDirectory()
        output_path = os.path.join(tmpdir.name, "hunyuan_output.mp4")
    _write_mp4(frames_np, output_path, fps)
    logger.info("hunyuan: output saved to %s", output_path)
    return output_path


def _write_mp4(frames_np, output_path, fps):
    try:
        import av

        container = av.open(output_path, mode="w")
        try:
            stream = container.add_stream("libx264", rate=fps)
            stream.width = frames_np.shape[2]
            stream.height = frames_np.shape[1]
            stream.pix_fmt = "yuv420p"
            for frame in frames_np:
                img = av.VideoFrame.from_ndarray(frame, format="rgb24")
                for packet in stream.encode(img):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        finally:
            container.close()
    except ImportError:
        raise RuntimeError(
            "av (PyAV) is required for MP4 output. Install with: pip install av"
        )
    except Exception as e:
        logger.error("hunyuan: mp4 write failed: %s", e)
        raise
