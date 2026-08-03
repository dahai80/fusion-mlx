# SPDX-License-Identifier: Apache-2.0
# Cosmos generate: dual-mode T2V (7B) + I2V (Predict2 2B).

import logging
import os

import mlx.core as mx
import numpy as np

from .dit import COSMOS_2B_CONFIG, COSMOS_7B_CONFIG, CosmosDiT
from .scheduler import CosmosFlowScheduler, CosmosPredict2Scheduler
from .vae import CosmosVideoVAE

logger = logging.getLogger(__name__)

_DEFAULT_FPS = 24
_DEFAULT_NUM_FRAMES = 121
_DEFAULT_WIDTH = 848
_DEFAULT_HEIGHT = 480


def _load_text_embeddings(prompt, model_path, max_length=512):
    # Cosmos uses T5-XXL text embeddings, typically pre-computed
    # For pure-MLX we attempt to load from cache or use UMT5 fallback
    emb_path = os.path.join(model_path, "text_embeddings")
    if os.path.exists(emb_path):
        npz_files = [f for f in os.listdir(emb_path) if f.endswith(".npz")]
        if npz_files:
            data = np.load(os.path.join(emb_path, npz_files[0]), allow_pickle=False)
            keys = list(data.keys())
            if keys:
                emb = mx.array(data[keys[0]], dtype=mx.float32)
                logger.info("cosmos: loaded pre-computed text emb shape=%s", emb.shape)
                return emb
    # Fallback: random placeholder (for testing / no T5 available)
    logger.warning("cosmos: no pre-computed text embeddings, using zero placeholder")
    text_dim = COSMOS_7B_CONFIG["text_embed_dim"]
    emb = mx.zeros((1, max_length, text_dim), dtype=mx.float32)
    return emb


def _load_clip_vision(image, model_path):
    # Cosmos Predict2 uses CLIP vision for image conditioning
    # Return zero placeholder for now; real CLIP loading requires mlx-vlm
    logger.info("cosmos predict2: using zero CLIP vision placeholder")
    vision_dim = 1024
    emb = mx.zeros((1, 1, vision_dim), dtype=mx.float32)
    return emb


def generate_video(
    model_path,
    prompt,
    num_frames=_DEFAULT_NUM_FRAMES,
    width=_DEFAULT_WIDTH,
    height=_DEFAULT_HEIGHT,
    fps=_DEFAULT_FPS,
    seed=None,
    image=None,
    cfg_scale=7.0,
    num_inference_steps=50,
    is_predict2=False,
    on_step=None,
    output_path=None,
):
    logger.info(
        "cosmos generate: prompt='%s' frames=%d %dx%d fps=%d seed=%s i2v=%s steps=%d",
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
        logger.info("cosmos: auto seed=%d", seed)

    # Determine config
    if is_predict2:
        config = COSMOS_2B_CONFIG
        scheduler = CosmosPredict2Scheduler()
    else:
        config = COSMOS_7B_CONFIG
        scheduler = CosmosFlowScheduler()

    # Load models
    dit_path = (
        os.path.join(model_path, "dit")
        if os.path.isdir(os.path.join(model_path, "dit"))
        else model_path
    )
    dit = CosmosDiT.from_pretrained(dit_path, config=config)
    vae_path = (
        os.path.join(model_path, "vae")
        if os.path.isdir(os.path.join(model_path, "vae"))
        else model_path
    )
    vae = CosmosVideoVAE.from_pretrained(vae_path)

    mx.eval(dit.parameters())
    mx.eval(vae.parameters())

    # Text embeddings
    text_emb = _load_text_embeddings(prompt, model_path)
    text_emb_null = mx.zeros_like(text_emb)

    # Latent shape
    latent_ch = config["in_channels"]
    pt, ph, pw = config["patch_size"]
    # Cosmos VAE: 8x spatial, 4x temporal
    t_latent = num_frames // 4
    h_latent = height // 8
    w_latent = width // 8
    # Ensure divisibility by patch_size
    t_latent = max(pt, (t_latent // pt) * pt)
    h_latent = max(ph, (h_latent // ph) * ph)
    w_latent = max(pw, (w_latent // pw) * pw)

    logger.info(
        "cosmos: latent shape=(1,%d,%d,%d,%d)", latent_ch, t_latent, h_latent, w_latent
    )

    # Initial noise
    noise = mx.random.normal(
        (1, latent_ch, t_latent, h_latent, w_latent), dtype=mx.float32
    )

    # I2V conditioning
    image_cond = None
    if image is not None and is_predict2:
        # VAE encode image -> replicate across time -> blend with noise
        import PIL.Image as PILImage

        if isinstance(image, str):
            img = PILImage.open(image).convert("RGB")
        elif isinstance(image, PILImage.Image):
            img = image
        else:
            img = image
        img_np = np.array(img.resize((width, height))).astype(np.float32) / 255.0
        img_arr = mx.array(img_np, dtype=mx.float32)
        img_arr = img_arr.transpose(2, 0, 1)[None]  # (1, 3, H, W)
        img_arr = img_arr[:, :, None, :, :]  # (1, 3, 1, H, W)
        # Pad to latent time dim
        img_latent = vae.encode(img_arr)
        # Replicate across temporal dim
        img_cond = mx.broadcast_to(
            img_latent, (1, latent_ch, t_latent, h_latent, w_latent)
        )
        # Blend with noise: Cosmos Predict2 uses noise + conditioning
        noise = noise + img_cond * 0.1  # gentle conditioning blend
        image_cond = img_latent  # Latent-space for DiT patch_embed (C=16)

    # Scheduler
    scheduler.set_timesteps(num_inference_steps)

    # Denoise
    latents = noise
    total_steps = len(scheduler.timesteps)
    cfg = float(cfg_scale) if cfg_scale is not None else 7.0
    for i, t in enumerate(scheduler.timesteps):
        timestep = mx.array([float(t)] * 1, dtype=mx.float32)
        # CFG: uncond + cond
        noise_pred_uncond = dit(latents, timestep, text_emb_null, image_cond=image_cond)
        noise_pred_cond = dit(latents, timestep, text_emb, image_cond=image_cond)
        noise_pred = noise_pred_uncond + cfg * (noise_pred_cond - noise_pred_uncond)
        latents = scheduler.step(noise_pred, t, latents)
        mx.eval(latents)
        if on_step is not None:
            on_step(i + 1, total_steps)
        logger.debug("cosmos denoise step %d/%d", i + 1, total_steps)

    # I2V: blend conditioning back
    if image is not None and is_predict2:
        latents = latents + img_cond * 0.05

    # VAE decode — use tiled for large latents
    _, _, t_l, h_l, w_l = latents.shape
    if h_l > 64 or w_l > 64 or t_l > 16:
        logger.info(
            "cosmos: using tiled VAE decode for large latent %s", latents.shape
        )
        video = vae.decode_tiled(latents)
    else:
        video = vae.decode(latents)
    mx.eval(video)

    # Convert to frames
    frames = video[0]  # (C, T, H, W)
    frames = frames.transpose(1, 2, 3, 0)  # (T, H, W, C)
    frames = mx.clip(frames * 255.0, 0, 255).astype(mx.uint8)
    frames_np = np.array(frames)

    # Write MP4
    if output_path is None:
        import tempfile

        tmpdir = tempfile.TemporaryDirectory()
        output_path = os.path.join(tmpdir.name, "cosmos_output.mp4")
        try:
            _write_mp4(frames_np, output_path, fps)
            logger.info("cosmos: output saved to %s", output_path)
        finally:
            tmpdir.cleanup()
    else:
        _write_mp4(frames_np, output_path, fps)
        logger.info("cosmos: output saved to %s", output_path)
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
        logger.error("cosmos: mp4 write failed: %s", e)
        raise
