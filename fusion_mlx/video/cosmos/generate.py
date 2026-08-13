# SPDX-License-Identifier: Apache-2.0
# Cosmos generate: dual-mode T2V (7B) + I2V (Predict2 2B).

import logging
import os
import time

import mlx.core as mx
import numpy as np

from .dit import COSMOS_2B_CONFIG, COSMOS_7B_CONFIG, CosmosDiT
from .scheduler import CosmosFlowScheduler, CosmosPredict2Scheduler
from .text_encoder import CosmosT5Encoder
from .vae import CosmosVideoVAE

logger = logging.getLogger(__name__)

_DEFAULT_FPS = 24
_DEFAULT_NUM_FRAMES = 121
_DEFAULT_WIDTH = 848
_DEFAULT_HEIGHT = 480


def _encode_prompt(prompt, text_encoder_path, max_length=512):
    te_path = (
        os.path.join(text_encoder_path, "text_encoder")
        if os.path.isdir(os.path.join(text_encoder_path, "text_encoder"))
        else text_encoder_path
    )
    te = CosmosT5Encoder.from_pretrained(te_path)
    try:
        from transformers import T5Tokenizer

        tokenizer = T5Tokenizer.from_pretrained(te_path, local_files_only=True)
    except Exception:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(te_path, local_files_only=True)
    text_emb = te.encode(prompt, tokenizer, max_length=max_length)
    return text_emb


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
    text_emb = _encode_prompt(prompt, model_path)
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
    condition_mask = None
    if image is not None and is_predict2:
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
        img_latent = vae.encode(img_arr)
        img_cond = mx.broadcast_to(
            img_latent, (1, latent_ch, t_latent, h_latent, w_latent)
        )
        noise = noise + img_cond * 0.1
        condition_mask = mx.ones((1, 1, t_latent, h_latent, w_latent), dtype=mx.float32)
        condition_mask = mx.concatenate(
            [mx.zeros((1, 1, 1, h_latent, w_latent)), condition_mask[:, :, 1:]], axis=2
        )

    # Scheduler
    scheduler.set_timesteps(num_inference_steps)

    padding_mask = mx.ones((1, 1, h_latent, w_latent), dtype=mx.float32)
    latents = noise
    total_steps = len(scheduler.timesteps)
    cfg = float(cfg_scale) if cfg_scale is not None else 7.0
    # #367 perf: CFG batched guidance — fuse uncond+cond into a single B=2
    # forward (DiT is batch-safe along dim 0) instead of two separate full
    # forwards. ~2x throughput, no quality change. cfg<=1.0 skips the uncond
    # branch entirely (single-forward shortcut).
    use_single_forward = cfg <= 1.0
    if use_single_forward:
        logger.info("cosmos: cfg=%.2f <=1.0, single-forward (no uncond branch)", cfg)
    logger.info(
        "cosmos: denoise start steps=%d cfg=%.2f batched_cfg=%s",
        total_steps,
        cfg,
        not use_single_forward,
    )
    step_t0 = time.time()
    for i, t in enumerate(scheduler.timesteps):
        if use_single_forward:
            timestep = mx.array([float(t)], dtype=mx.float32)
            noise_pred = dit(
                latents,
                timestep,
                text_emb,
                fps=fps,
                padding_mask=padding_mask,
                condition_mask=condition_mask,
            )
            mx.eval(noise_pred)
        else:
            timestep = mx.array([float(t)] * 2, dtype=mx.float32)
            latents_2 = mx.concatenate([latents, latents], axis=0)
            text_emb_2 = mx.concatenate([text_emb_null, text_emb], axis=0)
            padding_mask_2 = (
                mx.broadcast_to(padding_mask, (2, 1, h_latent, w_latent))
                if padding_mask is not None
                else None
            )
            condition_mask_2 = (
                mx.broadcast_to(condition_mask, (2, 1, t_latent, h_latent, w_latent))
                if condition_mask is not None
                else None
            )
            noise_pred_2 = dit(
                latents_2,
                timestep,
                text_emb_2,
                fps=fps,
                padding_mask=padding_mask_2,
                condition_mask=condition_mask_2,
            )
            mx.eval(noise_pred_2)
            noise_pred_uncond, noise_pred_cond = noise_pred_2[0:1], noise_pred_2[1:2]
            noise_pred = noise_pred_uncond + cfg * (noise_pred_cond - noise_pred_uncond)
            del latents_2, text_emb_2, padding_mask_2, condition_mask_2
            del noise_pred_2, noise_pred_uncond, noise_pred_cond
        latents = scheduler.step(noise_pred, t, latents)
        mx.eval(latents)
        mx.clear_cache()
        if on_step is not None:
            on_step(i + 1, total_steps)
        step_dt = time.time() - step_t0
        if (i + 1) % 5 == 0 or i == 0 or i == total_steps - 1:
            logger.info(
                "cosmos denoise step %d/%d dt=%.2fs it/s=%.2f",
                i + 1,
                total_steps,
                step_dt / (i + 1),
                (i + 1) / step_dt,
            )
        else:
            logger.debug("cosmos denoise step %d/%d", i + 1, total_steps)
    logger.info(
        "cosmos: denoise done steps=%d total=%.2fs avg_it/s=%.2f",
        total_steps,
        time.time() - step_t0,
        total_steps / (time.time() - step_t0),
    )

    # I2V: blend conditioning back
    if image is not None and is_predict2 and img_cond is not None:
        latents = latents + img_cond * 0.05

    # VAE decode — use tiled for large latents to stay within memory/time limits
    mx.eval(vae.parameters())
    B, C_l, T_l, H_l, W_l = latents.shape
    need_tiled = H_l > 64 or W_l > 64 or T_l > 16
    if need_tiled:
        logger.info("cosmos: using tiled VAE decode for large latent %s", latents.shape)
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

    # Convert to frames — Cosmos VAE outputs [-1,1] (diffusers convention);
    # denormalize to [0,1] before scaling to uint8, matching diffusers
    # VaeImageProcessor.denormalize: (x * 0.5 + 0.5).clamp(0,1)
    frames = video[0]  # (C, T, H, W)
    frames = frames.transpose(1, 2, 3, 0)  # (T, H, W, C)
    frames = mx.clip(frames * 0.5 + 0.5, 0.0, 1.0)
    frames = (frames * 255.0).astype(mx.uint8)
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
