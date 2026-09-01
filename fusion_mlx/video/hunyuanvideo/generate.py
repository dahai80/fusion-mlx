# SPDX-License-Identifier: Apache-2.0
# HunyuanVideo generate: T2V + I2V.

import gc
import logging
import os
import time

import mlx.core as mx
import numpy as np

from fusion_mlx.engines.video_backends._inpaint import apply_inpaint_mask

from .dit import HUNYUAN_VIDEO_CONFIG, HunyuanVideoDiT
from .scheduler import HunyuanVideoScheduler
from .text_encoder import HunyuanDualTextEncoder
from .vae import HunyuanVideoVAE

logger = logging.getLogger(__name__)

_DEFAULT_FPS = 24
_DEFAULT_NUM_FRAMES = 33
_DEFAULT_WIDTH = 720
_DEFAULT_HEIGHT = 480


_CLIP_TOKENIZER_ID = "openai/clip-vit-large-patch14"
_LLAMA_TOKENIZER_ID = "unsloth/llama-3-8b-bnb-4bit"
_LLAMA_MAX_LENGTH = 256

_clip_tokenizer = None
_llama_tokenizer = None


def _get_clip_tokenizer():
    global _clip_tokenizer
    if _clip_tokenizer is None:
        from transformers import CLIPTokenizer

        _clip_tokenizer = CLIPTokenizer.from_pretrained(_CLIP_TOKENIZER_ID)
        logger.info("hunyuan: loaded CLIP tokenizer from %s", _CLIP_TOKENIZER_ID)
    return _clip_tokenizer


def _get_llama_tokenizer():
    global _llama_tokenizer
    if _llama_tokenizer is None:
        from transformers import AutoTokenizer

        _llama_tokenizer = AutoTokenizer.from_pretrained(_LLAMA_TOKENIZER_ID)
        logger.info("hunyuan: loaded Llama3 tokenizer from %s", _LLAMA_TOKENIZER_ID)
    return _llama_tokenizer


def _tokenize_clip(prompt, max_length=77):
    tok = _get_clip_tokenizer()
    ids = tok(
        prompt,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="np",
    )
    return mx.array(ids["input_ids"], dtype=mx.int32)


def _tokenize_llama(prompt, max_length=_LLAMA_MAX_LENGTH):
    tok = _get_llama_tokenizer()
    ids = tok(
        prompt,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="np",
    )
    return mx.array(ids["input_ids"], dtype=mx.int32)


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
    controlnet_image=None,
    controlnet_adapter=None,
    controlnet_latent=None,
    inpaint_mask=None,
    init_latent=None,
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

    clip_ids = _tokenize_clip(prompt)
    llama_ids = _tokenize_llama(prompt)
    text_emb, text_pooled = text_encoder(clip_ids, llama_ids)
    text_emb_null = mx.zeros_like(text_emb)
    text_pooled_null = mx.zeros_like(text_pooled)

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
    # #367 perf: CFG batched guidance — fuse uncond+cond into a single B=2
    # forward (DiT is batch-safe along dim 0) instead of two separate full
    # forwards. ~2x throughput, no quality change. cfg<=1.0 skips the uncond
    # branch entirely (single-forward shortcut, useful for guidance-distilled
    # / low-cfg workflows).
    use_single_forward = cfg <= 1.0
    if use_single_forward:
        logger.info("hunyuan: cfg=%.2f <=1.0, single-forward (no uncond branch)", cfg)
    logger.info(
        "hunyuan: denoise start steps=%d cfg=%.2f batched_cfg=%s",
        total_steps,
        cfg,
        not use_single_forward,
    )
    step_t0 = time.time()
    if controlnet_image is not None:
        raise RuntimeError(
            "hunyuanvideo: ControlNet (Surface B) not available for this backend"
            " — no per-backend ControlNet model (see issue #733 follow-up)."
            " Refusing to silently degrade to T2V (#733)."
        )
    logger.info(
        "hunyuanvideo denoise: inpaint=%s controlnet=%s",
        inpaint_mask is not None,
        controlnet_image is not None,
    )
    for i, t in enumerate(scheduler.timesteps):
        if use_single_forward:
            timestep = mx.array([float(t)], dtype=mx.float32)
            guidance = mx.array([cfg], dtype=mx.float32)
            noise_pred = dit(
                latents,
                timestep,
                text_emb,
                pooled_emb=text_pooled,
                guidance=guidance,
                image_cond=image_cond,
            )
            mx.eval(noise_pred)
        else:
            timestep = mx.array([float(t)] * 2, dtype=mx.float32)
            guidance = mx.array([cfg, cfg], dtype=mx.float32)
            latents_2 = mx.concatenate([latents, latents], axis=0)
            text_emb_2 = mx.concatenate([text_emb_null, text_emb], axis=0)
            pooled_2 = mx.concatenate([text_pooled_null, text_pooled], axis=0)
            image_cond_2 = (
                mx.concatenate([image_cond, image_cond], axis=0)
                if image_cond is not None
                else None
            )
            noise_pred_2 = dit(
                latents_2,
                timestep,
                text_emb_2,
                pooled_emb=pooled_2,
                guidance=guidance,
                image_cond=image_cond_2,
            )
            mx.eval(noise_pred_2)
            noise_pred_uncond, noise_pred_cond = noise_pred_2[0:1], noise_pred_2[1:2]
            noise_pred = noise_pred_uncond + cfg * (noise_pred_cond - noise_pred_uncond)
            del latents_2, text_emb_2, pooled_2, image_cond_2
            del noise_pred_2, noise_pred_uncond, noise_pred_cond
        latents = scheduler.step(noise_pred, t, latents)
        del noise_pred
        mx.eval(latents)
        if inpaint_mask is not None and init_latent is not None:
            latents = apply_inpaint_mask(latents, init_latent, inpaint_mask)
            mx.eval(latents)
        mx.clear_cache()
        if on_step is not None:
            on_step(i + 1, total_steps)
        step_dt = time.time() - step_t0
        if (i + 1) % 5 == 0 or i == 0 or i == total_steps - 1:
            logger.info(
                "hunyuan denoise step %d/%d dt=%.2fs it/s=%.2f",
                i + 1,
                total_steps,
                step_dt / (i + 1),
                (i + 1) / step_dt,
            )
        else:
            logger.debug("hunyuan denoise step %d/%d", i + 1, total_steps)
    logger.info(
        "hunyuan: denoise done steps=%d total=%.2fs avg_it/s=%.2f",
        total_steps,
        time.time() - step_t0,
        total_steps / (time.time() - step_t0),
    )

    # Free DiT and text encoder before VAE decode to reduce peak memory
    del dit, text_encoder, text_emb, text_emb_null, text_pooled, text_pooled_null
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
