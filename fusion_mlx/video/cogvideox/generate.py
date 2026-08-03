# SPDX-License-Identifier: Apache-2.0
import argparse
import gc
import logging
import random
import time
from collections.abc import Callable

import mlx.core as mx
import numpy as np

logger = logging.getLogger(__name__)

from .scheduler import FlowMatchScheduler
from .utils import (
    get_model_path,
    load_config,
    load_transformer,
    load_vae_decoder,
    load_vae_encoder,
)

try:
    from fusion_mlx.custom_kernels.xfuser_attention import fast_attn_step as _fa_step
except Exception:
    from contextlib import nullcontext as _fa_step


class Colors:
    CYAN = "\033[96m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    MAGENTA = "\033[95m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"


def generate_video(
    model_dir: str,
    prompt: str,
    negative_prompt: str | None = None,
    image: str | None = None,
    width: int = 720,
    height: int = 480,
    num_frames: int = 49,
    steps: int | None = None,
    guide_scale: float | None = None,
    shift: float | None = None,
    seed: int = -1,
    output_path: str = "output.mp4",
    no_compile: bool = True,
    on_step_sync: Callable[[int, int], None] | None = None,
    session_id: str | None = None,
):
    model_dir = get_model_path(model_dir)
    config, quantization = load_config(model_dir)

    is_i2v = image is not None
    if steps is None:
        steps = config.sample_steps
    if shift is None:
        shift = config.sample_shift
    if guide_scale is None:
        guide_scale = config.sample_guide_scale

    cfg_disabled = guide_scale <= 1.0
    if negative_prompt is None:
        negative_prompt = ""

    if seed < 0:
        seed = random.randint(0, 2**32 - 1)
    mx.random.seed(seed)
    np.random.seed(seed)

    p = config.patch_size
    pt = config.patch_size_t if config.patch_size_t is not None else p
    tcr = config.vae_temporal_compression_ratio
    align_h = p * tcr
    align_w = p * tcr
    if height % align_h != 0:
        height = (height // align_h) * align_h
    if width % align_w != 0:
        width = (width // align_w) * align_w

    latent_t = (num_frames - 1) // tcr + 1
    latent_h = height // tcr
    latent_w = width // tcr
    target_shape = (config.vae_latent_channels, latent_t, latent_h, latent_w)

    pipeline_str = "Image-to-Video" if is_i2v else "Text-to-Video"
    variant = "5B" if config.num_attention_heads > 30 else "2B"
    print(f"{Colors.CYAN}{'=' * 60}")
    print(f"  CogVideoX-{variant} {pipeline_str} Generation (MLX)")
    print(f"{'=' * 60}{Colors.RESET}")
    print(f"{Colors.DIM}  Prompt: {prompt}")
    print(f"  Size: {width}x{height}, Frames: {num_frames}")
    print(f"  Steps: {steps}, Guide: {guide_scale}, Shift: {shift}")
    print(f"  Seed: {seed}{Colors.RESET}")

    # Load T5 encoder
    t1 = time.time()
    print(f"\n{Colors.BLUE}Loading T5 encoder...{Colors.RESET}")
    from .text_encoder import encode_text, load_t5_encoder

    t5_path = model_dir / "t5_encoder.safetensors"
    t5_encoder = load_t5_encoder(t5_path, config)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("google/umt5-xxl")

    print(f"{Colors.BLUE}Encoding text...{Colors.RESET}")
    context = encode_text(t5_encoder, tokenizer, prompt, config.max_text_seq_length)
    if not cfg_disabled:
        context_null = encode_text(
            t5_encoder, tokenizer, negative_prompt, config.max_text_seq_length
        )
        mx.eval(context, context_null)
    else:
        mx.eval(context)

    del t5_encoder
    gc.collect()
    mx.clear_cache()
    print(f"{Colors.DIM}  T5 encoding: {time.time() - t1:.1f}s{Colors.RESET}")

    # I2V: encode image
    z_img = None
    if is_i2v:
        print(f"\n{Colors.BLUE}Encoding input image...{Colors.RESET}")
        t_img = time.time()
        from PIL import Image as PILImage

        img = PILImage.open(image).convert("RGB")
        scale = max(width / img.width, height / img.height)
        img = img.resize(
            (round(img.width * scale), round(img.height * scale)), PILImage.LANCZOS
        )
        x1, y1 = (img.width - width) // 2, (img.height - height) // 2
        img = img.crop((x1, y1, x1 + width, y1 + height))
        img_arr = mx.array(np.array(img, dtype=np.float32) / 255.0 * 2.0 - 1.0)
        img_chw = img_arr.transpose(2, 0, 1)

        video = mx.concatenate(
            [img_chw[:, None, :, :], mx.zeros((3, num_frames - 1, height, width))],
            axis=1,
        )
        vae_path = model_dir / "vae.safetensors"
        vae_enc = load_vae_encoder(vae_path, config)
        z_video = vae_enc.encode(video[None])
        mx.eval(z_video)
        z_img = z_video[0]
        del vae_enc, img_arr, img_chw, video, z_video
        gc.collect()
        mx.clear_cache()
        print(f"{Colors.DIM}  Image encoding: {time.time() - t_img:.1f}s{Colors.RESET}")

    # Load transformer
    print(f"\n{Colors.BLUE}Loading transformer model...{Colors.RESET}")
    t2 = time.time()
    transformer_path = model_dir / "model.safetensors"
    model = load_transformer(transformer_path, config, quantization)
    print(f"{Colors.DIM}  Model loaded: {time.time() - t2:.1f}s{Colors.RESET}")

    # Precompute RoPE
    rope_cos, rope_sin = model._precompute_rope(num_frames, height, width)
    mx.eval(rope_cos, rope_sin)

    # Scheduler
    sched = FlowMatchScheduler(
        num_train_timesteps=config.num_train_timesteps, shift=shift
    )
    sched.set_timesteps(steps, shift=shift)

    # Noise
    noise = mx.random.normal(target_shape)
    latents = noise

    # Compile
    if not no_compile:
        model._compiled = mx.compile(model)

    # Denoise
    print(f"\n{Colors.GREEN}Denoising ({steps} steps)...{Colors.RESET}")
    t3 = time.time()
    timestep_list = sched.timesteps.tolist()

    for i in range(steps):
        t_val = timestep_list[i]
        t_batch = mx.array([t_val] if cfg_disabled else [t_val, t_val])

        if cfg_disabled:
            _call = getattr(model, "_compiled", model)
            with _fa_step(i):
                pred = _call(
                    latents[None],
                    encoder_hidden_states=context[None],
                    timestep=t_batch[:1],
                    rope_cos=rope_cos,
                    rope_sin=rope_sin,
                )
            noise_pred = pred[0]
        else:
            _call = getattr(model, "_compiled", model)
            with _fa_step(i):
                pred_cond = _call(
                    latents[None],
                    encoder_hidden_states=context[None],
                    timestep=t_batch[:1],
                    rope_cos=rope_cos,
                    rope_sin=rope_sin,
                )
                pred_uncond = _call(
                    latents[None],
                    encoder_hidden_states=context_null[None],
                    timestep=t_batch[:1],
                    rope_cos=rope_cos,
                    rope_sin=rope_sin,
                )
            noise_pred_cond = pred_cond[0]
            noise_pred_uncond = pred_uncond[0]
            noise_pred = noise_pred_uncond + guide_scale * (
                noise_pred_cond - noise_pred_uncond
            )
            del noise_pred_cond, noise_pred_uncond

        latents = sched.step(noise_pred[None], t_val, latents[None]).squeeze(0)
        del noise_pred
        mx.eval(latents)
        if on_step_sync is not None:
            on_step_sync(i + 1, steps)

    print(f"{Colors.DIM}  Denoising: {time.time() - t3:.1f}s{Colors.RESET}")

    # Session tail
    if session_id is not None:
        from fusion_mlx.cache.latent_cache import put_session_tail

        tail = latents[:, -1:, :, :]
        put_session_tail(session_id, str(model_dir), tail)

    del model, context
    if not cfg_disabled:
        del context_null
    gc.collect()
    mx.clear_cache()

    # VAE decode
    print(f"\n{Colors.BLUE}Decoding with VAE...{Colors.RESET}")
    t4 = time.time()
    vae_path = model_dir / "vae.safetensors"
    vae = load_vae_decoder(vae_path, config)
    video = vae.decode(latents[None])
    mx.eval(video)
    print(f"{Colors.DIM}  VAE decode: {time.time() - t4:.1f}s{Colors.RESET}")

    del latents
    mx.clear_cache()

    video = np.array(video[0])
    if video.shape[1] == 3:
        video = (video + 1.0) / 2.0
        video = np.clip(video * 255.0, 0, 255).astype(np.uint8)
        video = video.transpose(1, 2, 3, 0)
    else:
        video = (video + 1.0) / 2.0
        video = np.clip(video * 255.0, 0, 255).astype(np.uint8)

    from fusion_mlx.video.wan2.postprocess import save_video

    save_video(video, output_path, fps=config.sample_fps)
    print(f"\n{Colors.GREEN}✓ Video saved to {output_path}{Colors.RESET}")
    print(f"{Colors.DIM}  Total time: {time.time() - t1:.1f}s{Colors.RESET}")


def main():
    parser = argparse.ArgumentParser(description="CogVideoX Video Generation (MLX)")
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--negative-prompt", type=str, default=None)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--guide-scale", type=float, default=None)
    parser.add_argument("--shift", type=float, default=None)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--output-path", type=str, default="output.mp4")
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()

    generate_video(
        model_dir=args.model_dir,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        image=args.image,
        width=args.width,
        height=args.height,
        num_frames=args.num_frames,
        steps=args.steps,
        guide_scale=args.guide_scale,
        shift=args.shift,
        seed=args.seed,
        output_path=args.output_path,
        no_compile=args.no_compile,
    )


if __name__ == "__main__":
    main()
