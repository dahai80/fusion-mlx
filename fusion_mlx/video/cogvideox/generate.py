# SPDX-License-Identifier: Apache-2.0
import argparse
import gc
import logging
import random
import time
from collections.abc import Callable
from typing import Any

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

from fusion_mlx.engines.video_backends._inpaint import apply_inpaint_mask


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


def run_denoise(
    model,
    latents,
    context,
    context_null,
    cfg_disabled: bool,
    guide_scale: float,
    steps: int,
    timestep_list,
    sched,
    rope_cos,
    rope_sin,
    on_step_sync: Callable[[int, int], None] | None = None,
    # #731 Surface B: ControlNet adapter + preprocessed control latent.
    # Threaded for contract parity with Wan2 run_denoise (#653); cogvideox has
    # no per-backend ControlNet model so generate_video gates this to None and
    # fails visibly when controlnet_image is set (no silent T2V degrade).
    controlnet_adapter: Any = None,
    controlnet_latent: Any = None,
    # #731 Surface C: inpaint-mask re-composite after each sched.step.
    # mask=1 -> reactive (keep denoised); mask=0 -> frozen (restore init).
    # None -> T2V passthrough, bit-identical to pre-#731 behavior.
    inpaint_mask: Any = None,
    init_latent: Any = None,
):
    # Separable per-step denoise loop extracted from generate_video (#731),
    # mirroring the Wan2 stage.py:run_denoise precedent (#653). Returns the
    # denoised latent (4D C-first). All-None control -> pure-noise path.
    logger.info(
        "cogvideox run_denoise: steps=%d cfg_disabled=%s inpaint=%s controlnet=%s",
        steps,
        cfg_disabled,
        inpaint_mask is not None,
        controlnet_adapter is not None,
    )

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

        # #731 Surface C: frozen-region re-composite after each step.
        # DiT-agnostic, latent-space only; None -> passthrough.
        if inpaint_mask is not None and init_latent is not None:
            latents = apply_inpaint_mask(latents, init_latent, inpaint_mask)
            mx.eval(latents)

        if on_step_sync is not None:
            on_step_sync(i + 1, steps)

    return latents


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
    # #731 Surface B+C: ControlNet residual threading (B) and inpaint-mask
    # re-composite (C). All default None -> T2V pure-noise path, bit-identical
    # to pre-#731 behavior. controlnet_image fails visibly (no backend model).
    controlnet_image: str | None = None,
    controlnet_adapter: Any = None,
    controlnet_latent: Any = None,
    inpaint_mask: Any = None,
    init_latent: Any = None,
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

    # Surface B (#731): ControlNet residual injection is gated behind a real
    # per-backend ControlNet adapter. cogvideox has no ControlNet model yet
    # (the shared ControlNet adapter is Wan2-arch). controlnet_image on this
    # backend must fail visibly (Rule 12) rather than silently degrade to T2V.
    if controlnet_image is not None:
        raise RuntimeError(
            "cogvideox: ControlNet (Surface B) not available for this backend — "
            "no per-backend ControlNet model (see issue #731 follow-up). "
            "Refusing to silently degrade to T2V (#731)."
        )

    # Denoise (#731): extracted to run_denoise so Surface C (inpaint re-composite)
    # and Surface B (controlnet threading) land in a separable, testable loop.
    print(f"\n{Colors.GREEN}Denoising ({steps} steps)...{Colors.RESET}")
    t3 = time.time()
    latents = run_denoise(
        model=model,
        latents=latents,
        context=context,
        context_null=context_null if not cfg_disabled else None,
        cfg_disabled=cfg_disabled,
        guide_scale=guide_scale,
        steps=steps,
        timestep_list=sched.timesteps.tolist(),
        sched=sched,
        rope_cos=rope_cos,
        rope_sin=rope_sin,
        on_step_sync=on_step_sync,
        inpaint_mask=inpaint_mask,
        init_latent=init_latent,
        controlnet_adapter=controlnet_adapter,
        controlnet_latent=controlnet_latent,
    )
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
