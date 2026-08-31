# SPDX-License-Identifier: Apache-2.0
# Wan2 staged pipeline (issue #410). Extracts the T2V denoise + VAE decode
# path from generate_video() into reusable stage functions so Wan2Backend can
# expose the issue #170 pipeline stage API (load_text_encoder / encode_text /
# load_dit / denoise / load_vae / decode / decode_tiled / unload_*).
#
# Scope: T2V plus I2V / VACE / camera conditioning (#652). The stage API
# serves the Phase-2 FusionTextEncoder -> FusionKSampler -> FusionVAEDecode
# sequential-offload flow. T2V stays the pure-noise path; I2V/VACE/camera
# conditioning is encoded up front by Wan2Backend.encode_control into a
# ControlState, then threaded into run_denoise which mirrors generate.py's
# per-step conditioning bit-exactly (channel-concat y, VACE control_hidden,
# camera y_camera, TI2V mask-blend init + per-step re-apply). Sharing the
# same load_* helpers (load_t5_encoder, load_wan_model, load_vae_decoder) as
# generate.py keeps one weight-loading code path (Rule 7).

import gc
import logging
import math
from pathlib import Path

import mlx.core as mx
import numpy as np

from .scheduler import (
    FlowDPMPP2MScheduler,
    FlowMatchEulerScheduler,
    FlowUniPCScheduler,
)
from .utils import encode_text

from fusion_mlx.engines.video_backends._inpaint import apply_inpaint_mask

logger = logging.getLogger(__name__)

try:
    from fusion_mlx.custom_kernels.xfuser_attention import (
        fast_attn_step as _fa_step,
    )
except Exception:  # pragma: no cover - xfuser strategy optional
    from contextlib import nullcontext as _fa_step


from dataclasses import dataclass
from typing import Any


@dataclass
class ControlState:
    # Conditioning produced by Wan2Backend.encode_control (#652) and threaded
    # into run_denoise so the staged path reproduces generate.py bit-exactly.
    # All fields default to None/False so a T2V call (control=None) stays the
    # pure-noise path — run_denoise treats a None control as an all-default
    # ControlState.
    control_hidden_states: Any = None  # VACE [list of (z,...,t,h,w)] or None
    control_scales: Any = None  # VACE per-layer scales; DiT self-defaults
    y_camera: Any = None  # Fun-Camera [list of (C_cam,F,H,W)] or None
    y_i2v: Any = None  # I2V-14B channel-concat y tensor or None
    z_img: Any = None  # TI2V-5B encoded first-frame latent [z,1,h,w]
    i2v_mask: Any = None  # TI2V-5B blend mask [z,t,h,w]
    i2v_mask_tokens: Any = None  # TI2V-5B per-frame t-token weights [1,t_lat]
    is_i2v_mask_blend: bool = False
    is_i2v_channel_concat: bool = False


_SCHEDULERS = {
    "euler": FlowMatchEulerScheduler,
    "dpm++": FlowDPMPP2MScheduler,
    "unipc": FlowUniPCScheduler,
}


def load_wan_config(model_dir: str | Path):
    # Mirror generate_video() config load (335-435): read config.json if
    # present, else auto-detect from weight shapes. Returns (config,
    # quantization). Auto-corrects stale Wan2.2 VAE params so stage and
    # monolith agree.
    import json

    from .config import WanModelConfig

    model_dir = Path(model_dir)
    config_path = model_dir / "config.json"
    quantization = None
    if config_path.exists():
        with open(config_path) as f:
            config_dict = json.load(f)
        quantization = config_dict.pop("quantization", None)
        for key in ("patch_size", "vae_stride", "window_size", "sample_guide_scale"):
            if key in config_dict and isinstance(config_dict[key], list):
                config_dict[key] = tuple(config_dict[key])
        config = WanModelConfig(
            **{
                k: v
                for k, v in config_dict.items()
                if k in WanModelConfig.__dataclass_fields__
            }
        )
    else:
        if (model_dir / "low_noise_model.safetensors").exists():
            config = WanModelConfig.wan22_t2v_14b()
        else:
            model_path = model_dir / "model.safetensors"
            if not model_path.exists() and (model_dir / "dit").is_dir():
                model_path = model_dir / "dit"
            if model_path.exists():
                from .utils import _load_safetensors

                probe = _load_safetensors(model_path)
                for k, v in probe.items():
                    if "patch_embedding_proj.weight" in k:
                        dim = v.shape[0]
                        if dim <= 2048:
                            config = WanModelConfig.wan21_t2v_1_3b()
                        else:
                            config = WanModelConfig.wan21_t2v_14b()
                        break
                else:
                    config = WanModelConfig.wan21_t2v_14b()
                del probe
            else:
                config = WanModelConfig.wan21_t2v_14b()

    if config.in_dim == 48 and config.vae_z_dim != 48:
        logger.info(
            "stage: auto-correcting Wan2.2 VAE params (in_dim=48, vae_z_dim=%s)",
            config.vae_z_dim,
        )
        config = WanModelConfig(
            **{
                **{
                    f.name: getattr(config, f.name)
                    for f in config.__dataclass_fields__.values()
                },
                "vae_z_dim": 48,
                "vae_stride": (4, 16, 16),
            }
        )
    from .utils import correct_in_dim

    config = correct_in_dim(config, model_dir)
    return config, quantization


def resolve_t5_path(model_dir: Path) -> Path:
    # Replicates generate.py _resolve_model_file for the T5 encoder: flat
    # t5_encoder.safetensors, else text_encoder/ subdir.
    flat = model_dir / "t5_encoder.safetensors"
    if flat.exists():
        return flat
    sub = model_dir / "text_encoder"
    if sub.is_dir() or sub.is_symlink():
        if (sub / "config.json").exists():
            return sub
        safetensors = sorted(sub.glob("*.safetensors"))
        if len(safetensors) == 1:
            return safetensors[0]
        if safetensors:
            return sub
    return flat


def resolve_vae_path(model_dir: Path) -> Path:
    flat = model_dir / "vae.safetensors"
    if flat.exists():
        return flat
    sub = model_dir / "vae"
    if sub.is_dir() or sub.is_symlink():
        safetensors = sorted(sub.glob("*.safetensors"))
        if len(safetensors) == 1:
            return safetensors[0]
        if safetensors:
            return sub
    return flat


def encode_text_stage(
    t5_encoder,
    tokenizer,
    prompt: str,
    text_len: int,
):
    # Stage encode_text: T5-encode a single prompt (positive or negative).
    # Mirrors generate_video() lines 580-588 but returns just this prompt's
    # context (cond and uncond are produced by two separate encode_text calls
    # from two FusionTextEncoder nodes, matching the skyreels stage contract).
    context = encode_text(t5_encoder, tokenizer, prompt, text_len)
    mx.eval(context)
    return context


def _align_dims(config, height: int, width: int):
    # generate.py lines 511-553: align to patch_size*vae_stride, enforce
    # max_area.
    vae_stride = config.vae_stride
    patch_size = config.patch_size
    align_h = patch_size[1] * vae_stride[1]
    align_w = patch_size[2] * vae_stride[2]
    if height % align_h != 0 or width % align_w != 0:
        height = (height // align_h) * align_h
        width = (width // align_w) * align_w
        if height == 0:
            height = align_h
        if width == 0:
            width = align_w
    if config.max_area > 0 and height * width > config.max_area:
        from .generate import _best_output_size

        width, height = _best_output_size(
            width, height, align_w, align_h, config.max_area
        )
    return height, width


def compute_target_shape(config, num_frames: int, height: int, width: int):
    height, width = _align_dims(config, height, width)
    vae_stride = config.vae_stride
    patch_size = config.patch_size
    z_dim = config.vae_z_dim
    t_latent = (num_frames - 1) // vae_stride[0] + 1
    h_latent = height // vae_stride[1]
    w_latent = width // vae_stride[2]
    target_shape = (z_dim, t_latent, h_latent, w_latent)
    seq_len = math.ceil(
        (h_latent * w_latent) / (patch_size[1] * patch_size[2]) * t_latent
    )
    return target_shape, seq_len, height, width


def run_denoise(
    config,
    models,
    context,
    context_null,
    target_shape,
    seq_len,
    steps,
    guide_scale,
    shift,
    scheduler,
    seed,
    no_compile,
    on_step=None,
    control_hidden_states=None,
    control_scales=None,
    y_camera=None,
    y_i2v=None,
    z_img=None,
    i2v_mask=None,
    i2v_mask_tokens=None,
    is_i2v_mask_blend=False,
    is_i2v_channel_concat=False,
    inpaint_mask=None,
    init_latent=None,
):
    # T2V + I2V/VACE/camera denoise loop. T2V body extracted from
    # generate_video() lines 854-1096; conditioning threading mirrors
    # generate.py lines 1047-1184 (#652). models is [single] or [low, high]
    # for dual. Returns the 4D denoised latent (z_dim, t_latent, h_latent,
    # w_latent) — NOT batched; the stage denoise() wrapper adds the batch dim
    # to satisfy the 5D contract. Conditioning arrives as flat kwargs (so
    # callers/tests can assert each); folded into a ControlState here — the
    # per-step body reads control.X. All-None -> T2V pure-noise path,
    # bit-identical to the pre-#652 behavior.
    import random

    control = ControlState(
        control_hidden_states=control_hidden_states,
        control_scales=control_scales,
        y_camera=y_camera,
        y_i2v=y_i2v,
        z_img=z_img,
        i2v_mask=i2v_mask,
        i2v_mask_tokens=i2v_mask_tokens,
        is_i2v_mask_blend=is_i2v_mask_blend,
        is_i2v_channel_concat=is_i2v_channel_concat,
    )

    is_dual = config.dual_model
    cfg_disabled = (
        guide_scale <= 1.0
        if isinstance(guide_scale, (int, float))
        else all(gs <= 1.0 for gs in guide_scale)
    )

    if seed < 0:
        seed = random.randint(0, 2**32 - 1)
    mx.random.seed(seed)
    np.random.seed(seed)

    if is_dual:
        low_noise_model, high_noise_model = models
    else:
        single_model = models[0]

    # Precompute text embeddings (854-882)
    if cfg_disabled:
        if is_dual:
            context_cond_low = low_noise_model.embed_text([context])[0:1]
            context_cond_high = high_noise_model.embed_text([context])[0:1]
            mx.eval(context_cond_low, context_cond_high)
        else:
            context_cond = single_model.embed_text([context])[0:1]
            mx.eval(context_cond)
    else:
        if is_dual:
            context_emb_low = low_noise_model.embed_text([context, context_null])
            context_emb_high = high_noise_model.embed_text([context, context_null])
            mx.eval(context_emb_low, context_emb_high)
            context_cfg_low = mx.concatenate(
                [context_emb_low[0:1], context_emb_low[1:2]], axis=0
            )
            context_cfg_high = mx.concatenate(
                [context_emb_high[0:1], context_emb_high[1:2]], axis=0
            )
        else:
            context_emb = single_model.embed_text([context, context_null])
            mx.eval(context_emb)
            context_cfg = mx.concatenate([context_emb[0:1], context_emb[1:2]], axis=0)

    # Precompute cross-attention K/V (884-900)
    if cfg_disabled:
        if is_dual:
            cross_kv_low = low_noise_model.prepare_cross_kv(context_cond_low)
            cross_kv_high = high_noise_model.prepare_cross_kv(context_cond_high)
            mx.eval(cross_kv_low, cross_kv_high)
        else:
            cross_kv = single_model.prepare_cross_kv(context_cond)
            mx.eval(cross_kv)
    else:
        if is_dual:
            cross_kv_low = low_noise_model.prepare_cross_kv(context_cfg_low)
            cross_kv_high = high_noise_model.prepare_cross_kv(context_cfg_high)
            mx.eval(cross_kv_low, cross_kv_high)
        else:
            cross_kv = single_model.prepare_cross_kv(context_cfg)
            mx.eval(cross_kv)

    # Precompute RoPE (902-916)
    patch_size = config.patch_size
    t_latent = target_shape[1]
    h_latent = target_shape[2]
    w_latent = target_shape[3]
    f_grid = t_latent // patch_size[0]
    h_grid = h_latent // patch_size[1]
    w_grid = w_latent // patch_size[2]
    rope_grid_sizes = (
        [(f_grid, h_grid, w_grid)]
        if cfg_disabled
        else [(f_grid, h_grid, w_grid), (f_grid, h_grid, w_grid)]
    )
    if is_dual:
        rope_cos_sin_low = low_noise_model.prepare_rope(rope_grid_sizes)
        rope_cos_sin_high = high_noise_model.prepare_rope(rope_grid_sizes)
        mx.eval(rope_cos_sin_low, rope_cos_sin_high)
    else:
        rope_cos_sin = single_model.prepare_rope(rope_grid_sizes)
        mx.eval(rope_cos_sin)

    # Scheduler + initial noise (918-938). T2V: pure noise; TI2V-5B mask-blend
    # blends the encoded first-frame latent with noise (generate.py 1047-1054).
    sched_cls = _SCHEDULERS.get(scheduler, FlowUniPCScheduler)
    sched = sched_cls(num_train_timesteps=config.num_train_timesteps)
    sched.set_timesteps(steps, shift=shift)
    noise = mx.random.normal(target_shape)
    if (
        control.is_i2v_mask_blend
        and control.i2v_mask is not None
        and control.z_img is not None
    ):
        latents = (1.0 - control.i2v_mask) * control.z_img + control.i2v_mask * noise
    else:
        latents = noise
    boundary = (config.boundary * config.num_train_timesteps) if is_dual else None

    if not no_compile:
        models_to_compile = (
            [high_noise_model, low_noise_model] if is_dual else [single_model]
        )
        for m in models_to_compile:
            m._compiled = mx.compile(m)

    timestep_list = sched.timesteps.tolist()
    logger.info(
        "stage denoise: steps=%d cfg_disabled=%s dual=%s seed=%d shape=%s",
        steps,
        cfg_disabled,
        is_dual,
        seed,
        target_shape,
    )

    for i, _t in enumerate(range(steps)):
        timestep_val = timestep_list[i]

        if is_dual:
            if timestep_val >= boundary:
                model = high_noise_model
                kv = cross_kv_high
                rcs = rope_cos_sin_high
            else:
                model = low_noise_model
                kv = cross_kv_low
                rcs = rope_cos_sin_low
        else:
            model = single_model
            kv = cross_kv
            rcs = rope_cos_sin

        _call = getattr(model, "_compiled", model)

        if cfg_disabled:
            # TI2V-5B mask-blend: per-step t-token weights (generate.py 1097-1104).
            if control.is_i2v_mask_blend and control.i2v_mask_tokens is not None:
                t_tokens = control.i2v_mask_tokens * timestep_val
                pad_len = seq_len - t_tokens.shape[1]
                if pad_len > 0:
                    t_tokens = mx.concatenate(
                        [t_tokens, mx.full((1, pad_len), timestep_val)], axis=1
                    )
                t_batch = t_tokens
            else:
                t_batch = mx.array([timestep_val])

            y_arg = [control.y_i2v] if control.is_i2v_channel_concat else None

            ctx = (
                context_cond_high
                if (is_dual and timestep_val >= boundary)
                else (context_cond_low if is_dual else context_cond)
            )
            with _fa_step(i):
                preds = _call(
                    [latents],
                    t=t_batch,
                    context=ctx,
                    seq_len=seq_len,
                    cross_kv_caches=kv,
                    y=y_arg,
                    rope_cos_sin=rcs,
                    control_hidden_states=control.control_hidden_states,
                    control_scales=control.control_scales,
                    y_camera=control.y_camera,
                )
            noise_pred = preds[0]
            del preds
        else:
            if is_dual:
                gs = (
                    guide_scale
                    if isinstance(guide_scale, (int, float))
                    else (
                        guide_scale[1] if timestep_val >= boundary else guide_scale[0]
                    )
                )
            else:
                gs = (
                    guide_scale
                    if isinstance(guide_scale, (int, float))
                    else guide_scale[0]
                )
            t_batch = mx.array([timestep_val, timestep_val])
            # TI2V-5B mask-blend CFG: duplicate t-token weights for B=2 (1145-1152).
            if control.is_i2v_mask_blend and control.i2v_mask_tokens is not None:
                t_tokens = control.i2v_mask_tokens * timestep_val
                pad_len = seq_len - t_tokens.shape[1]
                if pad_len > 0:
                    t_tokens = mx.concatenate(
                        [t_tokens, mx.full((1, pad_len), timestep_val)], axis=1
                    )
                t_batch = mx.concatenate([t_tokens, t_tokens], axis=0)

            y_arg = (
                [control.y_i2v, control.y_i2v]
                if control.is_i2v_channel_concat
                else None
            )

            ctx = (
                context_cfg_high
                if (is_dual and timestep_val >= boundary)
                else (context_cfg_low if is_dual else context_cfg)
            )
            with _fa_step(i):
                preds = _call(
                    [latents, latents],
                    t=t_batch,
                    context=ctx,
                    seq_len=seq_len,
                    cross_kv_caches=kv,
                    y=y_arg,
                    rope_cos_sin=rcs,
                    control_hidden_states=control.control_hidden_states,
                    control_scales=control.control_scales,
                    y_camera=control.y_camera,
                )
            noise_pred_cond, noise_pred_uncond = preds[0], preds[1]
            noise_pred = noise_pred_uncond + gs * (noise_pred_cond - noise_pred_uncond)
            del noise_pred_cond, noise_pred_uncond, preds

        latents = sched.step(noise_pred[None], timestep_val, latents[None]).squeeze(0)

        # #653 Surface C: frozen-region re-composite. Orthogonal to the
        # TI2V mask blend below — runs even when is_i2v_mask_blend is False.
        # mask=1 -> reactive (keep denoised); mask=0 -> frozen (restore init).
        if inpaint_mask is not None and init_latent is not None:
            latents = apply_inpaint_mask(latents, init_latent, inpaint_mask)

        # TI2V-5B: re-apply mask to keep first frame frozen (generate.py 1183).
        if (
            control.is_i2v_mask_blend
            and control.i2v_mask is not None
            and control.z_img is not None
        ):
            latents = (
                1.0 - control.i2v_mask
            ) * control.z_img + control.i2v_mask * latents

        del noise_pred
        mx.eval(latents)

        if i <= 2 or i == steps - 1:
            _nan_c = int(np.isnan(np.array(latents)).sum())
            if _nan_c > 0:
                logger.warning(
                    "stage denoise step %d/%d: %d NaN in latents — aborting",
                    i + 1,
                    steps,
                    _nan_c,
                )
                break
        if on_step is not None:
            on_step(i + 1, steps)

    # Release precomputed tensors (1130-1147)
    if is_dual:
        del cross_kv_low, cross_kv_high, rope_cos_sin_low, rope_cos_sin_high
        if cfg_disabled:
            del context_cond_low, context_cond_high
        else:
            del context_cfg_low, context_cfg_high
    else:
        del cross_kv, rope_cos_sin
        if cfg_disabled:
            del context_cond
        else:
            del context_cfg
    del model, kv
    gc.collect()
    mx.clear_cache()
    # Ensure the returned latents are fully materialized on this executor
    # thread. The staged path runs VAE decode in a *separate* executor call and
    # round-trips this array through the event-loop main thread; MLX Metal
    # streams are thread-local, so a still-lazy array evaluated later on the
    # decode thread raises "There is no Stream(gpu, N) in current thread". The
    # caller (wan2.py denoise()) further evaluates the batch-dim projection on
    # this same thread before returning. The monolith shares one executor call
    # so this is a no-op for it (still correct).
    mx.eval(latents)
    return latents


def decode_wan_vae(latent, config, vae, tiling_config=None):
    # VAE decode extracted from generate_video() lines 1149-1254 (T2V branch,
    # no I2V mask_blend). latent is 4D (z_dim, t_latent, h_lat, w_lat).
    # Returns uint8 frames [T, H, W, 3].
    # Materialize the incoming latent on this executor thread. In the staged
    # path it is produced by denoise() in a *separate* executor call and
    # round-trips through the event-loop main thread; MLX Metal streams are
    # thread-local, so a still-lazy array (or a slice built off-thread) raises
    # "There is no Stream(gpu, N) in current thread" at the decode-side
    # mx.eval. The caller (Wan2Backend.decode) eval's the array on the
    # caller's thread BEFORE dispatching here, which materializes the data
    # and detaches stream affinity so this eval is portable.
    mx.eval(latent)
    is_wan22_vae = config.vae_z_dim == 48
    if is_wan22_vae:
        from .vae22 import denormalize_latents

        z = latent.transpose(1, 2, 3, 0)[None]
        z = denormalize_latents(z)
        if tiling_config is not None:
            video = vae.decode_tiled(z, tiling_config)
        else:
            video = vae(z)
        mx.eval(video)
        video = np.array(video[0])
        video = (video + 1.0) / 2.0
        nan_count = int(np.isnan(video).sum())
        if nan_count > 0:
            logger.warning(
                "stage VAE decode (wan22): %d NaN, replacing with 0", nan_count
            )
            video = np.nan_to_num(video, nan=0.0)
        video = np.clip(video * 255.0, 0, 255).astype(np.uint8)
    else:
        if tiling_config is not None:
            video = vae.decode_tiled(latent[None], tiling_config)
        else:
            video = vae.decode(latent[None])
        mx.eval(video)
        video = np.array(video[0])
        video = (video + 1.0) / 2.0
        nan_count = int(np.isnan(video).sum())
        if nan_count > 0:
            logger.warning(
                "stage VAE decode (wan21): %d NaN, replacing with 0", nan_count
            )
            video = np.nan_to_num(video, nan=0.0)
        video = np.clip(video * 255.0, 0, 255).astype(np.uint8)
        video = video.transpose(1, 2, 3, 0)
    return video


def encode_wan_vae(x_ncthw, config, vae_encoder):
    # VAE encode — inverse of decode_wan_vae. Input is NCTHW (1,3,T,H,W)
    # float32 on the video executor thread. Returns raw 4D latent
    # (z_dim, t_lat, h_lat, w_lat) from vae_encoder.encode. Caller
    # (Wan2Backend.encode) wraps to 5D + materializes. The caller eval's the
    # input on the caller's thread before dispatching so this eval is
    # portable across thread-local MLX streams (see decode_wan_vae).
    logger.info("stage VAE encode wan2 in_shape=%s", tuple(x_ncthw.shape))
    mx.eval(x_ncthw)
    lat = vae_encoder.encode(x_ncthw)
    mx.eval(lat)
    return lat
