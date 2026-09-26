# SPDX-License-Identifier: Apache-2.0
# LTX-2.5 generate_video orchestration (T2V distilled E2E)。
# 两阶段 distilled 生成编排：text-encoder(Gemma4-12b) → connector →
# stage1 denoise → spatial upsampler → stage2 denoise → temporal upsampler →
# VAE decode → mp4 bytes。
#
# 复用策略：positions / VAE decode / upsample_latents / sigmas 直接调用 ltx2
# 原语（纯数学/共享架构，见 memory 独立规则仅约束 transformer/MODEL 代码）。
# 去噪骨架用 ltx2_5/denoise.py（ltx2_5.Modality 与 ltx2.Modality 跨模块不兼容）。
# connector 显式在 generate 中运行（LTX2_5Model.__call__/prepare 不调用它）。
#
# 真实模型验证 (CLAUDE.md「须真实加载模型」)：22B transformer + 12b TE ≈ 50GB，
# 跑前须 start.sh stop 释放显存。
from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from fusion_mlx.cache.latent_cache import get_image_latent_cache, image_latent_key

from ..ltx2.conditioning import (
    LatentState,
    VideoConditionByLatentIndex,
    apply_conditioning,
)
from ..ltx2.denoise import denoise_dev_av
from ..ltx2.positions import (
    AUDIO_LATENT_CHANNELS,
    AUDIO_MEL_BINS,
    compute_audio_frames,
    create_audio_position_grid,
    create_position_grid,
)
from ..ltx2.upsampler import upsample_latents
from ..ltx2.utils import load_image, prepare_image_for_encoding
from .audio import (
    decode_audio,
    load_audio_decoder,
    load_vocoder_model,
    mux_video_audio,
    save_audio,
)
from .config import LTX2_5Variant
from .denoise import denoise_distilled_av, denoise_distilled_t2v
from .ltx2_5_model import LTX2_5Model
from .scheduler import (
    DISTILLED_STAGE_1_SIGMAS,
    DISTILLED_STAGE_2_SIGMAS,
    dev_sigmas,
)
from .text_encoder import load_text_encoder
from .upsampler import load_spatial_upsampler_2_5, load_temporal_upsampler
from .utils import get_model_path, is_split_layout, resolve_component
from .video_vae import load_video_decoder, load_video_encoder

logger = logging.getLogger(__name__)


def _encode_image_latent(
    src,
    h,
    w,
    model_repo,
    root,
    model_dtype,
    latent_cache,
    vae_encoder,
):
    # #782: VAE-encode a single image at (h,w) -> 128-channel latent for I2V
    # conditioning。ltx2_5 用 load_video_encoder(path) (conv VAE) + 复用 ltx2
    # 的 load_image / prepare_image_for_encoding (纯图像 helper, 架构无关)。
    # latent_cache (UMA Radix) hit 时零拷贝复用, 跳过 VAE encoder load+forward。
    key = image_latent_key(model_repo, src, h, w, model_dtype)
    if latent_cache is not None:
        cached = latent_cache.get(key)
        if cached is not None:
            logger.info("ltx2_5 latent cache hit: %dx%d (%s)", h, w, key)
            return cached, vae_encoder
    if vae_encoder is None:
        if is_split_layout(root):
            enc_path = resolve_component(root, "video_vae_conv_encoder")
        else:
            enc_path = resolve_component(root, "video_vae_conv")
        vae_encoder = load_video_encoder(enc_path)
        mx.eval(vae_encoder.parameters())
    loaded = load_image(src, height=h, width=w, dtype=model_dtype)
    latent = vae_encoder(prepare_image_for_encoding(loaded, h, w, dtype=model_dtype))
    mx.eval(latent)
    if latent_cache is not None:
        latent_cache.put(key, latent)
        logger.info("ltx2_5 latent cache miss+insert: %dx%d (%s)", h, w, key)
    return latent, vae_encoder


def _build_i2v_conditionings(
    image_latent,
    image_frame_idx: int,
    image_strength: float,
    end_image_latent=None,
    end_image_strength: float = 1.0,
):
    # 与 ltx2 同构: 首帧条件 frame_idx (有 end_image 时固定 0), 尾帧 frame_idx=-1。
    conditionings = []
    if image_latent is not None:
        idx = 0 if end_image_latent is not None else image_frame_idx
        conditionings.append(
            VideoConditionByLatentIndex(
                latent=image_latent, frame_idx=idx, strength=image_strength
            )
        )
    if end_image_latent is not None:
        conditionings.append(
            VideoConditionByLatentIndex(
                latent=end_image_latent, frame_idx=-1, strength=end_image_strength
            )
        )
    return conditionings


def _build_i2v_state(
    image_latent,
    latent_shape,
    noise_scale,
    image_frame_idx: int,
    image_strength: float,
    model_dtype,
):
    # #977: shared I2V latent-state constructor (dev single-stage + distilled
    # stages share this pattern). zeros latent + image clean_latent at
    # image_frame_idx, denoise_mask=1-strength (condition frame frozen clean).
    # renoise free frames (mask=1) to noise_scale; condition frames (mask=0)
    # stay at the (zero) base latent. Returns LatentState ready for
    # denoise_distilled_t2v(state=).
    state = LatentState(
        latent=mx.zeros(latent_shape, dtype=model_dtype),
        clean_latent=mx.zeros(latent_shape, dtype=model_dtype),
        denoise_mask=mx.ones((1, 1, latent_shape[2], 1, 1), dtype=model_dtype),
    )
    conditionings = _build_i2v_conditionings(
        image_latent, image_frame_idx, image_strength
    )
    state = apply_conditioning(state, conditionings)
    noise = mx.random.normal(latent_shape, dtype=model_dtype)
    scaled_mask = state.denoise_mask * mx.array(noise_scale, dtype=model_dtype)
    state = LatentState(
        latent=noise * scaled_mask
        + state.latent * (mx.array(1.0, dtype=model_dtype) - scaled_mask),
        clean_latent=state.clean_latent,
        denoise_mask=state.denoise_mask,
    )
    return state


_LTX2_5_TEMPORAL_TILE_FRAMES = 128
_LTX2_5_TEMPORAL_OVERLAP_FRAMES = 64
_LTX2_5_AUTO_TILING_FRAME_THRESHOLD = 65


def _resolve_ltx2_5_tiling_config(tiling: str, num_frames: int):
    # Map tiling string -> TilingConfig for ltx2_5 conv VAE decoder.
    # ltx2_5 CausalConv3d REFLECT padding at spatial tile edges produces visible
    # color seams (#937 reverted -> #939). Spatial tiling DISABLED for ltx2_5:
    # every config with spatial_config -> spatial_config=None (temporal-only).
    # Temporal-only keeps full spatial intact -> no spatial seams; chunks frames
    # with overlap+blend -> bounds memory. NOT bit-exact vs full decode (temporal
    # RF ~40 latent frames; 64f overlap only partially masks boundary drift) but
    # avoids the hard OOM crash that #945 reports.
    from ..ltx2.video_vae.tiling import TilingConfig

    if tiling == "none":
        return None
    needs_temporal = num_frames > _LTX2_5_AUTO_TILING_FRAME_THRESHOLD
    if tiling == "auto":
        if not needs_temporal:
            return None
        return TilingConfig.temporal_only(
            tile_size=_LTX2_5_TEMPORAL_TILE_FRAMES,
            overlap=_LTX2_5_TEMPORAL_OVERLAP_FRAMES,
        )
    if tiling == "temporal":
        return TilingConfig.temporal_only(
            tile_size=_LTX2_5_TEMPORAL_TILE_FRAMES,
            overlap=_LTX2_5_TEMPORAL_OVERLAP_FRAMES,
        )
    if tiling in ("default", "aggressive", "conservative", "spatial"):
        logger.warning(
            "ltx2_5: tiling=%s requests spatial tiling, but spatial tiling is "
            "disabled for ltx2_5 (conv decoder REFLECT seams, #937/#939). "
            "Falling back to temporal-only.",
            tiling,
        )
        return TilingConfig.temporal_only(
            tile_size=_LTX2_5_TEMPORAL_TILE_FRAMES,
            overlap=_LTX2_5_TEMPORAL_OVERLAP_FRAMES,
        )
    logger.warning("Unknown tiling mode %r, using auto", tiling)
    if needs_temporal:
        return TilingConfig.temporal_only(
            tile_size=_LTX2_5_TEMPORAL_TILE_FRAMES,
            overlap=_LTX2_5_TEMPORAL_OVERLAP_FRAMES,
        )
    return None


def _debug_log_latents(stage: str, latents: mx.array) -> None:
    # #946 diagnostic: log latent magnitude + NaN/Inf after each denoise stage.
    # Gate on FUSION_LTX_DEBUG_LATENTS=1 (default OFF, no prod overhead).
    if os.environ.get("FUSION_LTX_DEBUG_LATENTS", "0") != "1":
        return
    try:
        mx.eval(latents)
        mag = float(mx.max(mx.abs(latents)))
        has_nan = bool(mx.any(mx.isnan(latents)))
        has_inf = bool(mx.any(mx.isinf(latents)))
        logger.info(
            "ltx2_5 debug latent[%s]: shape=%s max_abs=%.6f nan=%s inf=%s",
            stage,
            latents.shape,
            mag,
            has_nan,
            has_inf,
        )
    except Exception:
        logger.debug("ltx2_5 debug latent[%s] log failed", stage, exc_info=True)


def generate_video(
    model_repo: str,
    prompt: str,
    *,
    text_encoder_weights: str | Path | None = None,
    transformer_weights: str | Path | None = None,
    video_vae_weights: str | Path | None = None,
    duration_head_weights: str | Path | None = None,
    spatial_upscaler_weights: str | Path | None = None,
    temporal_upscaler_weights: str | Path | None = None,
    audio_vae_weights: str | Path | None = None,
    variant: LTX2_5Variant | str = LTX2_5Variant.DISTILLED,
    num_frames: int | None = None,
    width: int = 768,
    height: int = 512,
    fps: int = 24,
    seed: int = 42,
    num_inference_steps: int | None = None,
    cfg_scale: float = 4.0,
    negative_prompt: str | None = None,
    image: str | None = None,
    image_strength: float = 1.0,
    image_frame_idx: int = 0,
    two_stage: bool = True,
    tiling: str = "auto",
    output_path: str | None = None,
    verbose: bool = True,
    controlnet_image: str | None = None,
    inpaint_mask=None,
    init_latent=None,
    session_id: str | None = None,
    audio: bool = False,
    audio_frozen: bool = False,
) -> bytes:
    # LTX-2.5 两阶段 distilled 生成。audio=True 走联合 A/V denoise (denoise_distilled_av),
    # 生成视频 + 48kHz 立体声 wav, ffmpeg mux 进 mp4。audio_frozen=True (A2V) 冻结输入音频。
    start_time = time.time()
    variant = LTX2_5Variant.from_str(variant)
    logger.info(
        "ltx2_5 generate_video: variant=%s prompt=%r frames=%s %dx%d fps=%d",
        variant.value,
        prompt[:60],
        num_frames,
        width,
        height,
        fps,
    )

    # 帧约束 (AR §2.3)：num_frames % 8 == 1。
    if num_frames is None:
        raise NotImplementedError(
            "LTX-2.5 duration-head driven num_frames inference requires real "
            "duration-head weights. Pass explicit num_frames "
            "(satisfying num_frames % 8 == 1)."
        )
    if num_frames % 8 != 1:
        adjusted = round((num_frames - 1) / 8) * 8 + 1
        logger.warning(
            "ltx2_5: num_frames %d -> %d (must be 1 + 8*k)", num_frames, adjusted
        )
        num_frames = adjusted
    if width % 32 != 0 or height % 32 != 0:
        raise ValueError(
            f"LTX-2.5 width/height must be divisible by 32, got {width}x{height}"
        )
    if not two_stage:
        raise NotImplementedError(
            "LTX-2.5 single-stage path not supported; distilled is two-stage."
        )

    root = get_model_path(model_repo)
    # Fail visible (Rule 12): a missing or bogus repo root must surface as a
    # FileNotFoundError here, not crash deep inside a load_* call with an
    # unrelated AttributeError. get_model_path may return a non-existent path
    # when snapshot_download is mocked or the repo id is invalid; validate the
    # resolved root before descending into component resolution.
    if not Path(root).exists():
        raise FileNotFoundError(
            f"LTX-2.5 model repo not found at {root} (resolved from {model_repo!r})"
        )
    var_str = variant.value

    # #968 part 2: num_inference_steps is a no-op on the distilled pipeline —
    # the sigma tables are baked (stage1=8, stage2=3, 11 total). Surface this
    # loudly instead of silently ignoring (the dev variant will honor steps
    # once its sigma table is extracted; see resolve_dev_sigmas).
    if num_inference_steps is not None and var_str == "distilled":
        logger.warning(
            "ltx2_5: num_inference_steps=%d ignored on distilled variant "
            "(fixed 8+3=11 steps; use pipeline=dev for step control).",
            num_inference_steps,
        )

    # ---- 1. text encoder (Gemma4-12b) ----
    te_path = (
        Path(text_encoder_weights)
        if text_encoder_weights
        else resolve_component(root, "text_encoder", variant=var_str)
    )
    logger.info("Loading text encoder: %s", te_path.name)
    # split 布局 (flat #762 / mlxcomm #786) connector 独立文件。mlxcomm projection
    # 在 connector.safetensors, TE 加载时需传 projection_weights_path。
    te_conn_path = None
    if is_split_layout(root):
        te_conn_path = resolve_component(root, "connector", variant=var_str)
        if not te_conn_path.exists():
            raise FileNotFoundError(
                f"LTX-2.5 split layout requires connector.safetensors at {te_conn_path}"
            )
    text_encoder = load_text_encoder(te_path, projection_weights_path=te_conn_path)
    mx.eval(text_encoder.parameters())
    # #946: the q8 Gemma4-12b TE overflows in bf16 activations -> NaN
    # video_features -> black frames (3/3 runs, std=0). Root-caused via per-layer
    # bisect: embed is finite (max 43), layer 0 output NaN in bf16; fp32
    # activations + fp32 weights + fp32 attention -> layer 0 finite,
    # video_features finite (max_abs 24.2). Fix (DEFAULT ON): dequant q8 ->
    # Linear, then set_dtype fp32. Memory cost ~12b fp32 (48GB) transient — TE
    # freed after encode (del + mx.clear_cache at L307). Opt OUT with
    # FUSION_LTX_TE_DEQUANT=0 only for memory-constrained debug (produces black
    # frames — not a usable mode).
    if os.environ.get("FUSION_LTX_TE_DEQUANT", "1") == "1":
        from .text_encoder import dequantize_te

        n = dequantize_te(text_encoder)
        text_encoder.set_dtype(mx.float32)
        mx.eval(text_encoder.parameters())
        logger.info(
            "TE dequantized %d layers + set_dtype fp32 (default ON, "
            "FUSION_LTX_TE_DEQUANT=0 to opt out) — #946 NaN fix",
            n,
        )
    # encode 返回 pre-connector (video_features[4096], audio_features[2048])。
    # connector 在 transformer 内, generate 显式运行。T2V 只需 video。
    # audio=True 时 encode 返回 3-tuple (video, audio, additive_mask)。
    audio_features = None
    if audio:
        video_features, audio_features, additive_mask = text_encoder.encode(
            prompt, return_audio_embeddings=True
        )
    else:
        video_features, additive_mask = text_encoder.encode(
            prompt, return_audio_embeddings=False
        )
    model_dtype = video_features.dtype
    mx.eval(video_features, additive_mask)
    _debug_log_latents("te_video_features", video_features)
    if audio_features is not None:
        mx.eval(audio_features)
        _debug_log_latents("te_audio_features", audio_features)
        logger.info(
            "Text encoder loaded: video_features=%s audio_features=%s mask=%s",
            video_features.shape,
            audio_features.shape,
            additive_mask.shape,
        )
    else:
        logger.info(
            "Text encoder loaded: video_features=%s mask=%s",
            video_features.shape,
            additive_mask.shape,
        )
    # dev CFG：负向 prompt 也过 TE（distilled guidance_scale=1 用不到）。
    # 空串必须走真实编码（#957 教训：零填充 embedding ≠ 空串编码）。
    negative_video_features = None
    negative_additive_mask = None
    negative_audio_features = None
    if var_str == "dev" and cfg_scale != 1.0:
        if audio:
            (
                negative_video_features,
                negative_audio_features,
                negative_additive_mask,
            ) = text_encoder.encode(negative_prompt or "", return_audio_embeddings=True)
            mx.eval(negative_video_features, negative_audio_features)
        else:
            negative_video_features, negative_additive_mask = text_encoder.encode(
                negative_prompt or "", return_audio_embeddings=False
            )
            mx.eval(negative_video_features, negative_additive_mask)
        logger.info("Negative prompt encoded: %s", negative_video_features.shape)
    del text_encoder
    mx.clear_cache()

    # ---- 2. transformer (22B) ----
    tx_path = (
        Path(transformer_weights)
        if transformer_weights
        else resolve_component(root, "transformer", variant=var_str)
    )
    logger.info("Loading transformer: %s", tx_path.name)
    # split 布局 (flat #762 / mlxcomm #786) connector 独立文件 (步骤 1 已解析为
    # te_conn_path)。Comfy 布局 connector 嵌在 transformer 文件内。
    transformer = LTX2_5Model.from_pretrained(
        tx_path, variant=var_str, connector_weights=te_conn_path
    )
    mx.eval(transformer.parameters())
    logger.info("Transformer loaded")

    # ---- 3. connector (显式运行; LTX2_5Model.__call__ 不调用它) ----
    # has_prompt_adaln=True -> 无 caption_projection -> connector 输出即 context
    # (inner_dim=4096)。connector 返回 (hidden_states, additive_attention_mask)。
    context, context_mask = transformer.video_embeddings_connector(
        video_features.astype(model_dtype), additive_mask
    )
    mx.eval(context, context_mask)
    logger.info(
        "Connector run: context=%s context_mask=%s",
        context.shape,
        context_mask.shape,
    )
    negative_context = None
    if negative_video_features is not None:
        negative_context, _ = transformer.video_embeddings_connector(
            negative_video_features.astype(model_dtype), negative_additive_mask
        )
        mx.eval(negative_context)
        logger.info("Negative connector run: context=%s", negative_context.shape)

    # ---- 3b. audio connector (audio=True only) ----
    # audio_embeddings_connector 与 video 同构 (Embeddings1DConnector, inner_dim=2048)。
    # 返回 (audio_context, audio_context_mask) -> audio Modality.context。
    audio_context = None
    audio_context_mask = None
    negative_audio_context = None
    if audio and audio_features is not None:
        audio_context, audio_context_mask = transformer.audio_embeddings_connector(
            audio_features.astype(model_dtype), additive_mask
        )
        mx.eval(audio_context, audio_context_mask)
        logger.info(
            "Audio connector run: context=%s mask=%s",
            audio_context.shape,
            audio_context_mask.shape,
        )
        if negative_audio_features is not None:
            negative_audio_context, _ = transformer.audio_embeddings_connector(
                negative_audio_features.astype(model_dtype), negative_additive_mask
            )
            mx.eval(negative_audio_context)
    del video_features, additive_mask, negative_video_features, negative_additive_mask
    del audio_features, negative_audio_features
    mx.clear_cache()

    # ---- 4. dims ----
    # stage1 在半分辨率生成, spatial upsampler x2 -> stage2 全分辨率。
    stage1_h, stage1_w = height // 2 // 32, width // 2 // 32
    latent_frames = 1 + (num_frames - 1) // 8
    # audio latent frames: 25 latent-frames/sec * duration (arch-agnostic, ltx2 reuse)。
    audio_frames = compute_audio_frames(num_frames, fps) if audio else 0
    audio_positions = create_audio_position_grid(1, audio_frames) if audio else None

    # ---- 5. spatial upsampler (stage1 -> stage2) ----
    spatial_path = (
        Path(spatial_upscaler_weights)
        if spatial_upscaler_weights
        else resolve_component(root, "spatial_upscaler", variant=var_str)
    )
    logger.info("Loading spatial upsampler: %s", spatial_path.name)
    spatial_up, spatial_scale = load_spatial_upsampler_2_5(spatial_path)
    mx.eval(spatial_up.parameters())
    stage2_h = int(stage1_h * spatial_scale)
    stage2_w = int(stage1_w * spatial_scale)
    logger.info(
        "Spatial upsampler loaded: scale=%sx stage2=%dx%d",
        spatial_scale,
        stage2_w * 32,
        stage2_h * 32,
    )

    # ---- 6. VAE decoder (conv 变体; load_video_decoder 拒绝 det 变体) ----
    # split 布局 (flat #762 / mlxcomm #786) enc/dec 各一文件 -> 取 decoder;
    # Comfy 单文件含两者。
    if video_vae_weights:
        vae_path = Path(video_vae_weights)
    elif is_split_layout(root):
        vae_path = resolve_component(root, "video_vae_conv_decoder", variant=var_str)
    else:
        vae_path = resolve_component(root, "video_vae_conv", variant=var_str)
    logger.info("Loading VAE decoder (conv): %s", vae_path.name)
    vae_decoder = load_video_decoder(vae_path)
    mx.eval(vae_decoder.parameters())
    latent_mean = vae_decoder.per_channel_statistics.mean
    latent_std = vae_decoder.per_channel_statistics.std
    logger.info("VAE decoder loaded")

    # ---- 6.2 dev single-stage path (Full/SFT DiT, 官方 diffusers 形态) ----
    # dev 无 baked sigma 表、无 spatial/temporal upsampler：全分辨率单阶段
    # 去噪（schedule 由 dev_sigmas() 按 token 数动态 shift，对拍 diffusers
    # max|Δ|<3e-8），CFG 走真实负向分支（guidance_scale>1 有意义）。
    if var_str == "dev":
        # #995: dev single-stage A/V. denoise_dev_av does real CFG per-modality
        # (video cfg_scale + audio_cfg_scale=7.0 default, mirrors distilled) on a
        # shared sigma grid. Negative audio context = empty-prompt encode (zeros
        # negative convention, #964). Video branch bit-identical to audio-off run
        # when modality_scale=1.0 (no cross-modal skip pass).
        dev_steps = num_inference_steps if num_inference_steps else 20
        dev_h, dev_w = height // 32, width // 32
        dev_tokens = latent_frames * dev_h * dev_w
        sig = dev_sigmas(dev_steps, num_tokens=dev_tokens)
        logger.info(
            "dev single-stage: %dx%d (%d steps, tokens=%d, cfg=%s, i2v=%s, audio=%s)",
            dev_w * 32,
            dev_h * 32,
            dev_steps,
            dev_tokens,
            cfg_scale,
            image is not None,
            audio,
        )
        mx.random.seed(seed)
        positions = create_position_grid(1, latent_frames, dev_h, dev_w)
        mx.eval(positions)
        # #977: dev I2V single-stage conditioning. Same latent-level injection
        # pattern as distilled stage1 — encode image at dev (full) resolution,
        # inject as clean_latent at image_frame_idx, freeze via denoise_mask,
        # renoise free frames to sig[0]. condition frames mask=0 -> timesteps=0
        # -> transformer treats as clean; apply_denoise_mask clamps each step.
        dev_state = None
        if image is not None:
            latent_cache = get_image_latent_cache(model_repo)
            dev_image_latent, vae_encoder = _encode_image_latent(
                image,
                height,
                width,
                model_repo,
                root,
                model_dtype,
                latent_cache,
                None,
            )
            if vae_encoder is not None:
                del vae_encoder
                mx.clear_cache()
            logger.info(
                "ltx2_5 dev I2V: image latent encoded %s (frame_idx=%d strength=%s)",
                dev_image_latent.shape,
                image_frame_idx,
                image_strength,
            )
            latent_shape = (1, 128, latent_frames, dev_h, dev_w)
            dev_state = _build_i2v_state(
                dev_image_latent,
                latent_shape,
                sig[0],
                image_frame_idx,
                image_strength,
                model_dtype,
            )
            latents = dev_state.latent
        else:
            latents = mx.random.normal(
                (1, 128, latent_frames, dev_h, dev_w), dtype=model_dtype
            )
        mx.eval(latents)
        # audio latent init: (1, 8, audio_frames, 16) noise. audio_frozen (A2V)
        # caller pre-fills condition audio; T2V audio path uses random noise.
        audio_latents = None
        if audio:
            audio_latents = mx.random.normal(
                (1, AUDIO_LATENT_CHANNELS, audio_frames, AUDIO_MEL_BINS),
                dtype=model_dtype,
            )
            mx.eval(audio_latents, audio_positions)
            logger.info(
                "dev audio latents init: %s frames=%d (frozen=%s)",
                audio_latents.shape,
                audio_frames,
                audio_frozen,
            )
        if audio:
            latents, audio_latents = denoise_dev_av(
                latents,
                audio_latents,
                positions,
                audio_positions,
                context,
                negative_context,
                audio_context,
                negative_audio_context,
                transformer,
                sig,
                verbose=verbose,
                cfg_scale=cfg_scale,
                video_state=dev_state,
                audio_frozen=audio_frozen,
                controlnet_image=controlnet_image,
                inpaint_mask=inpaint_mask,
                init_latent=init_latent,
            )
        else:
            latents = denoise_distilled_t2v(
                latents,
                positions,
                context,
                transformer,
                sig,
                verbose=verbose,
                controlnet_image=controlnet_image,
                inpaint_mask=inpaint_mask,
                init_latent=init_latent,
                state=dev_state,
                negative_context=negative_context,
                cfg_scale=cfg_scale,
            )
        mx.eval(latents)
        if audio_latents is not None:
            mx.eval(audio_latents)
        mx.clear_cache()
        del transformer
        mx.clear_cache()
        logger.info("Decoding latents %s (full decode)...", latents.shape)
        video = vae_decoder(latents)
        mx.eval(video)
        mx.clear_cache()
        del vae_decoder
        video = mx.squeeze(video, axis=0)
        video = mx.transpose(video, (1, 2, 3, 0))
        video = mx.clip((video + 1.0) / 2.0, 0.0, 1.0)
        video = (video * 255).astype(mx.uint8)
        video_np = np.array(video)
        logger.info(
            "Decoded %d frames %dx%d",
            video_np.shape[0],
            video_np.shape[2],
            video_np.shape[1],
        )
        if audio and audio_latents is not None:
            mp4_bytes = _write_mp4_av(
                video_np,
                fps,
                audio_latents,
                root,
                audio_vae_weights,
                var_str,
                output_path,
            )
        else:
            mp4_bytes = _write_mp4(video_np, fps, output_path)
        logger.info("ltx2_5 dev generate_video: %.1fs", time.time() - start_time)
        return mp4_bytes

    # ---- 6.5 I2V image encode (#782) ----
    # 两阶段 distilled 各分辨率独立 VAE-encode 同一图像: stage1 半分辨率,
    # stage2 全分辨率 (spatial upsampler x2 之间)。条件是 latent-level 注入,
    # transformer 无需改动。无图像时 is_i2v=False, 走原 T2V 路径不变。
    is_i2v = image is not None
    stage1_image_latent = None
    stage2_image_latent = None
    vae_encoder = None
    _session_tail_reused = False
    if is_i2v:
        # Session-tail latent cache (multi-shot continuity): if a previous
        # shot in this session left a tail-frame latent, reuse it as the
        # first-frame image conditioning and skip the VAE image encode
        # entirely — mirrors wan2/generate.py get_session_tail. The cache
        # no-ops when FUSION_SESSION_TAIL_CACHE != 1.
        if session_id is not None:
            try:
                from fusion_mlx.cache.latent_cache import get_session_tail

                tail_latent = get_session_tail(session_id, model_repo)
            except Exception:
                logger.debug("ltx2_5 session-tail get failed", exc_info=True)
                tail_latent = None
            if tail_latent is not None:
                # tail is [1, 128, 1, H, W] at the final (stage2) latent
                # resolution. Only reuse if its spatial dims match the current
                # shot's stage2 grid — a different width/height in a later shot
                # of the same session must fall back to VAE encode.
                sh = tuple(tail_latent.shape[-2:])
                if sh == (stage2_h, stage2_w):
                    stage2_image_latent = tail_latent.astype(model_dtype)
                    # stage1 is stage2 downsampled by the spatial_scale stride
                    # (stage2_h == stage1_h * spatial_scale). Compute the stride
                    # from the actual dims rather than hardcoding 2x so a
                    # non-2x upsampler stays correct.
                    stride_h = stage2_h // stage1_h if stage1_h else 1
                    stride_w = stage2_w // stage1_w if stage1_w else 1
                    stage1_image_latent = stage2_image_latent[
                        :, :, :, ::stride_h, ::stride_w
                    ]
                    _session_tail_reused = True
                    logger.info(
                        "ltx2_5 I2V: session tail reused (session_id=%s) "
                        "stage1=%s stage2=%s stride=%dx%d — VAE image encode skipped",
                        session_id,
                        stage1_image_latent.shape,
                        stage2_image_latent.shape,
                        stride_h,
                        stride_w,
                    )
                else:
                    logger.info(
                        "ltx2_5 I2V: session tail shape %s != stage2 %dx%d — "
                        "falling back to VAE image encode",
                        sh,
                        stage2_h,
                        stage2_w,
                    )
        if not _session_tail_reused:
            logger.info("ltx2_5 I2V: encoding image at stage resolutions...")
            latent_cache = get_image_latent_cache(model_repo)
            s1_h, s1_w = stage1_h * 32, stage1_w * 32
            s2_h, s2_w = stage2_h * 32, stage2_w * 32
            stage1_image_latent, vae_encoder = _encode_image_latent(
                image,
                s1_h,
                s1_w,
                model_repo,
                root,
                model_dtype,
                latent_cache,
                vae_encoder,
            )
            stage2_image_latent, vae_encoder = _encode_image_latent(
                image,
                s2_h,
                s2_w,
                model_repo,
                root,
                model_dtype,
                latent_cache,
                vae_encoder,
            )
            if vae_encoder is not None:
                del vae_encoder
                mx.clear_cache()
            logger.info("ltx2_5 I2V: image latents encoded")

    # ---- 7. stage1 denoise ----
    logger.info(
        "Stage 1: Generating at %dx%d (%d steps)",
        stage1_w * 32,
        stage1_h * 32,
        len(DISTILLED_STAGE_1_SIGMAS) - 1,
    )
    mx.random.seed(seed)
    positions = create_position_grid(1, latent_frames, stage1_h, stage1_w)
    mx.eval(positions)

    state1 = None
    if is_i2v and stage1_image_latent is not None:
        # stage1 从 zeros latent 出发; 条件帧 (image_frame_idx) 注入 clean_latent,
        # denoise_mask=1-strength (条件帧保持干净, 不去噪)。
        latent_shape = (1, 128, latent_frames, stage1_h, stage1_w)
        state1 = LatentState(
            latent=mx.zeros(latent_shape, dtype=model_dtype),
            clean_latent=mx.zeros(latent_shape, dtype=model_dtype),
            denoise_mask=mx.ones((1, 1, latent_frames, 1, 1), dtype=model_dtype),
        )
        conditionings = _build_i2v_conditionings(
            stage1_image_latent, image_frame_idx, image_strength
        )
        state1 = apply_conditioning(state1, conditionings)
        # 按 denoise_mask 重新加噪到 STAGE_1_SIGMAS[0]: 条件帧 mask=0 -> 不加噪,
        # 自由帧 mask=1 -> 全噪声。
        noise = mx.random.normal(latent_shape, dtype=model_dtype)
        noise_scale = mx.array(DISTILLED_STAGE_1_SIGMAS[0], dtype=model_dtype)
        scaled_mask = state1.denoise_mask * noise_scale
        state1 = LatentState(
            latent=noise * scaled_mask
            + state1.latent * (mx.array(1.0, dtype=model_dtype) - scaled_mask),
            clean_latent=state1.clean_latent,
            denoise_mask=state1.denoise_mask,
        )
        latents = state1.latent
        mx.eval(latents)
    else:
        latents = mx.random.normal(
            (1, 128, latent_frames, stage1_h, stage1_w), dtype=model_dtype
        )
        mx.eval(latents)

    _debug_log_latents("stage1_in", latents)
    _debug_log_latents("context_in", context)
    # audio latent init: (1, 8, audio_frames, 16) noise。audio_frozen (A2V) 时由
    # 调用方预填条件音频 latent (本轮 T2V audio 生成路径走随机噪声)。
    audio_latents = None
    if audio:
        audio_latents = mx.random.normal(
            (1, AUDIO_LATENT_CHANNELS, audio_frames, AUDIO_MEL_BINS),
            dtype=model_dtype,
        )
        mx.eval(audio_latents, audio_positions)
        logger.info(
            "Audio latents init: %s frames=%d (frozen=%s)",
            audio_latents.shape,
            audio_frames,
            audio_frozen,
        )

    if audio:
        latents, audio_latents = denoise_distilled_av(
            latents,
            audio_latents,
            positions,
            audio_positions,
            context,
            audio_context,
            transformer,
            DISTILLED_STAGE_1_SIGMAS,
            verbose=verbose,
            controlnet_image=controlnet_image,
            inpaint_mask=inpaint_mask,
            init_latent=init_latent,
            video_state=state1,
            audio_frozen=audio_frozen,
        )
    else:
        latents = denoise_distilled_t2v(
            latents,
            positions,
            context,
            transformer,
            DISTILLED_STAGE_1_SIGMAS,
            verbose=verbose,
            controlnet_image=controlnet_image,
            inpaint_mask=inpaint_mask,
            init_latent=init_latent,
            state=state1,
        )
    mx.eval(latents)
    if audio_latents is not None:
        mx.eval(audio_latents)
    _debug_log_latents("stage1_out", latents)
    mx.clear_cache()

    # ---- 8. spatial upsample (stage1 -> stage2) ----
    logger.info("Upsampling latents %dx...", spatial_scale)
    latents = upsample_latents(latents, spatial_up, latent_mean, latent_std)
    mx.eval(latents)
    del spatial_up
    mx.clear_cache()
    logger.info("Latents upsampled -> %s", latents.shape)

    # ---- 9. stage2 denoise ----
    logger.info(
        "Stage 2: Refining at %dx%d (%d steps)",
        stage2_w * 32,
        stage2_h * 32,
        len(DISTILLED_STAGE_2_SIGMAS) - 1,
    )
    positions = create_position_grid(1, latent_frames, stage2_h, stage2_w)
    mx.eval(positions)

    state2 = None
    # #946: re-seed before stage2 noise draw. Stage1 output drift (bf16 SDPA
    # nondeterminism) feeds stage2 input, but the stage2 noise RNG should not
    # also drift from accumulated RNG state. Deterministic offset removes one
    # confounder for the nondeterminism diagnosis.
    mx.random.seed(seed + 1)
    if is_i2v and stage2_image_latent is not None:
        # stage2 从 stage1 上采样 latents 出发; 同样注入条件帧 clean_latent,
        # 按 denoise_mask 重新加噪到 STAGE_2_SIGMAS[0]。
        state2 = LatentState(
            latent=latents,
            clean_latent=mx.zeros_like(latents),
            denoise_mask=mx.ones((1, 1, latent_frames, 1, 1), dtype=model_dtype),
        )
        conditionings = _build_i2v_conditionings(
            stage2_image_latent, image_frame_idx, image_strength
        )
        state2 = apply_conditioning(state2, conditionings)
        noise = mx.random.normal(latents.shape).astype(model_dtype)
        noise_scale = mx.array(DISTILLED_STAGE_2_SIGMAS[0], dtype=model_dtype)
        scaled_mask = state2.denoise_mask * noise_scale
        state2 = LatentState(
            latent=noise * scaled_mask
            + state2.latent * (mx.array(1.0, dtype=model_dtype) - scaled_mask),
            clean_latent=state2.clean_latent,
            denoise_mask=state2.denoise_mask,
        )
        latents = state2.latent
        mx.eval(latents)
    else:
        # stage2 从 stage1 上采样结果出发, 重新加噪到 STAGE_2_SIGMAS[0]。
        noise_scale = mx.array(DISTILLED_STAGE_2_SIGMAS[0], dtype=mx.float32)
        one_minus_scale = mx.array(1.0 - DISTILLED_STAGE_2_SIGMAS[0], dtype=mx.float32)
        noise = mx.random.normal(latents.shape).astype(mx.float32)
        latents = noise * noise_scale + latents.astype(mx.float32) * one_minus_scale
        mx.eval(latents)

    # audio 重新加噪到 STAGE_2_SIGMAS[0] (同 video, 镜像 ltx2 generate stage2)。
    # audio_frozen (A2V) 不加噪 — 输入音频跨阶段保持干净。
    if audio and audio_latents is not None and not audio_frozen:
        audio_noise_scale = mx.array(DISTILLED_STAGE_2_SIGMAS[0], dtype=mx.float32)
        audio_one_minus = mx.array(1.0 - DISTILLED_STAGE_2_SIGMAS[0], dtype=mx.float32)
        audio_noise = mx.random.normal(audio_latents.shape).astype(mx.float32)
        audio_latents = (
            audio_noise * audio_noise_scale
            + audio_latents.astype(mx.float32) * audio_one_minus
        )
        mx.eval(audio_latents)

    if audio:
        latents, audio_latents = denoise_distilled_av(
            latents,
            audio_latents,
            positions,
            audio_positions,
            context,
            audio_context,
            transformer,
            DISTILLED_STAGE_2_SIGMAS,
            verbose=verbose,
            controlnet_image=controlnet_image,
            inpaint_mask=inpaint_mask,
            init_latent=init_latent,
            video_state=state2,
            audio_frozen=audio_frozen,
        )
    else:
        latents = denoise_distilled_t2v(
            latents,
            positions,
            context,
            transformer,
            DISTILLED_STAGE_2_SIGMAS,
            verbose=verbose,
            controlnet_image=controlnet_image,
            inpaint_mask=inpaint_mask,
            init_latent=init_latent,
            state=state2,
        )
    mx.eval(latents)
    if audio_latents is not None:
        mx.eval(audio_latents)
    _debug_log_latents("stage2_out", latents)
    del transformer
    mx.clear_cache()

    # ---- 10. temporal upsampler (frames x2) ----
    temporal_path = (
        Path(temporal_upscaler_weights)
        if temporal_upscaler_weights
        else resolve_component(root, "temporal_upscaler", variant=var_str)
    )
    logger.info("Loading temporal upsampler: %s", temporal_path.name)
    temporal_up, temporal_scale = load_temporal_upsampler(temporal_path)
    mx.eval(temporal_up.parameters())
    logger.info(
        "Temporal upsampler loaded: scale=%sx latents=%s",
        temporal_scale,
        latents.shape,
    )
    # Mirror the spatial wrapper in upsample_latents(): the upsampler operates
    # in raw (denormalized) latent space, so denorm -> upsample -> renorm.
    # Feeding normalized-space latents bare produces 6-7 sigma outliers that
    # burn into the VAE decode as grid artifacts + oversaturation.
    t_mean = latent_mean.reshape(1, -1, 1, 1, 1)
    t_std = latent_std.reshape(1, -1, 1, 1, 1)
    latents = (temporal_up(latents * t_std + t_mean) - t_mean) / t_std
    mx.eval(latents)
    del temporal_up
    mx.clear_cache()
    logger.info("Temporal upsampled -> %s", latents.shape)

    # Session-tail latent cache (multi-shot continuity): store the last
    # temporal frame of the final denoised latent so the next shot in this
    # session can reuse it as first-frame conditioning (skips VAE image
    # encode). Mirrors wan2/generate.py put_session_tail. No-op when
    # FUSION_SESSION_TAIL_CACHE != 1.
    if session_id is not None:
        try:
            from fusion_mlx.cache.latent_cache import put_session_tail

            tail = latents[:, :, -1:, :, :]
            put_session_tail(session_id, model_repo, tail)
            logger.info(
                "ltx2_5 session tail put: session_id=%s shape=%s",
                session_id,
                tail.shape,
            )
        except Exception:
            logger.debug("ltx2_5 session-tail put failed", exc_info=True)

    # ---- 11. VAE decode -> frames -> mp4 ----
    # #945: wire the (previously dead) tiling param. Temporal-only tiling
    # bounds peak memory on long high-res videos that otherwise OOM via
    # memory_enforcer. Spatial tiling disabled for ltx2_5 conv decoder
    # (REFLECT seam artifacts, #937/#939). NOT bit-exact vs full decode at
    # chunk boundaries (temporal RF > overlap) but avoids the hard crash.
    tiling_config = _resolve_ltx2_5_tiling_config(tiling, num_frames)
    if tiling_config is not None:
        t_info = (
            f"{tiling_config.temporal_config.tile_size_in_frames}f"
            if tiling_config.temporal_config
            else "none"
        )
        logger.info(
            "Decoding latents %s (temporal-only tiling=%s tile=%s)...",
            latents.shape,
            tiling,
            t_info,
        )
        video = vae_decoder.decode_tiled(latents, tiling_config=tiling_config)
    else:
        logger.info("Decoding latents %s (full decode, tiling=none)...", latents.shape)
        video = vae_decoder(latents)
    mx.eval(video)
    mx.clear_cache()
    del vae_decoder

    video = mx.squeeze(video, axis=0)
    video = mx.transpose(video, (1, 2, 3, 0))
    video = mx.clip((video + 1.0) / 2.0, 0.0, 1.0)
    video = (video * 255).astype(mx.uint8)
    video_np = np.array(video)
    logger.info(
        "Decoded %d frames %dx%d",
        video_np.shape[0],
        video_np.shape[2],
        video_np.shape[1],
    )

    if audio and audio_latents is not None:
        mp4_bytes = _write_mp4_av(
            video_np,
            fps,
            audio_latents,
            root,
            audio_vae_weights,
            var_str,
            output_path,
        )
    else:
        mp4_bytes = _write_mp4(video_np, fps, output_path)

    logger.info("ltx2_5 generate_video: %.1fs", time.time() - start_time)
    return mp4_bytes


def _write_mp4_av(
    video_np: np.ndarray,
    fps: int,
    audio_latents: mx.array,
    root: Path,
    audio_vae_weights: str | Path | None,
    var_str: str,
    output_path: str | None,
) -> bytes:
    # video -> temp mp4 (no audio) + audio_latents -> wav -> ffmpeg mux -> final。
    # output_path 给定写最终, 否则写 temp 最终再读 bytes (清理过程文件, Rule)。
    import tempfile

    tmp_dir = Path(tempfile.mkdtemp(prefix="ltx2_5_av_"))
    try:
        video_tmp = tmp_dir / "video_only.mp4"
        _write_mp4(video_np, fps, str(video_tmp))
        logger.info("AV mux: video-only written to %s", video_tmp)

        audio_vae_path = (
            Path(audio_vae_weights)
            if audio_vae_weights
            else resolve_component(root, "audio_vae", variant=var_str)
        )
        if not audio_vae_path.exists():
            raise FileNotFoundError(
                f"LTX-2.5 audio VAE weights not found at {audio_vae_path}"
            )
        logger.info("Loading audio decoder + vocoder: %s", audio_vae_path.name)
        decoder = load_audio_decoder(audio_vae_path)
        vocoder = load_vocoder_model(audio_vae_path)
        mx.eval(decoder.parameters(), vocoder.parameters())
        audio_np = decode_audio(audio_latents, decoder, vocoder)
        del decoder, vocoder
        mx.clear_cache()
        wav_tmp = tmp_dir / "audio.wav"
        save_audio(audio_np, str(wav_tmp))
        logger.info("AV mux: audio wav written to %s", wav_tmp)

        if output_path is not None:
            final_path = Path(output_path)
            final_path.parent.mkdir(parents=True, exist_ok=True)
            mux_video_audio(video_tmp, wav_tmp, str(final_path))
            logger.info("AV mux: final written to %s", final_path)
            return final_path.read_bytes()
        final_tmp = tmp_dir / "final.mp4"
        mux_video_audio(video_tmp, wav_tmp, str(final_tmp))
        return final_tmp.read_bytes()
    finally:
        import shutil

        shutil.rmtree(tmp_dir, ignore_errors=True)
        logger.info("AV mux: temp dir cleaned")


def _write_mp4(video_np: np.ndarray, fps: int, output_path: str | None) -> bytes:
    # frames (F,H,W,3) uint8 RGB -> avc1 mp4。output_path 给定时写盘, 否则返内存。
    import cv2

    h, w = video_np.shape[1], video_np.shape[2]
    if output_path is not None:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        out = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
        for frame in video_np:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        out.release()
        logger.info("Saved video to %s", out_path)
        return out_path.read_bytes()

    # 无 output_path -> 内存 mp4 (cv2 不支持内存写入, 走 temp 文件)。
    import tempfile

    tmp = Path(tempfile.mkstemp(suffix=".mp4")[1])
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    out = cv2.VideoWriter(str(tmp), fourcc, fps, (w, h))
    for frame in video_np:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    out.release()
    data = tmp.read_bytes()
    tmp.unlink(missing_ok=True)
    return data


__all__ = ["generate_video"]
