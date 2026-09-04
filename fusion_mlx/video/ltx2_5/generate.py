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
from ..ltx2.positions import create_position_grid
from ..ltx2.upsampler import upsample_latents
from ..ltx2.utils import load_image, prepare_image_for_encoding
from .config import LTX2_5Variant
from .denoise import denoise_distilled_t2v
from .ltx2_5_model import LTX2_5Model
from .scheduler import DISTILLED_STAGE_1_SIGMAS, DISTILLED_STAGE_2_SIGMAS
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
    variant: LTX2_5Variant | str = LTX2_5Variant.DISTILLED,
    num_frames: int | None = None,
    width: int = 768,
    height: int = 512,
    fps: int = 24,
    seed: int = 42,
    num_inference_steps: int | None = None,
    cfg_scale: float = 4.0,
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
) -> bytes:
    # LTX-2.5 两阶段 distilled T2V 生成。I2V/audio/duration-head 不在本轮路径，
    # 留空 fail visible (Rule 12)。
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
    var_str = variant.value

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
    # encode 返回 pre-connector (video_features[4096], audio_features[2048])。
    # connector 在 transformer 内, generate 显式运行。T2V 只需 video。
    # return_audio_embeddings=False 时 encode 返回 (video_features, additive_mask)。
    video_features, additive_mask = text_encoder.encode(
        prompt, return_audio_embeddings=False
    )
    model_dtype = video_features.dtype
    mx.eval(video_features, additive_mask)
    logger.info(
        "Text encoder loaded: video_features=%s mask=%s",
        video_features.shape,
        additive_mask.shape,
    )
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
    del video_features, additive_mask
    mx.clear_cache()

    # ---- 4. dims ----
    # stage1 在半分辨率生成, spatial upsampler x2 -> stage2 全分辨率。
    stage1_h, stage1_w = height // 2 // 32, width // 2 // 32
    latent_frames = 1 + (num_frames - 1) // 8

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

    # ---- 6.5 I2V image encode (#782) ----
    # 两阶段 distilled 各分辨率独立 VAE-encode 同一图像: stage1 半分辨率,
    # stage2 全分辨率 (spatial upsampler x2 之间)。条件是 latent-level 注入,
    # transformer 无需改动。无图像时 is_i2v=False, 走原 T2V 路径不变。
    is_i2v = image is not None
    stage1_image_latent = None
    stage2_image_latent = None
    vae_encoder = None
    if is_i2v:
        logger.info("ltx2_5 I2V: encoding image at stage resolutions...")
        latent_cache = get_image_latent_cache(model_repo)
        s1_h, s1_w = stage1_h * 32, stage1_w * 32
        s2_h, s2_w = stage2_h * 32, stage2_w * 32
        stage1_image_latent, vae_encoder = _encode_image_latent(
            image, s1_h, s1_w, model_repo, root, model_dtype, latent_cache, vae_encoder
        )
        stage2_image_latent, vae_encoder = _encode_image_latent(
            image, s2_h, s2_w, model_repo, root, model_dtype, latent_cache, vae_encoder
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
    latents = temporal_up(latents)
    mx.eval(latents)
    del temporal_up
    mx.clear_cache()
    logger.info("Temporal upsampled -> %s", latents.shape)

    # ---- 11. VAE decode -> frames -> mp4 ----
    logger.info("Decoding latents %s ...", latents.shape)
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

    mp4_bytes = _write_mp4(video_np, fps, output_path)

    logger.info("ltx2_5 generate_video: %.1fs", time.time() - start_time)
    return mp4_bytes


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
