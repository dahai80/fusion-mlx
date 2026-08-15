# SPDX-License-Identifier: Apache-2.0
# LTX-2.5 generate_video orchestration (P6 skeleton).
# 两阶段 distilled 生成编排：text-encoder(Gemma4) → duration-head(可选) →
# stage1 denoise → spatial upsampler → stage2 denoise → temporal upsampler →
# VAE decode → audio mux → mp4 bytes。
#
# 复用策略：去噪骨架 / VAE decode / I2V conditioning / audio mux 与 ltx2 同族，
# 通过 ltx2 原语（denoise_distilled / upsample_latents / create_position_grid /
# LatentState / apply_conditioning）直接调用，避免复制 69KB generate.py。
#
# UNVERIFIED against real 22B weights (gated 403)。本骨架在「加载单文件权重」
# 边界 fail visible（raise NotImplementedError），真实模型首跑需：
#   1. 接受 HF gate → 下载 distilled 权重（hf-mirror）
#   2. 验证 LTX2_5Model / VAE / duration-head / upsampler 键树（首跑审计告警）
#   3. 修正键前缀拆分（video_vae._split_vae_weights / ltx2_5_model.sanitize）
#   4. 跑通 T2V → I2V → audio
from __future__ import annotations

import logging
import time
from pathlib import Path

from .config import LTX2_5Variant

logger = logging.getLogger(__name__)


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
) -> bytes:
    # LTX-2.5 两阶段 distilled 生成。真实权重加载为未验证边界，fail visible。
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

    # 帧约束（AR §2.3）：num_frames % 8 == 1；省略时由 duration-head 决定。
    if num_frames is None:
        raise NotImplementedError(
            "LTX-2.5 duration-head driven num_frames inference requires real "
            "duration-head weights (gated 403). Pass explicit num_frames "
            "(satisfying num_frames % 8 == 1) until weights land."
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

    # 真实权重加载边界：22B 单文件 checkpoint gated，未下载。
    # 以下每个 load_* 在缺权重时 fail visible（FileNotFoundError / RuntimeError）。
    raise NotImplementedError(
        "LTX-2.5 generate_video E2E path requires real 22B weights "
        "(gated repo Lightricks/LTX-2.5, 403). Structural skeleton complete; "
        "download distilled weights via hf-mirror after accepting the HF gate, "
        "then wire load_text_encoder / LTX2_5Model.from_pretrained / "
        "load_video_encoder / load_video_decoder / load_duration_head / "
        "load_spatial_upsampler_2_5 / load_temporal_upsampler into the "
        "stage1 -> spatial-up -> stage2 -> temporal-up -> VAE-decode flow "
        "(mirror ltx2 generate.py:555-700). Variant=" + variant.value
    )
    # 以下为真实模型首跑需接通的流程骨架（unreachable until weights land）：
    # 1. text_encoder = load_text_encoder(text_encoder_weights)  # Gemma4-12b
    # 2. text_embeddings, audio_embeddings = text_encoder.encode(...)
    # 3. transformer = LTX2_5Model.from_pretrained(transformer_weights, variant)
    # 4. vae_encoder = load_video_encoder(video_vae_weights)
    # 5. vae_decoder = load_video_decoder(video_vae_weights)
    # 6. stage1: denoise_distilled(latents, positions, text_embeddings,
    #      transformer, STAGE_1_SIGMAS, ...)
    # 7. spatial upsampler: upsample_latents(latents, spatial_up, mean, std)
    # 8. stage2: denoise_distilled(..., STAGE_2_SIGMAS, ...)
    # 9. temporal upsampler: load_temporal_upsampler + 沿时间轴 x2
    # 10. vae_decoder.decode(latents) -> frames -> denormalize -> mp4
    logger.info("ltx2_5 generate_video: %.1fs", time.time() - start_time)


__all__ = ["generate_video"]
