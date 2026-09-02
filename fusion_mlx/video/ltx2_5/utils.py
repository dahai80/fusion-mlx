# SPDX-License-Identifier: Apache-2.0
# Pure-MLX port of LTX-2.5 utility functions.
# Path resolution for the Lightricks/LTX-2.5 Comfy single-file repo. The repo
# layout is: diffusion_models/, vae/, text_encoders/, model_patches/,
# latent_upscale_models/. We resolve either a local dir or an HF snapshot.
import logging
import math
from pathlib import Path

import mlx.core as mx

logger = logging.getLogger(__name__)


def rms_norm(x: mx.array, eps: float = 1e-6) -> mx.array:
    return mx.fast.rms_norm(x, mx.ones((x.shape[-1],), dtype=x.dtype), eps)


def to_denoised(noisy: mx.array, velocity: mx.array, sigma) -> mx.array:
    original_dtype = noisy.dtype
    noisy_f32 = noisy.astype(mx.float32)
    velocity_f32 = velocity.astype(mx.float32)
    if isinstance(sigma, (int, float)):
        sigma_f32 = mx.array(sigma, dtype=mx.float32)
    else:
        sigma_f32 = sigma.astype(mx.float32)
        while sigma_f32.ndim < velocity_f32.ndim:
            sigma_f32 = mx.expand_dims(sigma_f32, axis=-1)
    result = noisy_f32 - sigma_f32 * velocity_f32
    return result.astype(original_dtype)


def get_timestep_embedding(
    timesteps: mx.array,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1.0,
    scale: float = 1.0,
    max_period: int = 10000,
) -> mx.array:
    assert timesteps.ndim == 1, "Timesteps should be 1D"
    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * mx.arange(0, half_dim, dtype=mx.float32)
    exponent = exponent / (half_dim - downscale_freq_shift)
    emb = mx.exp(exponent)
    emb = (timesteps[:, None].astype(mx.float32) * scale) * emb[None, :]
    if flip_sin_to_cos:
        emb = mx.concatenate([mx.cos(emb), mx.sin(emb)], axis=-1)
    else:
        emb = mx.concatenate([mx.sin(emb), mx.cos(emb)], axis=-1)
    if embedding_dim % 2 == 1:
        emb = mx.pad(emb, [(0, 0), (0, 1)])
    return emb


# LTX-2.5 Comfy 单文件仓的子目录与权重文件名（附录 A）。
_LTX2_5_REPO = "Lightricks/LTX-2.5"
_LTX2_5_FILES = {
    "transformer_distilled": "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors",
    "transformer_dev": "diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors",
    "video_vae": "vae/ltx-2.5-video-vae-bf16.safetensors",
    "video_vae_conv": "vae/ltx-2.5-video-vae-conv-bf16.safetensors",
    "audio_vae": "vae/ltx-2.5-audio-vae-bf16.safetensors",
    "text_encoder": "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
    "duration_head": "model_patches/ltx-2.5-duration-head-bf16.safetensors",
    "spatial_upscaler": "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
    "temporal_upscaler": "latent_upscale_models/ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors",
}

# Flat diffusers 布局根级权重文件名（issue #762，dgrauet/ltx-2.5-mlx-q8 等
# 量化分支）：transformer + connector 分立两文件，VAE enc/dec 分立，上采样器
# 单独命名，与 Comfy 子目录布局一一对应。connertor 键独立组件。
_LTX2_5_FLAT_FILES = {
    "transformer_distilled": "transformer-distilled.safetensors",
    "transformer_dev": "transformer-dev.safetensors",
    "connector": "connector.safetensors",
    "video_vae_conv_encoder": "vae_encoder_conv.safetensors",
    "video_vae_conv_decoder": "vae_decoder_conv.safetensors",
    "text_encoder": "text_encoder.safetensors",
    "duration_head": "duration_head.safetensors",
    "spatial_upscaler": "spatial_upscaler_x2_v1_0.safetensors",
    "temporal_upscaler": "temporal_upscaler_x2_v1_0.safetensors",
}


def is_flat_layout(root: Path) -> bool:
    # 检测根目录是否为 flat diffusers 布局：split_model.json recipe=ltx-2.5 +
    # 根级 transformer-*.safetensors（与 model_discovery._is_flat_ltx2_5_layout
    # 同构，避免 backend 模块反向依赖 pool）。
    split_model = root / "split_model.json"
    if not split_model.exists():
        return False
    try:
        import json

        with open(split_model) as f:
            if json.load(f).get("recipe") != "ltx-2.5":
                return False
    except (OSError, json.JSONDecodeError):
        return False
    return any(
        p.name.startswith("transformer-")
        and p.name.endswith(".safetensors")
        and (p.is_file() or p.is_symlink())
        for p in root.iterdir()
    )


def get_model_path(model_repo: str = _LTX2_5_REPO) -> Path:
    # 解析 LTX-2.5 仓根目录：本地路径优先，否则 HF snapshot 下载。
    # 只拉取 _LTX2_5_FILES 中实际用到的组件文件 (bf16), 跳过 nvfp4/int8 等未用变体,
    # 避免 local_files_only 因仓内多余文件缺失而触发全量重下 (20GB+ 浪费).
    if Path(model_repo).exists():
        logger.info("ltx2_5 get_model_path: local dir %s", model_repo)
        return Path(model_repo)
    from huggingface_hub import snapshot_download

    allow = list(_LTX2_5_FILES.values()) + ["*.json"]
    try:
        return Path(
            snapshot_download(
                repo_id=model_repo, local_files_only=True, allow_patterns=allow
            )
        )
    except Exception as exc:
        logger.info(
            "ltx2_5 get_model_path: local incomplete (%s), fetching %s", exc, model_repo
        )
        return Path(
            snapshot_download(
                repo_id=model_repo,
                local_files_only=False,
                allow_patterns=allow,
            )
        )


def resolve_component(
    root: Path,
    key: str,
    *,
    variant: str = "distilled",
) -> Path:
    # 在已解析的仓根目录下定位单个组件文件。flat diffusers 布局（#762）用
    # 根级单文件名 + 独立 connector 组件；Comfy 布局用子目录映射。
    if key == "transformer":
        key = "transformer_distilled" if variant == "distilled" else "transformer_dev"
    files = _LTX2_5_FLAT_FILES if is_flat_layout(root) else _LTX2_5_FILES
    if key not in files:
        raise ValueError(f"unknown LTX-2.5 component key: {key!r}")
    rel = files[key]
    candidate = root / rel
    if not candidate.exists():
        logger.warning("ltx2_5 component %s not found at %s", key, candidate)
    return candidate


def component_keys() -> list[str]:
    return list(_LTX2_5_FILES.keys())
