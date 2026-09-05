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

# mlx-community 布局 (#786, mlx-community/ltx-2.5-mlx-q8 等公开 4/8-bit 仓)：
# 无 split_model.json, 根 config.json model_type=AudioVideo + model_version=2.5.0。
# VAE 文件名无 _conv 后缀, spatial upscaler _v1_1, text_encoder 为子目录分片
# (gemma4-12b-ltx-v1/model-*.safetensors), transformer 在独立 dit 仓 (调用方
# 须显式传 transformer_weights, resolve_component 不解析)。
_LTX2_5_MLXCOMM_FILES = {
    "connector": "connector.safetensors",
    "video_vae_conv_encoder": "vae_encoder.safetensors",
    "video_vae_conv_decoder": "vae_decoder.safetensors",
    "text_encoder": "gemma4-12b-ltx-v1",
    "duration_head": "duration_head.safetensors",
    "spatial_upscaler": "spatial_upscaler_x2_v1_1.safetensors",
    "temporal_upscaler": "temporal_upscaler_x2_v1_0.safetensors",
}

# 三种布局: "comfy" (子目录), "flat" (dgrauet 根级单文件), "mlxcomm" (#786)。
_LAYOUT_COMFY = "comfy"
_LAYOUT_FLAT = "flat"
_LAYOUT_MLXCOMM = "mlxcomm"


def detect_layout(root: Path) -> str:
    # 统一布局探测: 返回 _LAYOUT_COMFY / _LAYOUT_FLAT / _LAYOUT_MLXCOMM。
    # flat 优先 (split_model.json recipe=ltx-2.5), 次 mlx-community (config.json
    # model_type=AudioVideo + model_version=2.5.0, 无 diffusion_models/ 子目录),
    # 兜底 comfy。
    root = Path(root)
    if is_flat_layout(root):
        return _LAYOUT_FLAT
    if is_mlx_community_layout(root):
        return _LAYOUT_MLXCOMM
    return _LAYOUT_COMFY


def is_flat_layout(root: Path) -> bool:
    # 检测根目录是否为 flat diffusers 布局：split_model.json recipe=ltx-2.5 +
    # 根级 transformer-*.safetensors（与 model_discovery._is_flat_ltx2_5_layout
    # 同构，避免 backend 模块反向依赖 pool）。
    root = Path(root)
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


def is_mlx_community_layout(root: Path) -> bool:
    # 检测 mlx-community 布局 (#786): 根 config.json model_type=AudioVideo +
    # model_version=2.5.0, 无 diffusion_models/ 子目录 (排除 Comfy), 无
    # split_model.json (排除 flat)。验证来源 mlx-community/ltx-2.5-mlx-q8。
    root = Path(root)
    config = root / "config.json"
    if not config.exists():
        return False
    if (root / "split_model.json").exists():
        return False
    if (root / "diffusion_models").is_dir():
        return False
    try:
        import json

        with open(config) as f:
            cfg = json.load(f)
        return (
            cfg.get("model_type") == "AudioVideo"
            and str(cfg.get("model_version")) == "2.5.0"
        )
    except (OSError, json.JSONDecodeError):
        return False


def is_split_layout(root: Path) -> bool:
    # VAE enc/dec 分立 + connector 独立文件的布局: flat (dgrauet, #762) 或
    # mlxcomm (#786)。Comfy 布局 VAE 单文件含 enc+dec, connector 嵌 transformer。
    layout = detect_layout(root)
    return layout in (_LAYOUT_FLAT, _LAYOUT_MLXCOMM)


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
    # 在已解析的仓根目录下定位单个组件文件。三布局: flat (dgrauet 根级单文件 +
    # 独立 connector, #762), mlxcomm (mlx-community, #786), comfy (子目录)。
    # mlxcomm text_encoder 为子目录分片 (gemma4-12b-ltx-v1/), 其余为根级单文件。
    # mlxcomm 不含 transformer (在独立 dit 仓, 调用方须显式传 transformer_weights)。
    if key == "transformer":
        key = "transformer_distilled" if variant == "distilled" else "transformer_dev"
    layout = detect_layout(root)
    if layout == _LAYOUT_FLAT:
        files = _LTX2_5_FLAT_FILES
    elif layout == _LAYOUT_MLXCOMM:
        files = _LTX2_5_MLXCOMM_FILES
    else:
        files = _LTX2_5_FILES
    if key not in files:
        if layout == _LAYOUT_MLXCOMM and key in (
            "transformer_distilled",
            "transformer_dev",
        ):
            raise ValueError(
                "mlx-community layout has no transformer weights; pass "
                "transformer_weights explicitly from the dit repo"
            )
        raise ValueError(f"unknown LTX-2.5 component key: {key!r}")
    rel = files[key]
    candidate = root / rel
    if not candidate.exists():
        logger.warning("ltx2_5 component %s not found at %s", key, candidate)
    return candidate


def component_keys() -> list[str]:
    return list(_LTX2_5_FILES.keys())
