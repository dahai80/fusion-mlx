# SPDX-License-Identifier: Apache-2.0
# Pure-MLX port of LTX-2 utility functions (vendored from mlx-video).
# Phase 4 LTX-2 direct-MLX port: model-layer foundation.
import json
import logging
import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from PIL import Image

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


def apply_quantization(model: nn.Module, weights: dict, quantization: dict | None):
    if quantization is not None:

        def get_class_predicate(p, m):
            if p in quantization:
                return quantization[p]
            if not hasattr(m, "to_quantized"):
                return False
            if hasattr(m, "weight") and m.weight.shape[0] % 64 != 0:
                return False
            return f"{p}.scales" in weights

        nn.quantize(
            model,
            group_size=quantization["group_size"],
            bits=quantization["bits"],
            mode=quantization.get("mode", "affine"),
            class_predicate=get_class_predicate,
        )
        logger.info(
            "apply_quantization: group_size=%s bits=%s mode=%s",
            quantization.get("group_size"),
            quantization.get("bits"),
            quantization.get("mode", "affine"),
        )


def get_model_path(model_repo: str) -> Path:
    try:
        if Path(model_repo).exists():
            return Path(model_repo)
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(repo_id=model_repo, local_files_only=True))
    except Exception:
        logger.info("Downloading model weights for %s", model_repo)
        from huggingface_hub import snapshot_download

        return Path(
            snapshot_download(
                repo_id=model_repo,
                local_files_only=False,
                resume_download=True,
                allow_patterns=["*.safetensors", "*.json"],
            )
        )


def load_image(
    image_path: str | Path,
    height: int | None = None,
    width: int | None = None,
    dtype: mx.Dtype = mx.float32,
) -> mx.array:
    image = Image.open(image_path).convert("RGB")

    if height is not None and width is not None:
        image = image.resize((width, height), Image.Resampling.LANCZOS)
    elif height is not None or width is not None:
        orig_w, orig_h = image.size
        if height is not None:
            scale = height / orig_h
            new_w = int(orig_w * scale)
            new_w = (new_w // 32) * 32
            image = image.resize((new_w, height), Image.Resampling.LANCZOS)
        else:
            scale = width / orig_w
            new_h = int(orig_h * scale)
            new_h = (new_h // 32) * 32
            image = image.resize((width, new_h), Image.Resampling.LANCZOS)
    else:
        orig_w, orig_h = image.size
        new_w = (orig_w // 32) * 32
        new_h = (orig_h // 32) * 32
        if new_w != orig_w or new_h != orig_h:
            image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

    image_np = np.array(image).astype(np.float32) / 255.0
    return mx.array(image_np, dtype=dtype)


def resize_image_aspect_ratio(
    image: mx.array,
    long_side: int = 512,
) -> mx.array:
    h, w = image.shape[:2]

    if h > w:
        new_h = long_side
        new_w = int(w * long_side / h)
    else:
        new_w = long_side
        new_h = int(h * long_side / w)

    new_h = (new_h // 32) * 32
    new_w = (new_w // 32) * 32

    image_np = np.array(image)
    if image_np.max() <= 1.0:
        image_np = (image_np * 255).astype(np.uint8)
    pil_image = Image.fromarray(image_np)
    pil_image = pil_image.resize((new_w, new_h), Image.Resampling.LANCZOS)

    return mx.array(np.array(pil_image).astype(np.float32) / 255.0)


def prepare_image_for_encoding(
    image: mx.array,
    target_height: int,
    target_width: int,
    dtype: mx.Dtype = mx.float32,
) -> mx.array:
    h, w = image.shape[:2]

    if h != target_height or w != target_width:
        image_np = np.array(image)
        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)
        pil_image = Image.fromarray(image_np)
        pil_image = pil_image.resize(
            (target_width, target_height), Image.Resampling.LANCZOS
        )
        image = mx.array(np.array(pil_image).astype(np.float32) / 255.0)

    image = image * 2.0 - 1.0

    image = mx.transpose(image, (2, 0, 1))
    image = mx.expand_dims(image, axis=0)
    image = mx.expand_dims(image, axis=2)

    return image.astype(dtype)


def convert_audio_encoder(
    model_path: str | Path,
    source_repo: str = "Lightricks/LTX-2",
) -> Path:
    model_path = Path(model_path)
    encoder_dir = model_path / "audio_vae" / "encoder"

    if (encoder_dir / "model.safetensors").exists():
        logger.info("convert_audio_encoder: reusing %s", encoder_dir)
        return encoder_dir

    from huggingface_hub import hf_hub_download

    vae_path = hf_hub_download(
        source_repo,
        "audio_vae/diffusion_pytorch_model.safetensors",
    )

    raw_weights = mx.load(vae_path)

    from .audio_vae import AudioEncoder
    from .config import AudioEncoderModelConfig

    decoder_config_path = model_path / "audio_vae" / "decoder" / "config.json"
    if decoder_config_path.exists():
        with open(decoder_config_path) as f:
            dec_cfg = json.load(f)
        enc_config = {
            "ch": dec_cfg.get("ch", 128),
            "in_channels": dec_cfg.get("out_ch", 2),
            "ch_mult": dec_cfg.get("ch_mult", [1, 2, 4]),
            "num_res_blocks": dec_cfg.get("num_res_blocks", 2),
            "attn_resolutions": dec_cfg.get("attn_resolutions", []),
            "resolution": dec_cfg.get("resolution", 256),
            "z_channels": dec_cfg.get("z_channels", 8),
            "double_z": True,
            "n_fft": 1024,
            "norm_type": dec_cfg.get("norm_type", "pixel"),
            "causality_axis": dec_cfg.get("causality_axis", "height"),
            "dropout": dec_cfg.get("dropout", 0.0),
            "mid_block_add_attention": dec_cfg.get("mid_block_add_attention", False),
            "sample_rate": dec_cfg.get("sample_rate", 16000),
            "mel_hop_length": dec_cfg.get("mel_hop_length", 160),
            "is_causal": dec_cfg.get("is_causal", True),
            "mel_bins": dec_cfg.get("mel_bins", 64) or 64,
            "resamp_with_conv": dec_cfg.get("resamp_with_conv", True),
            "attn_type": dec_cfg.get("attn_type", "vanilla"),
        }
    else:
        enc_config = {
            "in_channels": 2,
            "double_z": True,
            "n_fft": 1024,
            "mel_bins": 64,
        }

    config = AudioEncoderModelConfig.from_dict(enc_config)
    encoder = AudioEncoder(config)
    sanitized = encoder.sanitize(raw_weights)

    encoder_dir.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(encoder_dir / "model.safetensors"), sanitized)
    with open(encoder_dir / "config.json", "w") as f:
        json.dump(enc_config, f, indent=2)

    logger.info(
        "convert_audio_encoder: saved %d weights to %s",
        len(sanitized),
        encoder_dir,
    )
    return encoder_dir
