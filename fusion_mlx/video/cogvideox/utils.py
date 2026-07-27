# SPDX-License-Identifier: Apache-2.0
import json
import logging
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from .config import CogVideoXConfig
from .transformer import CogVideoXTransformer3DModel
from .vae import AutoencoderKLCogVideoX

logger = logging.getLogger(__name__)


def load_config(model_dir: Path) -> CogVideoXConfig:
    config_path = model_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            d = json.load(f)
        quantization = d.pop("quantization", None)
        config = CogVideoXConfig.from_dict(d)
        return config, quantization
    logger.warning("No config.json found, auto-detecting variant from weights")
    return _auto_detect_config(model_dir)


def _auto_detect_config(model_dir: Path) -> tuple[CogVideoXConfig, None]:
    model_path = model_dir / "model.safetensors"
    if model_path.exists():
        probe = mx.load(str(model_path), return_metadata=False)
        for k, v in probe.items():
            if "transformer_blocks" in k and "attn1.q_proj.weight" in k:
                dim = v.shape[0]
                del probe
                if dim <= 1920:
                    return CogVideoXConfig.cogvideox_2b(), None
                else:
                    return CogVideoXConfig.cogvideox_5b(), None
        del probe
    return CogVideoXConfig.cogvideox_2b(), None


def load_transformer(
    model_path: Path, config: CogVideoXConfig, quantization: dict | None = None
) -> CogVideoXTransformer3DModel:
    model = CogVideoXTransformer3DModel(config)
    if model_path.exists():
        weights = mx.load(str(model_path), return_metadata=False)
        weights = _remap_weights(weights, "transformer")
        model = _load_weights_with_quant(model, weights, quantization)
    mx.eval(model.parameters())
    return model


def load_vae_decoder(vae_path: Path, config: CogVideoXConfig) -> AutoencoderKLCogVideoX:
    vae = AutoencoderKLCogVideoX(config)
    if vae_path.exists():
        weights = mx.load(str(vae_path), return_metadata=False)
        weights = _remap_weights(weights, "vae")
        vae.load_weights(list(weights.items()))
    mx.eval(vae.parameters())
    return vae


def load_vae_encoder(vae_path: Path, config: CogVideoXConfig) -> AutoencoderKLCogVideoX:
    return load_vae_decoder(vae_path, config)


def _remap_weights(weights: dict, prefix: str) -> dict:
    remapped = {}
    for k, v in weights.items():
        new_k = k
        if k.startswith(f"{prefix}."):
            new_k = k[len(prefix) + 1 :]
        remapped[new_k] = v
    return remapped


def _load_weights_with_quant(
    model: nn.Module, weights: dict, quantization: dict | None
) -> nn.Module:
    if quantization is not None:
        from mlx.utils import quantize_model

        bits = quantization.get("bits", 8)
        group_size = quantization.get("group_size", 128)
        model = quantize_model(model, bits=bits, group_size=group_size)
    model.load_weights(list(weights.items()))
    return model


def get_model_path(model_dir: str) -> Path:
    p = Path(model_dir)
    if p.exists():
        return p
    candidates = [
        Path.home() / ".fusion-mlx" / "models" / model_dir,
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / f"models--{model_dir.replace('/', '--')}"
        / "snapshots",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"CogVideoX model not found: {model_dir}")
