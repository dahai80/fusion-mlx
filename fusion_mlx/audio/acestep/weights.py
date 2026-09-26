# SPDX-License-Identifier: Apache-2.0
# ACE-Step MLX weight loader (issue #988): maps safetensors keys to the
# MLX module tree. Handles naming divergences (proj_in.1 -> proj_in_weight,
# quantizer project_in/out -> nn.Linear).
from __future__ import annotations

import logging
from pathlib import Path

import mlx.core as mx

from .config import AceStepConfig
from .pipeline import AceStepModel

logger = logging.getLogger(__name__)


def _map_key(key: str) -> str | None:
    # safetensors key -> MLX module tree dot-path string.
    # Returns None for keys we don't load.
    parts = key.split(".")
    if (
        len(parts) >= 4
        and parts[0] == "decoder"
        and parts[1] == "proj_in"
        and parts[2] == "1"
    ):
        return f"decoder.proj_in_{parts[3]}"
    if (
        len(parts) >= 4
        and parts[0] == "decoder"
        and parts[1] == "proj_out"
        and parts[2] == "1"
    ):
        return f"decoder.proj_out_{parts[3]}"
    return key


def load_acestep_weights(
    model: AceStepModel,
    safetensors_path: str | Path,
    strict: bool = False,
) -> tuple[int, int]:
    # Load weights from a single safetensors file into the model.
    # Returns (loaded, skipped) counts.
    pairs = []
    skipped = 0
    tensors = mx.load(str(safetensors_path))  # dict[str, mx.array], bf16-native
    for key, arr in tensors.items():
        mapped = _map_key(key)
        if mapped is None:
            skipped += 1
            continue
        pairs.append((mapped, arr))
    loaded = model.load_weights(pairs, strict=strict)
    logger.info(
        "acestep weights loaded: %d tensors from %s (skipped %d, strict=%s)",
        len(pairs),
        safetensors_path,
        skipped,
        strict,
    )
    return loaded, skipped


def load_acestep_model(
    config: AceStepConfig,
    safetensors_path: str | Path,
) -> AceStepModel:
    model = AceStepModel(config)
    load_acestep_weights(model, safetensors_path, strict=False)
    mx.eval(model.parameters())
    return model


def load_silence_latent(safetensors_path: str | Path) -> mx.array:
    # silence_latent.safetensors: {silence_latent: (1, 15000, 64)} float32.
    # Converted from ACE-Step silence_latent.pt (transposed to T-last).
    data = mx.load(str(safetensors_path))
    key = "silence_latent" if "silence_latent" in data else next(iter(data))
    lat = data[key].astype(mx.float32)
    logger.info("silence_latent loaded: %s dtype=%s", lat.shape, lat.dtype)
    return lat


def load_acestep_orchestrator(
    checkpoint_dir: str | Path,
    text_encoder_repo: str = "Qwen/Qwen3-Embedding-0.6B",
    load_text_encoder: bool = True,
):
    # checkpoint_dir: ACE-Step1.5 snapshot dir containing acestep-v15-turbo/ + vae/.
    from .orchestration import AceStepOrchestrator
    from .vae import AutoencoderOobleckMLX

    checkpoint_dir = Path(checkpoint_dir)
    turbo_dir = checkpoint_dir / "acestep-v15-turbo"
    vae_dir = checkpoint_dir / "vae"
    config = AceStepConfig.from_json(turbo_dir / "config.json")
    model = load_acestep_model(config, turbo_dir / "model.safetensors")
    vae = AutoencoderOobleckMLX.from_pretrained(vae_dir)
    silence = load_silence_latent(turbo_dir / "silence_latent.safetensors")
    orch = AceStepOrchestrator(config, model, vae, silence)
    if load_text_encoder:
        orch.load_text_encoder(text_encoder_repo)
    return orch


__all__ = [
    "load_acestep_weights",
    "load_acestep_model",
    "load_silence_latent",
    "load_acestep_orchestrator",
]
