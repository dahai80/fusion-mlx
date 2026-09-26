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


__all__ = ["load_acestep_weights", "load_acestep_model"]
