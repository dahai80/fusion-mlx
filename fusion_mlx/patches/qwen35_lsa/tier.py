# SPDX-License-Identifier: Apache-2.0
"""Three-tier prefill weights for qwen3_5 full-attention layers (HSA M3).

Loads the trained tier projections (w_q_lat / w_k_lat per full-attention
layer) exported from the HSA distillation checkpoint, and holds the tier
geometry (window, block_size, latent_dim). Prefill routing: single-stream
chunks whose absolute end position exceeds TIER_THRESHOLD run the three-tier
kernel; shorter sequences keep the dense path (quality floor below the
trained regime).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

import mlx.core as mx

logger = logging.getLogger(__name__)

# Prefill at or below this many total tokens stays dense; the tier kernel
# takes over above it. Trained regime was 4K sequences; RULER floor cases
# (4K) must ride the dense path for the 0.99 one-vote-veto.
TIER_THRESHOLD = 4096


@dataclass
class TierConfig:
    window: int
    block_size: int
    latent_dim: int
    # layer index in the decoder layer list -> (w_q_lat, w_k_lat) as mx arrays
    weights: dict[int, tuple[mx.array, mx.array]]


def load_tier_config(path: str, num_layers: int) -> TierConfig:
    # Fail-loud at boot: a bad path or a layer mismatch must stop the serve,
    # not degrade to dense prefill silently mid-flight.
    from safetensors import safe_open

    with safe_open(path, framework="numpy") as handle:
        metadata = handle.metadata() or {}
        for field in ("window", "block_size", "latent_dim", "tier_layers"):
            if field not in metadata:
                raise ValueError(f"tier weights {path} is missing metadata field {field}")
        window = int(metadata["window"])
        block_size = int(metadata["block_size"])
        latent_dim = int(metadata["latent_dim"])
        weights: dict[int, tuple[mx.array, mx.array]] = {}
        for layer_name in sorted(json.loads(metadata["tier_layers"])):
            index = int(layer_name.removeprefix("layer"))
            if index >= num_layers:
                raise ValueError(
                    f"tier weights {path} names {layer_name} but the model has {num_layers} layers"
                )
            weights[index] = (
                mx.array(handle.get_tensor(f"{layer_name}.w_q_lat")),
                mx.array(handle.get_tensor(f"{layer_name}.w_k_lat")),
            )
            weights[index] = tuple(w.astype(mx.float16) for w in weights[index])
    if not weights:
        raise ValueError(f"tier weights {path} carries no layer projections")
    config = TierConfig(window, block_size, latent_dim, weights)
    logger.info(
        "tier prefill armed: %d layers, window=%d block=%d latent=%d threshold=%d (%s)",
        len(weights), window, block_size, latent_dim, TIER_THRESHOLD, path,
    )
    return config
