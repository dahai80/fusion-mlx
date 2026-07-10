# SPDX-License-Identifier: Apache-2.0
# Wan2 LoRA + quantize helpers (runtime subset of mlx_video convert.py).
# The torch-based PyTorch->MLX checkpoint conversion (load_torch_weights,
# convert_wan_checkpoint, sanitize_wan_*) is CLI-only and dropped: the port
# loads pre-converted mlx-community weights directly at runtime.

import logging
from pathlib import Path

import mlx.core as mx

logger = logging.getLogger(__name__)


def _load_lora_configs(lora_configs: list[tuple[str, float]]) -> dict[str, list]:
    from .lora import LoRAConfig, load_multiple_loras

    logger.info("Loading %d LoRA(s)...", len(lora_configs))
    configs = []
    for lora_path, strength in lora_configs:
        config = LoRAConfig(path=lora_path, strength=strength)
        configs.append(config)
        logger.info("  - %s (strength: %s)", Path(lora_path).name, strength)
    module_to_loras = load_multiple_loras(configs)
    if not module_to_loras:
        logger.warning("No LoRA weights matched model layers")
    return module_to_loras


def _quantize_predicate(path: str, module) -> bool:
    if not hasattr(module, "to_quantized"):
        return False
    quantize_patterns = (
        ".self_attn.q",
        ".self_attn.k",
        ".self_attn.v",
        ".self_attn.o",
        ".cross_attn.q",
        ".cross_attn.k",
        ".cross_attn.v",
        ".cross_attn.o",
        ".ffn.fc1",
        ".ffn.fc2",
    )
    return any(path.endswith(p) for p in quantize_patterns)


def load_and_apply_loras(
    model_weights: dict[str, mx.array],
    lora_configs: list[tuple[str, float]] | None = None,
    verbose: bool = False,
    quantization_bits: int = 0,
) -> dict[str, mx.array]:
    from .lora import apply_loras_to_weights

    if not lora_configs:
        return model_weights
    module_to_loras = _load_lora_configs(lora_configs)
    if not module_to_loras:
        return model_weights
    logger.info("Applying LoRAs to %d modules...", len(module_to_loras))
    if verbose:
        logger.info("  Model has %d weight keys", len(model_weights))
    modified_weights = apply_loras_to_weights(
        model_weights,
        module_to_loras,
        verbose=verbose,
        quantization_bits=quantization_bits,
    )
    logger.info("LoRAs applied successfully")
    return modified_weights
