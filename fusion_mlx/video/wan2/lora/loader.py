import re
from pathlib import Path

import mlx.core as mx

from .types import LoRAConfig, LoRAWeights


def load_lora_weights(lora_path: Path) -> dict[str, LoRAWeights]:
    if not lora_path.exists():
        raise FileNotFoundError(f"LoRA file not found: {lora_path}")

    all_weights = mx.load(str(lora_path))

    # Group weights by module name, handling both naming conventions
    lora_weights = {}
    module_names = set()

    for key in all_weights:
        # Format 1: {module}.lora_A.weight / {module}.lora_B.weight
        match = re.match(r"(.+)\.lora_([AB])\.weight$", key)
        if match:
            module_names.add(match.group(1))
            continue
        # Format 2: {module}.lora_down.weight / {module}.lora_up.weight
        match = re.match(r"(.+)\.lora_(down|up)\.weight$", key)
        if match:
            module_names.add(match.group(1))

    for module_name in module_names:
        # Try both key conventions
        key_a = f"{module_name}.lora_A.weight"
        key_b = f"{module_name}.lora_B.weight"
        if key_a not in all_weights or key_b not in all_weights:
            key_a = f"{module_name}.lora_down.weight"
            key_b = f"{module_name}.lora_up.weight"
        if key_a not in all_weights or key_b not in all_weights:
            continue

        lora_a = all_weights[key_a]
        lora_b = all_weights[key_b]

        if lora_a.ndim != 2 or lora_b.ndim != 2:
            raise ValueError(
                f"Invalid LoRA shape for {module_name}: "
                f"lora_A={lora_a.shape}, lora_B={lora_b.shape}"
            )

        rank = lora_a.shape[0]
        if lora_b.shape[1] != rank:
            raise ValueError(
                f"LoRA rank mismatch for {module_name}: "
                f"lora_A rank={rank}, lora_B rank={lora_b.shape[1]}"
            )

        # Check for per-module alpha stored as a scalar tensor
        alpha_key = f"{module_name}.alpha"
        if alpha_key in all_weights:
            alpha = float(all_weights[alpha_key].item())
        else:
            alpha = float(rank)

        lora_weights[module_name] = LoRAWeights(
            lora_A=lora_a,
            lora_B=lora_b,
            rank=rank,
            alpha=alpha,
            module_name=module_name,
        )

    if not lora_weights:
        raise ValueError(f"No valid LoRA weights found in {lora_path}")

    return lora_weights


def load_multiple_loras(
    configs: list[LoRAConfig],
) -> dict[str, list[tuple]]:
    module_to_loras: dict[str, list] = {}

    for config in configs:
        lora_weights = load_lora_weights(config.path)

        for module_name, weights in lora_weights.items():
            if config.target_modules is not None:
                if module_name not in config.target_modules:
                    continue

            if module_name not in module_to_loras:
                module_to_loras[module_name] = []

            module_to_loras[module_name].append((weights, config.strength))

    return module_to_loras
