import logging
from pathlib import Path

import mlx.core as mx

from .ltx2_model import LTXModel
from .utils import get_model_path

logger = logging.getLogger(__name__)


def load_and_merge_lora(
    model: LTXModel,
    lora_path: str,
    strength: float = 1.0,
) -> None:
    lora_file = Path(lora_path)
    if lora_file.is_file():
        pass
    elif lora_file.is_dir():
        candidates = sorted(lora_file.glob("*.safetensors"))
        if not candidates:
            raise FileNotFoundError(f"No .safetensors files found in {lora_path}")
        lora_candidates = [c for c in candidates if "distilled-lora" in c.name]
        lora_file = lora_candidates[0] if lora_candidates else candidates[0]
        logger.info("Using LoRA file: %s", lora_file.name)
    else:
        lora_dir = get_model_path(lora_path)
        candidates = sorted(lora_dir.glob("*.safetensors"))
        if not candidates:
            raise FileNotFoundError(f"No .safetensors files found in {lora_dir}")
        lora_candidates = [c for c in candidates if "distilled-lora" in c.name]
        lora_file = lora_candidates[0] if lora_candidates else candidates[0]
        logger.info("Using LoRA from repo: %s (%s)", lora_path, lora_file.name)

    lora_weights = mx.load(str(lora_file))

    has_prefix = any(k.startswith("diffusion_model.") for k in lora_weights)

    lora_pairs = {}
    for key in lora_weights:
        module_key = key
        if has_prefix:
            if not key.startswith("diffusion_model."):
                continue
            module_key = key.replace("diffusion_model.", "")

        if module_key.endswith(".lora_A.weight"):
            base_key = module_key.replace(".lora_A.weight", "")
            lora_pairs.setdefault(base_key, {})["A"] = lora_weights[key]
        elif module_key.endswith(".lora_B.weight"):
            base_key = module_key.replace(".lora_B.weight", "")
            lora_pairs.setdefault(base_key, {})["B"] = lora_weights[key]

    _LORA_KEY_REPLACEMENTS = [
        (".to_out.0", ".to_out"),
        (".ff.net.0.proj", ".ff.proj_in"),
        (".ff.net.2", ".ff.proj_out"),
        (".audio_ff.net.0.proj", ".audio_ff.proj_in"),
        (".audio_ff.net.2", ".audio_ff.proj_out"),
        (".linear_1", ".linear1"),
        (".linear_2", ".linear2"),
    ]
    if has_prefix:
        sanitized_pairs = {}
        for key, pair in lora_pairs.items():
            new_key = key
            for old, new in _LORA_KEY_REPLACEMENTS:
                if new_key.endswith(old):
                    new_key = new_key[: -len(old)] + new
                else:
                    new_key = new_key.replace(old + ".", new + ".")
            sanitized_pairs[new_key] = pair
    else:
        sanitized_pairs = lora_pairs

    def flatten_params(params, prefix=""):
        flat = {}
        for k, v in params.items():
            full_key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                flat.update(flatten_params(v, full_key))
            else:
                flat[full_key] = v
        return flat

    flat_weights = flatten_params(dict(model.parameters()))

    merged_count = 0
    batch = []
    batch_size = 100

    for module_key, pair in sanitized_pairs.items():
        if "A" not in pair or "B" not in pair:
            continue

        weight_key = f"{module_key}.weight"
        if weight_key not in flat_weights:
            continue

        lora_a = pair["A"].astype(mx.float32)
        lora_b = pair["B"].astype(mx.float32)

        delta = (lora_b * strength) @ lora_a

        base_weight = flat_weights.pop(weight_key)
        merged_weight = (base_weight.astype(mx.float32) + delta).astype(
            base_weight.dtype
        )
        batch.append((weight_key, merged_weight))
        del base_weight
        merged_count += 1

        if len(batch) >= batch_size:
            model.load_weights(batch, strict=False)
            mx.eval(model.parameters())
            batch.clear()

    if batch:
        model.load_weights(batch, strict=False)
        mx.eval(model.parameters())
        batch.clear()

    del flat_weights, lora_weights
    mx.clear_cache()
    logger.info("Merged %d LoRA pairs (strength=%s)", merged_count, strength)
