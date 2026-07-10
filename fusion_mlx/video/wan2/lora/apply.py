import mlx.core as mx
import mlx.nn as nn

from .types import LoRAWeights


def apply_lora_to_linear(
    linear_weight: mx.array,
    lora_weights_and_strengths: list[tuple[LoRAWeights, float]],
) -> mx.array:
    orig_dtype = linear_weight.dtype
    modified_weight = linear_weight

    for weights, strength in lora_weights_and_strengths:
        scale = weights.scale
        # Compute delta in float32 for precision, then cast back to avoid
        # promoting model weights (e.g. bfloat16 → float32 causes ~1.5x slowdown)
        delta = (weights.lora_B @ weights.lora_A) * (scale * strength)
        modified_weight = modified_weight + delta.astype(orig_dtype)

    return modified_weight


def _normalize_wan_lora_key(lora_key: str, model_keys: set) -> str:
    # Try the key as-is first
    if f"{lora_key}.weight" in model_keys or lora_key in model_keys:
        return lora_key

    # Common prefixes to strip
    prefixes_to_strip = [
        "model.diffusion_model.",
        "diffusion_model.",
        "base_model.model.",
        "model.",
    ]

    candidates = [lora_key]
    for prefix in prefixes_to_strip:
        if lora_key.startswith(prefix):
            candidates.append(lora_key[len(prefix) :])

    for candidate in candidates:
        # Try as-is
        if f"{candidate}.weight" in model_keys or candidate in model_keys:
            return candidate

        # Apply Wan2.2 key transformations
        transformed = candidate

        # FFN: ffn.0 → ffn.fc1, ffn.2 → ffn.fc2
        transformed = transformed.replace(".ffn.0.", ".ffn.fc1.")
        transformed = transformed.replace(".ffn.2.", ".ffn.fc2.")
        if transformed.endswith(".ffn.0"):
            transformed = transformed[: -len(".ffn.0")] + ".ffn.fc1"
        if transformed.endswith(".ffn.2"):
            transformed = transformed[: -len(".ffn.2")] + ".ffn.fc2"

        # Text embedding: text_embedding.0 → text_embedding_0
        transformed = transformed.replace("text_embedding.0.", "text_embedding_0.")
        transformed = transformed.replace("text_embedding.2.", "text_embedding_1.")
        if transformed.endswith("text_embedding.0"):
            transformed = transformed[: -len("text_embedding.0")] + "text_embedding_0"
        if transformed.endswith("text_embedding.2"):
            transformed = transformed[: -len("text_embedding.2")] + "text_embedding_1"

        # Time embedding: time_embedding.0 → time_embedding_0
        transformed = transformed.replace("time_embedding.0.", "time_embedding_0.")
        transformed = transformed.replace("time_embedding.2.", "time_embedding_1.")
        if transformed.endswith("time_embedding.0"):
            transformed = transformed[: -len("time_embedding.0")] + "time_embedding_0"
        if transformed.endswith("time_embedding.2"):
            transformed = transformed[: -len("time_embedding.2")] + "time_embedding_1"

        # Time projection: time_projection.1 → time_projection
        transformed = transformed.replace("time_projection.1.", "time_projection.")
        if transformed.endswith("time_projection.1"):
            transformed = transformed[: -len("time_projection.1")] + "time_projection"

        # Patch embedding: patch_embedding → patch_embedding_proj
        if (
            "patch_embedding" in transformed
            and "patch_embedding_proj" not in transformed
        ):
            transformed = transformed.replace("patch_embedding", "patch_embedding_proj")

        if f"{transformed}.weight" in model_keys or transformed in model_keys:
            return transformed

    # Return best attempt with prefix stripped
    for prefix in prefixes_to_strip:
        if lora_key.startswith(prefix):
            return lora_key[len(prefix) :]

    return lora_key


# Also support LTX-style key normalization
def _normalize_ltx_lora_key(lora_key: str, model_keys: set) -> str:
    if f"{lora_key}.weight" in model_keys or lora_key in model_keys:
        return lora_key

    prefixes_to_strip = [
        "model.diffusion_model.",
        "diffusion_model.",
        "model.",
    ]

    for prefix in prefixes_to_strip:
        if lora_key.startswith(prefix):
            normalized = lora_key[len(prefix) :]

            if f"{normalized}.weight" in model_keys or normalized in model_keys:
                return normalized

            transformed = normalized
            if transformed.endswith(".to_out.0"):
                transformed = transformed[: -len(".to_out.0")] + ".to_out"
            transformed = transformed.replace(".to_out.0.", ".to_out.")
            transformed = transformed.replace(".ff.net.0.proj.", ".ff.proj_in.")
            transformed = transformed.replace(".ff.net.0.proj", ".ff.proj_in")
            transformed = transformed.replace(".ff.net.2.", ".ff.proj_out.")
            transformed = transformed.replace(".ff.net.2", ".ff.proj_out")
            transformed = transformed.replace(
                ".audio_ff.net.0.proj.", ".audio_ff.proj_in."
            )
            transformed = transformed.replace(
                ".audio_ff.net.0.proj", ".audio_ff.proj_in"
            )
            transformed = transformed.replace(".audio_ff.net.2.", ".audio_ff.proj_out.")
            transformed = transformed.replace(".audio_ff.net.2", ".audio_ff.proj_out")

            if f"{transformed}.weight" in model_keys or transformed in model_keys:
                return transformed

    # Try transformations on the original key
    transformed = lora_key
    if transformed.endswith(".to_out.0"):
        transformed = transformed[: -len(".to_out.0")] + ".to_out"
    transformed = transformed.replace(".to_out.0.", ".to_out.")
    transformed = transformed.replace(".ff.net.0.proj.", ".ff.proj_in.")
    transformed = transformed.replace(".ff.net.0.proj", ".ff.proj_in")
    transformed = transformed.replace(".ff.net.2.", ".ff.proj_out.")
    transformed = transformed.replace(".ff.net.2", ".ff.proj_out")

    if f"{transformed}.weight" in model_keys or transformed in model_keys:
        return transformed

    for prefix in prefixes_to_strip:
        if lora_key.startswith(prefix):
            return lora_key[len(prefix) :]

    return lora_key


def _normalize_lora_key(lora_key: str, model_keys: set) -> str:
    # Detect model architecture from keys
    is_wan = any("self_attn.q.weight" in k for k in model_keys)

    if is_wan:
        return _normalize_wan_lora_key(lora_key, model_keys)
    else:
        return _normalize_ltx_lora_key(lora_key, model_keys)


def apply_loras_to_weights(
    model_weights: dict[str, mx.array],
    module_to_loras: dict[str, list[tuple[LoRAWeights, float]]],
    verbose: bool = False,
    quantization_bits: int = 0,
) -> dict[str, mx.array]:
    modified_weights = dict(model_weights)
    model_keys = set(model_weights.keys())

    applied_count = 0
    skipped_count = 0
    skipped_modules = []

    for module_name, loras in module_to_loras.items():
        normalized_name = _normalize_lora_key(module_name, model_keys)
        weight_key = f"{normalized_name}.weight"

        if weight_key not in modified_weights:
            if normalized_name not in modified_weights:
                skipped_count += 1
                skipped_modules.append(module_name)
                if verbose and skipped_count <= 5:
                    print(
                        f"    DEBUG: '{module_name}' -> '{normalized_name}' -> NOT FOUND"
                    )
                    similar = [
                        k
                        for k in list(model_keys)[:1000]
                        if normalized_name.split(".")[-1] in k
                    ][:3]
                    if similar:
                        print(f"      Similar keys: {similar}")
                continue
            weight_key = normalized_name

        original_weight = modified_weights[weight_key]

        # Handle quantized weights: dequantize → apply delta → re-quantize
        scales_key = f"{normalized_name}.scales"
        biases_key = f"{normalized_name}.biases"
        is_quantized = (
            original_weight.dtype == mx.uint32
            and scales_key in modified_weights
            and biases_key in modified_weights
        )

        if is_quantized:
            scales = modified_weights[scales_key]
            biases = modified_weights[biases_key]
            group_size = (original_weight.shape[-1] * 32) // (
                scales.shape[-1] * quantization_bits
            )
            dequantized = mx.dequantize(
                original_weight,
                scales,
                biases,
                group_size=group_size,
                bits=quantization_bits,
            )
            modified = apply_lora_to_linear(dequantized, loras)
            # Re-quantize with same parameters
            new_w, new_scales, new_biases = mx.quantize(
                modified, group_size=group_size, bits=quantization_bits
            )
            modified_weights[weight_key] = new_w
            modified_weights[scales_key] = new_scales
            modified_weights[biases_key] = new_biases
        else:
            modified_weights[weight_key] = apply_lora_to_linear(original_weight, loras)

        applied_count += 1

    if applied_count > 0:
        print(f"  ✓ Applied to {applied_count} modules")
    if skipped_count > 0:
        print(f"  ⚠ Skipped {skipped_count} incompatible modules")

    return modified_weights


class LoRALinear(nn.Module):

    def __init__(
        self,
        linear: nn.Module,
        lora_weights_and_strengths: list[tuple[LoRAWeights, float]],
    ):
        super().__init__()
        self.linear = linear
        self.lora_weights_and_strengths = lora_weights_and_strengths

    def __call__(self, x: mx.array) -> mx.array:
        output = self.linear(x)
        for weights, strength in self.lora_weights_and_strengths:
            scale = weights.scale
            lora_out = x @ weights.lora_A.T @ weights.lora_B.T
            output = output + (scale * strength * lora_out)
        return output


def apply_loras_to_model(
    model: nn.Module,
    module_to_loras: dict[str, list[tuple[LoRAWeights, float]]],
    verbose: bool = False,
) -> int:
    # Build a set of model module paths for key normalization
    module_paths = set()
    for name, _ in model.named_modules():
        module_paths.add(name)
        module_paths.add(f"{name}.weight")

    # Map LoRA keys → model module paths
    lora_to_module = {}
    for lora_key in module_to_loras:
        normalized = _normalize_lora_key(lora_key, module_paths)
        if normalized.endswith(".weight"):
            normalized = normalized[: -len(".weight")]
        lora_to_module[lora_key] = normalized

    applied_count = 0
    dequant_count = 0
    skipped = []

    for lora_key, loras in module_to_loras.items():
        module_path = lora_to_module[lora_key]
        parts = module_path.split(".")

        # Traverse to the parent module
        parent = model
        try:
            for part in parts[:-1]:
                parent = (
                    getattr(parent, part) if not part.isdigit() else parent[int(part)]
                )
            leaf_name = parts[-1]
            target = (
                getattr(parent, leaf_name)
                if not leaf_name.isdigit()
                else parent[int(leaf_name)]
            )
        except (AttributeError, IndexError, TypeError):
            skipped.append(lora_key)
            if verbose:
                print(f"    DEBUG: '{lora_key}' -> '{module_path}' -> module not found")
            continue

        if isinstance(target, nn.QuantizedLinear):
            # Dequantize → merge LoRA → replace with bf16 Linear
            weight = mx.dequantize(
                target.weight,
                target.scales,
                target.biases,
                group_size=target.group_size,
                bits=target.bits,
            )
            merged = apply_lora_to_linear(weight, loras)
            new_linear = nn.Linear(merged.shape[1], merged.shape[0])
            new_linear.weight = merged
            if "bias" in target:
                new_linear.bias = target.bias
            if leaf_name.isdigit():
                parent[int(leaf_name)] = new_linear
            else:
                setattr(parent, leaf_name, new_linear)
            dequant_count += 1
            applied_count += 1
        elif isinstance(target, nn.Linear):
            # Merge directly into weight
            target.weight = apply_lora_to_linear(target.weight, loras)
            applied_count += 1
        else:
            skipped.append(lora_key)
            if verbose:
                print(
                    f"    DEBUG: '{module_path}' is {type(target).__name__}, not Linear"
                )
            continue

    if applied_count > 0:
        msg = f"  ✓ Applied to {applied_count} modules"
        if dequant_count > 0:
            msg += f" ({dequant_count} dequantized to bf16)"
        print(msg)
    if skipped:
        print(f"  ⚠ Skipped {len(skipped)} incompatible modules")

    return applied_count
