# SPDX-License-Identifier: Apache-2.0
"""oQ: FusionMLX Universal Dynamic Quantization.

Mixed-precision quantization combining GGUF K-quant layer position strategy,
unsloth Dynamic 2.0 selective non-quantization, and BnB MSE-optimal clipping.

Supported levels: oQ2, oQ2.5, oQ2.7, oQ3, oQ3.5, oQ4, oQ5, oQ6, oQ8 (base
bits differ, same predicate). Fractional levels keep the lower level's base
bits and add a mandatory boost for routed expert down_proj (Super Weights
protection; see _LEVEL_EXPERT_DOWN_BOOST) plus a higher bpw budget.
"""

import json
import logging
import re
import shutil
import tempfile
import time as _time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    from mlx_lm.models.base import create_attention_mask

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

from fusion_mlx.pool.model_discovery import _has_vision_subconfig

logger = logging.getLogger(__name__)

OQ_LEVELS = {2, 2.5, 2.7, 3, 3.5, 4, 5, 6, 8}

OQ_DTYPES: tuple[str, ...] = ("bfloat16", "float16")

_OQ_DEFAULT_GROUP_SIZE = 64

_MAX_MODEL_RAM_FRACTION = 0.8

# Auto-built proxy for sensitivity measurement when the source model
# exceeds available RAM. Uniform 4-bit affine quant — same shape as a
# user-supplied --sensitivity-model, but built on demand.
_PROXY_QUANT_BITS = 4
_PROXY_QUANT_GROUP_SIZE = 64

_LEVEL_BITS: dict[float, int] = {
    2: 2,
    2.5: 2,
    2.7: 2,
    3: 3,
    3.5: 3,
    4: 4,
    5: 5,
    6: 6,
    8: 8,
}

_LEVEL_PROTECTION: dict[float, str] = {
    2: "full",
    2.5: "full",
    2.7: "full",
    3: "full",
    3.5: "full",
    4: "full",
    5: "full",
    6: "full",
    8: "full",
}

# Fractional levels: mandatory protection for routed expert down_proj
# (Super Weights), expressed as bits above the level's base bits.
# 2.5 -> 3-bit, 2.7 -> 4-bit, 3.5 -> 4-bit.
_LEVEL_EXPERT_DOWN_BOOST: dict[float, int] = {2.5: 1, 2.7: 2, 3.5: 1}

_OQ_BPW_TARGETS: dict[float, tuple[float, float]] = {
    2: (2.8, 3.0),
    2.5: (3.1, 3.3),
    2.7: (3.35, 3.45),
    3: (3.5, 3.7),
    3.5: (3.8, 4.0),
    4: (4.6, 4.7),
    5: (5.5, 5.7),
    6: (6.5, 6.7),
}


from ._core import _TrackedTensor
def _bpw_targets_for_level(oq_level: float) -> tuple[float, float] | None:
    """Return (target_bpw, hard_cap_bpw) for the given oQ level, or None."""
    return _OQ_BPW_TARGETS.get(oq_level)


def _is_deepseek_v4_config(config: dict) -> bool:
    model_type = str(config.get("model_type", "")).lower()
    if model_type.startswith("deepseek_v4"):
        return True

    architectures = config.get("architectures") or []
    return any(
        str(arch).lower().replace("_", "") == "deepseekv4forcausallm"
        for arch in architectures
    )


def _validate_oq_dtype_for_model(config: dict, dtype: str) -> None:
    if dtype == "float16" and _is_deepseek_v4_config(config):
        raise ValueError(
            "oQ dtype=float16 is unsupported for deepseek_v4. "
            "DeepSeek V4 fp16 oQ can collapse to repeated BOS tokens during "
            "generation; use dtype='bfloat16' instead."
        )


@dataclass
class QuantPlan:
    """Byte-budgeted mixed-precision plan for a single quantization run."""

    boost_map: dict[str, dict]
    effective_bpw: float
    target_bpw: float
    hard_cap_bpw: float


def universal_quant_predicate(
    path: str, module, config: dict, oq_level: int = 4
) -> bool | dict:
    """Per-tensor quantization decision based on GGUF/unsloth/llama.cpp rules.

    Protection levels vary by oQ level:
        oQ2: minimal protection (router fp16, lm_head 4-bit only) → ~2.5 bpw
        oQ2.5/oQ2.7/oQ3.5: fractional levels — lower level's base bits,
            routed expert down_proj protected above base per
            _LEVEL_EXPERT_DOWN_BOOST (Super Weights protection)
        oQ3: base 2-bit + full protection → ~3.3 bpw
        oQ4-oQ6: base N-bit + full protection
        oQ7: base 8-bit + full protection
        oQ8: near-uniform 8-bit (router fp16 only) → ~8.0 bpw

    Args:
        path: Dot-separated module path (e.g. "model.layers.0.self_attn.v_proj").
        module: The nn.Module being quantized.
        config: Model config.json dict.
        oq_level: oQ quantization level (2-8).

    Returns:
        False to skip quantization (keep fp16),
        True to use default bits,
        dict with {"bits": N, "group_size": M} for per-layer override.
    """
    path = _normalize_quant_path(path)
    path_l = path.lower()

    non_quantizable = config.get("_oq_non_quantizable", set())
    if path in non_quantizable:
        return False

    tc = config.get("text_config", {})
    num_layers = config.get("num_hidden_layers") or tc.get("num_hidden_layers", 32)
    num_experts = (
        config.get("num_local_experts")
        or tc.get("num_local_experts")
        or config.get("num_experts")
        or tc.get("num_experts", 0)
        or 0
    )
    hidden_size = config.get("hidden_size") or tc.get("hidden_size", 0)
    is_moe = num_experts > 0

    base_bits = int(_LEVEL_BITS.get(oq_level, oq_level))
    protection = _LEVEL_PROTECTION.get(oq_level, "full")
    full_protection = protection == "full"

    def gs():
        if _is_moe_router(path):
            return 64
        if num_experts >= 150:
            return 128
        return 64

    def bits(n):
        effective = int(max(n, base_bits))
        return {
            "bits": effective,
            "group_size": _gs_for_mode(effective, gs()),
            "mode": _mode_for_bits(effective),
        }

    if _is_moe_router(path):
        return False  # fp16 — tiny weights, some models (MoEGate) lack to_quantized()

    if "shared_expert_gate" in path and "gate_proj" not in path:
        return {"bits": 8, "group_size": 64, "mode": "affine"}

    if _is_vision_tensor(path):
        return False

    if _is_audio_tensor(path):
        return False

    if any(
        p in path_l
        for p in ("ssm_alpha", "ssm_beta", "a_log", "time_decay", "time_faaaa")
    ):
        return False

    if path.endswith(".D"):
        return False

    # Gated DeltaNet / Mamba-like SSM sensitive params (Qwen3_5 hybrid arch).
    # dt_bias drives the discretization step, keep fp16/fp32 like A_log.
    # conv1d is a small depth-wise causal conv, very sensitive to low bits.
    # linear_attn.out_proj mirrors self_attn.o_proj sensitivity.
    if path_l.endswith("dt_bias"):
        return False
    if "conv1d" in path_l and "linear_attn" in path_l:
        return bits(8)
    if "linear_attn.out_proj" in path_l:
        return bits(5)

    boost_map = config.get("_oq_boost_map") or {}
    if path in boost_map:
        return dict(boost_map[path])

    if config.get("_oq_use_budget_plan"):
        if any(p in path for p in ("ssm_output", "ssm_out")):
            return bits(8)
        if "lora.2" in path:
            return bits(8)
        return True

    if not full_protection:
        if any(p in path for p in ("lm_head", "output.weight", "classifier")):
            return bits(6)

        if any(p in path for p in ("ssm_output", "ssm_out")):
            return bits(8)

        if any(p in path for p in ("embed_tokens", "wte", "word_embeddings")):
            return bits(base_bits + 2)

        if num_experts >= 512 and hidden_size >= 4096:
            if "gate_proj" in path and "shared_expert" not in path:
                return bits(4)

        layer_idx = _extract_layer_index(path)
        if layer_idx >= 0:
            sensitive = layer_idx < num_layers // 8 or layer_idx >= 7 * num_layers // 8
            is_expert = "switch_mlp" in path or "experts" in path
            if sensitive and not is_expert:
                return bits(base_bits + 1)

        return True

    if any(p in path for p in ("ssm_output", "ssm_out")):
        return bits(8)

    if "lora.2" in path:
        return bits(8)

    if any(p in path for p in ("lm_head", "output.weight", "classifier")):
        return bits(6)

    if "cross_attn" in path and "o_proj" in path:
        return bits(6)

    if any(
        p in path for p in ("kv_a_proj_with_mqa", "kv_b_proj", "q_a_proj", "q_b_proj")
    ):
        return bits(6)

    if "o_proj" in path and "shared_expert" not in path:
        if not is_moe:
            return bits(5)

    if "shared_expert" in path and not path.endswith("shared_expert_gate"):
        return bits(8)

    if num_experts >= 512 and hidden_size >= 4096:
        if "gate_proj" in path and "shared_expert" not in path:
            return bits(4)
        if "down_proj" in path and "shared_expert" not in path:
            return bits(3)

    layer_idx = _extract_layer_index(path)

    sensitivity_map = config.get("_oq_sensitivity_map")
    if sensitivity_map and layer_idx >= 0:
        scores = list(sensitivity_map.values())
        scores.sort(reverse=True)
        threshold = scores[max(0, len(scores) // 4 - 1)] if scores else 0
        sensitive = sensitivity_map.get(str(layer_idx), 0) >= threshold
    else:
        sensitive = layer_idx >= 0 and (
            layer_idx < num_layers // 8 or layer_idx >= 7 * num_layers // 8
        )

    if any(p in path for p in ("v_proj", "v_a_proj", "v_b_proj")):
        if sensitive:
            return bits(6)
        return True

    if any(p in path for p in ("down_proj", "w2", "mlp.fc2", "wo")):
        is_routed_expert = (
            is_moe
            and "shared_expert" not in path
            and ("switch_mlp" in path or "experts" in path)
        )
        if is_routed_expert:
            down_boost = _LEVEL_EXPERT_DOWN_BOOST.get(oq_level)
            if down_boost:
                # Fractional levels protect routed expert down_proj above
                # the base bits (Super Weights protection).
                return bits(base_bits + down_boost)
            return True
        if sensitive:
            return bits(6)
        return bits(5)

    if any(p in path for p in ("q_proj", "k_proj")):
        if sensitive:
            return bits(5)

    if any(p in path for p in ("qkv_proj", "in_proj_qkv", "attn_qkv")):
        if sensitive:
            return bits(5)

    if any(p in path for p in ("in_proj_z", "in_proj_a", "in_proj_b", "delta_net")):
        return bits(5)

    if any(p in path for p in ("mixer.in_proj", "mixer.out_proj", "x_proj", "dt_proj")):
        return bits(5)

    return True


def _is_vision_tensor(name: str) -> bool:
    """Check if a tensor belongs to the vision encoder/projector."""
    return any(
        p in name
        for p in (
            "visual.",
            "vision_",
            "patch_embed",
            "pos_embed",
            "image_newline",
            "multi_modal_projector",
            "visual.merger",
            "image_norm",
            "temporal_embed",
        )
    )


def _is_audio_tensor(name: str) -> bool:
    """Check if a tensor belongs to the audio encoder.

    Mirrors `_is_vision_tensor`: matches `audio_tower.*` only, not
    `embed_audio.*` (the projection from audio output to text hidden size,
    which is quantized like `embed_vision.embedding_projection`).
    """
    return "audio_tower" in name


def _is_moe_router(path: str) -> bool:
    """Detect MoE router/gate layers (distinct from gate_proj)."""
    if path.endswith(("mlp.gate", ".router", ".router.layer")):
        return True
    if path.endswith(".gate") and "gate_proj" not in path:
        return True
    if ".gate." in path and "gate_proj" not in path:
        return True
    return False


def _extract_layer_index(path: str) -> int:
    """Extract transformer layer index from module path. Returns -1 if absent."""
    m = re.search(r"layers\.(\d+)\.", path)
    return int(m.group(1)) if m else -1


def _default_bits(config: dict) -> int:
    """Read default quantization bits from config."""
    q = config.get("quantization", {})
    return q.get("bits", 4)


def _normalize_quant_path(path: str) -> str:
    """Normalize tensor/module names to the module path used in configs."""
    if path.endswith(".weight"):
        return path[:-7]
    if path.endswith(".scales"):
        return path[:-7]
    if path.endswith(".biases"):
        return path[:-7]
    return path


def _base_bits_for_level(oq_level: int) -> int:
    return int(_LEVEL_BITS.get(oq_level, oq_level))


def _bytes_per_group(mode: str) -> int:
    if mode == "mxfp4":
        return 1
    if mode == "mxfp8":
        return 2
    return 4


def _tensor_quantized_bytes(shape: tuple, bits: int, group_size: int, mode: str) -> int:
    """Estimate serialized bytes for a quantized tensor."""
    n_elements = 1
    for dim in shape:
        n_elements *= dim
    if len(shape) < 2:
        return n_elements * 2
    if shape[-1] % group_size != 0:
        return n_elements * 2
    rows = n_elements // max(shape[-1], 1)
    n_groups = shape[-1] // group_size
    weight_bytes = (n_elements * bits + 7) // 8
    overhead_bytes = rows * n_groups * _bytes_per_group(mode)
    return weight_bytes + overhead_bytes


def _estimate_effective_bpw(
    named_shapes: dict[str, tuple],
    base_bits: int,
    base_group_size: int,
    base_mode: str,
    overrides: dict[str, dict] | None = None,
) -> float:
    """Estimate effective bpw for quantizable weights only."""
    overrides = overrides or {}
    total_bits = 0
    total_params = 0

    for path, shape in named_shapes.items():
        n_elements = 1
        for dim in shape:
            n_elements *= dim
        total_params += n_elements

        override = overrides.get(path)
        if override is None:
            bits = base_bits
            gs = base_group_size
            mode = base_mode
        else:
            bits = int(override.get("bits", base_bits))
            gs = int(override.get("group_size", base_group_size))
            mode = override.get("mode", _mode_for_bits(bits))

        total_bits += 8 * _tensor_quantized_bytes(shape, bits, gs, mode)

    return total_bits / max(total_params, 1)


def _collect_named_weight_shapes_from_model(model) -> dict[str, tuple]:
    """Collect quantizable weight shapes from the in-memory model."""
    named_shapes = {}
    for path, module in tree_flatten(model.leaf_modules(), is_leaf=nn.Module.is_module):
        if not hasattr(module, "weight") or not hasattr(module, "to_quantized"):
            continue
        if getattr(module.weight, "ndim", 0) < 2:
            continue
        named_shapes[_normalize_quant_path(path)] = tuple(module.weight.shape)
    return named_shapes


def _collect_named_weight_shapes_from_weights(
    weights: dict[str, Any],
) -> dict[str, tuple]:
    """Collect quantizable weight shapes from sanitized weight tensors."""
    named_shapes = {}
    for name, tensor in weights.items():
        norm_name = _normalize_quant_path(name)
        if name != f"{norm_name}.weight":
            continue
        if getattr(tensor, "ndim", 0) < 2:
            continue
        named_shapes[norm_name] = tuple(tensor.shape)
    return named_shapes


def _is_routed_expert(path: str) -> bool:
    """Check if a tensor belongs to routed MoE experts (93-98% of params)."""
    if "switch_mlp" in path:
        return True
    if "experts" in path and "shared_expert" not in path:
        return True
    if "block_sparse_moe" in path and "shared_expert" not in path:
        return True
    return False


_MANDATORY_BOOST_PATTERNS = {
    "lm_head": {"bits": 8, "group_size": 64, "mode": "affine"},
    "embeddings": {"bits": 8, "group_size": 64, "mode": "affine"},
    "embed_tokens": {"bits": 8, "group_size": 64, "mode": "affine"},
    "wte": {"bits": 8, "group_size": 64, "mode": "affine"},
}


def _sensitivity_tier(layer_score: float, max_score: float) -> int:
    """Map sensitivity score to boost tier: +4 (top), +2 (high), +1 (moderate).

    Greedy allocator will fallback to lower tiers if budget can't fit the
    requested bits (e.g., 8-bit → try 6-bit → try 5-bit).
    """
    if max_score <= 0:
        return 1
    ratio = layer_score / max_score
    if ratio >= 0.5:
        return 4
    if ratio >= 0.2:
        return 2
    return 1


def _build_quant_plan(
    named_shapes: dict[str, tuple],
    config: dict,
    oq_level: int,
    target_bpw: float = 4.6,
    hard_cap_bpw: float = 4.7,
    fixed_overrides: dict[str, dict] | None = None,
) -> QuantPlan:
    """Allocate byte-budgeted boosts using sensitivity-driven allocation.

    Strategy:
    1. Mandatory pre-allocation: consensus-critical tensors (lm_head → 8-bit)
    2. Data-driven: all non-expert tensors compete equally, ranked by
       layer sensitivity score. Higher sensitivity → more bits.
    3. Routed experts always stay at base bits (93-98% of params).

    fixed_overrides marks tensors whose output format is fixed up front
    (pre-quantized source tensors passed through as mxfp4/mxfp8). They are
    priced into the baseline bpw at their true cost and excluded from every
    boost decision.
    """
    base_bits = _base_bits_for_level(oq_level)
    base_mode = _mode_for_bits(base_bits)
    base_group_size = _gs_for_mode(base_bits, _OQ_DEFAULT_GROUP_SIZE)
    boost_map: dict[str, dict] = {}
    fixed_overrides = fixed_overrides or {}

    layer_scores = config.get("_oq_sensitivity_map") or {}
    max_layer_score = max(layer_scores.values(), default=0.0)

    total_params = 0
    expert_params = 0
    for path, shape in named_shapes.items():
        n = 1
        for dim in shape:
            n *= dim
        total_params += n
        if _is_routed_expert(path):
            expert_params += n

    current_bpw = _estimate_effective_bpw(
        named_shapes,
        base_bits,
        base_group_size,
        base_mode,
        overrides=fixed_overrides,
    )
    total_bits_f = current_bpw * total_params

    module = None
    for path, shape in named_shapes.items():
        if path in fixed_overrides:
            continue
        pred = universal_quant_predicate(
            path, module, {**config, "_oq_boost_map": {}}, oq_level
        )
        if pred is False:
            continue
        for pattern, boost in _MANDATORY_BOOST_PATTERNS.items():
            if pattern in path:
                cand_bits = int(boost["bits"])
                if cand_bits <= base_bits:
                    break
                cand_gs = int(boost.get("group_size", base_group_size))
                cand_mode = boost.get("mode", _mode_for_bits(cand_bits))
                base_cost = _tensor_quantized_bytes(
                    shape, base_bits, base_group_size, base_mode
                )
                cand_cost = _tensor_quantized_bytes(
                    shape, cand_bits, cand_gs, cand_mode
                )
                delta = 8 * (cand_cost - base_cost)
                next_bpw = (total_bits_f + delta) / total_params
                if delta > 0 and next_bpw <= hard_cap_bpw:
                    boost_map[path] = dict(boost)
                    total_bits_f += delta
                    current_bpw = next_bpw
                break

    # Fractional levels (oQ2.5 / oQ2.7 / oQ3.5): mandatory expert down_proj
    # boost above base bits (Super Weights protection).
    _down_boost = _LEVEL_EXPERT_DOWN_BOOST.get(oq_level)
    if _down_boost:
        for path, shape in named_shapes.items():
            if path in boost_map or path in fixed_overrides:
                continue
            if not _is_routed_expert(path):
                continue
            if not any(p in path for p in ("down_proj", "w2")):
                continue
            cand_bits = base_bits + _down_boost
            if cand_bits not in (2, 3, 4, 5, 6, 8):
                continue
            cand_gs = _gs_for_mode(cand_bits, _OQ_DEFAULT_GROUP_SIZE)
            cand_mode = _mode_for_bits(cand_bits)
            base_cost = _tensor_quantized_bytes(
                shape, base_bits, base_group_size, base_mode
            )
            cand_cost = _tensor_quantized_bytes(shape, cand_bits, cand_gs, cand_mode)
            delta = 8 * (cand_cost - base_cost)
            if delta > 0:
                boost_map[path] = {
                    "bits": cand_bits,
                    "group_size": cand_gs,
                    "mode": cand_mode,
                }
                total_bits_f += delta
                current_bpw = total_bits_f / total_params

    # Protection floor: apply full protection rules as minimum bits for
    # non-expert tensors. This ensures attention, shared experts, etc. get
    # adequate precision even at aggressive base bits (e.g. oQ2 base=2).
    # Each floor boost is checked against hard_cap to avoid overshooting.
    floor_config = {**config, "_oq_use_budget_plan": False, "_oq_boost_map": {}}
    for path, shape in named_shapes.items():
        if path in boost_map or path in fixed_overrides:
            continue
        if _is_routed_expert(path):
            continue
        floor_pred = universal_quant_predicate(path, module, floor_config, oq_level)
        if not isinstance(floor_pred, dict):
            continue
        floor_bits = int(floor_pred["bits"])
        if floor_bits <= base_bits:
            continue
        floor_gs = int(
            floor_pred.get(
                "group_size", _gs_for_mode(floor_bits, _OQ_DEFAULT_GROUP_SIZE)
            )
        )
        floor_mode = floor_pred.get("mode", _mode_for_bits(floor_bits))
        old_cost = _tensor_quantized_bytes(shape, base_bits, base_group_size, base_mode)
        new_cost = _tensor_quantized_bytes(shape, floor_bits, floor_gs, floor_mode)
        delta = 8 * (new_cost - old_cost)
        if delta <= 0:
            continue
        next_bpw = (total_bits_f + delta) / total_params
        if next_bpw > hard_cap_bpw:
            continue
        boost_map[path] = {
            "bits": floor_bits,
            "group_size": floor_gs,
            "mode": floor_mode,
        }
        total_bits_f += delta
        current_bpw = next_bpw

    # Sensitivity-based greedy boost: boost tensors from their current bits
    # (which may already be elevated by the protection floor) using remaining
    # budget up to hard_cap_bpw.
    candidates = []
    for path, shape in named_shapes.items():
        if _is_routed_expert(path) or path in fixed_overrides:
            continue
        pred = universal_quant_predicate(
            path, module, {**config, "_oq_boost_map": {}}, oq_level
        )
        if pred is False:
            continue
        layer_idx = _extract_layer_index(path)
        if layer_idx < 0:
            continue
        layer_score = float(layer_scores.get(str(layer_idx), 0.0))
        # Current bits (floor or base)
        cur_bits = boost_map[path]["bits"] if path in boost_map else base_bits
        cur_gs = _gs_for_mode(cur_bits, _OQ_DEFAULT_GROUP_SIZE)
        cur_mode = _mode_for_bits(cur_bits)
        cur_cost = _tensor_quantized_bytes(shape, cur_bits, cur_gs, cur_mode)
        # Max target based on sensitivity
        ratio = layer_score / max_layer_score if max_layer_score > 0 else 0
        if ratio >= 0.5:
            max_target = 8
        elif ratio >= 0.2:
            max_target = min(cur_bits + 2, 8)
        else:
            max_target = min(cur_bits + 1, 8)
        if max_target <= cur_bits:
            continue
        candidates.append((layer_score, path, shape, cur_bits, cur_cost, max_target))

    _VALID_BITS = (2, 3, 4, 5, 6, 8)
    for _score, path, shape, cur_bits, cur_cost, max_target in sorted(
        candidates, key=lambda x: x[0], reverse=True
    ):
        for cand_bits in range(max_target, cur_bits, -1):
            if cand_bits not in _VALID_BITS or cand_bits <= cur_bits:
                continue
            cand_gs = _gs_for_mode(cand_bits, _OQ_DEFAULT_GROUP_SIZE)
            cand_mode = _mode_for_bits(cand_bits)
            cand_cost = _tensor_quantized_bytes(shape, cand_bits, cand_gs, cand_mode)
            delta = 8 * (cand_cost - cur_cost)
            if delta <= 0:
                continue
            next_bpw = (total_bits_f + delta) / total_params
            if next_bpw > hard_cap_bpw:
                continue
            boost_map[path] = {
                "bits": cand_bits,
                "group_size": cand_gs,
                "mode": cand_mode,
            }
            total_bits_f += delta
            current_bpw = next_bpw
            break

    # Fallback: if still under target, boost non-expert tensors toward 8-bit
    # regardless of sensitivity tier. On large MoE models, non-expert weights
    # are <6% of params so every bit counts to reach the target bpw.
    if current_bpw < target_bpw:
        fallback_candidates = []
        for path, shape in named_shapes.items():
            if _is_routed_expert(path) or path in fixed_overrides:
                continue
            cur = boost_map.get(path)
            if cur is None:
                continue
            cur_bits = cur["bits"]
            if cur_bits >= 8:
                continue
            cur_gs = _gs_for_mode(cur_bits, _OQ_DEFAULT_GROUP_SIZE)
            cur_mode = _mode_for_bits(cur_bits)
            cur_cost = _tensor_quantized_bytes(shape, cur_bits, cur_gs, cur_mode)
            layer_idx = _extract_layer_index(path)
            layer_score = float(layer_scores.get(str(layer_idx), 0.0))
            fallback_candidates.append((layer_score, path, shape, cur_bits, cur_cost))

        for _score, path, shape, cur_bits, cur_cost in sorted(
            fallback_candidates, key=lambda x: x[0], reverse=True
        ):
            for cand_bits in (8, 6, 5, 4, 3):
                if cand_bits <= cur_bits:
                    continue
                cand_gs = _gs_for_mode(cand_bits, _OQ_DEFAULT_GROUP_SIZE)
                cand_mode = _mode_for_bits(cand_bits)
                cand_cost = _tensor_quantized_bytes(
                    shape, cand_bits, cand_gs, cand_mode
                )
                delta = 8 * (cand_cost - cur_cost)
                if delta <= 0:
                    continue
                next_bpw = (total_bits_f + delta) / total_params
                if next_bpw > hard_cap_bpw:
                    continue
                boost_map[path] = {
                    "bits": cand_bits,
                    "group_size": cand_gs,
                    "mode": cand_mode,
                }
                total_bits_f += delta
                current_bpw = next_bpw
                break
            if current_bpw >= target_bpw:
                break

    if boost_map:
        from collections import Counter

        bits_dist = Counter(v["bits"] for v in boost_map.values())
        layer_bits = {}
        for k, v in boost_map.items():
            idx = _extract_layer_index(k)
            label = f"L{idx}" if idx >= 0 else k.split(".")[-1]
            if label not in layer_bits:
                layer_bits[label] = v["bits"]
            else:
                layer_bits[label] = max(layer_bits[label], v["bits"])
        bits_summary = ", ".join(
            f"{b}bit×{c}" for b, c in sorted(bits_dist.items(), reverse=True)
        )
        top_layers = sorted(layer_bits.items(), key=lambda x: -x[1])[:8]
        top_str = ", ".join(f"{l}={b}b" for l, b in top_layers)
        logger.info(f"  plan detail: {bits_summary} | top: {top_str}")

    return QuantPlan(
        boost_map=boost_map,
        effective_bpw=current_bpw,
        target_bpw=target_bpw,
        hard_cap_bpw=hard_cap_bpw,
    )


def resolve_output_name(
    model_name: str,
    oq_level: int,
    dtype: str = "bfloat16",
    preserve_mtp: bool = False,
) -> str:
    """Generate output model name: strip existing quant suffixes, append oQ tag.

    Appends `-fp16` suffix when dtype is float16. bfloat16 is the default and
    produces no dtype suffix (backwards compatible). When preserve_mtp is True,
    appends `-mtp` so the resulting name reflects that mtp.* tensors and
    config fields were preserved through quantization.

    Examples:
        "Qwen3.5-122B-A10B" + 4 + bfloat16 -> "Qwen3.5-122B-A10B-oQ4"
        "Qwen3.5-122B-A10B" + 4 + float16  -> "Qwen3.5-122B-A10B-oQ4-fp16"
        "Qwen3.5-122B-A10B-oQ6-fp16" + 2 + bfloat16 -> "Qwen3.5-122B-A10B-oQ2"
        "Qwen3.5-27B" + 4 + bfloat16 + preserve_mtp -> "Qwen3.5-27B-oQ4-mtp"
    """
    pattern = re.compile(
        r"-(oQ[\d.]+e?|[0-9]+[_-]?bit|fp\d+|bf\d+|mtp)$",
        flags=re.IGNORECASE,
    )
    base = model_name
    while True:
        new = pattern.sub("", base)
        if new == base:
            break
        base = new
    level_str = f"{oq_level:g}"
    suffix = f"-oQ{level_str}"
    if dtype == "float16":
        suffix += "-fp16"
    if preserve_mtp:
        suffix += "-mtp"
    return f"{base}{suffix}"


# ── Auto-discovery streaming sanitizer ──────────────────────────────────


