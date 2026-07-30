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


from ._core import _TrackedTensor, _DiscoveredPlan, _discover_sanitize_plan
def _is_qat_unquantized_config(qc) -> bool:
    """Return True if qc is a QAT training config with full-precision weights.

    Gemma 4 QAT configs carry quant_type (e.g. "q4_0") recording the training
    regime but store weights in bfloat16 — no quant_method means no actual
    weight quantization has been applied.
    """
    return (
        isinstance(qc, dict)
        and qc.get("quant_type") == "q4_0"
        and "quant_method" not in qc
    )


def validate_quantizable(config: dict) -> bool:
    """Check if a model config indicates it can be quantized.

    Models with 'quantization' key (mlx-lm quantized) are excluded.
    Models with 'quantization_config' are excluded UNLESS they are native FP8
    (e.g. MiniMax, DeepSeek) which are full-precision models stored in FP8 format,
    or QAT-trained models (e.g. Google Gemma 4 QAT variants) whose
    quantization_config records training-time settings but whose weights are
    stored in full precision (bfloat16/float16).
    """
    if "quantization" in config:
        return False
    if "quantization_config" in config:
        qc = config["quantization_config"]
        if isinstance(qc, dict):
            quant_method = qc.get("quant_method", "")
            # FP8 models are full-precision weights stored in FP8 format
            if quant_method == "fp8":
                return True
            # QAT models record training-time quant_type but weights are fp16/bf16
            if _is_qat_unquantized_config(qc):
                return True
        return False
    return True


def _sensitivity_lm_config_override(config: dict) -> dict | None:
    """Return a model_config override for mlx_lm.load when the model has a
    QAT quantization_config that mlx-lm cannot process (missing quant_method).

    mlx-lm does ``quantization_config["quant_method"]`` without a fallback, so
    QAT configs (e.g. Google Gemma 4 QAT) raise KeyError and abort the load.
    Passing ``{"quantization_config": None}`` via model_config causes
    config.update() to replace the offending key before that branch runs.
    """
    for qc in (
        config.get("quantization_config"),
        config.get("text_config", {}).get("quantization_config"),
    ):
        if _is_qat_unquantized_config(qc):
            return {"quantization_config": None}
    return None


def make_predicate(config: dict, oq_level: int = 4) -> Callable:
    """Create a quant_predicate closure for mlx-lm's quantize_model."""

    def predicate(path: str, module) -> bool | dict:
        return universal_quant_predicate(path, module, config, oq_level)

    return predicate


def estimate_bpw_and_size(
    model_path: str,
    oq_level: int,
    group_size: int = 64,
    preserve_mtp: bool = False,
) -> dict:
    """Calculate precise effective bpw and output size by scanning actual tensors.

    Applies the universal predicate to each tensor to determine its bit width,
    then computes weighted average bpw and estimated output size.

    Args:
        model_path: Path to source model directory.
        preserve_mtp: When True, mtp.* tensors are kept (counted toward
            output size) instead of being skipped. Mirrors the matching
            argument in ``quantize_oq_streaming``.
        oq_level: Target oQ level (base bits).
        group_size: Quantization group size.

    Returns:
        Dict with effective_bpw, output_size_bytes, output_size_formatted.
    """
    source = Path(model_path)
    config_path = source / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    weight_files = sorted(source.glob("*.safetensors"))
    if not weight_files:
        return {
            "effective_bpw": float(oq_level),
            "output_size_bytes": 0,
            "output_size_formatted": "?",
        }

    if preserve_mtp:
        from fusion_mlx.utils.model_loading import _checkpoint_has_mtp_weights

        if not _checkpoint_has_mtp_weights(source):
            preserve_mtp = False

    # Header-only scan: shapes/dtypes come from the safetensors headers, so
    # checkpoints with dtypes mx.load rejects (F8_E8M0 block scales) still
    # estimate. The logical view hides .scale companions and reports
    # pre-quantized weights at their unpacked logical shape.
    idx = _LazyTensorIndex(weight_files)
    logical = idx.logical_metadata()

    named_shapes = {}
    for name, (shape, _dtype) in logical.items():
        norm = _normalize_quant_path(name)
        if name == f"{norm}.weight" and len(shape) >= 2:
            named_shapes[norm] = tuple(shape)

    # Match quantize_oq_streaming: the budget-plan flag must be set BEFORE
    # any predicate evaluation so the fixed-override floors here agree with
    # the per-tensor pricing loop below (the flag changes which predicate
    # branch answers).
    config["_oq_use_budget_plan"] = oq_level in _OQ_BPW_TARGETS

    # Pre-quantized tensors that pass through in source precision (mirrors
    # the decision in quantize_oq_streaming, evaluated pre-boost).
    fixed_overrides = {}
    _pre_boost_config = {**config, "_oq_boost_map": {}}
    for _path in named_shapes:
        _info = idx.source_quant_info(f"{_path}.weight")
        if _info is None:
            continue
        _floor_bits, _, _ = _get_predicate_bits(
            f"{_path}.weight", _pre_boost_config, oq_level, group_size
        )
        if _floor_bits is not None and _floor_bits >= _info["bits"]:
            fixed_overrides[_path] = {
                "bits": _info["bits"],
                "group_size": _info["group_size"],
                "mode": _info["mode"],
            }

    # Build budget plan for accurate estimate (position-based sensitivity)
    _level_targets = _bpw_targets_for_level(oq_level)
    if _level_targets is not None:
        tc = config.get("text_config", {})
        num_layers = config.get("num_hidden_layers") or tc.get("num_hidden_layers", 32)
        pos_sens = {}
        for i in range(num_layers):
            if i < num_layers // 8 or i >= 7 * num_layers // 8:
                pos_sens[str(i)] = 0.05
            elif i < num_layers // 4 or i >= 3 * num_layers // 4:
                pos_sens[str(i)] = 0.02
            else:
                pos_sens[str(i)] = 0.01
        config["_oq_sensitivity_map"] = pos_sens

        plan = _build_quant_plan(
            named_shapes,
            config,
            oq_level,
            target_bpw=_level_targets[0],
            hard_cap_bpw=_level_targets[1],
            fixed_overrides=fixed_overrides,
        )
        config["_oq_boost_map"] = plan.boost_map
    else:
        config["_oq_boost_map"] = {}

    total_params = 0
    total_weighted_bits = 0
    total_output_bytes = 0

    for name, (shape, _dtype) in logical.items():
        n_elements = 1
        for d in shape:
            n_elements *= d

        if _should_skip_tensor(name, preserve_mtp=preserve_mtp):
            continue

        if not _should_quantize_tensor(name, shape):
            total_params += n_elements
            total_weighted_bits += n_elements * 16
            total_output_bytes += n_elements * 2
            continue

        bits, gs, _mode = _get_predicate_bits(name, config, oq_level, group_size)
        if bits is None:
            total_params += n_elements
            total_weighted_bits += n_elements * 16
            total_output_bytes += n_elements * 2
            continue

        total_params += n_elements
        src_info = idx.source_quant_info(name)
        if src_info is not None and bits >= src_info["bits"]:
            # Passthrough: packed weight at source bits plus one e8m0
            # uint8 scale byte per group.
            rows = n_elements // max(shape[-1], 1)
            n_groups = shape[-1] // src_info["group_size"]
            tensor_bytes = (n_elements * src_info["bits"] + 7) // 8
            tensor_bytes += rows * n_groups
            total_output_bytes += tensor_bytes
            total_weighted_bits += tensor_bytes * 8
        elif len(shape) >= 2:
            n_groups = (shape[-1] + gs - 1) // gs
            rows = n_elements // max(shape[-1], 1)
            weight_bytes = (n_elements * bits + 7) // 8
            if _mode == "mxfp4":
                bytes_per_group = 1
            elif _mode == "mxfp8":
                bytes_per_group = 2
            else:
                bytes_per_group = 4
            overhead_bytes = rows * n_groups * bytes_per_group
            tensor_bytes = weight_bytes + overhead_bytes
            total_output_bytes += tensor_bytes
            total_weighted_bits += tensor_bytes * 8
        else:
            total_output_bytes += n_elements * 2
            total_weighted_bits += n_elements * 16

    for k in ("_oq_use_budget_plan", "_oq_boost_map", "_oq_sensitivity_map"):
        config.pop(k, None)

    effective_bpw = total_weighted_bits / max(total_params, 1)

    # Fractional-level correction: the expert down_proj boost is not
    # visible in pre-sanitize scans of fused layouts (gate_up_proj-style
    # tensors don't have a .weight suffix). After sanitize, down_proj is
    # ~31% of routed expert params, so each boost bit adds roughly this
    # much effective bpw. When the scan DID see the down tensors the boost
    # is already priced by the plan, so the correction would double-count.
    _down_boost = _LEVEL_EXPERT_DOWN_BOOST.get(oq_level)
    if _down_boost:
        _down_visible = any(
            _is_routed_expert(p) and any(s in p for s in ("down_proj", "w2"))
            for p in named_shapes
        )
        if not _down_visible:
            effective_bpw += 0.3 * _down_boost
            total_output_bytes = int(effective_bpw * total_params / 8)

    source_total = sum(sf.stat().st_size for sf in source.glob("*.safetensors"))
    streaming_peak = int(source_total * 1.5) + 5 * 1024**3

    return {
        "effective_bpw": round(effective_bpw, 2),
        "output_size_bytes": total_output_bytes,
        "output_size_formatted": _format_size(total_output_bytes),
        "memory_streaming_bytes": streaming_peak,
        "memory_streaming_formatted": _format_size(streaming_peak),
    }


def estimate_memory(source_size_bytes: int) -> dict:
    """Estimate peak memory for quantization.

    This is a rough estimate used before precise calculation is available.
    The /api/oq/estimate endpoint provides precise values per tensor.

    Streaming: source (mmap) + 5GB output buffer + sanitize overhead
    """
    peak = source_size_bytes + 6 * 1024**3
    return {"peak_bytes": peak, "peak_formatted": _format_size(peak)}


