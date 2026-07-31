# SPDX-License-Identifier: Apache-2.0
"""Weight conversion orchestration — HF safetensors to MLX format.

Callers: fusion_mlx.admin.migrate_route
API: convert_model(hf_dir, output_dir, config, template, ...) -> ConvertResult
Schema: ConvertResult(dataclass) — output_dir, num_weights, total_params_b, orphans, missing, error
User instruction verbatim: "做一个端到端的功能，做模型迁移和量化的功能，以openpangu为例，把迁移的每个步骤展现在GUI上"
"""

import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import mlx.core as mx
import numpy as np
from safetensors import safe_open

from .architectures import ArchTemplate
from .weight_mapper import build_weight_map, find_orphan_keys, find_missing_keys

logger = logging.getLogger(__name__)


@dataclass
class ConvertResult:
    output_dir: str
    num_weights: int = 0
    total_params_b: float = 0.0
    orphans: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    error: Optional[str] = None


def _load_hf_weights(hf_dir: str) -> dict[str, mx.array]:
    weights = {}
    for fname in sorted(Path(hf_dir).glob("*.safetensors")):
        logger.info("Loading %s", fname.name)
        with safe_open(str(fname), framework="numpy") as f:
            for key in f.keys():
                arr = f.get_tensor(key)
                weights[key] = mx.array(arr)
    logger.info("Loaded %d tensors from %s", len(weights), hf_dir)
    return weights


def _remap_weights(
    hf_weights: dict[str, mx.array],
    weight_map: dict[str, str],
) -> dict[str, mx.array]:
    mlx_weights = {}
    for hf_name, mlx_name in weight_map.items():
        if hf_name in hf_weights:
            mlx_weights[mlx_name] = hf_weights[hf_name]
        else:
            logger.warning("Mapped key not found in HF weights: %s", hf_name)
    return mlx_weights


def _quantize_weights(
    weights: dict[str, mx.array],
    quant_bits: int = 4,
    quant_group_size: int = 64,
) -> dict[str, mx.array]:
    if quant_bits <= 0:
        return weights

    from mlx.utils import quantize as mlx_quantize

    quantized = {}
    skip_patterns = ("norm.weight", "embed_tokens.weight", "lm_head.weight")

    for name, tensor in weights.items():
        if any(name.endswith(p) for p in skip_patterns) or len(tensor.shape) < 2:
            quantized[name] = tensor
            continue

        try:
            q_names, q_weights = mlx_quantize(
                {name: tensor},
                group_size=quant_group_size,
                bits=quant_bits,
            )
            for qn, qw in zip(q_names, q_weights):
                quantized[qn] = qw
        except Exception:
            logger.warning("Quantize failed for %s, keeping bf16", name)
            quantized[name] = tensor

    return quantized


def _build_mlx_config(
    hf_config: dict,
    template: ArchTemplate,
) -> dict:
    mlx_config = {
        "model_type": template.name,
        "num_hidden_layers": hf_config.get("num_hidden_layers", hf_config.get("n_layer", 0)),
        "hidden_size": hf_config.get("hidden_size", hf_config.get("d_model", 0)),
        "intermediate_size": hf_config.get("intermediate_size", 0),
        "num_attention_heads": hf_config.get("num_attention_heads", 0),
        "num_key_value_heads": hf_config.get("num_key_value_heads",
                                               hf_config.get("num_attention_heads", 0)),
        "rms_norm_eps": hf_config.get("rms_norm_eps", 1e-6),
        "vocab_size": hf_config.get("vocab_size", 0),
        "tie_word_embeddings": hf_config.get("tie_word_embeddings", False),
    }

    rope_theta = hf_config.get("rope_theta")
    if rope_theta:
        mlx_config["rope_theta"] = rope_theta

    rope_traditional = hf_config.get("rope_traditional", False)
    mlx_config["rope_traditional"] = rope_traditional

    if template.has_bias:
        mlx_config["bias"] = True
    if template.has_mlp_bias:
        mlx_config["mlp_bias"] = True

    max_position_embeddings = hf_config.get("max_position_embeddings")
    if max_position_embeddings:
        mlx_config["max_position_embeddings"] = max_position_embeddings

    return mlx_config


def convert_model(
    hf_dir: str,
    output_dir: str,
    config: dict,
    template: ArchTemplate,
    quant_bits: int = 0,
    quant_group_size: int = 64,
    progress_cb: Optional[Callable[[float, str], None]] = None,
) -> ConvertResult:
    result = ConvertResult(output_dir=output_dir)

    try:
        if progress_cb:
            progress_cb(0.0, "Loading HF weights")
        hf_weights = _load_hf_weights(hf_dir)
        hf_keys = list(hf_weights.keys())

        if progress_cb:
            progress_cb(0.2, "Building weight map")
        weight_map = build_weight_map(config, template)
        result.orphans = find_orphan_keys(hf_keys, weight_map)
        result.missing = find_missing_keys(weight_map, hf_keys)

        if result.missing:
            logger.warning(
                "Missing %d expected keys — conversion may be incomplete",
                len(result.missing),
            )

        if progress_cb:
            progress_cb(0.4, "Remapping weights")
        mlx_weights = _remap_weights(hf_weights, weight_map)

        if quant_bits > 0:
            if progress_cb:
                progress_cb(0.6, "Quantizing weights")
            mlx_weights = _quantize_weights(mlx_weights, quant_bits, quant_group_size)

        if progress_cb:
            progress_cb(0.8, "Saving MLX model")
        os.makedirs(output_dir, exist_ok=True)

        mlx_config = _build_mlx_config(config, template)
        config_path = os.path.join(output_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(mlx_config, f, indent=2)
        logger.info("Wrote config.json to %s", output_dir)

        weights_path = os.path.join(output_dir, "weights.npz")
        mx.savez(weights_path, **mlx_weights)
        logger.info("Wrote weights.npz (%d tensors) to %s", len(mlx_weights), output_dir)

        tokenizer_src = Path(hf_dir)
        for tok_name in ("tokenizer.json", "tokenizer.model", "tokenizer_config.json",
                         "special_tokens_map.json", "added_tokens.json"):
            src = tokenizer_src / tok_name
            if src.exists():
                shutil.copy2(str(src), os.path.join(output_dir, tok_name))
                logger.info("Copied %s", tok_name)

        result.num_weights = len(mlx_weights)
        total_elements = sum(np.prod(w.shape) for w in mlx_weights.values())
        result.total_params_b = float(total_elements) / 1e9

        if progress_cb:
            progress_cb(1.0, "Conversion complete")

    except Exception as e:
        logger.exception("Conversion failed: %s", e)
        result.error = str(e)

    return result
