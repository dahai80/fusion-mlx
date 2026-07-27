# SPDX-License-Identifier: Apache-2.0
# UniWorld-V1 PyTorch -> MLX weight converter.
# Converts denoise_projector, vae_projector, siglip_projector, task_head
# from HuggingFace PyTorch format to MLX safetensors.

import json
import logging
import shutil
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

logger = logging.getLogger(__name__)


def convert_uniworld_weights(
    source_dir: str | Path,
    output_dir: str | Path,
    dtype: str = "float16",
) -> dict[str, Any]:
    """Convert UniWorld-V1 PyTorch weights to MLX format.

    Args:
        source_dir: Path to UniWorld-V1 HF repo (contains model*.safetensors)
        output_dir: Output directory for converted weights
        dtype: Target dtype (float16 or bfloat16)
    Returns:
        Dict with conversion stats
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = {"converted": 0, "skipped": 0, "errors": []}

    # Create subdirectories
    (output_dir / "vlm").mkdir(exist_ok=True)
    (output_dir / "siglip").mkdir(exist_ok=True)
    (output_dir / "flux").mkdir(exist_ok=True)
    (output_dir / "projectors").mkdir(exist_ok=True)

    # Load source weights
    source_weights = _load_source_weights(source_dir)
    if not source_weights:
        logger.error("No source weights found in %s", source_dir)
        stats["errors"].append("no source weights")
        _save_meta(output_dir, source_dir, dtype, stats)
        return stats

    logger.info("Loaded %d source weight keys", len(source_weights))

    # Split weights by component
    vlm_weights = {}
    siglip_weights = {}
    projector_weights = {}
    task_head_weights = {}
    other_weights = {}

    for k, v in source_weights.items():
        if k.startswith("model.vision_tower.") or k.startswith("vision_tower."):
            siglip_weights[k] = v
        elif k.startswith("model.denoise_tower."):
            projector_weights[k] = v
        elif k.startswith("task_head."):
            task_head_weights[k] = v
        elif k.startswith("model.") and "vision" not in k and "denoise" not in k:
            vlm_weights[k] = v
        else:
            other_weights[k] = v

    # Convert and save each component
    mx_dtype = mx.float16 if dtype == "float16" else mx.bfloat16

    if vlm_weights:
        _save_component(vlm_weights, output_dir / "vlm", mx_dtype, "VLM", stats)

    if siglip_weights:
        _save_component(siglip_weights, output_dir / "siglip", mx_dtype, "SigLIP", stats)

    if projector_weights:
        _save_component(projector_weights, output_dir / "projectors", mx_dtype,
                        "Projectors", stats)

    if task_head_weights:
        _save_component(task_head_weights, output_dir / "task_head.safetensors",
                        mx_dtype, "TaskHead", stats, single_file=True)

    # Copy config
    config_src = source_dir / "config.json"
    if config_src.exists():
        shutil.copy2(config_src, output_dir / "config.json")
        logger.info("Copied config.json")

    _save_meta(output_dir, source_dir, dtype, stats)

    logger.info(
        "Conversion complete: %d converted, %d skipped, %d errors",
        stats["converted"], stats["skipped"], len(stats["errors"]),
    )
    return stats


def _save_meta(output_dir, source_dir, dtype, stats):
    meta = {
        "source": str(source_dir),
        "dtype": dtype,
        "stats": stats,
    }
    with open(output_dir / "conversion_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(
        "Conversion complete: %d converted, %d skipped, %d errors",
        stats["converted"], stats["skipped"], len(stats["errors"]),
    )
    return stats


def _load_source_weights(source_dir: Path) -> dict[str, mx.array]:
    weights = {}
    safetensor_files = sorted(source_dir.glob("*.safetensors"))
    if not safetensor_files:
        # Try model subdirectory
        safetensor_files = sorted((source_dir / "model").glob("*.safetensors"))
    for sf in safetensor_files:
        logger.info("Loading %s", sf.name)
        w = mx.load(str(sf))
        weights.update(w)
    return weights


def _save_component(
    weights: dict[str, mx.array],
    output_path: Path,
    mx_dtype: Any,
    label: str,
    stats: dict[str, Any],
    single_file: bool = False,
) -> None:
    if single_file:
        target = output_path
    else:
        target = output_path / "weights.safetensors"
        output_path.mkdir(parents=True, exist_ok=True)

    converted = {}
    for k, v in weights.items():
        try:
            if isinstance(v, mx.array):
                arr = v.astype(mx_dtype)
            else:
                arr = mx.array(np.array(v), dtype=mx_dtype)
            converted[k] = arr
            stats["converted"] += 1
        except Exception as e:
            logger.warning("Failed to convert %s: %s", k, e)
            stats["skipped"] += 1
            stats["errors"].append(f"{k}: {e}")

    if converted:
        mx.save_safetensors(str(target), converted)
        logger.info("Saved %s: %d weights to %s", label, len(converted), target)
