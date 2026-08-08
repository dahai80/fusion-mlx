# SPDX-License-Identifier: Apache-2.0
"""GGUF load guard for MLX engines.

mlx-lm / mlx-vlm have no GGUF load path (mx.save_gguf is one-way export).
Loading a .gguf file or a GGUF-only directory crashes inside mlx_lm.load
with an opaque error. This guard detects GGUF targets up front and raises
a clear, actionable ValueError pointing the user at the MLX-native path.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

GGUF_SUFFIX = ".gguf"
_MLX_WEIGHT_SUFFIXES = (".safetensors", ".npz")


def _local_exists(model_name: str) -> bool:
    try:
        return Path(model_name).exists()
    except OSError:
        return False


def is_gguf_model(model_name: str) -> bool:
    """Return True if model_name points at a GGUF-only local target."""
    if not model_name:
        return False
    name = model_name.strip()
    if name.lower().endswith(GGUF_SUFFIX):
        return _local_exists(name)
    p = Path(name)
    try:
        if not p.is_dir():
            return False
    except OSError:
        return False
    children = list(p.iterdir())
    has_gguf = any(c.name.lower().endswith(GGUF_SUFFIX) for c in children)
    if not has_gguf:
        return False
    has_mlx_weights = any(
        c.name.lower().endswith(_MLX_WEIGHT_SUFFIXES) for c in children
    )
    has_config = (p / "config.json").exists()
    return not has_config and not has_mlx_weights


class GGUFLoadError(ValueError):
    """Raised when a GGUF target is given to an MLX engine."""


def assert_not_gguf(model_name: str, engine_kind: str = "MLX") -> None:
    """Raise GGUFLoadError if model_name is a GGUF-only target.

    Call this right before mlx_lm.load / mlx_vlm.load so the error is
    raised before any weight download or parse attempt.
    """
    if not is_gguf_model(model_name):
        return
    logger.warning("GGUF target rejected by %s engine: %s", engine_kind, model_name)
    raise GGUFLoadError(
        f"{model_name} is a GGUF model; {engine_kind} engines cannot load "
        f"GGUF (mlx-lm/mlx-vlm have no GGUF load path, only export). "
        f"Use an MLX-native checkpoint instead: download an "
        f"'mlx-community/<model>-mlx' repo, or convert via the "
        f"POST /v1/convert endpoint (pytorch|safetensors -> MLX). "
        f"GGUF is a terminal format and cannot be converted back to MLX."
    )
