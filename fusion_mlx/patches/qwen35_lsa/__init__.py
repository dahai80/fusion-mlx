# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5 LSA (latent sparse attention) pre-load patch.

Arms the LSA serving path for ``model_type=qwen3_5`` checkpoints (Qwen3.8-27B
family): full-attention layers replace the dense decode SDPA with
centroid-scored bucket selection (top-p buckets expanded to top-k + sink +
window) using a distilled d-lsa projection map, implemented against the
``fusion_lsa`` package. The map file is validated here, fail-loud, before any
model loads.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

PATCH_ID = "qwen35_lsa"
TARGET_MODEL_TYPE = "qwen3_5"

_APPLIED = False
_active = False
_map_path: str | None = None


def set_lsa_active(enabled: bool, map_path: str | None) -> None:
    global _active, _map_path
    _active = enabled
    _map_path = map_path


def is_active() -> bool:
    return _active


def map_path() -> str | None:
    return _map_path


def reset_state() -> None:
    global _active, _map_path, _APPLIED
    _active = False
    _map_path = None
    _APPLIED = False


def validate_map(path: str) -> dict[str, str]:
    # Fail-loud: the serve flag pointed at the map, so a bad path or a
    # truncated file must stop the boot here, not degrade silently later.
    map_file = Path(path)
    if not map_file.is_file():
        raise FileNotFoundError(f"LSA map not found: {path}")
    from safetensors import safe_open

    with safe_open(path, framework="numpy") as handle:
        metadata = handle.metadata() or {}
    for field in ("model_arch", "indexer_layers", "latent_dim", "index_dim"):
        if field not in metadata:
            raise ValueError(f"LSA map {path} is missing metadata field {field}")
    return metadata


def activate(map_path_value: str | None) -> dict[str, str]:
    # Called by the pre-load dispatch when lsa_enabled is set: validate the
    # map, arm the module state, log the armed confirmation.
    global _active, _map_path
    if not map_path_value:
        raise ValueError(
            "--enable-lsa set without --lsa-map; the LSA path needs a projection map"
        )
    metadata = validate_map(map_path_value)
    _active = True
    _map_path = map_path_value
    logger.info(
        "LSA armed for %s: map=%s arch=%s layers=%s latent=%s index=%s",
        TARGET_MODEL_TYPE,
        map_path_value,
        metadata["model_arch"],
        metadata["indexer_layers"],
        metadata["latent_dim"],
        metadata["index_dim"],
    )
    return metadata


def _apply_patch(_config=None) -> None:
    logger.info(
        "Qwen3.5 LSA mlx-lm patch applied (armed=%s, map=%s)", _active, _map_path
    )


def apply_qwen35_lsa_patch() -> bool:
    # Idempotent registration of the patch into the central registry so the
    # doctor audit surface lists it; the actual attention wrapping happens at
    # model load (see the model adapter in this package).
    global _APPLIED
    if _APPLIED:
        return False
    from ..registry import register

    register(
        PATCH_ID,
        TARGET_MODEL_TYPE,
        "LSA sparse decode for qwen3_5 full-attention layers (d-lsa map)",
        _apply_patch,
    )
    _APPLIED = True
    return True


apply_qwen35_lsa_patch()
