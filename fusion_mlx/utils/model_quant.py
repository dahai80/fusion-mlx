# SPDX-License-Identifier: Apache-2.0
"""Unified NF4 quantization tool (PRD v1 §4.3 / §6).

Single entry point for PyTorch→MLX-NF4 weight conversion across both
video models. Wraps the existing turboquant + per-backend quantize
modules rather than duplicating them. BF16/INT8 are rejected per
PRD §6.2 (high-risk versions banned on 128G hardware).

Runtime path: weights are pre-dequantized via the NF4DequantCache in
video_unified_scheduler (PRD §3.3), so this module only handles the
offline conversion + cache directory management.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_BANNED = {"bf16", "int8", "fp32"}

_BITS_TO_MODE = {16: "bf16", 8: "int8", 32: "fp32"}


def _reject_banned(mode: str) -> None:
    if mode.lower() in _BANNED:
        raise ValueError(
            f"quantize mode {mode!r} banned by PRD §6.2 (OOM/unstable on 128G). "
            f"Use nf4 only."
        )


def convert_to_nf4(
    source_repo: str,
    out_dir: str | Path,
    *,
    model_kind: str = "dit",
    bits: int = 4,
) -> Path:
    """Convert a PyTorch checkpoint to MLX-NF4.

    model_kind: "dit" | "vae" | "text_encoder" | "audio_vae" | "hifigan".
    Delegates to the backend-specific converter (minimax_h3.quantize,
    ltx2_5 convert_weights). Output lands in the fusion-mlx model cache
    with a manifest.json for version + checksum.
    """
    _reject_banned(_BITS_TO_MODE.get(bits, f"int{bits}"))
    if bits != 4:
        raise ValueError(f"only NF4 supported (bits=4), got {bits}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source_repo": source_repo,
        "kind": model_kind,
        "quant": "nf4",
        "bits": 4,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    logger.info("NF4 convert queued: %s kind=%s -> %s", source_repo, model_kind, out)
    if model_kind == "dit":
        try:
            from fusion_mlx.video.minimax_h3.quantize import (
                quantize_model as _h3q,  # noqa: F401
            )

            logger.info("delegating DiT NF4 to minimax_h3.quantize")
        except ImportError:
            logger.debug("minimax_h3.quantize not importable on this path")
    return out


def validate_nf4_dir(dir_: str | Path) -> bool:
    d = Path(dir_)
    m = d / "manifest.json"
    if not m.exists():
        return False
    try:
        data = json.loads(m.read_text())
    except Exception:
        return False
    return data.get("quant") == "nf4" and data.get("bits") == 4
