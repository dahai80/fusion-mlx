# SPDX-License-Identifier: Apache-2.0
"""High-level GGUF loader with ASFW layout conversion + layer fallback.

Workflow (v2 doc §3.3):
  GGUF file → GGUFReader.read_header() → per-tensor read_tensor_data()
  → ASFWConverter.convert() → ASFWLayout (simdf32-aligned bytes + FP16
  dequant for degrade path) → MLX array (degrade) or Metal kernel (PR-K)

Layer format validation: each tensor's dtype is checked against the
ASFW dispatch table. Unsupported dtypes (IQ series, Q2_K, etc.) raise
UnsupportedQuantError; the loader catches this and marks the layer for
native mlx_lm fallback rather than aborting the whole model.

Env switches:
  FUSION_SHIM_ASFW=1  — enable ASFW conversion at GGUF load time.
                        Default OFF (prototype: stock mlx_lm loading).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .asfw import ASFWConverter, ASFWLayout, UnsupportedQuantError
from .gguf_reader import GGUFMetadata, GGUFReader, TensorInfo

logger = logging.getLogger(__name__)


@dataclass
class GGUFLoadResult:
    """Result of loading a GGUF file with ASFW conversion."""

    metadata: GGUFMetadata
    tensors: list[TensorInfo] = field(default_factory=list)
    # ASFW layouts for successfully converted tensors.
    asfw_layouts: dict[str, ASFWLayout] = field(default_factory=dict)
    # Tensor names that fell back (unsupported quant or conversion error).
    fallback_tensors: dict[str, str] = field(default_factory=dict)
    # FP16 arrays for the degrade path (stock MLX matmul).
    f16_arrays: dict[str, Any] = field(default_factory=dict)

    @property
    def has_fallbacks(self) -> bool:
        return len(self.fallback_tensors) > 0

    @property
    def all_converted(self) -> bool:
        return len(self.fallback_tensors) == 0


def is_asfw_enabled() -> bool:
    """Whether ASFW conversion is enabled at GGUF load time."""
    return os.environ.get("FUSION_SHIM_ASFW", "0") == "1"


def load_gguf(
    path: str | Path,
    *,
    convert_asfw: bool | None = None,
    dequantize: bool = True,
) -> GGUFLoadResult:
    """Load a GGUF file: parse header + optionally ASFW-convert tensors.

    Args:
        path: Path to .gguf file.
        convert_asfw: Override the FUSION_SHIM_ASFW env switch. None =
            respect env (default OFF). When False, only the header +
            tensor descriptors are read (no weight data loaded).
        dequantize: Produce FP16 dequantized arrays for the degrade path.

    Returns:
        GGUFLoadResult with metadata, tensor descriptors, and ASFW
        layouts / FP16 arrays for converted tensors.
    """
    path = Path(path)
    if convert_asfw is None:
        convert_asfw = is_asfw_enabled()

    reader = GGUFReader(path)
    meta = reader.read_header()

    result = GGUFLoadResult(metadata=meta, tensors=list(reader.tensors))

    if not convert_asfw:
        logger.info(
            "GGUF load %s: header only (ASFW OFF), %d tensors described",
            path.name,
            len(result.tensors),
        )
        return result

    converter = ASFWConverter()
    for info in result.tensors:
        try:
            raw = reader.read_tensor_data(info)
        except Exception as exc:
            logger.warning(
                "GGUF tensor %s: data read failed: %s; marking for fallback",
                info.name,
                exc,
            )
            result.fallback_tensors[info.name] = f"read_error: {exc}"
            continue

        try:
            layout = converter.convert(info, raw, dequantize=dequantize)
            result.asfw_layouts[info.name] = layout
            if layout.f16_array is not None:
                result.f16_arrays[info.name] = layout.f16_array
        except UnsupportedQuantError as exc:
            logger.info(
                "GGUF tensor %s: %s; falling back to native mlx_lm",
                info.name,
                exc,
            )
            result.fallback_tensors[info.name] = f"unsupported_quant: {info.dtype_name}"
        except Exception as exc:
            logger.warning(
                "GGUF tensor %s: ASFW convert failed: %s; marking for fallback",
                info.name,
                exc,
            )
            result.fallback_tensors[info.name] = f"convert_error: {exc}"

    converted = len(result.asfw_layouts)
    fallback = len(result.fallback_tensors)
    logger.info(
        "GGUF load %s: ASFW converted %d/%d tensors, %d fallback",
        path.name,
        converted,
        len(result.tensors),
        fallback,
    )
    return result


def validate_layer_formats(tensors: list[TensorInfo]) -> dict[str, list[str]]:
    """Check tensor dtypes against ASFW support + report unsupported.

    Returns a dict with keys 'supported' and 'unsupported', each a list
    of tensor names. Used before loading to decide whether to attempt
    ASFW or skip straight to native mlx_lm.
    """
    supported_dtypes = set(ASFWConverter._dispatchers.keys())
    result: dict[str, list[str]] = {"supported": [], "unsupported": []}
    for info in tensors:
        if info.dtype_name in supported_dtypes:
            result["supported"].append(info.name)
        else:
            result["unsupported"].append(info.name)
    return result
