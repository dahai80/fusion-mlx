# SPDX-License-Identifier: Apache-2.0
# Pydantic request models for the /v1/convert + /v1/quantize async job API.
# Field names mirror the `fusion-mlx convert` CLI flags so the API reuses the
# CLI pipeline (fusion_mlx.cli_convert) without translation.

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

_ALLOWED_OUTPUT_PREFIXES: list[Path] | None = None


def _get_allowed_output_prefixes() -> list[Path]:
    global _ALLOWED_OUTPUT_PREFIXES
    if _ALLOWED_OUTPUT_PREFIXES is not None:
        return _ALLOWED_OUTPUT_PREFIXES
    home = Path.home()
    prefixes = [
        home / ".fusion-mlx" / "models",
        home / ".cache" / "huggingface",
    ]
    cwd = Path.cwd().resolve()
    if cwd != Path("/") and len(cwd.parts) >= 2:
        prefixes.append(cwd)
    _ALLOWED_OUTPUT_PREFIXES = prefixes
    return _ALLOWED_OUTPUT_PREFIXES


class _ConvertBase(BaseModel):
    model: str = Field(
        ..., description="HF repo (org/name), model alias, or local model path"
    )

    output_path: str | None = Field(
        None, description="Output directory (default: ./<model-basename>)"
    )

    @field_validator("output_path")
    @classmethod
    def _validate_output_path(cls, v):
        if v is None:
            return v
        resolved = Path(v).resolve()
        for prefix in _get_allowed_output_prefixes():
            try:
                if resolved.is_relative_to(prefix.resolve()):
                    return str(resolved)
            except Exception:
                pass
        logger.warning("output_path rejected (outside allowed dirs): %s", v)
        raise ValueError(
            "output_path must be within allowed model directories "
            "(~/.fusion-mlx/models, CWD, or HF cache)"
        )

    quant_bits: Literal[2, 3, 4, 6, 8] | None = Field(
        None, description="Weight quantization bits. Omit for a plain bf16 convert."
    )

    quant_mode: Literal["affine", "mxfp4", "nvfp4", "mxfp8"] = Field(
        "affine",
        description="affine uses quant_bits/quant_group_size; mxfp4/nvfp4/mxfp8 are "
        "fixed-width float modes that ignore quant_bits and enable quantization alone.",
    )

    quant_group_size: int = Field(
        64, ge=1, description="Group size for affine quantization"
    )

    dtype: Literal["bf16", "fp16", "fp32"] | None = Field(
        None, description="Cast weights to this dtype (default: keep source dtype)"
    )

    upload_repo: str | None = Field(
        None, description="Upload the converted model to this HF repo (org/name)"
    )

    trust_remote_code: bool = Field(
        False, description="Allow custom modeling code from the source repo"
    )


class ConvertRequest(_ConvertBase):
    dequantize: bool = Field(
        False, description="Dequantize a quantized model back to float"
    )


class QuantizeRequest(_ConvertBase):
    pass
