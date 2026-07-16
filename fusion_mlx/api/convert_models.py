# SPDX-License-Identifier: Apache-2.0
# Pydantic request models for the /v1/convert + /v1/quantize async job API.
# Field names mirror the `fusion-mlx convert` CLI flags so the API reuses the
# CLI pipeline (fusion_mlx.cli_convert) without translation.

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class _ConvertBase(BaseModel):
    model: str = Field(
        ..., description="HF repo (org/name), model alias, or local model path"
    )

    output_path: str | None = Field(
        None, description="Output directory (default: ./<model-basename>)"
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
