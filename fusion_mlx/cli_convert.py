# SPDX-License-Identifier: Apache-2.0
"""HF -> MLX model conversion + weight quantization (wraps mlx-lm convert)."""

import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _resolve_output_path(model: str, out: str | None) -> str:
    if out:
        return out
    # mlx-lm defaults to a cwd-relative ``mlx_model`` which collides across
    # repeated converts; default to ./<model-basename> instead so each model
    # lands in its own directory.
    base = model.rsplit("/", 1)[-1]
    return str(Path.cwd() / base)


def _build_convert_kwargs(args, hf_path: str) -> dict:
    bits = getattr(args, "quant_bits", None)
    quantize = bits is not None
    return {
        "mlx_path": _resolve_output_path(hf_path, getattr(args, "out", None)),
        "quantize": quantize,
        "q_group_size": getattr(args, "quant_group_size", 64),
        "q_bits": bits,
        "q_mode": getattr(args, "quant_mode", "affine"),
        "dtype": getattr(args, "dtype", None),
        "upload_repo": getattr(args, "upload_repo", None),
        "dequantize": getattr(args, "dequantize", False),
        "trust_remote_code": getattr(args, "trust_remote_code", False),
    }


def _run_convert(hf_path: str, **kwargs) -> str:
    from mlx_lm import convert as mlx_convert

    logger.info(
        "convert: %s -> %s (quantize=%s bits=%s mode=%s)",
        hf_path,
        kwargs["mlx_path"],
        kwargs["quantize"],
        kwargs["q_bits"],
        kwargs["q_mode"],
    )
    # mlx-lm's convert() return type is unstable across versions (ModelHolder
    # / None / path); we report the mlx_path we requested, not the return
    # value, so behavior is robust to upstream churn.
    mlx_convert(hf_path, **kwargs)
    return kwargs["mlx_path"]


def convert_command(args) -> int:
    from fusion_mlx.model_aliases import resolve_model

    model = args.model
    resolved = resolve_model(model)
    if resolved != model:
        logger.info("convert: alias %s -> %s", model, resolved)
        model = resolved

    try:
        out = _run_convert(model, **_build_convert_kwargs(args, model))
    except Exception as exc:
        logger.exception("convert failed for %s", model)
        print(f"Error: convert failed: {exc}", file=sys.stderr)
        return 1
    print(f"Converted model written to: {out}")
    return 0
