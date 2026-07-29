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


# Fixed-width float quant modes (mlx main): nvfp4/mxfp4 are 4-bit E2M1,
# mxfp8 is 8-bit. mlx-lm's quantize_model.defaults_for_mode fills the
# correct (group_size, bits) per mode, so we pass None and let it choose —
# a user-supplied --quant-group-size of 64 would otherwise break nvfp4
# (which requires 16). Affine still needs --quant-bits to enable.
_FP_QUANT_MODES = ("nvfp4", "mxfp4", "mxfp8")


def _build_convert_kwargs(args, hf_path: str) -> dict:
    bits = getattr(args, "quant_bits", None)
    mode = getattr(args, "quant_mode", "affine")
    if mode in _FP_QUANT_MODES:
        quantize = True
        q_group_size = None
        q_bits = None
    else:
        quantize = bits is not None
        q_group_size = getattr(args, "quant_group_size", 64)
        q_bits = bits
    return {
        "mlx_path": _resolve_output_path(hf_path, getattr(args, "out", None)),
        "quantize": quantize,
        "q_group_size": q_group_size,
        "q_bits": q_bits,
        "q_mode": mode,
        "dtype": getattr(args, "dtype", None),
        "upload_repo": getattr(args, "upload_repo", None),
        "dequantize": getattr(args, "dequantize", False),
        "trust_remote_code": getattr(args, "trust_remote_code", False),
    }


def _convert_pytorch_to_safetensors(model_path: Path) -> Path:
    import shutil
    import tempfile

    import mlx.core as mx
    import torch

    pytorch_files = sorted(
        list(model_path.glob("pytorch_model.bin"))
        + list(model_path.glob("pytorch_model*.bin"))
    )
    if not pytorch_files:
        raise FileNotFoundError(f"No pytorch_model.bin found in {model_path}")

    logger.info(
        "Pre-converting %d pytorch weight file(s) to safetensors",
        len(pytorch_files),
    )
    weights: dict[str, mx.array] = {}
    for pf in pytorch_files:
        state_dict = torch.load(str(pf), map_location="cpu", weights_only=True)
        for k, v in state_dict.items():
            v = v.detach().cpu().float().numpy()
            weights[k] = mx.array(v)

    tmp_dir = Path(tempfile.mkdtemp(prefix="fusion_mlx_convert_"))
    out_file = tmp_dir / "model.safetensors"
    mx.save_safetensors(str(out_file), weights)
    logger.info("Safetensors cache written to %s", out_file)

    for f in model_path.iterdir():
        if f.suffix in (".bin",) or f.name == "model.safetensors":
            continue
        target = tmp_dir / f.name
        if not target.exists():
            if f.is_file():
                shutil.copy2(str(f), str(target))
            elif f.is_dir():
                shutil.copytree(str(f), str(target))
    return tmp_dir


def _run_convert(hf_path: str, **kwargs) -> str:
    import shutil

    from mlx_lm import convert as mlx_convert

    logger.info(
        "convert: %s -> %s (quantize=%s bits=%s mode=%s)",
        hf_path,
        kwargs["mlx_path"],
        kwargs["quantize"],
        kwargs["q_bits"],
        kwargs["q_mode"],
    )
    converted_dir: Path | None = None
    try:
        mlx_convert(hf_path, **kwargs)
    except FileNotFoundError as exc:
        if "safetensors" not in str(exc).lower():
            raise
        model_path = Path(hf_path)
        if not model_path.is_dir():
            from huggingface_hub import snapshot_download

            model_path = Path(snapshot_download(hf_path))
        pytorch_files = list(model_path.glob("pytorch_model*.bin"))
        if not pytorch_files:
            raise
        logger.info("No safetensors found, attempting pytorch_model.bin conversion")
        converted_dir = _convert_pytorch_to_safetensors(model_path)
        mlx_convert(str(converted_dir), **kwargs)
    finally:
        if converted_dir is not None:
            try:
                shutil.rmtree(str(converted_dir), ignore_errors=True)
            except Exception:
                pass
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
