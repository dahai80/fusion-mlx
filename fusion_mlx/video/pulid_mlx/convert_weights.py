# SPDX-License-Identifier: Apache-2.0
"""Convert PuLID/EVA-CLIP PyTorch weights to MLX safetensors format.

Handles two sub-models:
  1. EVA-CLIP vision encoder (EVAVisionTransformer)
  2. IDFormer identity encoder

Key transformations:
  - EVA-CLIP: strip "visual." prefix, skip text encoder keys,
    transpose Conv2d (O,I,kH,kW) -> (O,kH,kW,I)
  - IDFormer: remap PyTorch key names to MLX module structure

Usage:
  python -m fusion_mlx.video.pulid_mlx.convert_weights \\
      --pulid-dir /path/to/PuLID \\
      --output-dir /path/to/pulid_mlx_weights
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import mlx.core as mx

logger = logging.getLogger(__name__)


def _convert_conv_weight(w):
    if w.ndim == 4:
        return w.permute(0, 2, 3, 1).contiguous().numpy()
    return w.numpy()


def convert_eva_clip(pt_weights: dict, output_path: str):
    logger.info("Converting EVA-CLIP weights (%d tensors)", len(pt_weights))
    skip_keys = {"text.", "logit_scale", "mask_token"}
    mlx_weights = {}
    for k, v in pt_weights.items():
        new_k = k
        if k.startswith("visual."):
            new_k = k[7:]
        if any(skip in new_k for skip in skip_keys):
            logger.debug("skip: %s", k)
            continue
        if v.ndim == 4:
            np_val = _convert_conv_weight(v)
        else:
            np_val = v.float().numpy()
        mlx_weights[new_k] = mx.array(np_val)

    logger.info("EVA-CLIP: %d tensors after filtering", len(mlx_weights))
    mx.save_safetensors(output_path, mlx_weights)
    logger.info("EVA-CLIP saved to %s", output_path)
    return mlx_weights


_IDFORMER_KEY_MAP = {
    "model.": "",
    "mapping.0.": "mappings.0.net.0.",
    "mapping.1.": "mappings.0.net.1.",
    "mapping.2.": "mappings.0.net.2.",
    "mapping.3.": "mappings.1.net.0.",
    "mapping.4.": "mappings.1.net.1.",
    "mapping.5.": "mappings.1.net.2.",
    "mapping.6.": "mappings.2.net.0.",
    "mapping.7.": "mappings.2.net.1.",
    "mapping.8.": "mappings.2.net.2.",
    "mapping.9.": "mappings.3.net.0.",
    "mapping.10.": "mappings.3.net.1.",
    "mapping.11.": "mappings.3.net.2.",
    "mapping.12.": "mappings.4.net.0.",
    "mapping.13.": "mappings.4.net.1.",
    "mapping.14.": "mappings.4.net.2.",
    "id_embedding_mapping.0.": "id_embedding_mapping.net.0.",
    "id_embedding_mapping.1.": "id_embedding_mapping.net.1.",
    "id_embedding_mapping.2.": "id_embedding_mapping.net.2.",
    "id_embedding_mapping.3.": "id_embedding_mapping.net.3.",
    "id_embedding_mapping.4.": "id_embedding_mapping.net.4.",
}


def convert_idformer(pt_weights: dict, output_path: str):
    logger.info("Converting IDFormer weights (%d tensors)", len(pt_weights))

    mlx_weights = {}
    for k, v in pt_weights.items():
        new_k = k
        for src, dst in _IDFORMER_KEY_MAP.items():
            if new_k.startswith(src):
                new_k = dst + new_k[len(src):]
                break
        if v.ndim == 4:
            np_val = _convert_conv_weight(v)
        else:
            np_val = v.float().numpy()
        mlx_weights[new_k] = mx.array(np_val)

    logger.info("IDFormer: %d tensors after remapping", len(mlx_weights))
    mx.save_safetensors(output_path, mlx_weights)
    logger.info("IDFormer saved to %s", output_path)
    return mlx_weights


def _load_pytorch_weights(path: str) -> dict:
    try:
        import torch
    except ImportError:
        logger.error("PyTorch required for weight conversion: pip install torch")
        sys.exit(1)

    path = Path(path)
    if path.is_dir():
        candidates = sorted(path.glob("*.safetensors")) + sorted(path.glob("*.bin")) + sorted(path.glob("*.pt"))
        if not candidates:
            raise FileNotFoundError(f"No weight files in {path}")
        logger.info("Loading from directory: %d file(s)", len(candidates))
        state_dict = {}
        for f in candidates:
            logger.info("  loading %s", f.name)
            if f.suffix == ".safetensors":
                from safetensors.torch import load_file
                state_dict.update(load_file(str(f)))
            else:
                ckpt = torch.load(str(f), map_location="cpu", weights_only=True)
                if "state_dict" in ckpt:
                    state_dict.update(ckpt["state_dict"])
                else:
                    state_dict.update(ckpt)
        return state_dict
    else:
        logger.info("Loading single file: %s", path)
        if path.suffix == ".safetensors":
            from safetensors.torch import load_file
            return load_file(str(path))
        ckpt = torch.load(str(path), map_location="cpu", weights_only=True)
        if "state_dict" in ckpt:
            return ckpt["state_dict"]
        return ckpt


def convert_pulid(pulid_dir: str, output_dir: str):
    pulid_dir = Path(pulid_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    eva_dir = pulid_dir / "eva_clip"
    if not eva_dir.exists():
        eva_dir = pulid_dir / "EVA02-CLIP-L-14-336"

    if eva_dir.exists():
        logger.info("=== EVA-CLIP ===")
        eva_weights = _load_pytorch_weights(str(eva_dir))
        eva_out = output_dir / "eva_clip" / "weights.safetensors"
        eva_out.parent.mkdir(parents=True, exist_ok=True)
        convert_eva_clip(eva_weights, str(eva_out))
    else:
        logger.warning("EVA-CLIP directory not found at %s, skipping", eva_dir)

    idformer_dir = pulid_dir / "id_former"
    if not idformer_dir.exists():
        idformer_dir = pulid_dir / "idformer"
    if not idformer_dir.exists():
        idformer_dir = pulid_dir

    idformer_weights = None
    for subpath in [idformer_dir, pulid_dir / "pulid_v1.safetensors", pulid_dir / "pulid.safetensors"]:
        p = Path(subpath) if not isinstance(subpath, Path) else subpath
        if p.exists():
            logger.info("=== IDFormer ===")
            idformer_weights = _load_pytorch_weights(str(p))
            break

    if idformer_weights is not None:
        idformer_out = output_dir / "id_former" / "weights.safetensors"
        idformer_out.parent.mkdir(parents=True, exist_ok=True)
        idformer_keys = {k: v for k, v in idformer_weights.items() if not k.startswith("visual.") and "text." not in k}
        convert_idformer(idformer_keys, str(idformer_out))
    else:
        logger.warning("IDFormer weights not found, skipping")

    meta = {
        "source": str(pulid_dir),
        "converted_by": "fusion_mlx.video.pulid_mlx.convert_weights",
        "components": [],
    }
    if eva_dir.exists():
        meta["components"].append("eva_clip")
    if idformer_weights is not None:
        meta["components"].append("id_former")
    (output_dir / "conversion_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    logger.info("Conversion complete: %s", output_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Convert PuLID/EVA-CLIP PyTorch weights to MLX safetensors")
    parser.add_argument("--pulid-dir", required=True, help="Path to PuLID PyTorch weights directory")
    parser.add_argument("--output-dir", required=True, help="Output directory for MLX safetensors")
    args = parser.parse_args()
    convert_pulid(args.pulid_dir, args.output_dir)
