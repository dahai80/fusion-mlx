# SPDX-License-Identifier: Apache-2.0
# Open-Sora V2 weight converter: PyTorch safetensors -> MLX.

import json
import logging
import os
import struct
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_KEY_REMAP = {
    "model.": "",
}


def _remap_key(key: str) -> str:
    for old, new in _KEY_REMAP.items():
        if old in key:
            key = key.replace(old, new)
    return key


def _split_fused_qkv(state_dict: dict, key_prefix: str) -> dict:
    out = {}
    for suffix in [".weight", ".bias"]:
        fused_key = key_prefix + suffix
        if fused_key not in state_dict:
            continue
        fused = state_dict.pop(fused_key)
        if fused.ndim == 1:
            q, k, v = np.split(fused, 3)
        else:
            q, k, v = np.split(fused, 3, axis=0)
        base = key_prefix.replace("qkv", "q_proj")
        out[base + suffix] = q
        out[key_prefix.replace("qkv", "k_proj") + suffix] = k
        out[key_prefix.replace("qkv", "v_proj") + suffix] = v
    return out


def convert_weights(
    src_dir: str | Path,
    dst_dir: str | Path,
    fused_qkv: bool = False,
    dtype: str = "bf16",
):
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Converting Open-Sora V2 weights: {src_dir} -> {dst_dir}")

    safetensor_files = sorted(src_dir.glob("*.safetensors"))
    if not safetensor_files:
        logger.error(f"No safetensors files found in {src_dir}")
        return

    all_weights = {}
    for sf in safetensor_files:
        logger.info(f"Reading {sf.name}")
        with open(sf, "rb") as f:
            header_size = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_size))
            offset = 8 + header_size
            for name, info in header.items():
                f.seek(offset + info["data_offsets"][0])
                data = f.read(info["data_offsets"][1] - info["data_offsets"][0])
                arr = np.frombuffer(data, dtype=np.float32).reshape(info["shape"])
                all_weights[name] = arr

    if not fused_qkv:
        new_splits = {}
        keys_to_check = list(all_weights.keys())
        for key in keys_to_check:
            if ".qkv." in key:
                prefix = key.rsplit(".", 1)[0]
                splits = _split_fused_qkv(all_weights, prefix)
                new_splits.update(splits)
        all_weights.update(new_splits)

    converted = {}
    dropped = []
    for key, val in all_weights.items():
        new_key = _remap_key(key)
        if new_key:
            converted[new_key] = val
        else:
            dropped.append(key)

    if dropped:
        logger.warning(f"Dropped {len(dropped)} keys: {dropped[:10]}...")

    weights_path = dst_dir / "weights.npz"
    np.savez(str(weights_path), **converted)
    logger.info(f"Saved {len(converted)} weights to {weights_path}")

    meta = {
        "source": str(src_dir),
        "dtype": dtype,
        "fused_qkv": fused_qkv,
        "num_keys": len(converted),
        "dropped_keys": dropped[:50],
    }
    with open(dst_dir / "conversion_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Conversion complete: {len(converted)} keys, {len(dropped)} dropped")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src", required=True, help="Source directory with safetensors"
    )
    parser.add_argument("--dst", required=True, help="Destination directory")
    parser.add_argument("--fused_qkv", action="store_true", help="Keep fused qkv")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    convert_weights(args.src, args.dst, args.fused_qkv, args.dtype)
