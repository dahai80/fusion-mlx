# SPDX-License-Identifier: Apache-2.0
# Open-Sora V2 weight converter: PyTorch safetensors -> MLX.

import json
import logging
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


_MAX_HEADER_SIZE = 100 * 1024 * 1024  # 100 MB sanity cap


def _read_safetensor_file(sf_path: Path) -> dict:
    weights = {}
    with open(sf_path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        if header_size > _MAX_HEADER_SIZE:
            raise ValueError(
                f"Safetensor header too large ({header_size} bytes): {sf_path}"
            )
        header = json.loads(f.read(header_size))
        offset = 8 + header_size
        for name, info in header.items():
            f.seek(offset + info["data_offsets"][0])
            data = f.read(info["data_offsets"][1] - info["data_offsets"][0])
            arr = np.frombuffer(data, dtype=np.float32).reshape(info["shape"])
            weights[name] = arr
    return weights


def _process_shard(weights: dict, fused_qkv: bool) -> dict:
    if not fused_qkv:
        keys_to_check = list(weights.keys())
        for key in keys_to_check:
            if ".qkv." in key:
                prefix = key.rsplit(".", 1)[0]
                splits = _split_fused_qkv(weights, prefix)
                weights.update(splits)

    converted = {}
    dropped = []
    for key, val in weights.items():
        new_key = _remap_key(key)
        if new_key:
            converted[new_key] = val
        else:
            dropped.append(key)
    del weights
    return converted, dropped


def convert_weights(
    src_dir: str | Path,
    dst_dir: str | Path,
    fused_qkv: bool = False,
    dtype: str = "bf16",
):
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Converting Open-Sora V2 weights: %s -> %s", src_dir, dst_dir)

    safetensor_files = sorted(src_dir.glob("*.safetensors"))
    if not safetensor_files:
        logger.error("No safetensors files found in %s", src_dir)
        return

    all_converted = {}
    all_dropped = []
    for sf in safetensor_files:
        logger.info("Reading %s", sf.name)
        shard = _read_safetensor_file(sf)
        logger.info("  %d tensors, processing...", len(shard))
        converted, dropped = _process_shard(shard, fused_qkv)
        all_converted.update(converted)
        all_dropped.extend(dropped)
        del converted
        logger.info("  Done, freed shard memory")

    if all_dropped:
        logger.warning("Dropped %d keys: %s...", len(all_dropped), all_dropped[:10])

    weights_path = dst_dir / "weights.npz"
    np.savez(str(weights_path), **all_converted)
    del all_converted
    logger.info("Saved weights to %s", weights_path)

    meta = {
        "source": str(src_dir),
        "dtype": dtype,
        "fused_qkv": fused_qkv,
        "num_keys": len(safetensor_files),
        "dropped_keys": all_dropped[:50],
    }
    with open(dst_dir / "conversion_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(
        "Conversion complete: %d shards, %d dropped",
        len(safetensor_files),
        len(all_dropped),
    )


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
