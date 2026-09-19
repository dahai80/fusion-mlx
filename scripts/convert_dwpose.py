#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Convert dw-ll_ucoco_384.pth (mmpose RTMPose) → safetensors for DWPose MLX (#909).

One-time offline conversion. Produces a safetensors whose keys EXACTLY match the
DWPoseMLX parameter tree (strict load at runtime). Transforms:
  - conv weights: torch OIHW (out,in,kh,kw) → MLX OHWI (out,kh,kw,in) via permute(0,2,3,1)
  - BatchNorm num_batches_tracked: dropped (MLX BatchNorm has no such field)
  - BN / Linear / ScaleNorm params: passed through unchanged

Two input modes:
  1. torch .pth:   python scripts/convert_dwpose.py <src.pth> <dst.safetensors>
  2. repair existing safetensors (same tensors, just fix layout + drop tracked):
     python scripts/convert_dwpose.py --repair <src.safetensors> <dst.safetensors>

Usage:
    # Download from yzd-v/DWPose HF (via mirror):
    HF_MIRROR=https://hf-mirror.com huggingface-cli download yzd-v/DWPose \
        dw-ll_ucoco_384.pth --local-dir /tmp/dwpose
    python scripts/convert_dwpose.py /tmp/dwpose/dw-ll_ucoco_384.pth \
        ~/.fusion-mlx/models/dwpose/dw-ll_ucoco_384.safetensors
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def _load_source(src: str) -> dict:
    """Load torch .pth or existing safetensors → {key: np.ndarray}."""
    src_path = Path(src)
    if src_path.suffix == ".pth":
        import torch

        sd = torch.load(str(src_path), map_location="cpu", weights_only=True)
        if "state_dict" in sd:
            sd = sd["state_dict"]
        return {k: v.detach().cpu().numpy() for k, v in sd.items()}
    # repair mode: existing safetensors
    from safetensors import safe_open

    out = {}
    with safe_open(str(src_path), framework="numpy") as f:
        for k in f.keys():  # noqa: SIM118
            out[k] = f.get_tensor(k)
    return out


def _is_conv_weight(name: str, arr: np.ndarray) -> bool:
    """4D .weight that is a conv (not a 2D Linear weight reshaped)."""
    return name.endswith(".weight") and arr.ndim == 4


def convert(src: str, dst: str):
    from fusion_mlx.video.dwpose import DWPoseMLX, _flatten_param_keys

    src_sd = _load_source(src)
    flat = {}
    for name, arr in src_sd.items():
        if "num_batches_tracked" in name:
            continue  # MLX BatchNorm has no num_batches_tracked
        arr = arr.astype(np.float32)
        if _is_conv_weight(name, arr):
            arr = np.transpose(arr, (0, 2, 3, 1))  # OIHW → OHWI
        flat[name] = np.ascontiguousarray(arr)

    # Strict verification against the MLX model parameter tree.
    model = DWPoseMLX()
    expected = set(_flatten_param_keys(model.parameters()))
    got = set(flat.keys())
    missing = expected - got
    extra = got - expected
    if missing or extra:
        print("[convert_dwpose] STRICT MISMATCH:", file=sys.stderr)
        print(f"  missing ({len(missing)}): {sorted(missing)[:12]}", file=sys.stderr)
        print(f"  extra   ({len(extra)}): {sorted(extra)[:12]}", file=sys.stderr)
        return 2

    from safetensors.numpy import save_file

    dst_path = Path(dst)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(flat, str(dst_path))
    print(f"[convert_dwpose] wrote {len(flat)} tensors to {dst_path} (strict OK)")
    return 0


def main(argv):
    if len(argv) == 4 and argv[1] == "--repair":
        return convert(argv[2], argv[3])
    if len(argv) == 3:
        return convert(argv[1], argv[2])
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
