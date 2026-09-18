#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Convert dw-ll_ucoco_384.pth (torch) → safetensors for DWPose MLX backend (#909).

One-time offline conversion (requires torch — NOT a runtime dep). The output
safetensors is loaded by fusion_mlx.video.dwpose.DWPose.from_pretrained at runtime.

Usage:
    # Download checkpoint from HF mirror first:
    HF_MIRROR=https://hf-mirror.com huggingface-cli download yzd-v/DWPose dw-ll_ucoco_384.pth --local-dir /tmp/dwpose
    python scripts/convert_dwpose.py /tmp/dwpose/dw-ll_ucoco_384.pth ~/.fusion-mlx/models/dwpose/dw-ll_ucoco_384.safetensors
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def convert(src_pth: str, dst_safetensors: str):
    import torch
    from safetensors.numpy import save_file

    sd = torch.load(src_pth, map_location="cpu", weights_only=True)
    if "state_dict" in sd:
        sd = sd["state_dict"]
    flat = {}
    for k, v in sd.items():
        arr = v.detach().cpu().numpy().astype(np.float32)
        flat[k] = arr
    dst = Path(dst_safetensors)
    dst.parent.mkdir(parents=True, exist_ok=True)
    save_file(flat, str(dst))
    print(f"[convert_dwpose] wrote {len(flat)} tensors to {dst}")


def main(argv):
    if len(argv) != 3:
        print(__doc__)
        return 2
    convert(argv[1], argv[2])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
