#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Convert resnet18-5c106cde.pth (torch) → safetensors for face-parsing MLX (#910).

One-time offline conversion (requires torch — NOT a runtime dep).

Usage:
    # Download from zllrunning/face-parsing.PyTorch GitHub releases:
    wget https://github.com/zllrunning/face-makeup.PyTorch/raw/master/resnet18-5c106cde.pth -O /tmp/face_parsing.pth
    # or HF mirror if mirrored
    python scripts/convert_face_parsing.py /tmp/face_parsing.pth ~/.fusion-mlx/models/face-parsing/resnet18-5c106cde.safetensors
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
    print(f"[convert_face_parsing] wrote {len(flat)} tensors to {dst}")


def main(argv):
    if len(argv) != 3:
        print(__doc__)
        return 2
    convert(argv[1], argv[2])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
