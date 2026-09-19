#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Convert 79999_iter.pth (zllrunning/MuseTalk BiSeNet face-parsing) → safetensors
# for the MLX BiSeNet (#910, #915).
#
# One-time offline conversion. Produces a safetensors whose keys EXACTLY match
# the BiSeNetMLX parameter tree (strict load at runtime). Transforms:
#   - conv weights: torch OIHW (out,in,kh,kw) → MLX OHWI (out,kh,kw,in)
#   - BatchNorm num_batches_tracked: dropped (MLX BatchNorm has no such field)
#   - everything else: passed through unchanged (keys already match — the
#     MuseTalk BiSeNet variant deletes the spatial path, and 79999_iter.pth
#     carries no spatial_path keys)
#
# Usage:
#   python scripts/convert_face_parsing.py <src.pth> <dst.safetensors>
#
# Source: modelscope aile1997/face-parse-bisenet 79999_iter.pth (Google Drive
# mirror — GDrive unreachable). Example:
#   curl -L "https://modelscope.cn/api/v1/models/aile1997/face-parse-bisenet/repo?Revision=master&FilePath=79999_iter.pth" -o /tmp/bisenet/79999_iter.pth
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def main(argv):
    if len(argv) != 3:
        print(__doc__)
        return 2
    src, dst = argv[1], argv[2]

    import torch

    from fusion_mlx.video.dwpose import _flatten_param_keys
    from fusion_mlx.video.face_parsing import BiSeNetMLX

    sd = torch.load(str(src), map_location="cpu", weights_only=True)
    if not isinstance(sd, dict) or "state_dict" in sd:
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
    flat = {}
    for name, tensor in sd.items():
        if "num_batches_tracked" in name:
            continue
        arr = tensor.detach().cpu().numpy().astype(np.float32)
        if arr.ndim == 4 and name.endswith(".weight"):
            arr = np.transpose(arr, (0, 2, 3, 1))  # OIHW → OHWI
        flat[name] = np.ascontiguousarray(arr)

    model = BiSeNetMLX()
    expected = set(_flatten_param_keys(model.parameters()))
    got = set(flat.keys())
    missing = expected - got
    extra = got - expected
    if missing or extra:
        print("[convert_face_parsing] STRICT MISMATCH:", file=sys.stderr)
        print(f"  missing ({len(missing)}): {sorted(missing)[:12]}", file=sys.stderr)
        print(f"  extra   ({len(extra)}): {sorted(extra)[:12]}", file=sys.stderr)
        return 2

    from safetensors.numpy import save_file

    dst_path = Path(dst)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(flat, str(dst_path))
    print(f"[convert_face_parsing] wrote {len(flat)} tensors to {dst_path} (strict OK)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
