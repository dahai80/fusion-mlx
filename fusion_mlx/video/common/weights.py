# SPDX-License-Identifier: Apache-2.0
# Directory weight loader shared by hand-ported video pipelines.
#
# mlx.utils.load_weights was removed in mlx 0.32 (#1219 discovery) — every
# svd/hunyuanvideo pipeline weight load silently raised ImportError and fell
# back to random init. This is the local flat-dir replacement: merge every
# *.safetensors / *.npz in `path` into one dict.

import glob
import logging
import os

import mlx.core as mx

logger = logging.getLogger(__name__)


def load_dir_weights(path: str) -> dict:
    files = sorted(glob.glob(os.path.join(path, "*.safetensors"))) + sorted(
        glob.glob(os.path.join(path, "*.npz"))
    )
    if not files:
        raise FileNotFoundError(f"no .safetensors/.npz files found at {path}")
    weights: dict = {}
    for f in files:
        weights.update(mx.load(f))
    logger.info(
        "load_dir_weights: %d tensors from %d file(s) in %s",
        len(weights),
        len(files),
        path,
    )
    return weights
