#!/usr/bin/env python3
# Dev-only: converts raw PyTorch Wan2.2 checkpoints -> MLX format. This torch
# conversion path was intentionally dropped from the fusion_mlx.video.wan2 port
# (runtime loads pre-converted mlx-community weights directly). To run, install
# the original mlx-video + torch: pip install mlx-video torch  (dev utility,
# NOT a fusion-mlx runtime dependency).
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CHECKPOINT_DIR = Path.home() / ".cache/modelscope/hub/models/Wan-AI/Wan2___2-TI2V-5B"
OUTPUT_DIR = Path.home() / ".cache/mlx-video-models/wan22-ti2v-5b"


def main():
    if not CHECKPOINT_DIR.exists():
        logger.error("checkpoint not found: %s", CHECKPOINT_DIR)
        sys.exit(1)
    logger.info("converting Wan2.2-TI2V-5B: %s -> %s", CHECKPOINT_DIR, OUTPUT_DIR)
    try:
        import torch
        logger.info("torch %s available", torch.__version__)
    except ImportError:
        logger.error("torch not installed")
        sys.exit(1)
    from mlx_video.models.wan_2.convert import convert_wan_checkpoint
    convert_wan_checkpoint(
        str(CHECKPOINT_DIR),
        str(OUTPUT_DIR),
        dtype="bfloat16",
        model_version="auto",
    )
    logger.info("DONE: %s", OUTPUT_DIR)
    for p in sorted(OUTPUT_DIR.iterdir()):
        size = p.stat().st_size / 1024 / 1024 / 1024
        logger.info("  %s (%.2f G)", p.name, size)


if __name__ == "__main__":
    main()
