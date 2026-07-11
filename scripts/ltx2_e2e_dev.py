#!/usr/bin/env python3
import logging
import os
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_DIR = os.environ.get(
    "LTX2_MODEL_DIR",
    str(Path.home() / ".fusion-mlx/models/mlx-community/LTX-2-dev-bf16"),
)
TEXT_ENCODER = os.environ.get("LTX2_TEXT_ENCODER") or None
OUTPUT_MP4 = os.environ.get("LTX2_OUTPUT", "/tmp/fusion_ltx2_e2e_dev.mp4")

PROMPT = "A serene mountain lake at sunrise, cinematic, detailed, gentle ripples"
HEIGHT = 256
WIDTH = 256
NUM_FRAMES = 9
NUM_STEPS = 10
SEED = 42


def main():
    model_dir = Path(MODEL_DIR)
    if not model_dir.exists():
        logger.error("LTX-2 model dir not found: %s", MODEL_DIR)
        sys.exit(1)

    if not (model_dir / "transformer" / "config.json").exists():
        logger.error(
            "Modular layout not found at %s/transformer/config.json. "
            "Run convert.py first to split the monolithic safetensors.",
            MODEL_DIR,
        )
        sys.exit(1)

    logger.info("E2E T2V (DEV pipeline): model=%s", MODEL_DIR)
    logger.info(
        "Params: %dx%d, %d frames, %d steps, seed=%d",
        WIDTH,
        HEIGHT,
        NUM_FRAMES,
        NUM_STEPS,
        SEED,
    )

    from fusion_mlx.video.ltx2.generate import PipelineType, generate_video

    t0 = time.time()
    generate_video(
        MODEL_DIR,
        TEXT_ENCODER,
        PROMPT,
        pipeline=PipelineType.DEV,
        height=HEIGHT,
        width=WIDTH,
        num_frames=NUM_FRAMES,
        num_inference_steps=NUM_STEPS,
        cfg_scale=4.0,
        seed=SEED,
        fps=24,
        output_path=OUTPUT_MP4,
        verbose=True,
        audio=False,
        stg_scale=0.0,
        modality_scale=1.0,
    )
    elapsed = time.time() - t0
    logger.info("Total wall time: %.1fs", elapsed)

    out = Path(OUTPUT_MP4)
    if out.exists() and out.stat().st_size > 0:
        logger.info("SUCCESS: %s (%d bytes)", OUTPUT_MP4, out.stat().st_size)
    else:
        logger.error("FAILED: no output video at %s", OUTPUT_MP4)
        sys.exit(1)


if __name__ == "__main__":
    main()
