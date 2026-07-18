#!/usr/bin/env python3
# LTX-2 pure-MLX port E2E smoke (text-to-video).
#
# Requires the LTX-2 19B (or 13B) weights in the port's native MLX format.
# The pure-MLX port (fusion_mlx/video/ltx2, Stages A-H) is green on 4408 tests;
# this harness is the real-weights E2E gate. diffusers LTX-2.3 weights are
# structurally incompatible (1396 unmatched / 720 missing keys) - must be the
# native format the port expects.
#
# Weights (user-provided, ~38 GB for 19B bf16 - fits 128 GB unified memory easily):
#   export LTX2_MODEL_DIR=/path/to/ltx-2-19b-mlx
#   export LTX2_TEXT_ENCODER=/path/to/text-encoder   # optional; None = port default
#
# 38 GB is NOT a hardware blocker (128 GB RAM / 622 GB disk free). The only
# difficulty is download time + MLX-format availability.
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_DIR = os.environ.get(
    "LTX2_MODEL_DIR", str(Path.home() / ".cache/mlx-video-models/ltx-2-19b")
)
TEXT_ENCODER = os.environ.get("LTX2_TEXT_ENCODER") or None
OUTPUT_MP4 = os.environ.get("LTX2_OUTPUT", "/tmp/fusion_ltx2_e2e.mp4")

PROMPT = "A serene mountain lake at sunrise, cinematic, detailed, gentle ripples"
HEIGHT = 512
WIDTH = 768
NUM_FRAMES = 25
NUM_STEPS = 10
SEED = 42


def main():
    model_dir = Path(MODEL_DIR)
    if not model_dir.exists():
        logger.error(
            "LTX-2 model dir not found: %s\n"
            "  -> Download the LTX-2 19B (or 13B) weights in the port's native "
            "MLX format and set LTX2_MODEL_DIR to that path.\n"
            "  -> 38 GB bf16 fits this machine (128 GB unified memory). "
            "diffusers-format weights will NOT load (structurally incompatible).",
            MODEL_DIR,
        )
        sys.exit(1)

    logger.info(
        "E2E T2V generate: model=%s text_encoder=%s", MODEL_DIR, TEXT_ENCODER
    )
    from fusion_mlx.video.ltx2.generate import PipelineType, generate_video

    generate_video(
        MODEL_DIR,
        TEXT_ENCODER,
        PROMPT,
        pipeline=PipelineType.DISTILLED,
        height=HEIGHT,
        width=WIDTH,
        num_frames=NUM_FRAMES,
        num_inference_steps=NUM_STEPS,
        cfg_scale=4.0,
        seed=SEED,
        fps=24,
        output_path=OUTPUT_MP4,
        verbose=True,
    )

    out = Path(OUTPUT_MP4)
    if out.exists() and out.stat().st_size > 0:
        logger.info("SUCCESS: %s (%d bytes)", OUTPUT_MP4, out.stat().st_size)
    else:
        logger.error("FAILED: no output video at %s", OUTPUT_MP4)
        sys.exit(1)


if __name__ == "__main__":
    main()
