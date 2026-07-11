import logging
import os
import sys
from pathlib import Path

from PIL import Image, ImageDraw

from fusion_mlx.video.ltx2.generate import PipelineType, generate_video

logger = logging.getLogger("ltx2_e2e_i2v_apg")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)

MODEL_DIR = Path(
    os.environ.get(
        "LTX2_MODEL_DIR",
        os.path.expanduser("~/.fusion-mlx/models/mlx-community/LTX-2-dev-bf16"),
    )
)
TEXT_ENCODER = MODEL_DIR / "text_encoder"
PROMPT = os.environ.get(
    "LTX2_PROMPT", "A neon city skyline at night, cinematic, high detail"
)
MODE = os.environ.get("LTX2_MODE", "i2v").lower()
HEIGHT = int(os.environ.get("LTX2_HEIGHT", "256"))
WIDTH = int(os.environ.get("LTX2_WIDTH", "256"))
NUM_FRAMES = int(os.environ.get("LTX2_NUM_FRAMES", "9"))
NUM_STEPS = int(os.environ.get("LTX2_NUM_STEPS", "10"))
SEED = int(os.environ.get("LTX2_SEED", "42"))
TEST_IMAGE = Path(os.environ.get("LTX2_TEST_IMAGE", "/tmp/ltx2_e2e_test_image.png"))


def make_test_image(path, width, height):
    img = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(img)
    for x in range(width):
        r = int(255 * x / max(width - 1, 1))
        draw.line([(x, 0), (x, height)], fill=(r, 64, 255 - r))
    draw.rectangle(
        [width // 4, height // 4, 3 * width // 4, 3 * height // 4],
        fill=(255, 220, 0),
    )
    img.save(path)
    logger.info("test image saved: %s (%dx%d)", path, width, height)
    return str(path)


def _verify(output):
    if not output.exists():
        logger.error("FAIL: output missing %s", output)
        sys.exit(1)
    size = output.stat().st_size
    logger.info("output: %s (%d bytes)", output, size)
    if size < 1000:
        logger.error("FAIL: output too small %d bytes", size)
        sys.exit(1)
    logger.info("PASS: %s", output)


def run_i2v():
    image_path = make_test_image(TEST_IMAGE, WIDTH, HEIGHT)
    output = Path(os.environ.get("LTX2_OUTPUT", "/tmp/ltx2_e2e_i2v.mp4"))
    logger.info("MODE=i2v -> %s", output)
    generate_video(
        str(MODEL_DIR),
        str(TEXT_ENCODER),
        PROMPT,
        pipeline=PipelineType.DEV,
        height=HEIGHT,
        width=WIDTH,
        num_frames=NUM_FRAMES,
        num_inference_steps=NUM_STEPS,
        cfg_scale=4.0,
        seed=SEED,
        fps=24,
        output_path=str(output),
        verbose=True,
        audio=False,
        stg_scale=0.0,
        modality_scale=1.0,
        image=image_path,
        image_strength=1.0,
        image_frame_idx=0,
    )
    _verify(output)


def run_apg():
    output = Path(os.environ.get("LTX2_OUTPUT", "/tmp/ltx2_e2e_apg.mp4"))
    logger.info("MODE=apg -> %s", output)
    generate_video(
        str(MODEL_DIR),
        str(TEXT_ENCODER),
        PROMPT,
        pipeline=PipelineType.DEV,
        height=HEIGHT,
        width=WIDTH,
        num_frames=NUM_FRAMES,
        num_inference_steps=NUM_STEPS,
        cfg_scale=4.0,
        seed=SEED,
        fps=24,
        output_path=str(output),
        verbose=True,
        audio=False,
        stg_scale=0.0,
        modality_scale=1.0,
        use_apg=True,
        apg_eta=1.0,
        apg_norm_threshold=0.0,
    )
    _verify(output)


def main():
    cfg = MODEL_DIR / "transformer" / "config.json"
    if not cfg.exists():
        logger.error("missing %s", cfg)
        sys.exit(1)
    logger.info("model_dir=%s text_encoder=%s mode=%s", MODEL_DIR, TEXT_ENCODER, MODE)
    if MODE == "i2v":
        run_i2v()
    elif MODE == "apg":
        run_apg()
    else:
        logger.error("unknown LTX2_MODE=%s (use i2v or apg)", MODE)
        sys.exit(1)


if __name__ == "__main__":
    main()
