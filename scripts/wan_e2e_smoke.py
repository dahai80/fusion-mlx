#!/usr/bin/env python3
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_DIR = Path.home() / ".cache/mlx-video-models/wan22-ti2v-5b"
OUTPUT_MP4 = "/tmp/fusion_wan_e2e.mp4"
TEST_IMAGE = "/tmp/fusion_wan_test_image.jpg"
LOCAL_TOK = Path.home() / ".cache/modelscope/hub/models/Wan-AI/Wan2___2-TI2V-5B/google/umt5-xxl"


def make_test_image():
    import numpy as np
    from PIL import Image
    arr = np.zeros((480, 832, 3), dtype=np.uint8)
    for x in range(832):
        arr[:, x, 0] = int(x / 832 * 255)
    for y in range(480):
        arr[y, :, 1] = int(y / 480 * 255)
    Image.fromarray(arr).save(TEST_IMAGE, quality=95)
    logger.info("test image: %s", TEST_IMAGE)


def patch_tokenizer():
    from transformers import AutoTokenizer
    _orig = AutoTokenizer.from_pretrained
    _local = str(LOCAL_TOK)
    def _patched(name, *a, **kw):
        if name == "google/umt5-xxl":
            logger.info("tokenizer redirect -> %s", _local)
            return _orig(_local, *a, **kw)
        return _orig(name, *a, **kw)
    AutoTokenizer.from_pretrained = _patched
    logger.info("tokenizer patch installed")


def main():
    if not MODEL_DIR.exists():
        logger.error("model dir not found: %s (run wan_convert.py first)", MODEL_DIR)
        sys.exit(1)
    make_test_image()
    patch_tokenizer()
    from mlx_video.models.wan_2.generate import generate_video
    logger.info("E2E I2V generate: model=%s", MODEL_DIR)
    generate_video(
        str(MODEL_DIR),
        prompt="A cat walking on a sunny beach, cinematic, detailed",
        image=TEST_IMAGE,
        width=832,
        height=480,
        num_frames=41,
        steps=15,
        seed=42,
        output_path=OUTPUT_MP4,
        scheduler="unipc",
        no_compile=True,
    )
    out = Path(OUTPUT_MP4)
    if out.exists() and out.stat().st_size > 0:
        logger.info("SUCCESS: %s (%d bytes)", OUTPUT_MP4, out.stat().st_size)
    else:
        logger.error("FAILED: no output video")
        sys.exit(1)


if __name__ == "__main__":
    main()
