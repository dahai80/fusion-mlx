import base64
import io
import logging
import os
import time

import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from fusion_mlx.image.sr import super_resolve
from fusion_mlx.middleware.auth import check_rate_limit, verify_api_key

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/images", tags=["images"])

_pool = None


def set_images_sr_context(pool) -> None:
    global _pool
    _pool = pool


class SRResponse(BaseModel):
    image_b64: str
    width: int
    height: int
    in_width: int
    in_height: int
    elapsed: float = Field(default=0.0)


def _to_numpy_hwc(png_bytes: bytes):
    from PIL import Image
    pil = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    arr = np.array(pil).astype(np.float32) / 255.0
    return arr[None, ...]


@router.post(
    "/super-resolution",
    dependencies=[Depends(verify_api_key), Depends(check_rate_limit)],
)
async def super_resolution(
    image: UploadFile = File(...),
    scale: int = Form(4),
    tile_size: int = Form(512),
) -> SRResponse:
    t0 = time.time()
    if scale not in (2, 4):
        raise HTTPException(422, "scale must be 2 or 4")
    model_path = os.path.expanduser(
        "~/.fusion-mlx/models/realesrgan/RealESRGAN_x4plus.safetensors"
    )
    if not os.path.exists(model_path):
        raise HTTPException(
            503,
            "RealESRGAN model not downloaded. Convert RealESRGAN_x4plus.pth "
            "to safetensors at " + model_path + " (see PR-A plan Task 3).",
        )
    raw = await image.read()
    if not raw:
        raise HTTPException(422, "empty image upload")
    try:
        inp = _to_numpy_hwc(raw)
    except Exception as exc:
        logger.warning("sr decode failed: %s", exc)
        raise HTTPException(422, "could not decode image") from exc
    try:
        sr = super_resolve(
            inp, model_path=model_path, scale=scale,
            tile_size=tile_size, tile_overlap=64,
        )
    except Exception as exc:
        logger.exception("sr inference failed")
        raise HTTPException(500, "super-resolution failed") from exc
    from PIL import Image
    out_hwc = sr[0]
    pil = Image.fromarray((np.clip(out_hwc, 0, 1) * 255).astype(np.uint8))
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    elapsed = time.time() - t0
    logger.info(
        "sr endpoint: in=%dx%d out=%dx%d scale=%d elapsed=%.2fs",
        inp.shape[2], inp.shape[1], out_hwc.shape[1],
        out_hwc.shape[0], scale, elapsed,
    )
    return SRResponse(
        image_b64=base64.b64encode(buf.getvalue()).decode(),
        width=out_hwc.shape[1], height=out_hwc.shape[0],
        in_width=inp.shape[2], in_height=inp.shape[1], elapsed=elapsed,
    )
