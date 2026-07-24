# SPDX-License-Identifier: Apache-2.0
"""Image generation API routes for fusion-mlx.

Provides FastAPI routes for:
- POST /v1/images/generate  - Text-to-image / variant image generation
"""

import base64
import logging

from fastapi import APIRouter, Depends, HTTPException

from ..middleware.auth import check_rate_limit, verify_api_key
from pydantic import BaseModel, Field

from ..engines import ImageGenEngine
from ..engines.image_gen import VARIANT_MAP
from ..pool import EnginePool

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/images", tags=["images"])

_pool: EnginePool | None = None


def set_images_context(pool: EnginePool) -> None:
    """Inject engine pool into this module."""
    global _pool
    _pool = pool


class ImageGenerateRequest(BaseModel):
    """Request for image generation."""

    prompt: str
    # Number of images to generate (default 1, max 4)
    n: int = Field(default=1, ge=1, le=4)
    # Image dimensions (default 1024x1024)
    width: int = Field(default=1024, ge=256, le=2048)
    height: int = Field(default=1024, ge=256, le=2048)
    # Diffusion steps (fewer = faster, more = higher quality)
    steps: int = Field(default=4, ge=1, le=50)
    # Random seed (None = random)
    seed: int | None = None
    # Guidance scale (None = variant default; txt2img=1.0, flux1 variants=4.0)
    guidance: float | None = Field(default=None, ge=1.0, le=20.0)
    # Response format
    response_format: str = Field(default="url", pattern="^(url|b64_json)$")
    # Model name (default: first available image gen model)
    model: str | None = None
    # Pipeline variant: txt2img|controlnet_canny|controlnet_upscaler|depth|fill|kontext|redux
    variant: str | None = Field(default=None)
    # Optional diffusion knobs
    scheduler: str | None = None
    negative_prompt: str | None = None
    # ControlNet: input image path for canny/upscaler
    control_image: str | None = None
    controlnet_strength: float | None = Field(default=None, ge=0.0, le=2.0)
    # Redux: reference image paths + strengths
    reference_images: list[str] | None = None
    reference_strengths: list[float] | None = None
    # Fill / Kontext: edit image + mask
    edit_image: str | None = None
    mask_image: str | None = None
    # Depth: depth map image
    depth_image: str | None = None
    # Img2img strength (used by depth/kontext/redux/txt2img i2i)
    image_strength: float | None = Field(default=None, ge=0.0, le=1.0)


class ImageOutput(BaseModel):
    """Single generated image output."""

    url: str | None = None
    b64_json: str | None = None


class ImageGenerateResponse(BaseModel):
    """Response from image generation."""

    data: list[ImageOutput]
    created: int = Field(default_factory=lambda: int(__import__("time").time()))


@router.post("/generate", dependencies=[Depends(verify_api_key), Depends(check_rate_limit)])
async def generate_image(request: ImageGenerateRequest) -> ImageGenerateResponse:
    """Generate images from a text prompt using Flux variants."""
    try:
        if _pool is None:
            raise HTTPException(450, "Engine pool not initialized")

        # Validate variant if provided
        variant = request.variant
        if variant is not None and variant not in VARIANT_MAP:
            raise HTTPException(
                422,
                f"Unknown variant '{variant}'. Available: {list(VARIANT_MAP.keys())}",
            )

        # Find an image gen engine
        model_name = request.model
        if not model_name:
            model_name = "flux-2"

        engine = await _pool.get_engine(model_name)
        if engine is None or not isinstance(engine, ImageGenEngine):
            raise HTTPException(
                404,
                f"Image generation model '{model_name}' not loaded. "
                "Load a Flux model first.",
            )

        # If engine was started with a different variant, warn
        if variant is not None and engine.variant != variant:
            logger.warning(
                "Request variant=%s but engine variant=%s; "
                "engine variant is fixed at start time",
                variant,
                engine.variant,
            )

        # Build generate kwargs
        gen_kwargs: dict = dict(
            prompt=request.prompt,
            width=request.width,
            height=request.height,
            steps=request.steps,
            seed=request.seed,
            guidance=request.guidance,
            n_images=request.n,
        )
        if request.scheduler is not None:
            gen_kwargs["scheduler"] = request.scheduler
        if request.negative_prompt is not None:
            gen_kwargs["negative_prompt"] = request.negative_prompt
        # Variant-specific image inputs
        if request.control_image is not None:
            gen_kwargs["control_image"] = request.control_image
        if request.controlnet_strength is not None:
            gen_kwargs["controlnet_strength"] = request.controlnet_strength
        if request.reference_images is not None:
            gen_kwargs["reference_images"] = request.reference_images
        if request.reference_strengths is not None:
            gen_kwargs["reference_strengths"] = request.reference_strengths
        if request.edit_image is not None:
            gen_kwargs["edit_image"] = request.edit_image
        if request.mask_image is not None:
            gen_kwargs["mask_image"] = request.mask_image
        if request.depth_image is not None:
            gen_kwargs["depth_image"] = request.depth_image
        if request.image_strength is not None:
            gen_kwargs["image_strength"] = request.image_strength

        image_bytes_list = await engine.generate(**gen_kwargs)

        # Format response
        outputs = []
        for img_bytes in image_bytes_list:
            if request.response_format == "b64_json":
                outputs.append(
                    ImageOutput(b64_json=base64.b64encode(img_bytes).decode())
                )
            else:
                b64 = base64.b64encode(img_bytes).decode()
                outputs.append(ImageOutput(url=f"data:image/png;base64,{b64}"))

        return ImageGenerateResponse(data=outputs)

    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    except Exception as exc:
        logger.exception("Image generation failed")
        raise HTTPException(500, "Internal server error")
