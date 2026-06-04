# SPDX-License-Identifier: Apache-2.0
"""Image generation API routes for fusion-mlx (Flux 2).

Provides FastAPI routes for:
- POST /v1/images/generate  - Text-to-image generation
"""

import base64
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..engines import ImageGenEngine
from ..pool import EnginePool

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/images", tags=["images"])

_pool: Optional[EnginePool] = None


from typing import Optional


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
    seed: Optional[int] = None
    # Guidance scale (higher = closer to prompt)
    guidance: float = Field(default=4.0, ge=1.0, le=20.0)
    # Response format
    response_format: str = Field(default="url", pattern="^(url|b64_json)$")
    # Model name (default: first available image gen model)
    model: Optional[str] = None


class ImageOutput(BaseModel):
    """Single generated image output."""

    url: Optional[str] = None
    b64_json: Optional[str] = None


class ImageGenerateResponse(BaseModel):
    """Response from image generation."""

    data: List[ImageOutput]
    created: int = Field(default_factory=lambda: int(__import__("time").time()))


@router.post("/generate")
async def generate_image(request: ImageGenerateRequest) -> ImageGenerateResponse:
    """Generate images from a text prompt using Flux 2."""
    try:
        if _pool is None:
            raise HTTPException(450, "Engine pool not initialized")

        # Find an image gen engine
        model_name = request.model
        if not model_name:
            # Use a default image gen model name
            model_name = "flux-2"

        engine = await _pool.get_engine(model_name)
        if engine is None or not isinstance(engine, ImageGenEngine):
            raise HTTPException(
                404,
                f"Image generation model '{model_name}' not loaded. "
                "Load a Flux model first.",
            )

        # Generate images
        image_bytes_list = await engine.generate(
            prompt=request.prompt,
            width=request.width,
            height=request.height,
            steps=request.steps,
            seed=request.seed,
            guidance=request.guidance,
            n_images=request.n,
        )

        # Format response
        outputs = []
        for img_bytes in image_bytes_list:
            if request.response_format == "b64_json":
                outputs.append(
                    ImageOutput(b64_json=base64.b64encode(img_bytes).decode())
                )
            else:
                # Return as base64 data URL
                b64 = base64.b64encode(img_bytes).decode()
                outputs.append(ImageOutput(url=f"data:image/png;base64,{b64}"))

        return ImageGenerateResponse(data=outputs)

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Image generation failed")
        raise HTTPException(500, str(exc))
