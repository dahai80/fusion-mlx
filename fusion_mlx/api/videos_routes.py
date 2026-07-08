# SPDX-License-Identifier: Apache-2.0
# Video generation API routes for fusion-mlx (LTX-2).
# POST /v1/videos/generate - Text-to-video generation.
import base64
import logging
import time

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, model_validator

from ..engines import VideoGenEngine
from ..exceptions import ModelNotFoundError
from ..pool import EnginePool

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/videos", tags=["videos"])

_pool: EnginePool | None = None


def set_videos_context(pool: EnginePool) -> None:
    global _pool
    _pool = pool


class VideoGenerateRequest(BaseModel):
    prompt: str
    # Number of videos to generate (default 1, max 4)
    n: int = Field(default=1, ge=1, le=4)
    # Number of frames per video (LTX-2 requires 1 + 8*k)
    num_frames: int = Field(default=97, ge=1, le=1024)
    # Frame dimensions (LTX-2 distilled requires divisibility by 64)
    width: int = Field(default=768, ge=256, le=2048)
    height: int = Field(default=512, ge=256, le=2048)
    # Frames per second
    fps: int = Field(default=24, ge=1, le=60)
    # Random seed (None = random)
    seed: int | None = None
    # Response format
    response_format: str = Field(default="url", pattern="^(url|b64_json)$")
    # Model name (default: first available video gen model)
    model: str | None = None

    @model_validator(mode="after")
    def _validate_ltx_constraints(self):
        # mlx-video LTX-2 enforces num_frames = 1 + 8*k (it silently adjusts
        # otherwise) and height/width divisible by 64 for the distilled
        # pipeline (it asserts otherwise). Reject early with 422 so the caller
        # learns the real constraint instead of getting an adjusted/asserted
        # result or a 500.
        if self.num_frames % 8 != 1:
            raise ValueError("num_frames must be 1 + 8*k (e.g. 9, 17, 33, 97)")
        if self.width % 64 != 0 or self.height % 64 != 0:
            raise ValueError("width and height must be divisible by 64")
        return self


class VideoOutput(BaseModel):
    url: str | None = None
    b64_json: str | None = None


class VideoGenerateResponse(BaseModel):
    data: list[VideoOutput]
    created: int = Field(default_factory=lambda: int(time.time()))


def _encode_video_output(vid_bytes: bytes, response_format: str) -> VideoOutput:
    b64 = base64.b64encode(vid_bytes).decode()
    if response_format == "b64_json":
        return VideoOutput(b64_json=b64)
    return VideoOutput(url=f"data:video/mp4;base64,{b64}")


@router.post("/generate")
async def generate_video(request: VideoGenerateRequest) -> VideoGenerateResponse:
    # Generate videos from a text prompt using LTX-2.
    try:
        if _pool is None:
            raise HTTPException(503, "Engine pool not initialized")

        model_name = request.model
        if not model_name:
            model_name = "ltx-2"

        try:
            engine = await _pool.get_engine(model_name)
        except ModelNotFoundError:
            engine = None
        if engine is None or not isinstance(engine, VideoGenEngine):
            raise HTTPException(
                404,
                f"Video generation model '{model_name}' not loaded. "
                "Load an LTX-2 model first.",
            )

        video_bytes_list = await engine.generate(
            prompt=request.prompt,
            num_frames=request.num_frames,
            width=request.width,
            height=request.height,
            fps=request.fps,
            seed=request.seed,
            n=request.n,
        )

        outputs = [
            _encode_video_output(vb, request.response_format) for vb in video_bytes_list
        ]
        return VideoGenerateResponse(data=outputs)

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Video generation failed")
        raise HTTPException(500, str(exc))
