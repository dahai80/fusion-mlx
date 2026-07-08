# SPDX-License-Identifier: Apache-2.0
# Video generation API routes for fusion-mlx (LTX-2).
# POST /v1/videos/generate - Text-to-video generation.
import base64
import logging
import time

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..engines import VideoGenEngine
from ..pool import EnginePool

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/videos", tags=["videos"])

_pool: EnginePool | None = None


def set_videos_context(pool: EnginePool) -> None:
    # Inject engine pool into this module.
    global _pool
    _pool = pool


class VideoGenerateRequest(BaseModel):
    prompt: str
    # Number of videos to generate (default 1, max 4)
    n: int = Field(default=1, ge=1, le=4)
    # Number of frames per video
    num_frames: int = Field(default=97, ge=1, le=1024)
    # Frame dimensions
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


class VideoOutput(BaseModel):
    url: str | None = None
    b64_json: str | None = None


class VideoGenerateResponse(BaseModel):
    data: list[VideoOutput]
    created: int = Field(default_factory=lambda: int(time.time()))


@router.post("/generate")
async def generate_video(request: VideoGenerateRequest) -> VideoGenerateResponse:
    # Generate videos from a text prompt using LTX-2.
    try:
        if _pool is None:
            raise HTTPException(450, "Engine pool not initialized")

        model_name = request.model
        if not model_name:
            model_name = "ltx-2"

        engine = await _pool.get_engine(model_name)
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

        outputs = []
        for vid_bytes in video_bytes_list:
            if request.response_format == "b64_json":
                outputs.append(
                    VideoOutput(b64_json=base64.b64encode(vid_bytes).decode())
                )
            else:
                b64 = base64.b64encode(vid_bytes).decode()
                outputs.append(VideoOutput(url=f"data:video/mp4;base64,{b64}"))

        return VideoGenerateResponse(data=outputs)

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Video generation failed")
        raise HTTPException(500, str(exc))
