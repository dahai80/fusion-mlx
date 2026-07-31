# SPDX-License-Identifier: Apache-2.0
"""OCR API routes for fusion-mlx.

Provides FastAPI routes for:
- POST /v1/ocr — Optical character recognition via VLM OCR models

Supports base64 data URI, URL, and local file path image inputs.
Output formats: text, markdown, json.

Called by: fusion_mlx.server (router registration + set_ocr_context).
No existing file serves this purpose (no /v1/ocr endpoint exists).
"""

import base64
import logging
import time

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..engines.vlm import OCR_MODEL_GENERATION_DEFAULTS, VLMBatchedEngine
from ..middleware.auth import check_rate_limit, verify_api_key
from ..pool import EnginePool
from ..server_metrics import get_server_metrics

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["ocr"])

_pool: EnginePool | None = None


def set_ocr_context(pool: EnginePool) -> None:
    global _pool
    _pool = pool


class OCRRequest(BaseModel):
    model: str
    image: str = Field(
        ...,
        description="Image input: base64 data URI, URL, or local file path",
    )
    output_format: str = Field(
        default="markdown",
        description="Output format: text, markdown, or json",
    )
    temperature: float | None = None
    max_tokens: int | None = None


class OCRResult(BaseModel):
    text: str
    format: str


class OCRUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0


class OCRResponse(BaseModel):
    id: str
    object: str = "ocr_result"
    model: str
    results: list[OCRResult]
    usage: OCRUsage


async def _resolve_ocr_engine(model_id: str) -> VLMBatchedEngine:
    if _pool is None:
        raise HTTPException(status_code=503, detail="Server not initialized")
    engine = await _pool.get_engine(model_id)
    if engine is None:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")
    if not isinstance(engine, VLMBatchedEngine):
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_id}' is not a VLM/OCR model",
        )
    if not engine.is_ocr_model:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_id}' is not an OCR model (model_type={engine.model_type})",
        )
    return engine


def _resolve_image_url(image_input: str) -> str:
    if image_input.startswith("data:"):
        return image_input
    if image_input.startswith(("http://", "https://")):
        return image_input
    import os

    path = os.path.expanduser(image_input)
    if not os.path.isfile(path):
        raise HTTPException(
            status_code=400,
            detail=f"Image file not found: {image_input}",
        )
    with open(path, "rb") as f:
        data = f.read()
    ext = os.path.splitext(path)[1].lower()
    mime_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".gif": "image/gif",
        ".bmp": "image/bmp",
        ".tiff": "image/tiff",
        ".tif": "image/tiff",
        ".pdf": "application/pdf",
    }
    mime = mime_map.get(ext, "image/png")
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"


@router.post(
    "/ocr",
    dependencies=[Depends(verify_api_key), Depends(check_rate_limit)],
)
async def create_ocr(request: OCRRequest) -> OCRResponse:
    start = time.monotonic()
    request_id = f"ocr-{id(request):012x}"

    engine = await _resolve_ocr_engine(request.model)
    model_type = engine.model_type or ""
    defaults = OCR_MODEL_GENERATION_DEFAULTS.get(model_type, {})

    temperature = (
        request.temperature
        if request.temperature is not None
        else defaults.get("temperature", 0.0)
    )
    max_tokens = (
        request.max_tokens
        if request.max_tokens is not None
        else defaults.get("max_tokens", 8192)
    )
    repetition_penalty = defaults.get("repetition_penalty", 1.0)

    image_url = _resolve_image_url(request.image)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        }
    ]

    gen = await engine.chat(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
    )

    text = gen.text or ""
    latency_ms = (time.monotonic() - start) * 1000

    try:
        metrics = get_server_metrics()
        if metrics and hasattr(metrics, "record_request"):
            metrics.record_request(
                model=request.model,
                prompt_tokens=gen.prompt_tokens,
                completion_tokens=gen.completion_tokens,
                latency_ms=latency_ms,
            )
    except Exception:
        logger.debug("metrics recording failed for OCR request", exc_info=True)

    output_format = request.output_format
    if output_format not in ("text", "markdown", "json"):
        output_format = "markdown"

    result_text = text
    if output_format == "text":
        import re

        result_text = re.sub(r"^[#*_>`\-]+\s*", "", text, flags=re.MULTILINE).strip()
    elif output_format == "json":
        import json

        result_text = json.dumps({"text": text}, ensure_ascii=False)

    return OCRResponse(
        id=request_id,
        model=request.model,
        results=[OCRResult(text=result_text, format=output_format)],
        usage=OCRUsage(
            prompt_tokens=gen.prompt_tokens,
            completion_tokens=gen.completion_tokens,
            total_tokens=gen.prompt_tokens + gen.completion_tokens,
            latency_ms=round(latency_ms, 1),
        ),
    )
