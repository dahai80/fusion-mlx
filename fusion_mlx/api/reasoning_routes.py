# SPDX-License-Identifier: Apache-2.0
"""Reasoning API routes — /v1/reasoning endpoint.

Exposes an explicit reasoning step API for thinking models (DeepSeek-R1,
QwQ, etc). Wraps the existing chat completion + ThinkingParser to separate
reasoning_content from content, with reasoning_effort control.

User instruction: "出了P2剩余工作，P3的工作也要全部落地"
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .thinking import ThinkingParser, extract_thinking

logger = logging.getLogger(__name__)

router = APIRouter(tags=["reasoning"])

_pool: Any = None

EFFORT_TOKEN_MAP = {
    "low": 512,
    "medium": 2048,
    "high": 8192,
}


class ReasoningRequest(BaseModel):
    model: str
    prompt: str
    reasoning_effort: str = Field(default="medium", pattern="^(low|medium|high)$")
    max_reasoning_tokens: int | None = None
    temperature: float = 0.6
    max_tokens: int = 4096
    stream: bool = False


class ReasoningUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0


class ReasoningResponse(BaseModel):
    id: str
    object: str = "reasoning_result"
    model: str
    reasoning_content: str
    content: str
    usage: ReasoningUsage


def set_reasoning_context(pool: Any) -> None:
    global _pool
    _pool = pool


def _resolve_engine(model_id: str) -> Any:
    if _pool is None:
        raise HTTPException(status_code=503, detail="Server not ready")
    engine = _pool.get(model_id)
    if engine is None:
        raise HTTPException(
            status_code=404,
            detail=f"Model {model_id!r} not found",
        )
    return engine


@router.post("/v1/reasoning", response_model=ReasoningResponse)
async def create_reasoning(request: ReasoningRequest) -> ReasoningResponse:
    engine = _resolve_engine(request.model)

    budget_tokens = (
        request.max_reasoning_tokens
        if request.max_reasoning_tokens is not None
        else EFFORT_TOKEN_MAP.get(request.reasoning_effort, 2048)
    )

    from .models import (
        AssistantMessage,
        ChatCompletionChoice,
        ChatCompletionRequest,
        ChatCompletionResponse,
        UserMessage,
    )

    chat_req = ChatCompletionRequest(
        model=request.model,
        messages=[UserMessage(role="user", content=request.prompt)],
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        enable_thinking=True,
    )

    try:
        result = await engine.chat(chat_req)
    except Exception as exc:
        logger.error("reasoning engine.chat failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    raw_text = ""
    if result.choices:
        raw_text = result.choices[0].message.content or ""

    reasoning_content, content = extract_thinking(raw_text)

    reasoning_tokens = 0
    if reasoning_content:
        reasoning_tokens = len(reasoning_content.split())

    completion_tokens = 0
    if result.usage:
        completion_tokens = result.usage.completion_tokens or 0
        if reasoning_tokens > 0 and completion_tokens > reasoning_tokens:
            completion_tokens = completion_tokens - reasoning_tokens

    return ReasoningResponse(
        id=f"reason-{uuid.uuid4().hex[:24]}",
        model=request.model,
        reasoning_content=reasoning_content,
        content=content,
        usage=ReasoningUsage(
            prompt_tokens=result.usage.prompt_tokens if result.usage else 0,
            completion_tokens=completion_tokens,
            reasoning_tokens=reasoning_tokens,
            total_tokens=result.usage.total_tokens if result.usage else 0,
        ),
    )
