# SPDX-License-Identifier: Apache-2.0
"""MarkItDown chat completion helpers extracted from chat.py."""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from ..openai_models import ChatCompletionRequest
from . import _common
from ._common import _get_settings, logger


async def _create_markitdown_chat_completion(
    request: ChatCompletionRequest,
) -> Any:
    from ..markitdown import (
        MARKITDOWN_MODEL_ID,
        MarkItDownRequestError,
        convert_messages_to_markdown_async,
        markitdown_model_visible,
        stream_messages_to_markdown_async,
    )

    settings = _get_settings()
    global_settings = getattr(settings, "global_settings", None) if settings else None

    if not markitdown_model_visible(global_settings):
        raise HTTPException(
            status_code=404,
            detail=f"Model not found: {MARKITDOWN_MODEL_ID}",
        )

    if request.stream:
        response_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        markdown_chunks = stream_messages_to_markdown_async(
            request.messages,
            global_settings=global_settings,
            engine_pool=_common._pool,
            settings_manager=None,
            get_sampling_params=None,
            latest_user_only=True,
        )
        return StreamingResponse(
            _stream_markitdown_response(request, markdown_chunks, response_id),
            media_type="text/event-stream",
            headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
        )

    async def _build_markitdown_completion():
        try:
            markdown = await convert_messages_to_markdown_async(
                request.messages,
                global_settings=global_settings,
                engine_pool=_common._pool,
                settings_manager=None,
                get_sampling_params=None,
                latest_user_only=True,
            )
        except MarkItDownRequestError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
        except RuntimeError as exc:
            raise HTTPException(
                status_code=500, detail="Internal server error"
            ) from exc

        if not markdown:
            raise HTTPException(
                status_code=400,
                detail="No text or supported file content found for MarkItDown.",
            )

        logger.info("MarkItDown completion converted request to markdown")
        return _build_markitdown_response(request, markdown, response_id)

    response_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    result = await _build_markitdown_completion()
    return result


def _build_markitdown_response(
    request: ChatCompletionRequest,
    markdown: str,
    response_id: str,
):
    from ..openai_models import ChatCompletionChoice, ChatCompletionResponse

    return ChatCompletionResponse(
        id=response_id,
        object="chat.completion",
        created=int(time.time()),
        model=request.model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message={"role": "assistant", "content": markdown},
                finish_reason="stop",
            )
        ],
        usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    )


async def _stream_markitdown_response(
    request: ChatCompletionRequest,
    markdown_chunks: AsyncIterator,
    response_id: str,
) -> AsyncIterator[str]:
    import json as _json

    model = request.model
    created = int(time.time())

    role_chunk = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
        ],
    }
    yield f"data: {_json.dumps(role_chunk)}\n\n"

    async for chunk in markdown_chunks:
        if not chunk:
            continue
        content_chunk = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {"index": 0, "delta": {"content": chunk}, "finish_reason": None}
            ],
        }
        yield f"data: {_json.dumps(content_chunk)}\n\n"

    done_chunk = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {_json.dumps(done_chunk)}\n\n"
    yield "data: [DONE]\n\n"
