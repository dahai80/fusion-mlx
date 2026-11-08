# SPDX-License-Identifier: Apache-2.0
# OpenAI-shape response builders for the replay provider (#1518).
# Mirrors the markitdown intercept: chat_completions calls
# _create_replay_chat_completion BEFORE any engine dispatch, so a
# replay request never touches the executor, the pool, or the GPU.

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi.responses import StreamingResponse

from ..replay import REPLAY_MODEL_ID, replay_content


def _replay_text(request: Any) -> str:
    messages = getattr(request, "messages", None) or []
    return replay_content(messages)


def _build_replay_response(request: Any, content: str, response_id: str):
    from ..openai_models import ChatCompletionChoice, ChatCompletionResponse

    return ChatCompletionResponse(
        id=response_id,
        object="chat.completion",
        created=int(time.time()),
        model=getattr(request, "model", None) or REPLAY_MODEL_ID,
        choices=[
            ChatCompletionChoice(
                index=0,
                message={"role": "assistant", "content": content},
                finish_reason="stop",
            )
        ],
        usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    )


async def _create_replay_chat_completion(request: Any):
    content = _replay_text(request)
    response_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    if getattr(request, "stream", False):
        return StreamingResponse(
            _stream_replay_response(request, content, response_id),
            media_type="text/event-stream",
            headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
        )
    return _build_replay_response(request, content, response_id)


def _sse(obj) -> str:
    import json as _json

    return f"data: {_json.dumps(obj)}\n\n"


async def _stream_replay_response(
    request: Any, content: str, response_id: str
) -> AsyncIterator[str]:
    model = getattr(request, "model", None) or REPLAY_MODEL_ID
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
    yield _sse(role_chunk)
    content_chunk = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
    }
    yield _sse(content_chunk)
    done_chunk = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield _sse(done_chunk)
    yield "data: [DONE]\n\n"
