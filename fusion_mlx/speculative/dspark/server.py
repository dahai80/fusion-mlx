# SPDX-License-Identifier: Apache-2.0
# DSpark dedicated FastAPI server.
#
# DSpark (DeepSeek DeepSpec) is self-contained: DSparkGenerator loads its
# own target + converted MLX draft and runs the lossless propose→verify
# loop. So this server is SIMPLER than the DFlash server (no mlx-vlm hook
# tapping an existing engine) — it wraps a single DSparkGenerator and
# serializes requests through a 1-worker thread pool (single-user serial,
# matching DFlash's concurrency contract: one Metal stream, no re-entrant
# KV cache). Multi-turn chat is rendered via the target tokenizer's chat
# template, then generate_from_tokens / stream_from_tokens runs the
# spec-decode loop on the pre-built token array (bypassing build_prompt,
# which is single-turn only — see dspark_metal/adapters.py).

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import mlx.core as mx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from .runtime import DSparkRuntime, load_runtime

logger = logging.getLogger(__name__)

# Single-user serial: DSpark's per-round target forward is not re-entrant
# (shared KV cache, single Metal stream). A 1-worker pool means at most
# one generation in flight; the rest queue. Mirrors DFlash's contract.
_dspark_executor: ThreadPoolExecutor | None = None


def _load_runtime_blocking(
    target_repo: str, draft_path: str, draft_quant_bits: int
) -> DSparkRuntime:
    return load_runtime(target_repo, draft_path, draft_quant_bits)


class _ChatMessage(BaseModel):
    role: str
    content: str | list[Any] | None = None


class _ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[_ChatMessage]
    max_tokens: int | None = None
    max_new_tokens: int | None = None
    temperature: float = 0.0
    top_p: float | None = None
    stream: bool = False
    n: int | None = None
    tools: Any | None = None
    tool_choice: Any | None = None
    logprobs: Any | None = None
    response_format: Any | None = None
    stop: Any | None = None
    seed: int | None = None
    confidence_threshold: float = 0.0
    enable_thinking: bool | None = None


def _coerce_content(content: str | list[Any] | None) -> str:
    # DSpark targets are text-only Qwen3 — collapse multimodal content
    # down to the concatenated text parts. Non-text parts are dropped
    # (the drafter has no vision head); an empty result raises so the
    # caller returns a 400 rather than silently generating from "".
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for part in content:
        if isinstance(part, str):
            parts.append(part)
        elif isinstance(part, dict) and part.get("type") == "text":
            parts.append(str(part.get("text", "")))
    return "".join(parts)


def _render_prompt(
    runtime: DSparkRuntime,
    messages: list[dict[str, str]],
    enable_thinking: bool,
) -> tuple[mx.array, int]:
    tokenizer = runtime.generator.target.tokenizer
    try:
        prompt_str = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        # Older tokenizers / templates without the enable_thinking kwarg.
        prompt_str = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    ids = tokenizer.encode(prompt_str, add_special_tokens=False)
    return mx.array(ids, dtype=mx.uint32), len(ids)


def _sse_chunk(
    cid: str,
    created: int,
    model: str,
    *,
    role: bool = False,
    delta: str | None = None,
    finish_reason: str | None = None,
    usage: dict[str, int] | None = None,
) -> str:
    choice: dict[str, Any] = {"index": 0, "delta": {}, "finish_reason": finish_reason}
    if role:
        choice["delta"] = {"role": "assistant"}
    elif delta is not None:
        choice["delta"] = {"content": delta}
    chunk: dict[str, Any] = {
        "id": cid,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [choice],
    }
    if usage is not None:
        chunk["usage"] = usage
    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"


def _build_app(
    *,
    runtime: DSparkRuntime,
    served_model_name: str,
    default_max_tokens: int,
    cors_origins: list[str] | None,
    enable_thinking_default: bool,
) -> FastAPI:
    app = FastAPI(title="fusion-mlx DSpark server")
    origins = cors_origins if cors_origins else ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/healthz")
    async def healthz() -> JSONResponse:
        return JSONResponse(
            {
                "status": "ok",
                "engine": "dspark",
                "mode": "single-user-serial",
                "target": runtime.target_repo,
                "draft_path": runtime.draft_path,
                "draft_quant_bits": runtime.draft_quant_bits,
                "draft_quantization": getattr(
                    runtime.generator, "draft_quantization", None
                ),
                "accept_lens": runtime.accept_lens_snapshot()[-16:],
            }
        )

    @app.get("/v1/models")
    async def list_models() -> JSONResponse:
        return JSONResponse(
            {
                "object": "list",
                "data": [
                    {
                        "id": served_model_name,
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "dspark",
                    }
                ],
            }
        )

    @app.post("/v1/chat/completions")
    async def chat_completions(req: _ChatCompletionRequest) -> Any:
        if not req.messages:
            raise HTTPException(status_code=400, detail="messages must be non-empty")
        if req.n is not None and req.n > 1:
            raise HTTPException(
                status_code=400,
                detail="DSpark is single-stream; n>1 is not supported",
            )
        if req.tools:
            raise HTTPException(
                status_code=400, detail="DSpark does not support tool calls"
            )
        if req.logprobs:
            raise HTTPException(
                status_code=400, detail="DSpark does not support logprobs"
            )
        if req.response_format:
            raise HTTPException(
                status_code=400,
                detail="DSpark does not support response_format (raw text only)",
            )

        messages: list[dict[str, str]] = []
        for m in req.messages:
            text = _coerce_content(m.content)
            if not text and m.role != "assistant":
                raise HTTPException(
                    status_code=400,
                    detail=f"message role={m.role!r} has empty/non-text content",
                )
            messages.append({"role": m.role, "content": text})

        enable_thinking = (
            req.enable_thinking
            if req.enable_thinking is not None
            else enable_thinking_default
        )
        cap = max(1, int(default_max_tokens))
        max_new_tokens = (
            req.max_new_tokens
            if req.max_new_tokens is not None
            else (req.max_tokens if req.max_tokens is not None else cap)
        )
        max_new_tokens = max(1, min(int(max_new_tokens), cap))
        temperature = float(req.temperature) if req.temperature is not None else 0.0
        conf = float(req.confidence_threshold) if req.confidence_threshold else 0.0

        prompt_tokens, prompt_len = _render_prompt(runtime, messages, enable_thinking)
        gen = runtime.generator

        if req.stream:
            return StreamingResponse(
                _stream_completion(
                    runtime=runtime,
                    prompt_tokens=prompt_tokens,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    confidence_threshold=conf,
                    served_model_name=served_model_name,
                    prompt_len=prompt_len,
                ),
                media_type="text/event-stream",
            )

        def _run() -> Any:
            return gen.generate_from_tokens(
                prompt_tokens=prompt_tokens,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                confidence_threshold=conf,
                skip_special_tokens=True,
            )

        assert _dspark_executor is not None
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(_dspark_executor, _run)

        metrics = result.metrics or {}
        avg_accept = metrics.get("avg_acceptance_length")
        runtime.record_accept(avg_accept)
        completion_tokens = len(result.generated_tokens)
        total = prompt_len + completion_tokens
        return JSONResponse(
            {
                "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": served_model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": result.text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_len,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total,
                },
                "metrics": {
                    "generation_tps": metrics.get("generation_tps"),
                    "avg_acceptance_length": avg_accept,
                },
            }
        )

    async def _stream_completion(
        *,
        runtime: DSparkRuntime,
        prompt_tokens: mx.array,
        max_new_tokens: int,
        temperature: float,
        confidence_threshold: float,
        served_model_name: str,
        prompt_len: int,
    ):
        gen = runtime.generator
        queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def _producer() -> None:
            try:
                for event in gen.stream_from_tokens(
                    prompt_tokens=prompt_tokens,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    confidence_threshold=confidence_threshold,
                    skip_special_tokens=True,
                ):
                    fut = asyncio.run_coroutine_threadsafe(
                        queue.put(("delta", event)), loop
                    )
                    fut.result()
            except Exception as exc:  # noqa: BLE001 — surface to client
                asyncio.run_coroutine_threadsafe(
                    queue.put(("error", exc)), loop
                ).result()
                return
            asyncio.run_coroutine_threadsafe(queue.put(("done", None)), loop).result()

        assert _dspark_executor is not None
        _dspark_executor.submit(_producer)

        cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())
        yield _sse_chunk(cid, created, served_model_name, role=True)

        while True:
            kind, payload = await queue.get()
            if kind == "delta":
                event = payload
                if event.delta:
                    yield _sse_chunk(
                        cid, created, served_model_name, delta=event.delta
                    )
                if event.finished:
                    metrics = event.metrics or {}
                    avg_accept = metrics.get("avg_acceptance_length")
                    runtime.record_accept(avg_accept)
                    completion_tokens = (
                        len(event.generated_tokens) if event.generated_tokens else 0
                    )
                    usage = {
                        "prompt_tokens": prompt_len,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_len + completion_tokens,
                    }
                    yield _sse_chunk(
                        cid,
                        created,
                        served_model_name,
                        finish_reason="stop",
                        usage=usage,
                    )
            elif kind == "error":
                yield _sse_chunk(
                    cid,
                    created,
                    served_model_name,
                    finish_reason="error",
                )
                logger.exception("DSpark stream failed", exc_info=payload)
                return
            elif kind == "done":
                yield "data: [DONE]\n\n"
                return

    return app


def run_dspark_server(
    *,
    target_model_repo: str,
    drafter_path: str,
    draft_quant_bits: int = 8,
    host: str,
    port: int,
    served_model_name: str | None = None,
    default_max_tokens: int = 32768,
    cors_origins: list[str] | None = None,
    uvicorn_log_level: str = "info",
    enable_thinking: bool = False,
) -> None:
    # Boot guards fail loud and early (before the multi-minute model load).
    from .eligibility import have_runtime

    if not have_runtime():
        print(
            "\n  Error: --enable-dspark requires dspark-metal (DeepSeek DeepSpec "
            "MLX port). Install with `uv add dspark-metal` or "
            "`pip install -e /path/to/dspark-metal`.\n"
        )
        sys.exit(1)
    import os

    if not drafter_path:
        print(
            "\n  Error: --enable-dspark requires --dspark-drafter-path "
            "<path-to-converted-mlx-draft>. Convert one with:\n"
            "    dspark-metal-convert deepseek-ai/dspark_qwen3_8b_block7 "
            "--target mlx-community/Qwen3-8B-bf16\n"
            "  → produces models/dspark_qwen3_8b_block7-mlx/\n"
        )
        sys.exit(1)
    if not os.path.exists(drafter_path):
        print(
            f"\n  Error: --dspark-drafter-path {drafter_path!r} does not exist. "
            "Convert one with:\n"
            "    dspark-metal-convert deepseek-ai/dspark_qwen3_8b_block7 "
            "--target mlx-community/Qwen3-8B-bf16\n"
        )
        sys.exit(1)

    global _dspark_executor
    _dspark_executor = ThreadPoolExecutor(max_workers=1)
    # Load on the single worker so the generator's Metal stream lives on
    # the same thread that will serve requests (avoids cross-thread Metal
    # context handoff on the first generate).
    logger.info(
        "DSpark boot: loading target=%s draft=%s q%d",
        target_model_repo,
        drafter_path,
        draft_quant_bits,
    )
    runtime = _dspark_executor.submit(
        _load_runtime_blocking, target_model_repo, drafter_path, draft_quant_bits
    ).result()

    served = served_model_name or target_model_repo
    app = _build_app(
        runtime=runtime,
        served_model_name=served,
        default_max_tokens=default_max_tokens,
        cors_origins=cors_origins,
        enable_thinking_default=enable_thinking,
    )

    import uvicorn

    logger.info(
        "DSpark server listening on %s:%d (served=%s target=%s draft=%s)",
        host,
        port,
        served,
        target_model_repo,
        drafter_path,
    )
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=uvicorn_log_level.lower(),
        timeout_keep_alive=30,
    )
