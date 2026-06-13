# SPDX-License-Identifier: Apache-2.0
"""
Cloud routing for large-context requests.

Routes requests with many new (uncached) tokens to a cloud LLM via litellm,
keeping short requests on local MLX inference. This avoids the O(n^2) attention
prefill bottleneck on Apple Silicon for cold, large-context prompts.
"""

import asyncio
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

logger = logging.getLogger(__name__)


class CloudRouter:
    """
    Routes requests to a cloud LLM when new token count exceeds a threshold.

    Uses litellm for provider-agnostic cloud calls (Anthropic, OpenAI, etc.).
    litellm is imported lazily — only when a request actually routes to cloud.
    """

    def __init__(
        self,
        cloud_model: str,
        threshold: int = 20000,
        api_base: str | None = None,
        api_key: str | None = None,
    ):
        """
        Args:
            cloud_model: litellm model string (e.g. "anthropic/claude-sonnet-4-5-20250929")
            threshold: Route to cloud when new_tokens exceeds this value
            api_base: Custom API base URL (for OpenAI-compatible providers like Zhipu)
            api_key: API key override (otherwise uses env vars)
        """
        self.cloud_model = cloud_model
        self.threshold = threshold
        self.api_base = api_base
        self.api_key = api_key
        self._litellm = None

         # Circuit breaker state - prevents local/cloud oscillation
        self._circuit_open = False
        self._circuit_failure_count = 0
        self._circuit_failure_threshold = 5
        self._circuit_reset_on_success = True
        self._circuit_open_at: float | None = None
        self._half_open_timeout: float = 30.0

    def _get_litellm(self):
        """Lazy import of litellm."""
        if self._litellm is None:
            import litellm

            self._litellm = litellm
        return self._litellm


    def report_local_failure(self) -> None:
        self._circuit_failure_count += 1
        if self._circuit_failure_count >= self._circuit_failure_threshold:
            self._circuit_open = True
            self._circuit_open_at = time.time()
            logger.warning("[CLOUD] Circuit breaker OPENED after %d consecutive local failures", self._circuit_failure_count)

    def report_local_success(self) -> None:
        if self._circuit_open and self._circuit_reset_on_success:
            self._circuit_open = False
            self._circuit_failure_count = 0
            logger.info("[CLOUD] Circuit breaker CLOSED - local inference recovered")

    def is_circuit_open(self) -> bool:
        return self._circuit_open

    def should_route_to_cloud(self, new_tokens: int) -> bool:
        """Return True if new_tokens exceeds threshold OR circuit breaker is open.

        Half-open: after _half_open_timeout seconds in OPEN, allow one probe
        request through to local. The caller (route_chat) must report success/failure
        back via report_local_success() / report_local_failure().
        """
        if self._circuit_open and self._circuit_open_at is not None:
            elapsed = time.time() - self._circuit_open_at
            if elapsed > self._half_open_timeout:
                self._circuit_open = False  # half-open: allow one probe
                logger.info("[CLOUD] Circuit breaker HALF-OPEN after %.0fs — allowing probe", elapsed)
        return self._circuit_open or new_tokens > self.threshold

    async def completion(
        self,
        messages: list[dict[str, Any]],
        **kwargs,
    ) -> dict:
        """
        Non-streaming cloud completion. Returns OpenAI-format response dict.

        Args:
            messages: OpenAI chat format messages
            **kwargs: temperature, max_tokens, top_p, tools, etc.
        """
        litellm = self._get_litellm()

        call_kwargs = self._build_call_kwargs(messages, stream=False, **kwargs)

        start = time.perf_counter()
        response = await self._call_cloud(litellm, call_kwargs, False)
        elapsed = time.perf_counter() - start

        logger.info(
            f"[CLOUD] Non-streaming completion from {self.cloud_model} "
            f"in {elapsed:.2f}s"
        )

        # litellm returns a ModelResponse — convert to dict
        return response.model_dump()

    async def stream_completion(
        self,
        messages: list[dict[str, Any]],
        model_name: str = "cloud-model",
        **kwargs,
    ) -> AsyncIterator[str]:
        """
        Streaming cloud completion. Yields SSE-formatted chunks.

        Args:
            messages: OpenAI chat format messages
            model_name: Model name to use in SSE chunk IDs
            **kwargs: temperature, max_tokens, top_p, tools, etc.
        """
        litellm = self._get_litellm()

        call_kwargs = self._build_call_kwargs(messages, stream=True, **kwargs)

        start = time.perf_counter()
        response = await self._call_cloud(litellm, call_kwargs, True)

        request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())
        token_count = 0

        async for chunk in response:
            # Extract the delta from litellm's streaming response
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta
            finish_reason = chunk.choices[0].finish_reason

            # Build SSE chunk in OpenAI format
            sse_chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": finish_reason,
                    }
                ],
            }

            # Populate delta fields
            if hasattr(delta, "role") and delta.role:
                sse_chunk["choices"][0]["delta"]["role"] = delta.role
            if hasattr(delta, "content") and delta.content:
                sse_chunk["choices"][0]["delta"]["content"] = delta.content
                token_count += 1
            if hasattr(delta, "tool_calls") and delta.tool_calls:
                sse_chunk["choices"][0]["delta"]["tool_calls"] = [
                    tc.model_dump() for tc in delta.tool_calls
                ]

            yield f"data: {json.dumps(sse_chunk)}\n\n"

        elapsed = time.perf_counter() - start
        logger.info(
            f"[CLOUD] Streamed {token_count} tokens from {self.cloud_model} "
            f"in {elapsed:.2f}s"
        )

        yield "data: [DONE]\n\n"

    async def _call_cloud(self, litellm, call_kwargs: dict, is_stream: bool):
        """Call litellm.acompletion with 30s timeout and exponential backoff retry (max 3)."""
        last_error = None
        for attempt in range(3):
            try:
                return await asyncio.wait_for(
                    litellm.acompletion(**call_kwargs), timeout=30.0
                )
            except (TimeoutError, Exception) as e:
                last_error = e
                if attempt < 2:
                    wait = 1.0 * (2 ** attempt)
                    logger.warning(
                        f"[CLOUD] Attempt {attempt + 1}/3 failed ({type(e).__name__}, "
                        f"{self.cloud_model}) — retrying in {wait:.0f}s"
                    )
                    await asyncio.sleep(wait)
        raise last_error

    def _build_call_kwargs(
        self,
        messages: list[dict[str, Any]],
        stream: bool,
        **kwargs,
    ) -> dict:
        """Build kwargs dict for litellm.acompletion."""
        call_kwargs: dict[str, Any] = {
            "model": self.cloud_model,
            "messages": messages,
            "stream": stream,
            "drop_params": True,  # Let litellm drop unsupported params per provider
        }

        # Custom API base and key for OpenAI-compatible providers (e.g. Zhipu)
        if self.api_base:
            call_kwargs["api_base"] = self.api_base
        if self.api_key:
            call_kwargs["api_key"] = self.api_key

        # Pass through standard OpenAI params (litellm handles provider compatibility)
        for key in (
            "temperature",
            "max_tokens",
            "top_p",
            "stop",
            "frequency_penalty",
            "presence_penalty",
            "tools",
            "tool_choice",
            "response_format",
        ):
            if key in kwargs and kwargs[key] is not None:
                call_kwargs[key] = kwargs[key]

        return call_kwargs
