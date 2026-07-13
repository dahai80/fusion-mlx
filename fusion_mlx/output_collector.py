# SPDX-License-Identifier: Apache-2.0
"""Output collector for streaming with low-latency optimizations."""

import asyncio
from collections import deque
from dataclasses import dataclass
from typing import Any

from .request import RequestOutput


class RequestOutputCollector:
    """Per-request output collector with smart buffering.

    Two modes:
    - aggregate=True (non-streaming): merges all outputs into one final result
    - aggregate=False (streaming): queues individual outputs in a deque for
      low-latency per-token delivery
    """

    _waiting_consumers: int = 0

    def __init__(self, aggregate: bool = True):
        self.aggregate = aggregate
        self.ready = asyncio.Event()
        self._is_waiting = False
        if aggregate:
            self._merged: RequestOutput | None = None
        else:
            self._queue: deque[RequestOutput] = deque()

    def put(self, output: RequestOutput) -> None:
        if self.aggregate:
            if self._merged is None:
                self._merged = output
            else:
                self._merged = self._merge_outputs(self._merged, output)
        else:
            self._queue.append(output)
        self.ready.set()

    def get_nowait(self) -> RequestOutput | None:
        if self.aggregate:
            output = self._merged
            if output is not None:
                self._merged = None
                self.ready.clear()
            return output
        else:
            if self._queue:
                output = self._queue.popleft()
                if not self._queue:
                    self.ready.clear()
                return output
            self.ready.clear()
            return None

    async def get(self) -> RequestOutput:
        if not self._is_waiting:
            self._is_waiting = True
            RequestOutputCollector._waiting_consumers += 1
        try:
            while True:
                while not (self._merged if self.aggregate else self._queue):
                    await self.ready.wait()
                output = self.get_nowait()
                if output is not None:
                    return output
        finally:
            if self._is_waiting:
                self._is_waiting = False
                RequestOutputCollector._waiting_consumers -= 1

    def _merge_outputs(
        self,
        existing: RequestOutput,
        new: RequestOutput,
    ) -> RequestOutput:
        return RequestOutput(
            request_id=new.request_id,
            new_token_ids=existing.new_token_ids + new.new_token_ids,
            new_text=existing.new_text + new.new_text,
            output_token_ids=new.output_token_ids,
            output_text=new.output_text,
            finished=new.finished,
            finish_reason=new.finish_reason,
            prompt_tokens=new.prompt_tokens,
            completion_tokens=new.completion_tokens,
            generated_at=(
                existing.generated_at
                if existing.generated_at is not None
                else new.generated_at
            ),
            generated_until=(
                new.generated_until
                if new.generated_until is not None
                else existing.generated_until
            ),
            tool_calls=new.tool_calls,
            cached_tokens=new.cached_tokens,
            logprobs=self._merge_logprobs(existing.logprobs, new.logprobs),
            error=new.error or existing.error,
            error_code=new.error_code or existing.error_code,
            error_metadata=new.error_metadata or existing.error_metadata,
        )

    @staticmethod
    def _merge_logprobs(existing: Any, new: Any) -> Any:
        if existing is None:
            return new
        if new is None:
            return existing
        if isinstance(existing, list) and isinstance(new, list):
            return existing + new
        if isinstance(existing, dict) and isinstance(new, dict):
            merged = dict(existing)
            ec = existing.get("content")
            nc = new.get("content")
            if isinstance(ec, list) and isinstance(nc, list):
                merged["content"] = ec + nc
            return merged

        # Single MLX arrays (one full-vocab vector per decode step):
        # accumulate into a list so aggregated outputs carry the full
        # per-token sequence aligned with ``new_token_ids``.
        def _as_list(x: Any) -> list:
            return x if isinstance(x, list) else [x]

        return _as_list(existing) + _as_list(new)

    def clear(self) -> None:
        if self.aggregate:
            self._merged = None
        else:
            self._queue.clear()
        self.ready.clear()
        if self._is_waiting:
            self._is_waiting = False
            RequestOutputCollector._waiting_consumers -= 1

    @classmethod
    def has_waiting_consumers(cls) -> bool:
        return cls._waiting_consumers > 0

    def __bool__(self) -> bool:
        if self.aggregate:
            return self._merged is not None
        return bool(self._queue)


@dataclass
class RequestStreamState:
    stream_interval: int = 1
    sent_tokens: int = 0

    def should_send(self, total_tokens: int, finished: bool) -> bool:
        if finished:
            return True
        if self.sent_tokens == 0:
            return True
        return (total_tokens - self.sent_tokens) >= self.stream_interval

    def mark_sent(self, total_tokens: int) -> None:
        self.sent_tokens = total_tokens
