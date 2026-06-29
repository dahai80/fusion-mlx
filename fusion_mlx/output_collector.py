# SPDX-License-Identifier: Apache-2.0
"""Output collector for streaming with low-latency optimizations."""

import asyncio
from dataclasses import dataclass

from .request import RequestOutput


class RequestOutputCollector:
    """Per-request output collector with smart buffering."""

    _waiting_consumers: int = 0

    def __init__(self, aggregate: bool = True):
        self.output: RequestOutput | None = None
        self.ready = asyncio.Event()
        self.aggregate = aggregate
        self._is_waiting = False

    def put(self, output: RequestOutput) -> None:
        if self.output is None:
            self.output = output
        elif self.aggregate:
            self.output = self._merge_outputs(self.output, output)
        else:
            self.output = output
        self.ready.set()

    def get_nowait(self) -> RequestOutput | None:
        output = self.output
        if output is not None:
            self.output = None
            self.ready.clear()
        return output

    async def get(self) -> RequestOutput:
        if not self._is_waiting:
            self._is_waiting = True
            RequestOutputCollector._waiting_consumers += 1
        try:
            while True:
                while self.output is None:
                    await self.ready.wait()
                output = self.get_nowait()
                if output is not None:
                    return output
                # clear() stole output; re-wait
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
            error=new.error or existing.error,
            error_code=new.error_code or existing.error_code,
            error_metadata=new.error_metadata or existing.error_metadata,
        )

    def clear(self) -> None:
        self.output = None
        self.ready.clear()
        if self._is_waiting:
            self._is_waiting = False
            RequestOutputCollector._waiting_consumers -= 1

    @classmethod
    def has_waiting_consumers(cls) -> bool:
        return cls._waiting_consumers > 0


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
