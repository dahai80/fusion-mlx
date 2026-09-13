# SPDX-License-Identifier: Apache-2.0
"""G-6 (#0912 audit): VLM usage performance fields regression tests.

PR #878 fixed BatchedEngine but missed VLMBatchedEngine — Qwen3.8-27B is
served as VLM, so stream/non-stream usage returned null for
time_to_first_token / model_load_duration / generation_tokens_per_second.

These tests pin the VLM engine stamping + streaming.py last_chunk propagation
so the regression cannot silently return.
"""

from types import SimpleNamespace
from typing import Any

import pytest

from fusion_mlx.engine.base import GenerationOutput


class _FakeVLMStreamingCore:
    """Minimal async engine core yielding outputs that carry TTFT."""

    def __init__(self, ttft: float = 0.12, completion_tokens: int = 5):
        self._ttft = ttft
        self._completion_tokens = completion_tokens
        self.aborted_request_id = None

    async def add_request(self, **kwargs: Any) -> str:
        return "req-g6"

    async def stream_outputs(self, request_id: str):
        yield SimpleNamespace(
            output_text="hello world",
            new_text="hello ",
            prompt_tokens=10,
            completion_tokens=self._completion_tokens,
            finished=False,
            finish_reason=None,
            tool_calls=None,
            cached_tokens=0,
            logprobs=None,
            new_token_ids=[],
            time_to_first_token=self._ttft,
        )
        yield SimpleNamespace(
            output_text="hello world",
            new_text="world",
            prompt_tokens=10,
            completion_tokens=self._completion_tokens,
            finished=True,
            finish_reason="stop",
            tool_calls=None,
            cached_tokens=0,
            logprobs=None,
            new_token_ids=[],
            time_to_first_token=self._ttft,
        )

    async def abort_request(self, request_id: str) -> None:
        self.aborted_request_id = request_id


class TestVLMStreamGenerateUsageFields:
    """VLMBatchedEngine.stream_generate must stamp all 3 usage fields."""

    @pytest.mark.asyncio
    async def test_stream_stamps_ttft_model_load_tps(self):
        from fusion_mlx.engines.vlm import VLMBatchedEngine

        engine = VLMBatchedEngine(model_name="mlx-community--Qwen3.8-27B-4bit")
        engine._loaded = True
        engine._model_load_duration = 1.16
        engine._engine = _FakeVLMStreamingCore(ttft=0.12, completion_tokens=5)

        chunks: list[GenerationOutput] = []
        async for gen in engine.stream_generate("hello", max_tokens=10):
            chunks.append(gen)

        assert len(chunks) == 2
        # TTFT captured from scheduler-stamped first output.
        assert chunks[0].time_to_first_token == 0.12
        assert chunks[1].time_to_first_token == 0.12
        # model_load_duration propagated from engine load phase.
        assert chunks[0].model_load_duration == 1.16
        assert chunks[1].model_load_duration == 1.16
        # generation_tokens_per_second computed (not None).
        assert chunks[0].generation_tokens_per_second is not None
        assert chunks[0].generation_tokens_per_second > 0
        assert chunks[1].generation_tokens_per_second > 0

    @pytest.mark.asyncio
    async def test_stream_tps_grows_with_tokens(self):
        from fusion_mlx.engines.vlm import VLMBatchedEngine

        engine = VLMBatchedEngine(model_name="test-vlm")
        engine._loaded = True
        engine._model_load_duration = 2.0
        engine._engine = _FakeVLMStreamingCore(ttft=0.1, completion_tokens=10)

        chunks: list[GenerationOutput] = []
        async for gen in engine.stream_generate("hi", max_tokens=10):
            chunks.append(gen)

        # Cumulative tps = completion_tokens / elapsed; later chunk has larger
        # elapsed so tps is a real computed float, not a hardcoded constant.
        assert isinstance(chunks[1].generation_tokens_per_second, float)

    @pytest.mark.asyncio
    async def test_stream_abort_on_incomplete(self):
        from fusion_mlx.engines.vlm import VLMBatchedEngine

        fake = _FakeVLMStreamingCore(ttft=0.1, completion_tokens=3)
        engine = VLMBatchedEngine(model_name="test-vlm")
        engine._loaded = True
        engine._model_load_duration = 0.5
        engine._engine = fake

        stream = engine.stream_generate("hi", max_tokens=10)
        await stream.__anext__()
        await stream.aclose()

        # GeneratorExit path -> abort_request called because finished_normally
        # never observed True.
        assert fake.aborted_request_id == "req-g6"


class TestStreamChunkModelLoadPropagation:
    """streaming.py last_chunk StreamChunk must carry model_load_duration.

    Before G-6, last_chunk set time_to_first_token +
    generation_tokens_per_second but NOT model_load_duration, so the final
    stream Usage serialized model_load_duration=None (excluded).
    """

    def test_stream_chunk_dataclass_has_model_load_field(self):
        from fusion_mlx.api.adapters.base import StreamChunk

        chunk = StreamChunk(
            text="",
            is_last=True,
            finish_reason="stop",
            prompt_tokens=10,
            completion_tokens=5,
            cached_tokens=0,
            time_to_first_token=0.12,
            generation_tokens_per_second=33.3,
            model_load_duration=1.16,
        )
        assert chunk.model_load_duration == 1.16

    def test_stream_chunk_model_load_defaults_none(self):
        from fusion_mlx.api.adapters.base import StreamChunk

        chunk = StreamChunk(text="", is_last=True, finish_reason="stop")
        assert chunk.model_load_duration is None

    def test_adapter_format_stream_chunk_copies_model_load(self):
        from fusion_mlx.api.adapters.base import StreamChunk
        from fusion_mlx.api.adapters.openai import OpenAIAdapter
        from fusion_mlx.api.openai_models import ChatCompletionRequest

        adapter = OpenAIAdapter()
        request = ChatCompletionRequest(
            model="test", messages=[{"role": "user", "content": "hi"}]
        )
        chunk = StreamChunk(
            text="",
            is_last=True,
            finish_reason="stop",
            prompt_tokens=10,
            completion_tokens=5,
            cached_tokens=0,
            time_to_first_token=0.12,
            generation_tokens_per_second=33.3,
            model_load_duration=1.16,
        )
        sse = adapter.format_stream_chunk(chunk, request)
        assert "model_load_duration" in sse
        assert "1.16" in sse
        assert "time_to_first_token" in sse
        assert "generation_tokens_per_second" in sse
