# SPDX-License-Identifier: Apache-2.0
"""#1692: generation_tokens_per_second must be a decode-span rate.

The old computation divided completion_tokens by wall-clock elapsed since
request start, so queue + prefill wait diluted the decode rate by the TTFT
fraction (~11% low at TTFT 0.17s / span 1.44s; worse for long prompts).
The fix starts the rate window at the first output carrying tokens
(streaming) or subtracts time_to_first_token (non-streaming).

A fake module clock lets the tests pin exact numbers: prefill 0.15s,
decode 1.0s for 10 tokens -> span rate must read 10.0 (not 8.7).
"""

from types import SimpleNamespace
from typing import Any

import pytest


class _FakeClock:
    def __init__(self):
        self.t = 100.0

    def perf_counter(self):
        return self.t


class _TimedStreamingCore:
    """Yields two outputs, advancing the fake clock: prefill then decode."""

    def __init__(self, clock, prefill_s=0.15, decode_s=1.0, tokens=10):
        self._clock = clock
        self._prefill = prefill_s
        self._decode = decode_s
        self._tokens = tokens
        self.aborted_request_id = None

    async def add_request(self, **kwargs: Any) -> str:
        return "req-1692"

    async def stream_outputs(self, request_id: str):
        self._clock.t += self._prefill
        yield SimpleNamespace(
            output_text="x",
            new_text="x",
            prompt_tokens=10,
            completion_tokens=1,
            finished=False,
            finish_reason=None,
            tool_calls=None,
            cached_tokens=0,
            logprobs=None,
            new_token_ids=[],
            time_to_first_token=self._prefill,
        )
        self._clock.t += self._decode
        yield SimpleNamespace(
            output_text="x" * self._tokens,
            new_text="y",
            prompt_tokens=10,
            completion_tokens=self._tokens,
            finished=True,
            finish_reason="stop",
            tool_calls=None,
            cached_tokens=0,
            logprobs=None,
            new_token_ids=[],
            time_to_first_token=self._prefill,
        )

    async def abort_request(self, request_id: str) -> None:
        self.aborted_request_id = request_id


class _TimedNonStreamCore:
    def __init__(self, prefill_s=0.15, decode_s=1.0, tokens=10):
        self._prefill = prefill_s
        self._decode = decode_s
        self._tokens = tokens

    async def generate(self, **kwargs: Any):
        return SimpleNamespace(
            output_text="x" * self._tokens,
            prompt_tokens=10,
            completion_tokens=self._tokens,
            finish_reason="stop",
            tool_calls=None,
            cached_tokens=0,
            logprobs=None,
            new_token_ids=[],
            time_to_first_token=self._prefill,
            generation_tokens_per_second=None,
        )


@pytest.mark.asyncio
async def test_batched_stream_rate_is_decode_span(monkeypatch):
    import fusion_mlx.engines.batched as batched_mod
    from fusion_mlx.engines.batched import BatchedEngine

    clock = _FakeClock()
    monkeypatch.setattr(
        batched_mod, "time", SimpleNamespace(perf_counter=clock.perf_counter)
    )
    eng = BatchedEngine(model_name="test-model")
    eng._loaded = True
    eng._engine = _TimedStreamingCore(clock)

    chunks = []
    async for gen in eng.stream_generate("hello", max_tokens=10):
        chunks.append(gen)

    final = chunks[-1]
    # Span window = decode 1.0s (prefill 0.15s excluded) -> 10 tok / 1.0s.
    assert final.generation_tokens_per_second == pytest.approx(10.0, rel=1e-3)
    assert final.time_to_first_token == pytest.approx(0.15, rel=1e-3)


@pytest.mark.asyncio
async def test_batched_nonstream_rate_excludes_ttft(monkeypatch):
    import fusion_mlx.engines.batched as batched_mod
    from fusion_mlx.engines.batched import BatchedEngine

    clock = _FakeClock()

    real_perf = clock.perf_counter

    def perf_two_phase():
        # First call (t0) then generate() bumps the clock once via the core.
        return real_perf()

    monkeypatch.setattr(
        batched_mod, "time", SimpleNamespace(perf_counter=perf_two_phase)
    )
    eng = BatchedEngine(model_name="test-model")
    eng._loaded = True
    core = _TimedNonStreamCore()
    _inner_generate = core.generate

    async def generate_with_clock(**kwargs):
        clock.t += 1.15
        return await _inner_generate(**kwargs)

    core.generate = generate_with_clock
    eng._engine = core

    out = await eng.generate(prompt="hello", max_tokens=10)
    # Elapsed 1.15s, TTFT 0.15s -> span 1.0s -> 10.0 tok/s.
    assert out.generation_tokens_per_second == pytest.approx(10.0, rel=1e-3)


@pytest.mark.asyncio
async def test_vlm_stream_rate_is_decode_span(monkeypatch):
    import fusion_mlx.engines.vlm as vlm_mod
    from fusion_mlx.engines.vlm import VLMBatchedEngine

    clock = _FakeClock()
    monkeypatch.setattr(
        vlm_mod, "time", SimpleNamespace(perf_counter=clock.perf_counter)
    )
    engine = VLMBatchedEngine(model_name="test-vlm")
    engine._loaded = True
    engine._model_load_duration = 0.0
    engine._engine = _TimedStreamingCore(clock)

    chunks = []
    async for gen in engine.stream_generate("hello", max_tokens=10):
        chunks.append(gen)

    final = chunks[-1]
    assert final.generation_tokens_per_second == pytest.approx(10.0, rel=1e-3)
    assert final.time_to_first_token == pytest.approx(0.15, rel=1e-3)


@pytest.mark.asyncio
async def test_batched_stream_rate_falls_back_without_tokens():
    """No token-carrying output -> rate stays 0.0, no ZeroDivision."""
    from fusion_mlx.engines.batched import BatchedEngine

    class _EmptyCore(_TimedStreamingCore):
        async def stream_outputs(self, request_id: str):
            self._clock.t += 0.2
            yield SimpleNamespace(
                output_text="",
                new_text="",
                prompt_tokens=10,
                completion_tokens=0,
                finished=True,
                finish_reason="length",
                tool_calls=None,
                cached_tokens=0,
                logprobs=None,
                new_token_ids=[],
                time_to_first_token=None,
            )

    eng = BatchedEngine(model_name="test-model")
    eng._loaded = True
    eng._engine = _EmptyCore(_FakeClock())

    chunks = []
    async for gen in eng.stream_generate("hello", max_tokens=1):
        chunks.append(gen)

    assert chunks[-1].generation_tokens_per_second == 0.0
