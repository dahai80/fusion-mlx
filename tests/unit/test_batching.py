# SPDX-License-Identifier: Apache-2.0
"""
Tests for continuous batching system.

These tests verify the scheduler, engine, and request handling
for the vLLM-style continuous batching implementation.
"""

import asyncio
import sys
import threading
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from fusion_mlx.request import (
    Request,
    RequestOutput,
    RequestStatus,
    SamplingParams,
)
from fusion_mlx.scheduler import (
    Scheduler,
    SchedulerConfig,
    SchedulingPolicy,
)


class TestRequest:
    """Tests for Request class."""

    def test_request_creation(self):
        """Test basic request creation."""
        params = SamplingParams(max_tokens=100, temperature=0.8)
        request = Request(
            request_id="test-1",
            prompt="Hello, world!",
            sampling_params=params,
        )

        assert request.request_id == "test-1"
        assert request.prompt == "Hello, world!"
        assert request.sampling_params.max_tokens == 100
        assert request.status == RequestStatus.WAITING
        assert not request.is_finished()

    def test_request_status_transitions(self):
        """Test request status transitions."""
        request = Request(
            request_id="test-1",
            prompt="Hello",
            sampling_params=SamplingParams(),
        )

        assert request.status == RequestStatus.WAITING
        assert not request.is_finished()

        request.status = RequestStatus.RUNNING
        assert not request.is_finished()

        request.set_finished(RequestStatus.FINISHED_STOPPED)
        assert request.is_finished()
        assert request.get_finish_reason() == "stop"

    def test_request_output_tokens(self):
        """Test appending output tokens."""
        request = Request(
            request_id="test-1",
            prompt="Hello",
            sampling_params=SamplingParams(),
        )
        request.prompt_token_ids = [1, 2, 3]
        request.num_prompt_tokens = 3

        assert request.num_output_tokens == 0
        assert request.num_tokens == 3

        request.append_output_token(100)
        request.append_output_token(101)

        assert request.num_output_tokens == 2
        assert request.num_tokens == 5
        assert request.output_token_ids == [100, 101]

    def test_request_comparison(self):
        """Test request comparison for priority queue."""
        req1 = Request(
            request_id="req-1",
            prompt="Hello",
            sampling_params=SamplingParams(),
            priority=0,
            arrival_time=1.0,
        )
        req2 = Request(
            request_id="req-2",
            prompt="World",
            sampling_params=SamplingParams(),
            priority=1,
            arrival_time=0.5,
        )
        req3 = Request(
            request_id="req-3",
            prompt="Test",
            sampling_params=SamplingParams(),
            priority=0,
            arrival_time=2.0,
        )

        # Lower priority value = higher priority
        assert req1 < req2
        # Same priority, earlier arrival = higher priority
        assert req1 < req3


class TestSamplingParams:
    """Tests for SamplingParams."""

    def test_default_params(self):
        """Test default sampling parameters."""
        params = SamplingParams()

        # Default max_tokens is env-tunable (FUSION_MLX_MAX_TOKENS), not a
        # hard-coded 256 any more.
        from fusion_mlx.request import get_default_max_tokens

        assert params.max_tokens == get_default_max_tokens()
        assert params.temperature == 0.7
        assert params.top_p == 0.9
        assert params.stop == []
        assert params.stop_token_ids == []

    def test_custom_params(self):
        """Test custom sampling parameters."""
        params = SamplingParams(
            max_tokens=100,
            temperature=0.5,
            top_p=0.95,
            top_k=50,
            stop=["END"],
            stop_token_ids=[1, 2],
        )

        assert params.max_tokens == 100
        assert params.temperature == 0.5
        assert params.top_p == 0.95
        assert params.top_k == 50
        assert params.stop == ["END"]
        assert params.stop_token_ids == [1, 2]


class TestRequestOutput:
    """Tests for RequestOutput."""

    def test_output_creation(self):
        """Test output creation."""
        output = RequestOutput(
            request_id="test-1",
            new_token_ids=[100, 101],
            new_text="Hello",
            output_token_ids=[100, 101],
            output_text="Hello",
            finished=True,
            finish_reason="stop",
            prompt_tokens=10,
            completion_tokens=2,
        )

        assert output.request_id == "test-1"
        assert output.finished
        assert output.finish_reason == "stop"

        usage = output.usage
        assert usage["prompt_tokens"] == 10
        assert usage["completion_tokens"] == 2
        assert usage["total_tokens"] == 12


class TestSchedulerConfig:
    """Tests for SchedulerConfig."""

    def test_default_config(self):
        """Test default scheduler config."""
        config = SchedulerConfig()

        assert config.max_num_seqs == 256
        assert config.policy == SchedulingPolicy.FCFS
        assert config.prefill_batch_size == 8
        assert config.completion_batch_size == 32

    def test_custom_config(self):
        """Test custom scheduler config."""
        config = SchedulerConfig(
            max_num_seqs=64,
            policy=SchedulingPolicy.PRIORITY,
            prefill_batch_size=4,
            completion_batch_size=16,
        )

        assert config.max_num_seqs == 64
        assert config.policy == SchedulingPolicy.PRIORITY


class TestSchedulerBasic:
    """Basic tests for Scheduler (without real model)."""

    # R-P2-3: local mock_tokenizer fixture removed — was shadowing the
    # global conftest.py fixture with different semantics. The global
    # fixture now includes eos_token_ids for scheduler compatibility.

    @pytest.fixture
    def mock_model(self):
        """Create a mock model."""
        return MagicMock()

    def test_scheduler_creation(self, mock_model, mock_tokenizer):
        """Test scheduler creation."""
        scheduler = Scheduler(
            model=mock_model,
            tokenizer=mock_tokenizer,
            config=SchedulerConfig(max_num_seqs=10),
        )

        assert scheduler.get_num_waiting() == 0
        assert scheduler.get_num_running() == 0
        assert not scheduler.has_requests()

    def test_add_request(self, mock_model, mock_tokenizer):
        """Test adding requests to scheduler."""
        scheduler = Scheduler(
            model=mock_model,
            tokenizer=mock_tokenizer,
        )

        request = Request(
            request_id="test-1",
            prompt="Hello world",
            sampling_params=SamplingParams(max_tokens=10),
        )

        scheduler.add_request(request)

        assert scheduler.get_num_waiting() == 1
        assert scheduler.has_requests()
        assert scheduler.get_request("test-1") is not None

    def test_add_duplicate_request(self, mock_model, mock_tokenizer):
        """Test adding duplicate request raises error."""
        scheduler = Scheduler(
            model=mock_model,
            tokenizer=mock_tokenizer,
        )

        request = Request(
            request_id="test-1",
            prompt="Hello",
            sampling_params=SamplingParams(),
        )

        scheduler.add_request(request)

        with pytest.raises(ValueError, match="already exists"):
            scheduler.add_request(request)

    def test_abort_waiting_request(self, mock_model, mock_tokenizer):
        """Test aborting a waiting request (deferred abort pattern)."""
        scheduler = Scheduler(
            model=mock_model,
            tokenizer=mock_tokenizer,
        )

        request = Request(
            request_id="test-1",
            prompt="Hello",
            sampling_params=SamplingParams(),
        )

        scheduler.add_request(request)
        assert scheduler.get_num_waiting() == 1

        # abort_request() enqueues for deferred processing
        result = scheduler.abort_request("test-1")
        assert result is True

        # Process pending aborts (normally happens inside step())
        scheduler._process_pending_aborts()

        assert scheduler.get_num_waiting() == 0
        assert "test-1" in scheduler.finished_req_ids

    def test_abort_nonexistent_request(self, mock_model, mock_tokenizer):
        """Aborting a non-existent request returns False (F-151 hardening).

        Pre-F-151 ``abort_request`` returned True for any string, even ones
        never admitted into the scheduler. That let the
        ``/v1/requests/{id}/cancel`` route respond ``{"cancelled": true}`` to
        attacker-supplied IDs — an info-leak + validation bypass. The route
        relies on the False return as the 404 signal.
        """
        scheduler = Scheduler(
            model=mock_model,
            tokenizer=mock_tokenizer,
        )

        result = scheduler.abort_request("nonexistent")
        assert result is False
        # Idempotency cross-check: a request that IS known returns True,
        # and a follow-up abort on the SAME id (now in _pending_abort_ids)
        # also returns True so double-cancel doesn't 404 the second caller.
        request = Request(
            request_id="known-1",
            prompt="Hello",
            sampling_params=SamplingParams(),
        )
        scheduler.add_request(request)
        assert scheduler.abort_request("known-1") is True
        assert scheduler.abort_request("known-1") is True

    def test_get_stats(self, mock_model, mock_tokenizer):
        """Test getting scheduler stats."""
        scheduler = Scheduler(
            model=mock_model,
            tokenizer=mock_tokenizer,
        )

        stats = scheduler.get_stats()

        assert "num_waiting" in stats
        assert "num_running" in stats
        assert "num_requests_processed" in stats
        assert stats["num_waiting"] == 0
        assert stats["num_running"] == 0

    def test_reset(self, mock_model, mock_tokenizer):
        """Test resetting scheduler."""
        scheduler = Scheduler(
            model=mock_model,
            tokenizer=mock_tokenizer,
        )

        # Add some requests
        for i in range(5):
            request = Request(
                request_id=f"test-{i}",
                prompt=f"Hello {i}",
                sampling_params=SamplingParams(),
            )
            scheduler.add_request(request)

        assert scheduler.get_num_waiting() == 5

        scheduler.reset()

        assert scheduler.get_num_waiting() == 0
        assert scheduler.get_num_running() == 0
        assert not scheduler.has_requests()


# Integration tests require actual MLX model
@pytest.mark.integration
class TestSchedulerIntegration:
    """Integration tests that require a real model."""

    @pytest.fixture
    def model_and_tokenizer(self):
        """Load a small test model."""
        try:
            from mlx_lm import load

            model, tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-4bit")
            return model, tokenizer
        except Exception as e:
            pytest.skip(f"Could not load test model: {e}")

    def test_scheduler_with_real_model(self, model_and_tokenizer):
        """Test scheduler with real model."""
        model, tokenizer = model_and_tokenizer

        scheduler = Scheduler(
            model=model,
            tokenizer=tokenizer,
            config=SchedulerConfig(
                max_num_seqs=4,
                prefill_batch_size=2,
                completion_batch_size=4,
            ),
        )

        # Add a request
        request = Request(
            request_id="test-1",
            prompt="What is 2+2?",
            sampling_params=SamplingParams(max_tokens=10),
        )
        scheduler.add_request(request)

        # Run a few steps
        outputs = []
        for _ in range(20):
            output = scheduler.step()
            if output.outputs:
                outputs.extend(output.outputs)
            if output.finished_request_ids:
                break

        assert len(outputs) > 0
        # Check we got at least one output
        final_output = outputs[-1]
        assert final_output.request_id == "test-1"

    def test_multiple_concurrent_requests(self, model_and_tokenizer):
        """Test handling multiple concurrent requests."""
        model, tokenizer = model_and_tokenizer

        scheduler = Scheduler(
            model=model,
            tokenizer=tokenizer,
            config=SchedulerConfig(
                max_num_seqs=8,
                prefill_batch_size=4,
                completion_batch_size=8,
            ),
        )

        # Add multiple requests
        prompts = [
            "What is 1+1?",
            "What is 2+2?",
            "What is 3+3?",
            "What is 4+4?",
        ]

        for i, prompt in enumerate(prompts):
            request = Request(
                request_id=f"test-{i}",
                prompt=prompt,
                sampling_params=SamplingParams(max_tokens=10),
            )
            scheduler.add_request(request)

        # Run until all complete
        finished = set()
        max_steps = 100
        steps = 0

        while len(finished) < len(prompts) and steps < max_steps:
            output = scheduler.step()
            finished.update(output.finished_request_ids)
            steps += 1

        assert len(finished) == len(prompts), f"Only {len(finished)} requests finished"


class TestEngineThreading:
    """Threading tests for EngineCore."""

    def test_mlx_step_thread_initializer_rebinds_generation_stream(self, monkeypatch):
        """The executor thread must own mlx-lm's generation stream.

        Updated for #170: the worker now ADOPTS its thread's auto-default
        stream (via `mx.default_stream`) rather than creating a fresh one,
        so any ad-hoc `mx.array(...)` allocation that falls back to the
        default and the captured `with mx.stream(...)` context converge on
        the same stream object.
        """
        from fusion_mlx import engine_core

        fake_generate = types.SimpleNamespace(generation_stream="old-stream")
        monkeypatch.setitem(sys.modules, "mlx_lm.generate", fake_generate)
        monkeypatch.setattr(engine_core.mx, "default_device", lambda: "gpu")
        monkeypatch.setattr(
            engine_core.mx, "default_stream", lambda device: f"default-stream:{device}"
        )

        engine_core._init_mlx_step_thread()

        assert fake_generate.generation_stream == "default-stream:gpu"


# REMOVED 2026-09-13 (#0913 audit): TestMetalCacheLimit pinned
# `fusion_mlx.engine.batched._compute_metal_cache_limit`, which no longer
# exists anywhere in the product tree — the duplicate BatchedEngine was
# removed from `fusion_mlx/engine/batched` in #422/#428 and the metal cache
# limit helper went with it. Deleted rather than re-homed against product
# internals.


@pytest.mark.asyncio
class TestEngineAsync:
    """Async tests for the engine."""

    @pytest.fixture
    def mock_model_and_tokenizer(self):
        """Create mock model and tokenizer."""
        model = MagicMock()
        tokenizer = MagicMock()
        tokenizer.encode = lambda x: list(range(len(x.split())))
        tokenizer.decode = lambda x: " ".join(str(t) for t in x)
        tokenizer.eos_token_id = 0
        tokenizer.eos_token_ids = {0}
        return model, tokenizer

    async def test_engine_loop_keeps_all_scheduler_steps_on_mlx_thread(
        self, mock_model_and_tokenizer
    ):
        """Prefill and decode steps must run on the same MLX worker thread."""
        from fusion_mlx import engine_core
        from fusion_mlx.engine_core import EngineConfig, EngineCore

        model, tokenizer = mock_model_and_tokenizer
        engine = EngineCore(model, tokenizer, EngineConfig(step_interval=0.001))

        class FakeScheduler:
            batch_generator = None

            def __init__(self):
                self.calls = 0
                self.thread_names = []

            def has_requests(self):
                return self.calls < 2

            def step(self):
                self.thread_names.append(threading.current_thread().name)
                self.calls += 1
                if self.calls >= 2:
                    engine._running = False
                return SimpleNamespace(
                    outputs=[],
                    finished_request_ids=[],
                    has_work=False,
                    prefill_eviction_request=None,
                )

            def deep_reset(self):
                pass

            def shutdown(self):
                pass

        fake_scheduler = FakeScheduler()
        engine.scheduler = fake_scheduler

        # Mirror what start() does — create the mlx-step worker executor so
        # _engine_loop() picks it up. Tests can't call start() directly here
        # because start() spawns a task and returns immediately.
        import concurrent.futures

        engine._mlx_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="mlx-step",
            initializer=engine_core._init_mlx_step_thread,
        )
        engine._running = True

        try:
            await asyncio.wait_for(engine._engine_loop(), timeout=2)
        finally:
            engine._running = False
            # close() submits scheduler.shutdown()/deep_reset() to the
            # executor, so shut the executor down after close(), not before.
            engine.close()  # parks the executor in _immortal_mlx_executors

        assert fake_scheduler.thread_names
        assert all(name.startswith("mlx-step") for name in fake_scheduler.thread_names)

    async def test_stream_interval_gates_aggregate_collector_puts(
        self, mock_model_and_tokenizer
    ):
        """Regression: stream_interval > 1 must not drop step deltas.

        Contract drift update (2026-09-13 #0913 audit): the
        ``_stream_buffers`` accumulate-and-flush machinery was removed in the
        F-012 context-registry redesign (see debt_modules.txt note on
        test_rst_mid_sse_zombie_kv). The engine now gates only
        ``aggregate=True`` (non-streaming) collectors via
        ``RequestStreamState.should_send()``; streaming (deque) collectors
        bypass the gate and get every step delta. This test feeds 6 step
        outputs through the loop with stream_interval=4 and asserts the
        gating invariants on all three delta fields: should_send() fires at
        step 1 (first-token rule), step 5 (4 >= stream_interval) and step 6
        (finished=True always flushes) → 3 puts, and the concatenated deltas
        across those puts must be lossless (nothing dropped between puts).
        """
        from fusion_mlx import engine_core
        from fusion_mlx.engine_core import (
            EngineConfig,
            EngineCore,
            RequestContext,
        )
        from fusion_mlx.output_collector import RequestStreamState

        model, tokenizer = mock_model_and_tokenizer
        engine = EngineCore(
            model,
            tokenizer,
            EngineConfig(step_interval=0.001, stream_interval=4),
        )

        rid = "stream-interval-buffer-test"
        # Per-step deltas. completion_tokens climbs by 1 each step (matches
        # scheduler's one-token-per-step contract). logprobs values are
        # sentinels — the test asserts the *list* is preserved, the type
        # of the entries is irrelevant.
        steps = [
            RequestOutput(
                request_id=rid,
                new_token_ids=[10],
                new_text="he",
                output_token_ids=[10],
                output_text="he",
                finished=False,
                completion_tokens=1,
                logprobs="lp1",
            ),
            RequestOutput(
                request_id=rid,
                new_token_ids=[20],
                new_text="llo",
                output_token_ids=[10, 20],
                output_text="hello",
                finished=False,
                completion_tokens=2,
                logprobs="lp2",
            ),
            RequestOutput(
                request_id=rid,
                new_token_ids=[30],
                new_text=" wo",
                output_token_ids=[10, 20, 30],
                output_text="hello wo",
                finished=False,
                completion_tokens=3,
                logprobs="lp3",
            ),
            RequestOutput(
                request_id=rid,
                new_token_ids=[40],
                new_text="rld",
                output_token_ids=[10, 20, 30, 40],
                output_text="hello world",
                finished=False,
                completion_tokens=4,
                logprobs="lp4",
            ),
            RequestOutput(
                request_id=rid,
                new_token_ids=[50],
                new_text="!",
                output_token_ids=[10, 20, 30, 40, 50],
                output_text="hello world!",
                finished=False,
                completion_tokens=5,
                logprobs="lp5",
            ),
            RequestOutput(
                request_id=rid,
                new_token_ids=[60],
                new_text=".",
                output_token_ids=[10, 20, 30, 40, 50, 60],
                output_text="hello world!.",
                finished=True,
                finish_reason="stop",
                completion_tokens=6,
                logprobs="lp6",
            ),
        ]

        class FakeScheduler:
            batch_generator = None

            def __init__(self):
                self.calls = 0

            def has_requests(self):
                return self.calls < len(steps)

            def step(self):
                output = steps[self.calls]
                self.calls += 1
                if self.calls >= len(steps):
                    engine._running = False
                return SimpleNamespace(
                    outputs=[output],
                    finished_request_ids=[rid] if output.finished else [],
                    has_work=False,
                    prefill_eviction_request=None,
                )

            def deep_reset(self):
                pass

            def shutdown(self):
                pass

        engine.scheduler = FakeScheduler()

        # Capture every collector.put rather than aggregating, so we can
        # see exactly which deltas reached the consumer.
        puts: list[RequestOutput] = []

        class RecordingCollector:
            # Aggregated collector: the engine only applies stream_interval
            # gating (should_send/mark_sent) to aggregate=True collectors;
            # streaming (deque) collectors bypass it and get every delta.
            aggregate = True

            def put(self, output):
                puts.append(output)

            def clear(self):
                pass

        engine._active_contexts[rid] = RequestContext(
            collector=RecordingCollector(),
            stream_state=RequestStreamState(stream_interval=4),
            finished_event=asyncio.Event(),
        )

        import concurrent.futures

        engine._mlx_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="mlx-step",
            initializer=engine_core._init_mlx_step_thread,
        )
        engine._running = True

        try:
            await asyncio.wait_for(engine._engine_loop(), timeout=2)
        finally:
            engine._running = False
            # close() submits scheduler.shutdown()/deep_reset() to the
            # executor, so shut the executor down after close(), not before.
            engine.close()  # parks the executor in _immortal_mlx_executors

        # should_send() with stream_interval=4 fires at:
        #   step 1 (sent_tokens==0 first-token rule)
        #   step 5 (5 - 1 == 4 >= stream_interval)
        #   step 6 (finished=True always flushes)
        # → 3 puts (steps 2-4 suppressed between put 1 and put 2).
        assert len(puts) == 3, f"expected 3 puts, got {len(puts)}: {puts!r}"

        # The three surviving puts must be the step-1, step-5 and step-6
        # outputs, with their full per-step deltas intact.
        assert puts[0].new_text == "he"
        assert puts[0].new_token_ids == [10]
        assert puts[0].logprobs == "lp1"

        assert puts[1].new_text == "!"
        assert puts[1].new_token_ids == [50]
        assert puts[1].logprobs == "lp5"

        # finished=True must always flush.
        assert puts[-1].finished is True
        assert puts[-1].new_token_ids == [60]
        assert puts[-1].new_text == "."

    async def test_engine_lifecycle(self, mock_model_and_tokenizer):
        """Test engine start/stop lifecycle."""
        from fusion_mlx.engine_core import AsyncEngineCore, EngineConfig

        model, tokenizer = mock_model_and_tokenizer

        engine = AsyncEngineCore(model, tokenizer, EngineConfig())

        assert not engine.engine.is_running()

        # Use async context manager
        async with engine:
            assert engine.engine.is_running()
            await asyncio.sleep(0.05)

        assert not engine.engine.is_running()

    async def test_engine_context_manager(self, mock_model_and_tokenizer):
        """Test engine as async context manager."""
        from fusion_mlx.engine_core import AsyncEngineCore

        model, tokenizer = mock_model_and_tokenizer

        async with AsyncEngineCore(model, tokenizer) as engine:
            assert engine.engine.is_running()

        assert not engine.engine.is_running()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
