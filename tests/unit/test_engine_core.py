# SPDX-License-Identifier: Apache-2.0
"""Tests for EngineCore module (migrated from omlx to fusion-mlx).

Adaptations:
- _output_collectors/_stream_states/_finished_events -> _active_contexts (RequestContext)
- engine.engine_id -> engine._engine_id (private, with property)
- close() error messages use FATAL_TEARDOWN_TIMEOUT_S variable
- SchedulerOutput from config.py lacks prefill_eviction_request (use MagicMock)
- PrefillMemoryExceededError moved to fusion_mlx.scheduler.types
"""

import asyncio
import concurrent.futures
from unittest.mock import MagicMock, patch

import pytest

from fusion_mlx.engine_core import (
    AsyncEngineCore,
    EngineConfig,
    EngineCore,
    RequestContext,
    get_mlx_executor,
)
from fusion_mlx.output_collector import RequestOutputCollector, RequestStreamState
from fusion_mlx.request import RequestOutput, SamplingParams
from fusion_mlx.scheduler import SchedulerConfig, SchedulerOutput
from fusion_mlx.scheduler.types import PrefillMemoryExceededError


class TestEngineConfig:
    def test_default_values(self):
        config = EngineConfig()
        assert config.model_name == ""
        assert config.scheduler_config is None
        assert config.step_interval == 0.05
        assert config.stream_interval == 1

    def test_custom_values(self):
        scheduler_config = SchedulerConfig(max_num_seqs=64)
        config = EngineConfig(
            model_name="test-model",
            scheduler_config=scheduler_config,
            step_interval=0.005,
            stream_interval=5,
        )
        assert config.model_name == "test-model"
        assert config.scheduler_config is scheduler_config
        assert config.scheduler_config.max_num_seqs == 64
        assert config.step_interval == 0.005
        assert config.stream_interval == 5


class TestEngineCoreInitialization:
    def test_init_with_defaults(self, mock_model, mock_tokenizer):
        with patch("fusion_mlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True
            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)
            try:
                assert engine.model is mock_model
                assert engine.tokenizer is mock_tokenizer
                assert isinstance(engine.config, EngineConfig)
                assert engine._running is False
                assert engine._task is None
                assert engine._steps_executed == 0
                assert engine._active_contexts == {}
            finally:
                engine.close()

    def test_init_with_custom_config(self, mock_model, mock_tokenizer):
        with patch("fusion_mlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True
            config = EngineConfig(
                model_name="custom-model",
                step_interval=0.01,
                stream_interval=3,
            )
            engine = EngineCore(
                model=mock_model, tokenizer=mock_tokenizer, config=config,
            )
            try:
                assert engine.config.model_name == "custom-model"
                assert engine.config.step_interval == 0.01
                assert engine.config.stream_interval == 3
            finally:
                engine.close()

    def test_init_generates_engine_id(self, mock_model, mock_tokenizer):
        with patch("fusion_mlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True
            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)
            try:
                assert engine.engine_id is not None
                assert len(engine.engine_id) > 0
            finally:
                engine.close()

    def test_init_with_custom_engine_id(self, mock_model, mock_tokenizer):
        with patch("fusion_mlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True
            engine = EngineCore(
                model=mock_model,
                tokenizer=mock_tokenizer,
                engine_id="custom-engine-123",
            )
            try:
                assert engine.engine_id == "custom-engine-123"
            finally:
                engine.close()


class TestEngineCoreStartStop:
    @pytest.mark.asyncio
    async def test_start_sets_running(self, mock_model, mock_tokenizer):
        with patch("fusion_mlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True
            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)
            try:
                await engine.start()
                assert engine._running is True
                assert engine._task is not None
            finally:
                await engine.stop()
                engine.close()

    @pytest.mark.asyncio
    async def test_stop_clears_running(self, mock_model, mock_tokenizer):
        with patch("fusion_mlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True
            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)
            try:
                await engine.start()
                await engine.stop()
                assert engine._running is False
                assert engine._task is None
            finally:
                engine.close()

    @pytest.mark.asyncio
    async def test_is_running(self, mock_model, mock_tokenizer):
        with patch("fusion_mlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True
            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)
            try:
                assert engine.is_running() is False
                await engine.start()
                assert engine.is_running() is True
                await engine.stop()
                assert engine.is_running() is False
            finally:
                engine.close()

    @pytest.mark.asyncio
    async def test_double_start_noop(self, mock_model, mock_tokenizer):
        with patch("fusion_mlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True
            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)
            try:
                await engine.start()
                first_task = engine._task
                await engine.start()
                assert engine._task is first_task
            finally:
                await engine.stop()
                engine.close()

    @pytest.mark.asyncio
    async def test_idle_loop_wakes_without_waiting_for_step_interval(
        self, mock_model, mock_tokenizer
    ):
        with patch("fusion_mlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True
            engine = EngineCore(
                model=mock_model,
                tokenizer=mock_tokenizer,
                config=EngineConfig(step_interval=10.0),
            )
            try:
                engine.scheduler.has_requests = MagicMock(return_value=False)
                await engine.start()
                for _ in range(20):
                    if engine.scheduler.has_requests.call_count >= 2:
                        break
                    await asyncio.sleep(0.01)
                calls_before = engine.scheduler.has_requests.call_count
                assert calls_before >= 2
                await asyncio.sleep(0.05)
                assert engine.scheduler.has_requests.call_count == calls_before
                engine._wake_engine_loop()
                for _ in range(20):
                    if engine.scheduler.has_requests.call_count > calls_before:
                        break
                    await asyncio.sleep(0.01)
                assert engine.scheduler.has_requests.call_count > calls_before
            finally:
                await engine.stop()
                engine.close()

    @pytest.mark.asyncio
    async def test_loop_sleeps_when_scheduler_reports_no_work(
        self, mock_model, mock_tokenizer
    ):
        with patch("fusion_mlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True
            engine = EngineCore(
                model=mock_model,
                tokenizer=mock_tokenizer,
                config=EngineConfig(step_interval=10.0),
            )
            try:
                engine.scheduler.has_requests = MagicMock(return_value=True)
                engine.scheduler.step = MagicMock(
                    return_value=SchedulerOutput(has_work=False)
                )
                await engine.start()
                for _ in range(20):
                    if engine.scheduler.step.call_count >= 1:
                        break
                    await asyncio.sleep(0.01)
                calls_before = engine.scheduler.step.call_count
                assert calls_before == 1
                await asyncio.sleep(0.05)
                assert engine.scheduler.step.call_count == calls_before
                engine._wake_engine_loop()
                for _ in range(20):
                    if engine.scheduler.step.call_count > calls_before:
                        break
                    await asyncio.sleep(0.01)
                assert engine.scheduler.step.call_count > calls_before
            finally:
                await engine.stop()
                engine.close()


class TestEngineCoreAddRequest:
    @pytest.mark.asyncio
    async def test_add_request_returns_id(self, mock_model, mock_tokenizer):
        with patch("fusion_mlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True
            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)
            try:
                await engine.start()
                request_id = await engine.add_request(
                    prompt="Hello, world!",
                    sampling_params=SamplingParams(max_tokens=50),
                )
                assert request_id is not None
                assert isinstance(request_id, str)
            finally:
                await engine.stop()
                engine.close()

    @pytest.mark.asyncio
    async def test_add_request_with_custom_id(self, mock_model, mock_tokenizer):
        with patch("fusion_mlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True
            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)
            try:
                await engine.start()
                request_id = await engine.add_request(
                    prompt="Hello",
                    request_id="custom-request-001",
                )
                assert request_id == "custom-request-001"
            finally:
                await engine.stop()
                engine.close()

    @pytest.mark.asyncio
    async def test_add_request_creates_collector(self, mock_model, mock_tokenizer):
        with patch("fusion_mlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True
            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)
            try:
                await engine.start()
                request_id = await engine.add_request(prompt="Hello")
                assert request_id in engine._active_contexts
                ctx = engine._active_contexts[request_id]
                assert ctx.collector is not None
                assert ctx.stream_state is not None
                assert ctx.finished_event is not None
            finally:
                await engine.stop()
                engine.close()

    @pytest.mark.asyncio
    async def test_add_request_cleans_up_if_scheduler_insert_fails(
        self, mock_model, mock_tokenizer
    ):
        with patch("fusion_mlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True
            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)
            try:
                await engine.start()
                engine.scheduler.add_request = MagicMock(
                    side_effect=RuntimeError("insert boom")
                )
                engine.scheduler.abort_request = MagicMock(return_value=True)
                with pytest.raises(RuntimeError):
                    await engine.add_request(prompt="Hello")
                assert engine._active_contexts == {}
                assert engine._finished_at == {}
                engine.scheduler.abort_request.assert_called_once()
            finally:
                await engine.stop()
                engine.close()

    @pytest.mark.asyncio
    async def test_add_request_with_default_sampling_params(
        self, mock_model, mock_tokenizer
    ):
        with patch("fusion_mlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True
            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)
            try:
                await engine.start()
                request_id = await engine.add_request(prompt="Hello")
                assert request_id is not None
            finally:
                await engine.stop()
                engine.close()


class TestEngineCoreAbortRequest:
    @pytest.mark.asyncio
    async def test_abort_request_after_close_returns_false(self):
        engine = EngineCore.__new__(EngineCore)
        engine._closed = True
        engine.scheduler = None
        engine._active_contexts = {}
        engine._finished_at = {}
        result = await engine.abort_request("request-after-close")
        assert result is False

    @pytest.mark.asyncio
    async def test_abort_request(self, mock_model, mock_tokenizer):
        with patch("fusion_mlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True
            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)
            try:
                await engine.start()
                request_id = await engine.add_request(prompt="Hello")
                result = await engine.abort_request(request_id)
                assert result is True
            finally:
                await engine.stop()
                engine.close()

    @pytest.mark.asyncio
    async def test_abort_request_signals_consumer(self, mock_model, mock_tokenizer):
        with patch("fusion_mlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True
            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)
            try:
                await engine.start()
                engine.scheduler.has_requests = lambda: False
                request_id = await engine.add_request(prompt="Hello")
                await engine.abort_request(request_id)
                assert request_id in engine._active_contexts
                ctx = engine._active_contexts[request_id]
                output = ctx.collector.get_nowait()
                assert output is not None
                assert output.finished is True
                assert output.finish_reason == "abort"
                assert output.error == "Request aborted"
                assert ctx.finished_event.is_set()
                engine._cleanup_request(request_id)
                assert request_id not in engine._active_contexts
            finally:
                await engine.stop()
                engine.close()

    @pytest.mark.asyncio
    async def test_abort_request_no_ghost_in_scheduler(
        self, mock_model, mock_tokenizer
    ):
        with patch("fusion_mlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True
            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)
            try:
                await engine.start()
                engine.scheduler.has_requests = lambda: False
                request_id = await engine.add_request(prompt="Hello")
                assert request_id in engine.scheduler.requests
                await engine.abort_request(request_id)
                assert request_id in engine._active_contexts
                engine.scheduler._process_pending_aborts()
                assert request_id not in engine.scheduler.requests
                assert request_id not in engine.scheduler.running
                assert request_id not in engine.scheduler.request_id_to_uid
                engine._cleanup_request(request_id)
                assert request_id not in engine._active_contexts
            finally:
                await engine.stop()
                engine.close()

    @pytest.mark.asyncio
    async def test_abort_request_wakes_blocked_stream_outputs(
        self, mock_model, mock_tokenizer
    ):
        with patch("fusion_mlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True
            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)
            try:
                await engine.start()
                engine.scheduler.has_requests = lambda: False
                request_id = await engine.add_request(prompt="Hello")
                outputs = []

                async def consume():
                    async for output in engine.stream_outputs(request_id):
                        outputs.append(output)

                task = asyncio.create_task(consume())
                await asyncio.sleep(0.02)
                await engine.abort_request(request_id)
                with pytest.raises(RuntimeError, match="Request aborted"):
                    await asyncio.wait_for(task, timeout=1.0)
                assert len(outputs) == 1
                assert outputs[0].finished is True
                assert outputs[0].error == "Request aborted"
                assert request_id not in engine._active_contexts
            finally:
                await engine.stop()
                engine.close()


class TestEngineCoreGetStats:
    @pytest.mark.asyncio
    async def test_get_stats_initial(self, mock_model, mock_tokenizer):
        with patch("fusion_mlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True
            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)
            try:
                await engine.start()
                engine.scheduler.get_stats = MagicMock(return_value={})
                stats = engine.get_stats()
                assert "running" in stats
                assert "uptime_seconds" in stats
                assert "steps_executed" in stats
                assert "active_requests" in stats
                assert "stream_interval" in stats
                assert stats["running"] is True
                assert stats["steps_executed"] == 0
                assert stats["active_requests"] == 0
            finally:
                await engine.stop()
                engine.close()

    @pytest.mark.asyncio
    async def test_get_stats_includes_scheduler_stats(self, mock_model, mock_tokenizer):
        with patch("fusion_mlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True
            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)
            try:
                engine.scheduler.get_stats = MagicMock(
                    return_value={"num_waiting": 0, "num_running": 0}
                )
                stats = engine.get_stats()
                assert "num_waiting" in stats
                assert "num_running" in stats
            finally:
                engine.close()


class TestEngineCoreClose:
    def test_close_releases_model(self, mock_model, mock_tokenizer):
        with patch("fusion_mlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True
            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)
            engine.close()
            mock_registry.return_value.release.assert_called()

    def test_close_idempotent(self, mock_model, mock_tokenizer):
        with patch("fusion_mlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True
            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)
            engine.close()
            engine.close()

    def test_close_fatal_exits_when_teardown_future_times_out(
        self, mock_model, mock_tokenizer
    ):
        with patch("fusion_mlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True
            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)
            engine._mlx_executor.shutdown(wait=False)
            future = MagicMock()
            future.result.side_effect = concurrent.futures.TimeoutError
            executor = MagicMock()
            executor.submit.return_value = future
            engine._mlx_executor = executor
            with (
                patch("fusion_mlx.engine_core.fatal_exit", side_effect=SystemExit) as fatal,
                pytest.raises(SystemExit),
            ):
                engine.close()
            future.result.assert_called_once_with(timeout=60.0)
            assert "scheduler teardown timed out" in fatal.call_args.args[0]

    def test_close_fatal_exits_when_compile_cache_clear_times_out(
        self, mock_model, mock_tokenizer
    ):
        with patch("fusion_mlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True
            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)
            engine._mlx_executor.shutdown(wait=False)
            ok_future = MagicMock()
            ok_future.result.return_value = None
            timeout_future = MagicMock()
            timeout_future.result.side_effect = concurrent.futures.TimeoutError
            executor = MagicMock()
            executor.submit.side_effect = [ok_future, ok_future, timeout_future]
            engine._mlx_executor = executor
            with (
                patch("fusion_mlx.engine_core.compile_cache_clear_available", return_value=True),
                patch("fusion_mlx.engine_core.fatal_exit", side_effect=SystemExit) as fatal,
                pytest.raises(SystemExit),
            ):
                engine.close()
            timeout_future.result.assert_called_once_with(timeout=60.0)
            assert "compile cache" in fatal.call_args.args[0].lower()


class TestEngineCoreGetCacheStats:
    def test_get_cache_stats(self, mock_model, mock_tokenizer):
        with patch("fusion_mlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True
            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)
            try:
                engine.scheduler.get_cache_stats = MagicMock(return_value=None)
                stats = engine.get_cache_stats()
                assert stats is None
            finally:
                engine.close()


class TestEngineCoreGenerateCancellation:
    @pytest.mark.asyncio
    async def test_generate_cancel_aborts_request(self, mock_model, mock_tokenizer):
        with patch("fusion_mlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True
            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)
            try:
                await engine.start()
                engine.scheduler.has_requests = lambda: False
                task = asyncio.create_task(
                    engine.generate(
                        prompt="Hello, world!",
                        sampling_params=SamplingParams(max_tokens=50),
                    )
                )
                await asyncio.sleep(0.05)
                assert len(engine._active_contexts) == 1
                request_id = list(engine._active_contexts.keys())[0]
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
                assert request_id not in engine._active_contexts
            finally:
                await engine.stop()
                engine.close()

    @pytest.mark.asyncio
    async def test_generate_cancel_multiple_requests(self, mock_model, mock_tokenizer):
        with patch("fusion_mlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True
            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)
            try:
                await engine.start()
                engine.scheduler.has_requests = lambda: False
                task1 = asyncio.create_task(
                    engine.generate(
                        prompt="Request 1",
                        sampling_params=SamplingParams(max_tokens=50),
                    )
                )
                task2 = asyncio.create_task(
                    engine.generate(
                        prompt="Request 2",
                        sampling_params=SamplingParams(max_tokens=50),
                    )
                )
                await asyncio.sleep(0.05)
                assert len(engine._active_contexts) == 2
                request_ids = list(engine._active_contexts.keys())
                task1.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task1
                assert request_ids[0] not in engine._active_contexts
                assert request_ids[1] in engine._active_contexts
                task2.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task2
            finally:
                await engine.stop()
                engine.close()


class TestEngineCoreErrorPropagation:
    @pytest.mark.asyncio
    async def test_error_output_propagates_to_collector(
        self, mock_model, mock_tokenizer
    ):
        with patch("fusion_mlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True
            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)
            try:
                await engine.start()
                request_id = await engine.add_request(
                    prompt="Hello",
                    sampling_params=SamplingParams(max_tokens=50),
                )
                engine.scheduler.running[request_id] = MagicMock()
                ctx = engine._active_contexts.get(request_id)
                assert ctx is not None
                error_output = RequestOutput(
                    request_id=request_id,
                    finished=True,
                    finish_reason="error",
                    error="Memory limit exceeded during prefill",
                )
                ctx.collector.put(error_output)
                result = ctx.collector.get_nowait()
                assert result is not None
                assert result.error == "Memory limit exceeded during prefill"
                assert result.finished is True
                assert result.finish_reason == "error"
            finally:
                await engine.stop()
                engine.close()

    @pytest.mark.asyncio
    async def test_stream_outputs_raises_on_error(self, mock_model, mock_tokenizer):
        with patch("fusion_mlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True
            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)
            try:
                await engine.start()
                request_id = await engine.add_request(
                    prompt="Hello",
                    sampling_params=SamplingParams(max_tokens=50),
                )
                ctx = engine._active_contexts[request_id]
                error_output = RequestOutput(
                    request_id=request_id,
                    finished=True,
                    finish_reason="error",
                    error="Memory limit exceeded during prefill",
                )
                ctx.collector.put(error_output)
                with pytest.raises(RuntimeError, match="Memory limit exceeded"):
                    async for _ in engine.stream_outputs(request_id):
                        pass
            finally:
                await engine.stop()
                engine.close()

    @pytest.mark.asyncio
    async def test_stream_outputs_restores_prefill_memory_error(
        self, mock_model, mock_tokenizer
    ):
        with patch("fusion_mlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True
            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)
            try:
                await engine.start()
                request_id = await engine.add_request(
                    prompt="Hello",
                    sampling_params=SamplingParams(max_tokens=50),
                )
                ctx = engine._active_contexts[request_id]
                ctx.collector.put(
                    RequestOutput(
                        request_id=request_id,
                        finished=True,
                        finish_reason="error",
                        error="Prefill context too large for available memory",
                        error_code="prefill_memory_exceeded",
                        error_metadata={
                            "request_id": request_id,
                            "estimated_bytes": 123,
                            "limit_bytes": 100,
                        },
                    )
                )
                with pytest.raises(PrefillMemoryExceededError) as exc:
                    async for _ in engine.stream_outputs(request_id):
                        pass
                assert exc.value.request_id == request_id
                assert exc.value.estimated_bytes == 123
                assert exc.value.limit_bytes == 100
            finally:
                await engine.stop()
                engine.close()

    @pytest.mark.asyncio
    async def test_generate_raises_on_error(self, mock_model, mock_tokenizer):
        with patch("fusion_mlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True
            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)
            try:
                await engine.start()
                request_id = await engine.add_request(
                    prompt="Hello",
                    sampling_params=SamplingParams(max_tokens=50),
                )
                ctx = engine._active_contexts[request_id]
                error_output = RequestOutput(
                    request_id=request_id,
                    finished=True,
                    finish_reason="error",
                    error="Memory limit exceeded during prefill",
                )
                ctx.collector.put(error_output)
                ctx.finished_event.set()
                final_output = None
                while True:
                    output = ctx.collector.get_nowait()
                    if output is None:
                        break
                    final_output = output
                assert final_output is not None
                assert final_output.error == "Memory limit exceeded during prefill"
            finally:
                await engine.stop()
                engine.close()

    @pytest.mark.asyncio
    async def test_generate_restores_prefill_memory_error(
        self, mock_model, mock_tokenizer
    ):
        with patch("fusion_mlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True
            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)
            try:
                await engine.start()
                request_id = await engine.add_request(
                    prompt="Hello",
                    sampling_params=SamplingParams(max_tokens=50),
                )
                ctx = engine._active_contexts[request_id]
                ctx.collector.put(
                    RequestOutput(
                        request_id=request_id,
                        finished=True,
                        finish_reason="error",
                        error="Prefill context too large for available memory",
                        error_code="prefill_memory_exceeded",
                        error_metadata={
                            "request_id": request_id,
                            "estimated_bytes": 123,
                            "limit_bytes": 100,
                        },
                    )
                )
                ctx.finished_event.set()

                async def _reuse_existing_request(*args, **kwargs):
                    return request_id

                engine.add_request = _reuse_existing_request  # type: ignore[method-assign]
                with pytest.raises(PrefillMemoryExceededError) as exc:
                    await engine.generate(
                        prompt="ignored",
                        sampling_params=SamplingParams(max_tokens=50),
                    )
                assert exc.value.request_id == request_id
                assert exc.value.estimated_bytes == 123
                assert exc.value.limit_bytes == 100
            finally:
                await engine.stop()
                engine.close()


class TestAsyncEngineCore:
    @pytest.mark.asyncio
    async def test_context_manager(self, mock_model, mock_tokenizer):
        with patch("fusion_mlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True
            async with AsyncEngineCore(
                model=mock_model, tokenizer=mock_tokenizer,
            ) as engine:
                assert engine.engine._running is True
            assert engine.engine._running is False

    @pytest.mark.asyncio
    async def test_add_request(self, mock_model, mock_tokenizer):
        with patch("fusion_mlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True
            async with AsyncEngineCore(
                model=mock_model, tokenizer=mock_tokenizer,
            ) as engine:
                request_id = await engine.add_request(prompt="Hello")
                assert request_id is not None

    @pytest.mark.asyncio
    async def test_abort_request(self, mock_model, mock_tokenizer):
        with patch("fusion_mlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True
            async with AsyncEngineCore(
                model=mock_model, tokenizer=mock_tokenizer,
            ) as engine:
                request_id = await engine.add_request(prompt="Hello")
                result = await engine.abort_request(request_id)
                assert result is True

    @pytest.mark.asyncio
    async def test_abort_request_after_close_returns_false(self):
        async_engine = AsyncEngineCore.__new__(AsyncEngineCore)
        setattr(async_engine, "engine", None)
        result = await async_engine.abort_request("request-after-close")
        assert result is False

    @pytest.mark.asyncio
    async def test_context_manager_exit_after_close_does_not_raise(self):
        async_engine = AsyncEngineCore.__new__(AsyncEngineCore)
        setattr(async_engine, "engine", None)
        await async_engine.__aexit__(None, None, None)

    @pytest.mark.asyncio
    async def test_abort_all_requests_after_close_returns_zero(self):
        async_engine = AsyncEngineCore.__new__(AsyncEngineCore)
        setattr(async_engine, "engine", None)
        count = await async_engine.abort_all_requests()
        assert count == 0

    @pytest.mark.asyncio
    async def test_get_stats(self, mock_model, mock_tokenizer):
        with patch("fusion_mlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True
            async with AsyncEngineCore(
                model=mock_model, tokenizer=mock_tokenizer,
            ) as engine:
                engine.engine.scheduler.get_stats = MagicMock(return_value={})
                stats = engine.get_stats()
                assert "running" in stats
                assert stats["running"] is True

    @pytest.mark.asyncio
    async def test_get_cache_stats(self, mock_model, mock_tokenizer):
        with patch("fusion_mlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True
            async with AsyncEngineCore(
                model=mock_model, tokenizer=mock_tokenizer,
            ) as engine:
                engine.engine.scheduler.get_cache_stats = MagicMock(return_value=None)
                stats = engine.get_cache_stats()
                assert stats is None

    @pytest.mark.asyncio
    async def test_start_stores_task_reference(self):
        with patch("fusion_mlx.engine_core.EngineCore") as MockEngine, \
             patch("fusion_mlx.engine_core.mx") as mock_mx:
            mock_engine = MagicMock()
            mock_engine.start = MagicMock(return_value=asyncio.sleep(0))
            MockEngine.return_value = mock_engine
            core = AsyncEngineCore(MagicMock(), MagicMock())
            assert core._start_task is None
            core.start()
            assert core._start_task is not None
            assert isinstance(core._start_task, asyncio.Task)


class TestEngineCoreAbortAllRequests:
    @pytest.mark.asyncio
    async def test_abort_all_requests(self, mock_model, mock_tokenizer):
        with patch("fusion_mlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True
            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)
            try:
                await engine.start()
                engine.scheduler.has_requests = lambda: False
                rid1 = await engine.add_request(prompt="Hello")
                rid2 = await engine.add_request(prompt="World")
                count = await engine.abort_all_requests()
                assert count == 2
                for rid in [rid1, rid2]:
                    ctx = engine._active_contexts.get(rid)
                    if ctx is not None:
                        output = ctx.collector.get_nowait()
                        assert output is not None
                        assert output.finished is True
                        assert output.finish_reason == "error"
                        assert "memory" in output.error.lower()
                        assert output.new_text is not None
                        assert "[Error:" in output.new_text
                        assert "memory" in output.new_text.lower()
            finally:
                await engine.stop()
                engine.close()

    @pytest.mark.asyncio
    async def test_abort_all_requests_empty(self, mock_model, mock_tokenizer):
        with patch("fusion_mlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True
            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)
            try:
                await engine.start()
                count = await engine.abort_all_requests()
                assert count == 0
            finally:
                await engine.stop()
                engine.close()

    @pytest.mark.asyncio
    async def test_abort_all_requests_engine_keeps_running(
        self, mock_model, mock_tokenizer
    ):
        with patch("fusion_mlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True
            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)
            try:
                await engine.start()
                engine.scheduler.has_requests = lambda: False
                rid = await engine.add_request(prompt="Hello")
                await engine.abort_all_requests()
                assert engine.is_running() is True
                new_rid = await engine.add_request(prompt="New request")
                assert new_rid in engine._active_contexts
            finally:
                await engine.stop()
                engine.close()


class TestGlobalMLXExecutor:
    def test_get_mlx_executor_returns_singleton(self):
        executor1 = get_mlx_executor()
        executor2 = get_mlx_executor()
        assert executor1 is executor2

    def test_engines_have_per_engine_executors(self, mock_model, mock_tokenizer):
        with patch("fusion_mlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True
            engine1 = EngineCore(model=mock_model, tokenizer=mock_tokenizer)
            engine2 = EngineCore(model=mock_model, tokenizer=mock_tokenizer)
            try:
                assert engine1._mlx_executor is not engine2._mlx_executor
            finally:
                engine1.close()
                engine2.close()

    @pytest.mark.asyncio
    async def test_shared_executor_serializes_concurrent_tasks(self):
        import threading
        import time

        executor = get_mlx_executor()
        loop = asyncio.get_running_loop()
        active_count = 0
        max_concurrent = 0
        lock = threading.Lock()

        def simulated_step(task_id: str, duration: float = 0.05):
            nonlocal active_count, max_concurrent
            with lock:
                active_count += 1
                if active_count > max_concurrent:
                    max_concurrent = active_count
            time.sleep(duration)
            with lock:
                active_count -= 1
            return task_id

        tasks = [
            loop.run_in_executor(executor, simulated_step, "engine_a_step1"),
            loop.run_in_executor(executor, simulated_step, "engine_b_step1"),
            loop.run_in_executor(executor, simulated_step, "engine_a_step2"),
            loop.run_in_executor(executor, simulated_step, "engine_b_step2"),
        ]
        results = await asyncio.gather(*tasks)
        assert set(results) == {
            "engine_a_step1", "engine_b_step1",
            "engine_a_step2", "engine_b_step2",
        }
        assert max_concurrent == 1, (
            f"Expected max 1 concurrent task, got {max_concurrent}. "
            f"Shared executor failed to serialize MLX operations."
        )

    @pytest.mark.asyncio
    async def test_two_engine_loops_run_concurrently_on_separate_executors(
        self, mock_model, mock_tokenizer
    ):
        import threading
        import time

        active_count = 0
        max_concurrent = 0
        total_steps = 0
        lock = threading.Lock()

        def make_tracked_step():
            def tracked_step():
                nonlocal active_count, max_concurrent, total_steps
                with lock:
                    active_count += 1
                    total_steps += 1
                    if active_count > max_concurrent:
                        max_concurrent = active_count
                time.sleep(0.01)
                with lock:
                    active_count -= 1
                return SchedulerOutput(outputs=[])
            return tracked_step

        with patch("fusion_mlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True
            engine1 = EngineCore(model=mock_model, tokenizer=mock_tokenizer)
            engine2 = EngineCore(model=mock_model, tokenizer=mock_tokenizer)
            engine1.scheduler.step = make_tracked_step()
            engine2.scheduler.step = make_tracked_step()
            engine1.scheduler.has_requests = lambda: True
            engine2.scheduler.has_requests = lambda: True
            try:
                await engine1.start()
                await engine2.start()
                await asyncio.sleep(0.3)
            finally:
                await engine1.stop()
                await engine2.stop()
                engine1.close()
                engine2.close()
        assert total_steps >= 4, f"Expected at least 4 steps, got {total_steps}"
        assert max_concurrent >= 2, (
            f"Expected concurrent execution (max_concurrent >= 2), got {max_concurrent}."
        )


class TestEngineCoreCloseReleasesSSDManager:
    def test_manager_closed_when_shutdown_raises(self, mock_model, mock_tokenizer):
        with patch("fusion_mlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True
            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)
            scheduler = engine.scheduler
            manager = MagicMock()
            scheduler.paged_ssd_cache_manager = manager
            scheduler.shutdown = MagicMock(side_effect=RuntimeError("boom"))
            engine.close()
            assert manager.close.call_count >= 1
            assert engine.scheduler is None

    def test_manager_closed_when_executor_fallback_raises(
        self, mock_model, mock_tokenizer
    ):
        with patch("fusion_mlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True
            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)
            scheduler = engine.scheduler
            manager = MagicMock()
            scheduler.paged_ssd_cache_manager = manager
            scheduler.shutdown = MagicMock(side_effect=RuntimeError("boom"))
            engine._mlx_executor.shutdown(wait=True)
            engine.close()
            assert manager.close.call_count >= 1
            assert engine.scheduler is None

    def test_manager_closed_on_normal_close(self, mock_model, mock_tokenizer):
        with patch("fusion_mlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True
            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)
            scheduler = engine.scheduler
            manager = MagicMock()
            scheduler.paged_ssd_cache_manager = manager
            engine.close()
            assert manager.close.call_count >= 1
            assert engine.scheduler is None


def _make_step_output(has_work=True, prefill_eviction_request=None, outputs=None):
    """Create a mock SchedulerOutput with prefill_eviction_request field."""
    out = MagicMock()
    out.has_work = has_work
    out.prefill_eviction_request = prefill_eviction_request
    out.outputs = outputs or []
    return out


class TestStepBurst:
    def _make_engine(self, mock_model, mock_tokenizer, max_steps, budget=0.2):
        with patch("fusion_mlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True
            config = EngineConfig(
                decode_burst_max_steps=max_steps,
                decode_burst_budget_single_s=budget,
                decode_burst_budget_s=budget,
            )
            return EngineCore(model=mock_model, tokenizer=mock_tokenizer, config=config)

    def test_max_steps_1_runs_single_step(self, mock_model, mock_tokenizer):
        engine = self._make_engine(mock_model, mock_tokenizer, max_steps=1)
        try:
            engine.scheduler.step = MagicMock(
                return_value=SchedulerOutput(has_work=True)
            )
            engine.scheduler.has_requests = MagicMock(return_value=True)
            outs = engine._step_burst()
            assert len(outs) == 1
            assert engine.scheduler.step.call_count == 1
        finally:
            engine.close()

    def test_runs_up_to_max_steps(self, mock_model, mock_tokenizer):
        engine = self._make_engine(mock_model, mock_tokenizer, max_steps=4)
        try:
            engine.scheduler.step = MagicMock(
                return_value=_make_step_output(has_work=True)
            )
            engine.scheduler.has_requests = MagicMock(return_value=True)
            outs = engine._step_burst()
            assert len(outs) == 4
            assert engine.scheduler.step.call_count == 4
        finally:
            engine.close()

    def test_breaks_when_no_requests(self, mock_model, mock_tokenizer):
        engine = self._make_engine(mock_model, mock_tokenizer, max_steps=4)
        try:
            engine.scheduler.step = MagicMock(
                return_value=_make_step_output(has_work=True)
            )
            engine.scheduler.has_requests = MagicMock(return_value=False)
            outs = engine._step_burst()
            assert len(outs) == 1
            assert engine.scheduler.step.call_count == 1
        finally:
            engine.close()

    def test_breaks_on_no_work(self, mock_model, mock_tokenizer):
        engine = self._make_engine(mock_model, mock_tokenizer, max_steps=4)
        try:
            engine.scheduler.step = MagicMock(
                side_effect=[
                    _make_step_output(has_work=True),
                    _make_step_output(has_work=False),
                    _make_step_output(has_work=True),
                ]
            )
            engine.scheduler.has_requests = MagicMock(return_value=True)
            outs = engine._step_burst()
            assert len(outs) == 2
            assert engine.scheduler.step.call_count == 2
        finally:
            engine.close()

    def test_breaks_on_eviction(self, mock_model, mock_tokenizer):
        engine = self._make_engine(mock_model, mock_tokenizer, max_steps=4)
        try:
            engine.scheduler.step = MagicMock(
                return_value=_make_step_output(
                    has_work=True, prefill_eviction_request=MagicMock()
                )
            )
            engine.scheduler.has_requests = MagicMock(return_value=True)
            outs = engine._step_burst()
            assert len(outs) == 1
            assert engine.scheduler.step.call_count == 1
        finally:
            engine.close()

    def test_breaks_on_budget(self, mock_model, mock_tokenizer):
        engine = self._make_engine(mock_model, mock_tokenizer, max_steps=8, budget=0.05)
        try:
            engine.scheduler.step = MagicMock(
                return_value=_make_step_output(has_work=True)
            )
            engine.scheduler.has_requests = MagicMock(return_value=True)
            with patch("fusion_mlx.engine_core.time.monotonic", side_effect=[100.0, 200.0]):
                outs = engine._step_burst()
            assert len(outs) == 1
            assert engine.scheduler.step.call_count == 1
        finally:
            engine.close()

    def test_budget_zero_disables_bursting(self, mock_model, mock_tokenizer):
        engine = self._make_engine(mock_model, mock_tokenizer, max_steps=8, budget=0.0)
        try:
            engine.scheduler.step = MagicMock(
                return_value=SchedulerOutput(has_work=True)
            )
            engine.scheduler.has_requests = MagicMock(return_value=True)
            outs = engine._step_burst()
            assert len(outs) == 1
            assert engine.scheduler.step.call_count == 1
        finally:
            engine.close()

    def test_adaptive_single_budget_when_solo(self, mock_model, mock_tokenizer):
        with patch("fusion_mlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True
            config = EngineConfig(
                decode_burst_max_steps=4,
                decode_burst_budget_single_s=10.0,
                decode_burst_budget_s=0.0,
            )
            engine = EngineCore(
                model=mock_model, tokenizer=mock_tokenizer, config=config
            )
        try:
            engine.scheduler.step = MagicMock(
                return_value=_make_step_output(has_work=True)
            )
            engine.scheduler.has_requests = MagicMock(return_value=True)
            engine.scheduler.running = {"a": object()}
            outs = engine._step_burst()
            assert len(outs) == 4
        finally:
            engine.close()

    def test_adaptive_concurrent_budget_when_busy(self, mock_model, mock_tokenizer):
        with patch("fusion_mlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True
            config = EngineConfig(
                decode_burst_max_steps=8,
                decode_burst_budget_single_s=10.0,
                decode_burst_budget_s=0.0,
            )
            engine = EngineCore(
                model=mock_model, tokenizer=mock_tokenizer, config=config
            )
        try:
            engine.scheduler.step = MagicMock(
                return_value=SchedulerOutput(has_work=True)
            )
            engine.scheduler.has_requests = MagicMock(return_value=True)
            engine.scheduler.running = {"a": object(), "b": object()}
            outs = engine._step_burst()
            assert len(outs) == 1
        finally:
            engine.close()


class TestOrphanedCollectorReaping:
    def test_reaps_only_stale_finished_collectors(self, mock_model, mock_tokenizer):
        with patch("fusion_mlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True
            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)
            try:
                now = 1000.0
                orphan_collector = RequestOutputCollector()
                orphan_collector.put(
                    RequestOutput(
                        request_id="orphan",
                        finished=True,
                        finish_reason="abort",
                        new_text="partial",
                    )
                )
                engine._active_contexts["orphan"] = RequestContext(
                    collector=orphan_collector,
                    stream_state=RequestStreamState(),
                    finished_event=asyncio.Event(),
                )
                engine._finished_at["orphan"] = now - 100.0
                engine._active_contexts["fresh"] = RequestContext(
                    collector=RequestOutputCollector(),
                    stream_state=RequestStreamState(),
                    finished_event=asyncio.Event(),
                )
                engine._finished_at["fresh"] = now - 1.0
                engine._active_contexts["active"] = RequestContext(
                    collector=RequestOutputCollector(),
                    stream_state=RequestStreamState(),
                    finished_event=asyncio.Event(),
                )
                reaped = engine._reap_orphaned_collectors(now=now, grace=5.0)
                assert reaped == 1
                assert "orphan" not in engine._active_contexts
                assert "orphan" not in engine._finished_at
                assert "fresh" in engine._active_contexts
                assert "active" in engine._active_contexts
            finally:
                engine.close()

    def test_mark_request_finished_stamps_once_and_signals(
        self, mock_model, mock_tokenizer
    ):
        with patch("fusion_mlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True
            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)
            try:
                event = asyncio.Event()
                engine._active_contexts["r1"] = RequestContext(
                    collector=RequestOutputCollector(),
                    stream_state=RequestStreamState(),
                    finished_event=event,
                )
                engine._mark_request_finished("r1")
                assert event.is_set()
                assert "r1" in engine._finished_at
                first = engine._finished_at["r1"]
                engine._mark_request_finished("r1")
                assert engine._finished_at["r1"] == first
            finally:
                engine.close()

    def test_cleanup_request_removes_finished_stamp(self, mock_model, mock_tokenizer):
        with patch("fusion_mlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True
            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)
            try:
                engine._active_contexts["r1"] = RequestContext(
                    collector=RequestOutputCollector(),
                    stream_state=RequestStreamState(),
                    finished_event=asyncio.Event(),
                )
                engine._finished_at["r1"] = 123.0
                engine._cleanup_request("r1")
                assert "r1" not in engine._finished_at
                assert "r1" not in engine._active_contexts
            finally:
                engine.close()

    @pytest.mark.asyncio
    async def test_generate_drains_via_held_reference_if_reaped(
        self, mock_model, mock_tokenizer
    ):
        with patch("fusion_mlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True
            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)
            try:
                await engine.start()
                engine.scheduler.has_requests = lambda: False
                task = asyncio.create_task(
                    engine.generate(
                        prompt="Hello",
                        sampling_params=SamplingParams(max_tokens=5),
                    )
                )
                await asyncio.sleep(0.05)
                request_id = list(engine._active_contexts.keys())[0]
                ctx = engine._active_contexts[request_id]
                collector = ctx.collector
                collector.put(
                    RequestOutput(
                        request_id=request_id,
                        finished=True,
                        finish_reason="stop",
                        new_text="done",
                    )
                )
                engine._active_contexts.pop(request_id)
                ctx.finished_event.set()
                result = await task
                assert result is not None
                assert result.new_text == "done"
            finally:
                await engine.stop()
                engine.close()
