# SPDX-License-Identifier: Apache-2.0
"""Engine Core for fusion-mlx continuous batching."""

import asyncio
import concurrent.futures
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional, Union

import mlx.core as mx

from .request import Request, RequestOutput, RequestStatus, SamplingParams
from .output_collector import RequestOutputCollector, RequestStreamState
from .model_registry import get_registry, ModelOwnershipError

logger = logging.getLogger(__name__)

_executor_config: dict[str, dict[str, Any]] = {
    "llm": {"max_workers": 1, "prefix": "mlx-llm"},
    "image": {"max_workers": 1, "prefix": "mlx-image"},
    "audio": {"max_workers": 2, "prefix": "mlx-audio"},
    "io": {"max_workers": 2, "prefix": "mlx-io"},
}
_global_executors: dict[str, concurrent.futures.ThreadPoolExecutor] = {}


def _init_mlx_thread() -> None:
    stream = mx.new_thread_local_stream(mx.default_device())
    import sys
    gen_mod = sys.modules.get("mlx_lm.generate")
    if gen_mod is not None:
        gen_mod.generation_stream = stream
    sched_mod = sys.modules.get("fusion_mlx.scheduler")
    if sched_mod is not None:
        sched_mod.generation_stream = stream


def get_executor(pool_type: str = "llm") -> concurrent.futures.ThreadPoolExecutor:
    if pool_type in _global_executors:
        return _global_executors[pool_type]
    cfg = _executor_config.get(pool_type, {"max_workers": 1, "prefix": f"mlx-{pool_type}"})
    exec_ = concurrent.futures.ThreadPoolExecutor(
        max_workers=cfg["max_workers"],
        thread_name_prefix=cfg["prefix"],
        initializer=_init_mlx_thread,
    )
    _global_executors[pool_type] = exec_
    return exec_


def get_mlx_executor() -> concurrent.futures.ThreadPoolExecutor:
    return get_executor("llm")


@dataclass
class EngineConfig:
    model_name: str = ""
    scheduler_config: Any | None = None
    step_interval: float = 0.001
    stream_interval: int = 1


class EngineCore:
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        config: Optional[EngineConfig] = None,
        engine_id: Optional[str] = None,
        force_model_ownership: bool = True,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or EngineConfig()
        self._engine_id = engine_id or str(uuid.uuid4())
        self._owns_model = False
        self._closed = False

        registry = get_registry()
        registry.acquire(model=model, engine=self, engine_id=self._engine_id, force=force_model_ownership)
        self._owns_model = True

        self._mlx_stream = mx.new_thread_local_stream(mx.default_device())
        self._mlx_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix=f"mlx-engine-{self._engine_id[:8]}",
        )

        # Create scheduler with per-engine stream
        from .scheduler import Scheduler, SchedulerConfig
        scheduler_config = self.config.scheduler_config or SchedulerConfig()
        self.scheduler = Scheduler(
            model=model,
            tokenizer=tokenizer,
            config=scheduler_config,
            stream=self._mlx_stream,
        )

        self._output_collectors: Dict[str, RequestOutputCollector] = {}
        self._stream_states: Dict[str, RequestStreamState] = {}
        self._finished_events: Dict[str, asyncio.Event] = {}

        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._start_time: Optional[float] = None
        self._steps_executed = 0

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._start_time = time.time()
        self._task = asyncio.create_task(self._engine_loop())

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    def is_running(self) -> bool:
        return self._running

    async def _engine_loop(self) -> None:
        """Main engine loop — steps the scheduler on the MLX thread."""
        step_interval = self.config.step_interval
        stream_interval = self.config.stream_interval
        use_simple_streaming = (stream_interval == 1)

        while self._running:
            if self.scheduler and self.scheduler.has_requests():
                try:
                    output = await asyncio.wait_for(
                        asyncio.get_running_loop().run_in_executor(
                            self._mlx_executor, self.scheduler.step
                        ), timeout=30.0)
                    self._steps_executed += 1
                    if output and output.outputs:
                        collectors = self._output_collectors
                        states = self._stream_states
                        events = self._finished_events
                        for req_output in output.outputs:
                            rid = req_output.request_id
                            collector = collectors.get(rid)
                            if collector is not None:
                                if use_simple_streaming:
                                    collector.put(req_output)
                                else:
                                    state = states.get(rid)
                                    if state and state.should_send(
                                        req_output.completion_tokens, req_output.finished
                                    ):
                                        collector.put(req_output)
                                        state.mark_sent(req_output.completion_tokens)
                            if req_output.finished:
                                event = events.get(rid)
                                if event:
                                    event.set()
                except Exception as e:
                    logger.error(f"Engine loop error: {e}", exc_info=True)
                    if self.scheduler:
                        try:
                            failed_ids = self.scheduler.fail_all_requests()
                            for rid in failed_ids:
                                collector = self._output_collectors.get(rid)
                                if collector is not None:
                                    collector.put(RequestOutput(
                                        request_id=rid, finished=True,
                                        finish_reason="error", error=str(e)
                                    ))
                                event = self._finished_events.get(rid)
                                if event:
                                    event.set()
                        except Exception:
                            logger.debug("fusion_mlx/engine_core.py:166: swallowed exception")
                            pass
            else:
                await asyncio.sleep(step_interval)


    async def add_request(
        self,
        prompt: Union[str, List[int]],
        sampling_params: Optional[SamplingParams] = None,
        request_id: Optional[str] = None,
        images: Optional[List[Any]] = None,
        videos: Optional[List[Any]] = None,
        vlm_inputs_embeds: Optional[Any] = None,
        vlm_extra_kwargs: Optional[Dict[str, Any]] = None,
        vlm_image_hash: Optional[str] = None,
        vlm_cache_key_start: int = 0,
        vlm_cache_key_ranges: Optional[list] = None,
        specprefill: Optional[bool] = None,
        specprefill_keep_pct: Optional[float] = None,
        specprefill_threshold: Optional[int] = None,
        specprefill_system_end: Optional[int] = None,
    ) -> str:
        if request_id is None:
            request_id = str(uuid.uuid4())
        if sampling_params is None:
            sampling_params = SamplingParams()

        request = Request(
            request_id=request_id, prompt=prompt, sampling_params=sampling_params,
            images=images, videos=videos,
            vlm_inputs_embeds=vlm_inputs_embeds, vlm_extra_kwargs=vlm_extra_kwargs,
            vlm_image_hash=vlm_image_hash, vlm_cache_key_start=vlm_cache_key_start,
            vlm_cache_key_ranges=vlm_cache_key_ranges,
        )
        if specprefill is not None:
            request._specprefill_enabled = specprefill
        elif self.scheduler and getattr(self.scheduler, "_specprefill_draft_model", None) is not None:
            request._specprefill_enabled = True
        if specprefill_keep_pct is not None:
            request._specprefill_keep_pct = specprefill_keep_pct
        if specprefill_threshold is not None:
            request._specprefill_threshold = specprefill_threshold
        if specprefill_system_end is not None and specprefill_system_end > 0:
            request.specprefill_system_end = specprefill_system_end

        self._output_collectors[request_id] = RequestOutputCollector(aggregate=True)
        self._stream_states[request_id] = RequestStreamState(stream_interval=self.config.stream_interval)
        self._finished_events[request_id] = asyncio.Event()

        if self.scheduler:
            self.scheduler.add_request(request)

        return request_id

    async def abort_request(self, request_id: str) -> bool:
        if getattr(self, "_closed", False) or not self.scheduler:
            return False
        result = self.scheduler.abort_request(request_id)
        collector = self._output_collectors.get(request_id)
        if collector is not None:
            collector.put(RequestOutput(request_id=request_id, finished=True, finish_reason="abort", error="Request aborted"))
        event = self._finished_events.get(request_id)
        if event is not None:
            event.set()
        return result

    async def abort_all_requests(self) -> int:
        from .utils.formatting import get_phys_footprint

        request_ids = list(self._output_collectors.keys())
        ceiling = 0
        sched = self.scheduler
        if sched is not None:
            ceiling = int(getattr(sched, "_memory_hard_limit_bytes", 0) or 0)
        usage = get_phys_footprint()
        usage_gb = usage / (1024**3)
        ceiling_gb = ceiling / (1024**3) if ceiling > 0 else 0.0
        for rid in request_ids:
            if self.scheduler:
                self.scheduler.abort_request(rid)
            collector = self._output_collectors.get(rid)
            if collector is not None:
                error_msg = (
                    f"Request aborted: process memory limit exceeded "
                    f"(usage {usage_gb:.1f} GB, ceiling {ceiling_gb:.1f} GB). "
                    "Reduce context size or lower memory_guard_tier."
                    if ceiling > 0
                    else f"Request aborted: process memory limit exceeded (usage {usage_gb:.1f} GB)."
                )
                collector.put(RequestOutput(request_id=rid, finished=True, finish_reason="error", new_text=f"\n\n[Error: {error_msg}]", error=error_msg))
            event = self._finished_events.get(rid)
            if event is not None:
                event.set()
        if request_ids:
            logger.warning(f"Aborted {len(request_ids)} requests due to memory pressure")
        return len(request_ids)

    def _cleanup_request(self, request_id: str) -> None:
        collector = self._output_collectors.pop(request_id, None)
        if collector:
            collector.clear()
        self._stream_states.pop(request_id, None)
        self._finished_events.pop(request_id, None)

    async def stream_outputs(self, request_id: str, timeout: Optional[float] = None) -> AsyncIterator[RequestOutput]:
        collector = self._output_collectors.get(request_id)
        if collector is None:
            return
        try:
            while True:
                try:
                    if timeout:
                        output = collector.get_nowait()
                        if output is None:
                            output = await asyncio.wait_for(collector.get(), timeout=timeout)
                    else:
                        output = collector.get_nowait() or await collector.get()
                    yield output
                    if output.error:
                        raise RuntimeError(output.error)
                    if output.finished:
                        break
                except asyncio.TimeoutError:
                    logger.warning(f"Timeout waiting for request {request_id}")
                    break
        finally:
            self._cleanup_request(request_id)

    async def generate(
        self,
        prompt: Union[str, List[int]],
        sampling_params: Optional[SamplingParams] = None,
        request_id: Optional[str] = None,
        **kwargs,
    ) -> RequestOutput:
        request_id = await self.add_request(prompt=prompt, sampling_params=sampling_params, request_id=request_id, **kwargs)
        event = self._finished_events.get(request_id)
        if event is None:
            raise RuntimeError(f"No event for request {request_id}")
        try:
            await event.wait()
        except asyncio.CancelledError:
            logger.info(f"Request {request_id} cancelled, aborting")
            await self.abort_request(request_id)
            self._cleanup_request(request_id)
            raise

        collector = self._output_collectors.get(request_id)
        if collector is None:
            raise RuntimeError(f"No collector for request {request_id}")
        final_output = None
        while True:
            output = collector.get_nowait()
            if output is None:
                break
            final_output = output
        self._cleanup_request(request_id)
        if final_output is None:
            raise RuntimeError(f"No output for request {request_id}")
        if final_output.error:
            raise RuntimeError(final_output.error)
        return final_output

    def generate_batch_sync(
        self,
        prompts: List[Union[str, List[int]]],
        sampling_params: Optional[SamplingParams] = None,
    ) -> List[RequestOutput]:
        if sampling_params is None:
            sampling_params = SamplingParams()
        request_ids = []
        for prompt in prompts:
            rid = str(uuid.uuid4())
            req = Request(request_id=rid, prompt=prompt, sampling_params=sampling_params)
            if self.scheduler:
                self.scheduler.add_request(req)
            request_ids.append(rid)
        results: Dict[str, RequestOutput] = {}
        if self.scheduler:
            while self.scheduler.has_requests():
                output = self.scheduler.step()
                for ro in output.outputs:
                    if ro.finished:
                        results[ro.request_id] = ro
        for rid in request_ids:
            if self.scheduler and rid in results:
                self.scheduler.remove_finished_request(rid)
        return [results[rid] for rid in request_ids if rid in results]

    def get_stats(self) -> Dict[str, Any]:
        scheduler_stats = self.scheduler.get_stats() if self.scheduler else {}
        uptime = time.time() - self._start_time if self._start_time else 0
        return {
            "running": self._running, "uptime_seconds": uptime,
            "steps_executed": self._steps_executed,
            "active_requests": len(self._output_collectors),
            "stream_interval": self.config.stream_interval, **scheduler_stats,
        }

    def get_cache_stats(self) -> Optional[Dict[str, Any]]:
        if self.scheduler:
            return self.scheduler.get_cache_stats()
        return None

    def close(self) -> None:
        if self._closed:
            return
        if self._owns_model:
            get_registry().release(self.model, self._engine_id)
            self._owns_model = False
        self._closed = True
        for fn in (self.scheduler.shutdown, self.scheduler.deep_reset) if self.scheduler else ():
            try:
                self._mlx_executor.submit(fn).result()
            except RuntimeError:
                try:
                    fn()
                except RuntimeError:
                    pass
        if self._mlx_executor is not None:
            self._mlx_executor.shutdown(wait=True)
            self._mlx_executor = None
        for c in self._output_collectors.values():
            c.clear()
        self._output_collectors.clear()
        self._stream_states.clear()
        self._finished_events.clear()
        self.model = None
        self.tokenizer = None
        self.scheduler = None

    def __del__(self):
        try:
            if self._owns_model and not self._closed:
                get_registry().release(self.model, self._engine_id)
        except Exception:
            logger.debug("swallowed exception at fusion_mlx/engine_core.py:403")

            pass

    @property
    def engine_id(self) -> str:
        return self._engine_id


class AsyncEngineCore:
    def __init__(self, model: Any, tokenizer: Any, config: Optional[EngineConfig] = None):
        self.engine = EngineCore(model, tokenizer, config)
        self._start_task: Optional[asyncio.Task] = None

    @property
    def _mlx_executor(self):
        return self.engine._mlx_executor

    async def __aenter__(self) -> "AsyncEngineCore":
        await self.engine.start()
        return self

    async def __aexit__(self, *args) -> None:
        await self.stop()

    def start(self) -> None:
        self._start_task = asyncio.create_task(self.engine.start())

    async def stop(self) -> None:
        engine = getattr(self, "engine", None)
        if engine is None:
            return
        await engine.stop()

    async def add_request(self, prompt: Union[str, List[int]], sampling_params: Optional[SamplingParams] = None, request_id: Optional[str] = None, **kwargs) -> str:
        return await self.engine.add_request(prompt=prompt, sampling_params=sampling_params, request_id=request_id, **kwargs)

    async def abort_request(self, request_id: str) -> bool:
        engine = getattr(self, "engine", None)
        if engine is None:
            return False
        return await engine.abort_request(request_id)

    async def abort_all_requests(self) -> int:
        engine = getattr(self, "engine", None)
        if engine is None:
            return 0
        return await engine.abort_all_requests()

    async def stream_outputs(self, request_id: str, timeout: Optional[float] = None) -> AsyncIterator[RequestOutput]:
        async for output in self.engine.stream_outputs(request_id, timeout):
            yield output

    async def generate(self, prompt: Union[str, List[int]], sampling_params: Optional[SamplingParams] = None, **kwargs) -> RequestOutput:
        return await self.engine.generate(prompt=prompt, sampling_params=sampling_params, **kwargs)

    def get_stats(self) -> Dict[str, Any]:
        return self.engine.get_stats()

    def get_cache_stats(self) -> Optional[Dict[str, Any]]:
        return self.engine.get_cache_stats()
