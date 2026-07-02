# SPDX-License-Identifier: Apache-2.0
"""Engine Core for fusion-mlx continuous batching."""

import asyncio
import concurrent.futures
import logging
import os
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

import mlx.core as mx

from .model_registry import get_registry
from .output_collector import RequestOutputCollector, RequestStreamState
from .request import Request, RequestOutput, SamplingParams
from .scheduler.types import PrefillMemoryExceededError
from .utils.compile_cache import (
    clear_thread_compile_cache,
    compile_cache_clear_available,
)
from .utils.fatal import FATAL_TEARDOWN_TIMEOUT_S, fatal_exit

logger = logging.getLogger(__name__)


def _raise_request_output_error(output: RequestOutput) -> None:
    if output.error_code == "prefill_memory_exceeded":
        metadata = output.error_metadata or {}
        request_id = metadata.get("request_id")
        estimated_bytes = metadata.get("estimated_bytes")
        limit_bytes = metadata.get("limit_bytes")
        raise PrefillMemoryExceededError(
            message=output.error or "Prefill memory exceeded",
            request_id=str(request_id) if request_id is not None else output.request_id,
            estimated_bytes=(
                int(estimated_bytes) if estimated_bytes is not None else None
            ),
            limit_bytes=int(limit_bytes) if limit_bytes is not None else None,
        )
    raise RuntimeError(output.error)


# Fallback only: used when the MLX compile-cache clear symbol is unavailable
# (see utils/compile_cache.py). In that case a per-engine MLX worker thread
# cannot exit safely (its thread_local ~CompilerCache would free @mx.compile
# graphs' Python objects without the GIL -> crash), so close() keeps the
# executor + stream alive here for the process lifetime instead.
_immortal_mlx_executors: list = []
_immortal_mlx_streams: list = []


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
    logger.debug("MLX executor thread initialized: generation_stream = %s", stream)


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
class RequestContext:
    collector: RequestOutputCollector
    stream_state: RequestStreamState
    finished_event: asyncio.Event


@dataclass
class EngineConfig:
    model_name: str = ""
    scheduler_config: Any | None = None
    step_interval: float = 0.05
    stream_interval: int = 1
    prefill_eviction_callback: Callable[[Any], Awaitable[bool]] | None = None
    # Decode burst: run several scheduler.step() calls per run_in_executor
    # hand-off instead of one. Each decode token otherwise bounces back to the
    # event loop, ping-ponging the GIL with asyncio + uvicorn on the main
    # thread; bursting keeps the MLX thread holding the GIL continuously.
    # scheduler.step() services aborts/admission/finish every step, so
    # correctness is unchanged. Budget is a TIME ceiling so event-loop pause
    # is bounded consistently across hardware.
    decode_burst_max_steps: int = field(
        default_factory=lambda: int(os.environ.get("OMLX_DECODE_BURST_MAX_STEPS", "16"))
    )
    decode_burst_budget_single_s: float = field(
        default_factory=lambda: float(
            os.environ.get("OMLX_DECODE_BURST_BUDGET_SINGLE_S", "0.5")
        )
    )
    decode_burst_budget_s: float = field(
        default_factory=lambda: float(
            os.environ.get("OMLX_DECODE_BURST_BUDGET_S", "0.1")
        )
    )


class EngineCore:
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        config: EngineConfig | None = None,
        engine_id: str | None = None,
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

        # Per-engine executor with dedicated mx.Stream (#1248).
        # Each EngineCore gets its own thread + GPU stream so different
        # models can run scheduler.step() concurrently.
        self._mlx_stream = mx.new_thread_local_stream(mx.default_device())
        self._mlx_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix=f"mlx-engine-{self._engine_id[:8]}",
        )

        # Scheduler must be created on the executor thread so it uses the
        # thread-local MLX stream (not the main thread's stream).
        from .scheduler import Scheduler, SchedulerConfig
        scheduler_config = self.config.scheduler_config or SchedulerConfig()
        self.scheduler: Scheduler | None = None
        _sched_result: list = []
        def _make_scheduler():
            _sched_result.append(Scheduler(
                model=model,
                tokenizer=tokenizer,
                config=scheduler_config,
                stream=self._mlx_stream,
            ))
        _fut = self._mlx_executor.submit(_make_scheduler)
        _fut.result()
        self.scheduler = _sched_result[0]

        # Initialize speculative decode draft model on the executor thread
        from .scheduler.spec_decode import SpecDecodeState, SPEC_DRAFT_MODEL_ENABLED
        if SPEC_DRAFT_MODEL_ENABLED:
            def _init_draft():
                from .speculative.draft_model import DraftModelDecoder
                draft = DraftModelDecoder()
                loaded = draft.load()
                if loaded:
                    self.scheduler._spec_decode_state = SpecDecodeState(draft_model_decoder=draft)
                    logger.info("Speculative decode: draft model enabled (%s)", draft.model_path)
                else:
                    logger.info("Speculative decode: draft model failed to load, disabled")
            _fut = self._mlx_executor.submit(_init_draft)
            _fut.result()

        # Initialize n-gram speculative decode (CPU-side, zero GPU overhead)
        from .scheduler.ngram_spec import NGramSpecState, NGRAM_SPEC_ENABLED
        if NGRAM_SPEC_ENABLED:
            self.scheduler._ngram_spec_state = NGramSpecState()
            logger.info("N-gram speculative decode: enabled (order=%d, num_draft=%d)",
                        self.scheduler._ngram_spec_state.predictor.order,
                        self.scheduler._ngram_spec_state.predictor.num_draft)

        self._active_contexts: dict[str, RequestContext] = {}

        # Finish timestamps for orphan-collector reaping (#1154).
        self._finished_at: dict[str, float] = {}
        self._last_reap = 0.0

        self._running = False
        self._task: asyncio.Task | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._wake_event: asyncio.Event | None = None
        self._start_time: float | None = None
        self._steps_executed = 0
        logger.debug("Engine %s initialized", self._engine_id)

    async def start(self) -> None:
        if self._running:
            return
        self._loop = asyncio.get_running_loop()
        self._wake_event = asyncio.Event()
        self._running = True
        self._start_time = time.time()
        self._task = asyncio.create_task(self._engine_loop())
        logger.info("Engine started")

    async def stop(self) -> None:
        self._running = False
        if self._wake_event is not None:
            self._wake_event.set()
        if self._task:
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        self._wake_event = None
        self._loop = None
        logger.info("Engine stopped")

    def is_running(self) -> bool:
        return self._running

    def _wake_engine_loop(self) -> None:
        """Wake the idle engine loop after scheduler-visible state changes."""
        event = getattr(self, "_wake_event", None)
        loop = getattr(self, "_loop", None)
        if event is None or loop is None or loop.is_closed():
            return
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None
        if running_loop is loop:
            event.set()
        else:
            loop.call_soon_threadsafe(event.set)

    def _step_burst(self) -> list:
        """Run scheduler.step() several times in one executor hand-off.

        Each decode token otherwise bounces back to the event loop, which
        ping-pongs the GIL with asyncio + uvicorn on the main thread.
        Chaining a few steps lets the MLX thread hold the GIL continuously.

        scheduler.step() services aborts/admission/finish every step, so
        correctness is unchanged; the only cost is event-loop responsiveness,
        bounded by decode_burst_budget_s. Stops early when no work remains, a
        prefill eviction needs the (async) callback, or the budget elapses.

        Runs on the MLX executor thread. Returns the SchedulerOutputs in order.
        """
        max_steps = self.config.decode_burst_max_steps
        outputs = [self.scheduler.step()]
        if max_steps <= 1:
            return outputs
        running = getattr(self.scheduler, "running", None)
        single = running is None or len(running) <= 1
        budget = (
            self.config.decode_burst_budget_single_s
            if single
            else self.config.decode_burst_budget_s
        )
        if budget <= 0:
            return outputs
        deadline = time.monotonic() + budget
        while len(outputs) < max_steps:
            last = outputs[-1]
            if (
                not last.has_work
                or not self.scheduler.has_requests()
                or last.prefill_eviction_request is not None
                or time.monotonic() >= deadline
            ):
                break
            outputs.append(self.scheduler.step())
        return outputs

    async def _engine_loop(self) -> None:
        """Main engine loop — runs scheduler steps on the MLX executor.

        All scheduler steps run on _mlx_executor (single-worker thread) to
        guarantee that MLX GPU operations are never concurrent.
        """
        loop = asyncio.get_running_loop()
        step_interval = self.config.step_interval
        stream_interval = self.config.stream_interval
        use_simple_streaming = stream_interval == 1

        while self._running:
            try:
                # Sweep collectors orphaned by client disconnects (throttled).
                now = time.monotonic()
                if now - self._last_reap >= 1.0:
                    self._last_reap = now
                    self._reap_orphaned_collectors(now)

                if self.scheduler.has_requests():
                    step_outputs = await loop.run_in_executor(
                        self._mlx_executor, self._step_burst
                    )
                    self._steps_executed += len(step_outputs)

                    contexts = self._active_contexts
                    eviction_request = None
                    has_streaming_consumer = False

                    for output in step_outputs:
                        if (
                            eviction_request is None
                            and output.prefill_eviction_request is not None
                        ):
                            eviction_request = output.prefill_eviction_request

                        outputs = output.outputs
                        if not outputs:
                            continue

                        for req_output in outputs:
                            rid = req_output.request_id
                            ctx = contexts.get(rid)
                            if ctx is not None:
                                is_streaming = not ctx.collector.aggregate
                                if use_simple_streaming or is_streaming:
                                    ctx.collector.put(req_output)
                                    has_streaming_consumer = True
                                else:
                                    if ctx.stream_state.should_send(
                                        req_output.completion_tokens,
                                        req_output.finished,
                                    ):
                                        ctx.collector.put(req_output)
                                        ctx.stream_state.mark_sent(
                                            req_output.completion_tokens
                                        )
                            if req_output.finished:
                                self._mark_request_finished(rid)

                    # Yield to event loop so SSE handlers can flush queued
                    # outputs. For streaming (deque-based collector), this must
                    # happen after every burst to avoid buffering all tokens.
                    if has_streaming_consumer or any(
                        o.outputs for o in step_outputs
                    ):
                        await asyncio.sleep(0)

                    if eviction_request is not None:
                        callback = self.config.prefill_eviction_callback
                        if callback is not None:
                            logger.info(
                                "Running prefill LRU eviction for request %s",
                                eviction_request.request_id,
                            )
                            evicted = await callback(eviction_request)
                            if evicted:
                                logger.info(
                                    "Prefill LRU eviction completed for request %s",
                                    eviction_request.request_id,
                                )
                            else:
                                logger.info(
                                    "No idle model evicted for request %s; "
                                    "scheduler will fall back to throttling",
                                    eviction_request.request_id,
                                )
                        else:
                            logger.debug(
                                "Prefill eviction requested for %s but no callback "
                                "is configured",
                                eviction_request.request_id,
                            )
                        continue
                    if not step_outputs[-1].has_work:
                        event = self._wake_event
                        if event is None:
                            await asyncio.sleep(step_interval)
                        else:
                            event.clear()
                            with suppress(TimeoutError):
                                await asyncio.wait_for(
                                    event.wait(), timeout=step_interval
                                )
                else:
                    event = self._wake_event
                    if event is None:
                        await asyncio.sleep(step_interval)
                    else:
                        event.clear()
                        if self.scheduler.has_requests():
                            continue
                        with suppress(TimeoutError):
                            await asyncio.wait_for(
                                event.wait(), timeout=step_interval
                            )

            except asyncio.CancelledError:
                break
            except Exception as e:
                import traceback
                logger.error("Engine loop error: %s\n%s", e, traceback.format_exc())
                # Fail all requests and remove from scheduler to prevent
                # infinite loop (has_requests() must return False).
                def _safe_fail():
                    try:
                        return self.scheduler.fail_all_requests()
                    except Exception:
                        return []
                failed_ids = await loop.run_in_executor(
                    self._mlx_executor, _safe_fail
                )
                for rid in failed_ids:
                    ctx = self._active_contexts.get(rid)
                    if ctx is not None:
                        ctx.collector.put(RequestOutput(
                            request_id=rid, finished=True,
                            finish_reason="error", error=str(e)
                        ))
                    self._mark_request_finished(rid)
                await asyncio.sleep(0.1)


    async def add_request(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams | None = None,
        request_id: str | None = None,
        images: list[Any] | None = None,
        videos: list[Any] | None = None,
        vlm_inputs_embeds: Any | None = None,
        vlm_extra_kwargs: dict[str, Any] | None = None,
        vlm_image_hash: str | None = None,
        vlm_cache_key_start: int = 0,
        vlm_cache_key_ranges: list | None = None,
        specprefill: bool | None = None,
        specprefill_keep_pct: float | None = None,
        specprefill_threshold: int | None = None,
        specprefill_system_end: int | None = None,
        streaming: bool = False,
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

        logger.info(
              "add_request: id=%s, prompt=%r, max_tokens=%d",
            request_id, str(prompt)[:100] if isinstance(prompt, str) else f"tokens({len(prompt)})",
            sampling_params.max_tokens,
              )
        self._active_contexts[request_id] = RequestContext(
            collector=RequestOutputCollector(aggregate=not streaming),
            stream_state=RequestStreamState(stream_interval=self.config.stream_interval),
            finished_event=asyncio.Event(),
        )

        if self.scheduler:
            # Route through the MLX executor so prefix cache reconstruction
            # (mx.load, mx.concatenate) never races with scheduler.step()
            # on the Metal stream.
            loop = asyncio.get_running_loop()
            try:
                await loop.run_in_executor(
                    self._mlx_executor, self.scheduler.add_request, request
                )
            except BaseException:
                # If the caller is cancelled or the insert fails, the request
                # never reaches stream_outputs()/generate()'s try/finally, so
                # nothing would mark it finished or clean it up. Drop tracking
                # and abort any partial scheduler insert before re-raising.
                try:
                    self.scheduler.abort_request(request_id)
                except Exception as abort_exc:
                    logger.debug(
                        "Abort of partial insert for %s failed: %s",
                        request_id, abort_exc,
                    )
                self._cleanup_request(request_id)
                raise
        self._wake_engine_loop()
        return request_id

    async def abort_request(self, request_id: str) -> bool:
        scheduler = getattr(self, "scheduler", None)
        if getattr(self, "_closed", False) or scheduler is None:
            logger.debug(
                "Skipping abort for request %s because engine is already closed",
                request_id,
            )
            return False
        result = scheduler.abort_request(request_id)
        ctx = self._active_contexts.get(request_id)
        if ctx is not None:
            ctx.collector.put(RequestOutput(
                request_id=request_id, finished=True,
                finish_reason="abort", error="Request aborted",
            ))
        self._mark_request_finished(request_id)
        self._wake_engine_loop()
        return result

    async def abort_all_requests(self) -> int:
        from .utils.proc_memory import get_phys_footprint

        request_ids = list(self._active_contexts.keys())
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
            ctx = self._active_contexts.get(rid)
            if ctx is not None:
                error_msg = (
                    f"Request aborted: process memory limit exceeded "
                    f"(usage {usage_gb:.1f} GB, ceiling {ceiling_gb:.1f} GB). "
                    "Reduce context size or lower memory_guard_tier."
                    if ceiling > 0
                    else (
                        f"Request aborted: process memory limit exceeded "
                        f"(usage {usage_gb:.1f} GB). "
                        "Reduce context size or lower memory_guard_tier."
                    )
                )
                ctx.collector.put(RequestOutput(
                    request_id=rid, finished=True, finish_reason="error",
                    new_text=f"\n\n[Error: {error_msg}]", error=error_msg,
                ))
            self._mark_request_finished(rid)
        if request_ids:
            logger.warning("Aborted %d requests due to memory pressure", len(request_ids))
            self._wake_engine_loop()
        return len(request_ids)

    def _cleanup_request(self, request_id: str) -> None:
        ctx = self._active_contexts.pop(request_id, None)
        if ctx:
            ctx.collector.clear()
        self._finished_at.pop(request_id, None)

    def _mark_request_finished(self, request_id: str) -> None:
        """Stamp finish time and signal the consumer.

        The timestamp lets _reap_orphaned_collectors() drop collectors whose
        consumer never cleaned up (e.g. client disconnected mid-stream).
        """
        self._finished_at.setdefault(request_id, time.monotonic())
        ctx = self._active_contexts.get(request_id)
        if ctx is not None:
            ctx.finished_event.set()

    def _reap_orphaned_collectors(self, now: float, grace: float = 5.0) -> int:
        """Drop tracking for finished requests whose consumer never cleaned up.

        Pop-only: never clear() the collector object. A live consumer holds its
        own reference, so dropping the dict entry cannot truncate output.
        """
        if not self._finished_at:
            return 0
        stale = [rid for rid, ts in self._finished_at.items() if now - ts >= grace]
        for rid in stale:
            ctx = self._active_contexts.pop(rid, None)
            if ctx:
                ctx.collector.clear()
            self._finished_at.pop(rid, None)
        if stale:
            logger.debug(
                "Reaped %d orphaned output collector(s) after disconnect: %s",
                len(stale), stale,
            )
        return len(stale)

    async def stream_outputs(self, request_id: str, timeout: float | None = None) -> AsyncIterator[RequestOutput]:
        ctx = self._active_contexts.get(request_id)
        if ctx is None:
            return
        collector = ctx.collector
        try:
            logger.info("stream_outputs start: %s", request_id)
            while True:
                try:
                    if timeout:
                        output = collector.get_nowait()
                        if output is None:
                            output = await asyncio.wait_for(
                                collector.get(), timeout=timeout
                            )
                    else:
                        output = collector.get_nowait() or await collector.get()
                    yield output
                    if output.error:
                        _raise_request_output_error(output)
                    if output.finished:
                        logger.info(
                            "stream_outputs done: %s, finish=%s, tokens=%d",
                            request_id, output.finish_reason,
                            output.completion_tokens,
                        )
                        break
                except asyncio.TimeoutError:
                    logger.warning("Timeout waiting for request %s", request_id)
                    break
        finally:
            self._cleanup_request(request_id)

    async def generate(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams | None = None,
        request_id: str | None = None,
        **kwargs,
    ) -> RequestOutput:
        request_id = await self.add_request(
            prompt=prompt, sampling_params=sampling_params,
            request_id=request_id, **kwargs,
        )
        ctx = self._active_contexts.get(request_id)
        if ctx is None:
            raise RuntimeError(f"No context for request {request_id}")
        # Capture the collector reference BEFORE awaiting — the orphan reaper
        # is pop-only and may drop the dict entry once the request is finished,
        # but a held reference still drains.
        collector = ctx.collector
        try:
            await ctx.finished_event.wait()
        except asyncio.CancelledError:
            logger.info("Request %s cancelled, aborting", request_id)
            await self.abort_request(request_id)
            self._cleanup_request(request_id)
            raise

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
            _raise_request_output_error(final_output)
        return final_output

    def generate_batch_sync(
        self,
        prompts: list[str | list[int]],
        sampling_params: SamplingParams | None = None,
    ) -> list[RequestOutput]:
        if sampling_params is None:
            sampling_params = SamplingParams()
        request_ids = []
        for prompt in prompts:
            rid = str(uuid.uuid4())
            req = Request(request_id=rid, prompt=prompt, sampling_params=sampling_params)
            if self.scheduler:
                self.scheduler.add_request(req)
            request_ids.append(rid)
        results: dict[str, RequestOutput] = {}
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

    async def generate_batch_async(
        self,
        prompts: list[str | list[int]],
        sampling_params: SamplingParams | None = None,
      ) -> list[RequestOutput]:
        """Non-blocking batch generation via asyncio.gather."""
        tasks = [self.generate(prompt, sampling_params) for prompt in prompts]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        outputs = []
        for i, r in enumerate(results):
            if isinstance(r, RequestOutput):
                outputs.append(r)
            else:
                logger.warning(f"generate_batch_async: prompt {i} failed: {r}")
        return outputs

    async def prefill(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams | None = None,
    ) -> dict[str, Any]:
        """Run prefill only: process prompt tokens, export KV state, skip decode."""
        request_id = await self.add_request(
            prompt=prompt, sampling_params=sampling_params,
            request_id=str(uuid.uuid4()),
        )
        sched = self.scheduler
        if not sched:
            raise RuntimeError("No scheduler for prefill")
        def _prefill_loop():
            for _ in range(1000):
                sched.step()
                req = sched.requests.get(request_id)
                if req is None:
                    break
                remaining = req.remaining_tokens if req.remaining_tokens is not None else []
                if len(remaining) == 0:
                    break

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._mlx_executor, _prefill_loop)
        kv_state = sched.export_kv_state(request_id)
        if kv_state is None:
            logger.warning("prefill %s: export_kv_state returned None", request_id)
        ctx = self._active_contexts.get(request_id)
        collector = ctx.collector if ctx else None
        final_output = None
        if collector:
            while True:
                output = collector.get_nowait()
                if output is None:
                    break
                final_output = output
        self._cleanup_request(request_id)
        return {
            "output": final_output,
            "kv_state": kv_state or {},
        }

    async def decode_with_handoff(
        self,
        prompt_token_ids: list[int],
        sampling_params: SamplingParams | None,
        kv_state: dict[str, Any],
    ) -> RequestOutput:
        """Start decode from prefill KV state, skip prefill entirely."""
        request_id = str(uuid.uuid4())
        await self.add_request(
            prompt=prompt_token_ids,
            sampling_params=sampling_params or SamplingParams(),
            request_id=request_id,
         )
        sched = self.scheduler
        if sched and kv_state:
            sched.import_kv_state(request_id, kv_state)
        ctx = self._active_contexts.get(request_id)
        if ctx is None:
            raise RuntimeError(f"No event for request {request_id}")
        collector = ctx.collector
        try:
            await ctx.finished_event.wait()
        except asyncio.CancelledError:
            logger.info("Request %s cancelled, aborting", request_id)
            await self.abort_request(request_id)
            self._cleanup_request(request_id)
            raise
        final_output = None
        while True:
            output = collector.get_nowait()
            if output is None:
                break
            final_output = output
        self._cleanup_request(request_id)
        if final_output is None:
            raise RuntimeError(f"No decode output for request {request_id}")
        if final_output.error:
            _raise_request_output_error(final_output)
        return final_output


    def get_stats(self) -> dict[str, Any]:
        scheduler_stats = self.scheduler.get_stats() if self.scheduler else {}
        uptime = time.time() - self._start_time if self._start_time else 0
        return {
            "running": self._running, "uptime_seconds": uptime,
            "steps_executed": self._steps_executed,
            "active_requests": len(self._active_contexts),
            "stream_interval": self.config.stream_interval, **scheduler_stats,
        }

    def get_cache_stats(self) -> dict[str, Any] | None:
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
        mgr = getattr(self.scheduler, "paged_ssd_cache_manager", None)
        if mgr is not None:
            try:
                mgr.close()
            except Exception:
                logger.debug("SSD cache manager close failed", exc_info=True)
        for fn in (self.scheduler.shutdown, self.scheduler.deep_reset) if self.scheduler else ():
            try:
                self._mlx_executor.submit(fn).result(
                    timeout=FATAL_TEARDOWN_TIMEOUT_S
                )
            except concurrent.futures.TimeoutError:
                fatal_exit(
                    f"scheduler teardown timed out after {FATAL_TEARDOWN_TIMEOUT_S}s"
                )
            except RuntimeError:
                try:
                    fn()
                except RuntimeError:
                    pass
        for ctx in list(self._active_contexts.values()):
            with suppress(Exception):
                ctx.collector.clear()
        self._active_contexts.clear()
        self._finished_at.clear()
        if self._mlx_executor is not None:
            if compile_cache_clear_available():
                try:
                    self._mlx_executor.submit(
                        clear_thread_compile_cache
                    ).result(timeout=FATAL_TEARDOWN_TIMEOUT_S)
                except concurrent.futures.TimeoutError:
                    fatal_exit(
                        "compile cache clear timed out after "
                        f"{FATAL_TEARDOWN_TIMEOUT_S}s"
                    )
            else:
                _immortal_mlx_executors.append(self._mlx_executor)
                if self._mlx_stream is not None:
                    _immortal_mlx_streams.append(self._mlx_stream)
                self._mlx_executor = None
                self._mlx_stream = None
                self.model = None
                self.tokenizer = None
                self.scheduler = None
                return
            self._mlx_executor.shutdown(wait=True)
            self._mlx_executor = None
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
    def __init__(self, model: Any, tokenizer: Any, config: EngineConfig | None = None):
        self.engine = EngineCore(model, tokenizer, config)
        self._start_task: asyncio.Task | None = None

    @property
    def _mlx_executor(self):
        return self.engine._mlx_executor

    async def __aenter__(self) -> "AsyncEngineCore":
        await self.engine.start()
        return self

    async def __aexit__(self, *args) -> None:
        await self.stop()

    def start(self) -> asyncio.Task:
        if self._start_task is not None and not self._start_task.done():
            return self._start_task
        self._start_task = asyncio.create_task(self.engine.start())
        return self._start_task

    async def stop(self) -> None:
        engine = getattr(self, "engine", None)
        if engine is None:
            return
        await engine.stop()

    async def add_request(self, prompt: str | list[int], sampling_params: SamplingParams | None = None, request_id: str | None = None, **kwargs) -> str:
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

    async def stream_outputs(self, request_id: str, timeout: float | None = None) -> AsyncIterator[RequestOutput]:
        async for output in self.engine.stream_outputs(request_id, timeout):
            yield output

    async def generate(self, prompt: str | list[int], sampling_params: SamplingParams | None = None, **kwargs) -> RequestOutput:
        return await self.engine.generate(prompt=prompt, sampling_params=sampling_params, **kwargs)

    def get_stats(self) -> dict[str, Any]:
        return self.engine.get_stats()

    def get_cache_stats(self) -> dict[str, Any] | None:
        return self.engine.get_cache_stats()
