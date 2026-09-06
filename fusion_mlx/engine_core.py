# SPDX-License-Identifier: Apache-2.0
"""Engine Core for fusion-mlx continuous batching."""

import asyncio
import concurrent.futures
import logging
import os
import threading
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx

from .exceptions import PrefillMemoryExceededError
from .model_registry import get_registry
from .output_collector import RequestOutputCollector, RequestStreamState
from .request import Request, RequestOutput, SamplingParams
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


def _resolve_video_max_workers() -> int:
    # E-2 (#811): video max_workers=1 was the ONLY thing preventing
    # concurrent diffusion OOM. It was implicit — a future "performance"
    # PR bumping it to 2 would OOM with no warning. Make it an explicit,
    # documented knob (env override for power users) so the serial-invariant
    # is visible, not accidental. Default stays 1.
    raw = os.environ.get("FUSION_MLX_MAX_CONCURRENT_VIDEO", "").strip()
    if not raw:
        return 1
    try:
        n = int(raw)
        if n >= 1:
            logger.warning(
                "FUSION_MLX_MAX_CONCURRENT_VIDEO=%d: concurrent video "
                "generation risks Metal OOM (stage-1 latent + 22B "
                "transformer + VAE per concurrent job)",
                n,
            )
            return n
    except ValueError:
        pass
    return 1


_executor_config: dict[str, dict[str, Any]] = {
    "llm": {"max_workers": 1, "prefix": "mlx-llm"},
    "image": {"max_workers": 1, "prefix": "mlx-image"},
    "video": {"max_workers": _resolve_video_max_workers(), "prefix": "mlx-video"},
    # audio must be max_workers=1: mlx-audio's Metal Stream is thread-local,
    # so load_model() and generate() must run on the same thread (else
    # "no Stream(gpu, N) in current thread").
    "audio": {"max_workers": 1, "prefix": "mlx-audio"},
    "io": {"max_workers": 2, "prefix": "mlx-io"},
}
_global_executors: dict[str, concurrent.futures.ThreadPoolExecutor] = {}


def _init_mlx_step_thread() -> None:
    # model load + scheduler creation + step all run on this thread (the
    # executor is reused as EngineCore._mlx_executor via AsyncEngineCore(
    # executor=...)). Must set generation_stream here too (same as
    # _init_mlx_thread) so mlx-lm BatchGenerator prefill uses the same
    # default stream model weights are bound to (#KV-0). Prior empty pass
    # left generation_stream unset -> cross-stream "no Stream(gpu, 0)".
    _init_mlx_thread()


def _init_mlx_thread() -> None:
    # Use the thread's DEFAULT stream (same stream model load binds weights to)
    # rather than a separate new_thread_local_stream. MLX 0.31.3+ binds model
    # weights to the stream that first touches them at load (the worker's
    # default stream 0). mlx-lm BatchGenerator runs prefill on generation_stream;
    # if that is a different stream (new_thread_local_stream, non-zero index),
    # cross-stream weight access raises "There is no Stream(gpu, 0) in current
    # thread" on every request (#KV-0).
    stream = mx.default_stream(mx.default_device())
    import sys

    gen_mod = sys.modules.get("mlx_lm.generate")
    if gen_mod is not None:
        gen_mod.generation_stream = stream
    sched_mod = sys.modules.get("fusion_mlx.scheduler")
    if sched_mod is not None:
        sched_mod.generation_stream = stream
    # VLM: mlx_vlm 前向也跑在 executor 线程, 需绑定同一线程局部 Metal Stream
    # (遗漏致 "There is no Stream(gpu, 1) in current thread" prefill 报错)
    vlm_gen_mod = sys.modules.get("mlx_vlm.generate")
    if vlm_gen_mod is not None:
        vlm_gen_mod.generation_stream = stream
    logger.debug("MLX executor thread initialized: generation_stream = %s", stream)


def get_executor(pool_type: str = "llm") -> concurrent.futures.ThreadPoolExecutor:
    if pool_type in _global_executors:
        return _global_executors[pool_type]
    cfg = _executor_config.get(
        pool_type, {"max_workers": 1, "prefix": f"mlx-{pool_type}"}
    )
    exec_ = concurrent.futures.ThreadPoolExecutor(
        max_workers=cfg["max_workers"],
        thread_name_prefix=cfg["prefix"],
        initializer=_init_mlx_thread,
    )
    _global_executors[pool_type] = exec_
    return exec_


def get_mlx_executor() -> concurrent.futures.ThreadPoolExecutor:
    return get_executor("llm")


# Video diffusion is long-running: SkyReels-V3 R2V 14B 720p 30-step is ~1hr
# (~115s/step x 30 + VAE decode). The prior hardcoded 600s (10min) ceiling
# killed in-progress jobs via TimeoutError before they could finish (#148).
# Default 7200s (2hr) covers 720p 30-step + VAE with headroom; override via
# FUSION_VIDEO_GEN_TIMEOUT (seconds). Invalid/non-positive values fall back
# to the default with a warning.
_VIDEO_GEN_TIMEOUT_DEFAULT_S = 7200.0


def get_video_gen_timeout() -> float:
    raw = os.environ.get("FUSION_VIDEO_GEN_TIMEOUT")
    if not raw:
        return _VIDEO_GEN_TIMEOUT_DEFAULT_S
    try:
        val = float(raw)
    except (TypeError, ValueError):
        logger.warning(
            "FUSION_VIDEO_GEN_TIMEOUT=%r is not a number, using default %.0fs",
            raw,
            _VIDEO_GEN_TIMEOUT_DEFAULT_S,
        )
        return _VIDEO_GEN_TIMEOUT_DEFAULT_S
    if val <= 0:
        logger.warning(
            "FUSION_VIDEO_GEN_TIMEOUT=%r <= 0, using default %.0fs",
            raw,
            _VIDEO_GEN_TIMEOUT_DEFAULT_S,
        )
        return _VIDEO_GEN_TIMEOUT_DEFAULT_S
    return val


# Poisoned-executor registry (#811 R-3). A hung video job on the single
# max_workers=1 video executor cannot be cancelled from Python (the worker
# thread keeps running the MLX pipeline), so every subsequent video request
# would queue and time out, making the video subsystem dead until process
# restart. Rather than silently queueing forever, we mark the executor
# poisoned the first time a generation times out (or runs past a watchdog
# deadline) and fast-fail later requests with a loud "restart required"
# error. A new executor replaces the old one in _global_executors so a fresh
# request starts a fresh worker thread (it will reload model weights on its
# own thread-local stream). The poisoned flag keeps anything still enqueued
# on the old executor from being mistaken for live work.
_video_executor_poisoned = False
_video_executor_poison_lock = threading.Lock()


def is_video_executor_poisoned() -> bool:
    return _video_executor_poisoned


def reset_video_executor_poison() -> None:
    # Called after a confirmed clean stop/restart of the video engine.
    global _video_executor_poisoned
    with _video_executor_poison_lock:
        _video_executor_poisoned = False


def poison_executor(pool_type: str = "video") -> None:
    # Mark a pool's executor poisoned and swap in a replacement so new work
    # is not queued behind a stuck worker (#811 R-3). The old executor is
    # NOT shut down (shutting it down would block on the hung thread); it is
    # abandoned to the process lifetime. MLX weights are thread-local, so
    # the replacement worker reloads on first use.
    global _video_executor_poisoned
    with _video_executor_poison_lock:
        if pool_type == "video" and _video_executor_poisoned:
            return
        exec_ = _global_executors.get(pool_type)
        if exec_ is None:
            return
        cfg = _executor_config.get(
            pool_type, {"max_workers": 1, "prefix": f"mlx-{pool_type}"}
        )
        new_exec = concurrent.futures.ThreadPoolExecutor(
            max_workers=cfg["max_workers"],
            thread_name_prefix=cfg["prefix"],
            initializer=_init_mlx_thread,
        )
        _global_executors[pool_type] = new_exec
        if pool_type == "video":
            _video_executor_poisoned = True
            logger.error(
                "video executor POISONED (#811 R-3): a generation hung and could "
                "not be cancelled. New worker spawned; the stuck thread is "
                "abandoned. RESTART fusion-mlx to reclaim its memory. "
                "Subsequent video requests will reload on the fresh worker."
            )
        else:
            logger.error("%s executor replaced after a hung job (#811 R-3).", pool_type)


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
        default_factory=lambda: int(
            os.environ.get("FUSION_DECODE_BURST_MAX_STEPS", "16")
        )
    )
    decode_burst_budget_single_s: float = field(
        default_factory=lambda: float(
            os.environ.get("FUSION_DECODE_BURST_BUDGET_SINGLE_S", "0.5")
        )
    )
    decode_burst_budget_s: float = field(
        default_factory=lambda: float(
            os.environ.get("FUSION_DECODE_BURST_BUDGET_S", "0.1")
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
        executor: Any = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or EngineConfig()
        self._engine_id = engine_id or str(uuid.uuid4())
        self._owns_model = False
        self._closed = False
        self._idle_event = None

        registry = get_registry()
        registry.acquire(
            model=model,
            engine=self,
            engine_id=self._engine_id,
            force=force_model_ownership,
        )
        self._owns_model = True

        # Per-engine executor with dedicated mx.Stream (#1248).
        # Each EngineCore gets its own thread + GPU stream so different
        # models can run scheduler.step() concurrently.
        # E-12 (#811): do NOT create the stream here on the main thread —
        # _make_scheduler() reassigns self._mlx_stream to the executor
        # thread's default stream (the one weights bind to). A stream
        # created here on the main thread would be leaked (never closed),
        # accumulating across reload churn. Defer; the scheduler create on
        # the executor thread sets the canonical stream.
        self._mlx_stream = None
        if executor is not None:
            # Reuse caller-provided executor (BatchedEngine._start_llm's
            # _model_load_executor) so scheduler creation + model load + step
            # run on the SAME thread. MLX 0.31.3+ binds model weights to the
            # stream of the thread that first touches them; creating the
            # scheduler on a different thread than load -> cross-thread weight
            # access -> "There is no Stream(gpu, N) in current thread" (#KV-0).
            self._mlx_executor = executor
        else:
            self._mlx_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix=f"mlx-engine-{self._engine_id[:8]}",
                initializer=_init_mlx_thread,
            )

        # Scheduler must be created on the executor thread so it uses the
        # thread-local MLX stream (not the main thread's stream).
        from .scheduler import Scheduler, SchedulerConfig

        scheduler_config = self.config.scheduler_config or SchedulerConfig()
        self.scheduler: Scheduler | None = None
        _sched_result: list = []

        def _make_scheduler():
            # Resolve the stream ON the executor thread so it is the SAME
            # default stream model load used there. MLX 0.31.3+ binds weights
            # to the load thread's default stream; using a different stream
            # (new_thread_local_stream created in __init__ on the main thread)
            # -> cross-stream "There is no Stream(gpu, N) in current thread"
            # on every forward (#KV-0).
            self._mlx_stream = mx.default_stream(mx.default_device())
            _sched_result.append(
                Scheduler(
                    model=model,
                    tokenizer=tokenizer,
                    config=scheduler_config,
                    stream=self._mlx_stream,
                )
            )

        _fut = self._mlx_executor.submit(_make_scheduler)
        try:
            _fut.result()
        except BaseException:
            # E-13 (#811): _make_scheduler raised (OOM, bad weights, config
            # error). The registry acquired the model above, but without a
            # scheduler the engine is unusable and start() would never be
            # reached to clean up. Release the model now so it is not
            # orphaned in the registry, then re-raise the original error.
            self._owns_model = False
            try:
                get_registry().release(self.model, self._engine_id)
            except Exception:
                logger.debug("registry release on failed init", exc_info=True)
            raise
        self.scheduler = _sched_result[0]

        # Draft-model speculative decode safety gate. The draft-model verify
        # path (``model([D1..DK], cache)``) has not been audited for hybrid
        # recurrent architectures (GatedDeltaNet / Mamba / ArraysCache layers),
        # so it stays disabled for recurrent models until its rejection path is
        # audited the same way n-gram spec's was (see the n-gram block below).
        # N-gram spec is NOT gated here — its rejection path is GDN-safe after
        # the trim + resample fixes. Uses the shared
        # ``model_has_recurrent_cache`` helper (same probe
        # ``enrich_model_config`` runs) so the boot gate and config gate can't
        # drift apart.
        from .model_auto_config import model_has_recurrent_cache

        spec_eligible = not model_has_recurrent_cache(model)
        if not spec_eligible:
            logger.info(
                "Draft-model speculative decode disabled: model has recurrent "
                "(ArraysCache) layers. N-gram spec remains enabled (GDN-safe)."
            )

        # Initialize speculative decode draft model on the executor thread
        from .scheduler.spec_decode import SPEC_DRAFT_MODEL_ENABLED, SpecDecodeState

        if spec_eligible and SPEC_DRAFT_MODEL_ENABLED:

            def _init_draft():
                spec_method = os.environ.get(
                    "FUSION_SPEC_METHOD", "draft_model"
                ).lower()
                draft = None
                if spec_method == "eagle3":
                    from .speculative.eagle3 import Eagle3Speculator

                    draft = Eagle3Speculator()
                    logger.info("Speculative decode: using EAGLE3 method")
                else:
                    from .speculative.draft_model import DraftModelDecoder

                    draft = DraftModelDecoder()

                loaded = draft.load()
                if loaded:
                    # model_name is an EngineConfig dataclass field, not an
                    # EngineCore attribute — use self.config.model_name.
                    target_name = getattr(self.config, "model_name", "") or ""
                    # Safety guard: refuse to spec-decode if the Eagle3
                    # draft family does not match the loaded target model
                    # (e.g. EAGLE3-LLaMA3 against a Qwen target). Produces
                    # garbage drafts silently otherwise.
                    if (
                        spec_method == "eagle3"
                        and hasattr(draft, "is_compatible")
                        and not draft.is_compatible(target_name)
                    ):
                        logger.warning(
                            "Speculative decode: eagle3 draft incompatible with "
                            "target model %r, disabling spec decode",
                            target_name,
                        )
                        return
                    hidden_capture = None
                    if spec_method == "eagle3" and hasattr(model, "model"):
                        target_embed = getattr(model.model, "embed_tokens", None)
                        if target_embed is not None:
                            draft.bind_target_embed_from_model(target_embed)
                        capture_layers = getattr(draft, "capture_layers", [8, 16, 31])
                        from .speculative.hidden_capture import HiddenStateCapture

                        hidden_capture = HiddenStateCapture(
                            model, layer_ids=capture_layers
                        )
                        hidden_capture.install()
                        draft.set_hidden_capture(hidden_capture)
                        logger.info(
                            "Speculative decode: eagle3 hidden_capture installed layers=%s",
                            capture_layers,
                        )
                    self.scheduler._spec_decode_state = SpecDecodeState(
                        draft_model_decoder=draft,
                        hidden_capture=hidden_capture,
                    )
                    logger.info(
                        "Speculative decode: draft model enabled (%s, method=%s, temp=%s)",
                        draft.model_path if hasattr(draft, "model_path") else "?",
                        spec_method,
                        getattr(draft.config, "temperature", "?"),
                    )
                else:
                    logger.info(
                        "Speculative decode: draft model failed to load, disabled"
                    )

            _fut = self._mlx_executor.submit(_init_draft)
            _fut.result()

        # Initialize n-gram speculative decode (CPU-side, zero GPU overhead).
        # GDN-safe: the batched verify ``model([D1..DK], cache)`` and the GDN
        # layer/kernel are correct for multi-token forwards (proven by layer
        # isolation tests — batched S=K equals sequential K×S=1 for conv_state,
        # ssm_state and output). The earlier corruption ("1,2,...,11,21,24,93"
        # repetition on count tasks) was traced to TWO rejection-path bugs in
        # ``_verify_drafts`` / ``ngram_spec_step``, both now fixed:
        #   (1) KVCache was not trimmed on rejection —
        #       ``mlx_cache.trim_prompt_cache`` is a no-op when ANY cache is
        #       non-trimmable (hybrid GDN models have ArraysCache layers), and
        #       it trimmed ``n_rejected`` instead of ``K`` (the replay writes
        #       ``n_accepted`` duplicate entries that must also be dropped). The
        #       rejection path now trims each trimmable cache directly by ``K``.
        #   (2) The bonus token used ``resample_idx = min(n_accepted, K-1)``,
        #       which on rejection selects the prediction AFTER the first
        #       rejected draft; the correct index is ``n_accepted - 1`` (pred
        #       after the last ACCEPTED draft).
        # With both fixed, n-gram spec is token-for-token coherent with pure
        # decode on the GDN model (verified standalone, K=3, count-1-to-50).
        # Losslessness is covered by tests/integration/test_ngram_spec_gdn_coherence.py.
        from .scheduler.ngram_spec import NGRAM_SPEC_ENABLED, NGramSpecState

        if NGRAM_SPEC_ENABLED:
            self.scheduler._ngram_spec_state = NGramSpecState()
            logger.info(
                "N-gram speculative decode: enabled (order=%d, num_draft=%d)",
                self.scheduler._ngram_spec_state.predictor.order,
                self.scheduler._ngram_spec_state.predictor.num_draft,
            )

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
        # P2-7: consecutive-error counter for the engine-loop circuit breaker.
        self._consecutive_loop_errors = 0
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
            with suppress(asyncio.CancelledError, asyncio.TimeoutError):
                # E-6 (#811): bound the wait for the cancelled engine loop.
                # A long prefill/video step on the executor keeps _task alive
                # past cancellation; an unbounded await here blocks close()'s
                # scheduler.shutdown submit behind it, which then hits its own
                # 60s timeout and fatal-exits, skipping the rest of teardown.
                # Bounded wait lets close() proceed to teardown instead of
                # stalling the whole shutdown.
                await asyncio.wait_for(self._task, timeout=5.0)
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
                # M4: reduced throttle from 1s to 0.2s; hard cap for burst.
                now = time.monotonic()
                force_reap = len(self._finished_at) > 100
                if force_reap or now - self._last_reap >= 0.2:
                    self._last_reap = now
                    self._reap_orphaned_collectors(now)

                if self.scheduler.has_requests():
                    step_outputs = await loop.run_in_executor(
                        self._mlx_executor, self._step_burst
                    )
                    self._steps_executed += len(step_outputs)
                    # P2-7: a successful step resets the circuit breaker.
                    if self._consecutive_loop_errors:
                        self._consecutive_loop_errors = 0

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
                    if has_streaming_consumer or any(o.outputs for o in step_outputs):
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
                            await asyncio.wait_for(event.wait(), timeout=step_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                import traceback

                logger.error("Engine loop error: %s\n%s", e, traceback.format_exc())

                # P2-7: circuit breaker. A persistent scheduler failure (e.g.
                # corrupted KV, broken model forward) would otherwise spin the
                # loop forever — fail_all_requests empties waiting/running so
                # has_requests() returns False, but a re-submitted or stuck
                # request re-triggers the same fault each iteration, burning
                # CPU and flooding logs. After N consecutive errors, stop the
                # loop loudly so the pool can evict and reload the engine.
                # E-7 (#811): the counter is global (not per-fault-signature)
                # so unrelated transient errors (e.g. 50 different prompts
                # each OOMing once) can trip it. Mitigate by keeping the
                # threshold modest — a genuine persistent fault repeats on
                # the SAME re-submitted request and crosses quickly, while
                # 20 independent one-shot transients are rare in practice.
                # The counter resets on every successful step, so a healthy
                # engine never accumulates.
                self._consecutive_loop_errors += 1
                if self._consecutive_loop_errors >= 20:
                    logger.critical(
                        "Engine loop hit %d consecutive errors — stopping "
                        "engine to break persistent-failure spin",
                        self._consecutive_loop_errors,
                    )
                    self._running = False
                    try:
                        failed_ids = await loop.run_in_executor(
                            self._mlx_executor, self.scheduler.fail_all_requests
                        )
                    except Exception:
                        failed_ids = []
                    for rid in failed_ids:
                        ctx = self._active_contexts.get(rid)
                        if ctx is not None:
                            try:
                                ctx.collector.put(
                                    RequestOutput(
                                        request_id=rid,
                                        finished=True,
                                        finish_reason="error",
                                        error="engine loop stopped: repeated failures",
                                    )
                                )
                            except Exception:
                                pass
                        self._mark_request_finished(rid)
                    # R-23 (#811): sweep any context fail_all_requests missed
                    # so no consumer hangs on an un-set finished_event.
                    leaked = self._fail_unfinished_contexts(
                        "engine loop stopped: repeated failures"
                    )
                    if leaked:
                        logger.critical(
                            "R-23: %d active context(s) missed by "
                            "fail_all_requests — force-finished to avoid hang",
                            leaked,
                        )
                    return

                # Fail all requests and remove from scheduler to prevent
                # infinite loop (has_requests() must return False).
                def _safe_fail():
                    try:
                        return self.scheduler.fail_all_requests()
                    except Exception:
                        return []

                failed_ids = await loop.run_in_executor(self._mlx_executor, _safe_fail)
                for rid in failed_ids:
                    ctx = self._active_contexts.get(rid)
                    if ctx is not None:
                        ctx.collector.put(
                            RequestOutput(
                                request_id=rid,
                                finished=True,
                                finish_reason="error",
                                error=str(e),
                            )
                        )
                    self._mark_request_finished(rid)
                # R-23 (#811): sweep any context fail_all_requests missed.
                leaked = self._fail_unfinished_contexts(str(e))
                if leaked:
                    logger.critical(
                        "R-23: %d active context(s) missed by "
                        "fail_all_requests — force-finished to avoid hang",
                        leaked,
                    )
                await asyncio.sleep(0.1)
            except BaseException as e:
                # P0-2: KeyboardInterrupt/SystemExit from inside scheduler.step
                # (run on the MLX executor) bypass the Exception branch above.
                # Without this, in-flight requests hang forever: their
                # _active_contexts entries are never cleaned and finished_event
                # is never set. Fail everything loudly, then re-raise so the
                # process interrupt propagates normally.
                logger.error("Engine loop terminating on %r", e)
                self._running = False
                try:
                    failed_ids = await loop.run_in_executor(
                        self._mlx_executor, self.scheduler.fail_all_requests
                    )
                except Exception:
                    failed_ids = []
                for rid in failed_ids:
                    ctx = self._active_contexts.get(rid)
                    if ctx is not None:
                        try:
                            ctx.collector.put(
                                RequestOutput(
                                    request_id=rid,
                                    finished=True,
                                    finish_reason="error",
                                    error=f"engine loop terminated: {e!r}",
                                )
                            )
                        except Exception:
                            pass
                    self._mark_request_finished(rid)
                # R-23 (#811): sweep any context fail_all_requests missed.
                leaked = self._fail_unfinished_contexts(
                    f"engine loop terminated: {e!r}"
                )
                if leaked:
                    logger.critical(
                        "R-23: %d active context(s) missed by "
                        "fail_all_requests — force-finished to avoid hang",
                        leaked,
                    )
                raise

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
        resume_prompt_cache: list | None = None,
        resume_cached_tokens: int = 0,
    ) -> str:
        # P1-7: reject new requests once stop() has torn the engine down.
        # Without this guard add_request enqueues into the scheduler after the
        # engine loop is cancelled, so no one processes the request and the
        # caller hangs forever waiting for output.
        if not self._running:
            logger.error(
                "add_request rejected: engine not running (request_id=%s)",
                request_id,
            )
            raise RuntimeError("engine is not running; request rejected")
        if request_id is None:
            request_id = str(uuid.uuid4())
        if sampling_params is None:
            sampling_params = SamplingParams()

        request = Request(
            request_id=request_id,
            prompt=prompt,
            sampling_params=sampling_params,
            images=images,
            videos=videos,
            vlm_inputs_embeds=vlm_inputs_embeds,
            vlm_extra_kwargs=vlm_extra_kwargs,
            vlm_image_hash=vlm_image_hash,
            vlm_cache_key_start=vlm_cache_key_start,
            vlm_cache_key_ranges=vlm_cache_key_ranges,
        )
        # Disconnect KV resume: seed externally-loaded KV (from a disk
        # checkpoint written on a prior disconnect) so add_request's
        # prefix-cache-prep is skipped and the cached tail survives.
        if resume_prompt_cache is not None and resume_cached_tokens > 0:
            request.prompt_cache = list(resume_prompt_cache)
            request.cached_tokens = int(resume_cached_tokens)
        if specprefill is not None:
            request._specprefill_enabled = specprefill
        elif (
            self.scheduler
            and getattr(self.scheduler, "_specprefill_draft_model", None) is not None
        ):
            request._specprefill_enabled = True
        if specprefill_keep_pct is not None:
            request._specprefill_keep_pct = specprefill_keep_pct
        if specprefill_threshold is not None:
            request._specprefill_threshold = specprefill_threshold
        if specprefill_system_end is not None and specprefill_system_end > 0:
            request.specprefill_system_end = specprefill_system_end

        logger.info(
            "add_request: id=%s, prompt=%r, max_tokens=%d",
            request_id,
            str(prompt)[:100] if isinstance(prompt, str) else f"tokens({len(prompt)})",
            sampling_params.max_tokens,
        )
        self._active_contexts[request_id] = RequestContext(
            collector=RequestOutputCollector(aggregate=not streaming),
            stream_state=RequestStreamState(
                stream_interval=self.config.stream_interval
            ),
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
                # P0-3: route the abort through the SAME executor step() runs
                # on, so abort_request cannot mutate scheduler.waiting/running
                # concurrently with a step() mid-iteration on the executor
                # thread (deque popleft/append are not atomic). Blocking here
                # is acceptable: we are about to re-raise anyway.
                try:
                    fut = self._mlx_executor.submit(
                        self.scheduler.abort_request, request_id
                    )
                    fut.result(timeout=5.0)
                except Exception as abort_exc:
                    logger.debug(
                        "Abort of partial insert for %s failed: %s",
                        request_id,
                        abort_exc,
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
            ctx.collector.put(
                RequestOutput(
                    request_id=request_id,
                    finished=True,
                    finish_reason="abort",
                    error="Request aborted",
                )
            )
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
                ctx.collector.put(
                    RequestOutput(
                        request_id=rid,
                        finished=True,
                        finish_reason="error",
                        new_text=f"\n\n[Error: {error_msg}]",
                        error=error_msg,
                    )
                )
            self._mark_request_finished(rid)
        if request_ids:
            logger.warning(
                "Aborted %d requests due to memory pressure", len(request_ids)
            )
            self._wake_engine_loop()
        return len(request_ids)

    def _cleanup_request(self, request_id: str) -> None:
        ctx = self._active_contexts.pop(request_id, None)
        if ctx:
            ctx.collector.clear()
        self._finished_at.pop(request_id, None)

    def _fail_unfinished_contexts(self, error_msg: str) -> int:
        # R-23 (#811): fail_all_requests only reports rids it found in the
        # scheduler queues. A rid can exist in _active_contexts but be missed
        # — narrow races during add_request's executor insert, scheduler queue
        # inconsistency mid-step, or a request that finished in the scheduler
        # but whose collector never received a terminal output. Without this
        # sweep those ctx's finished_event stays unset and generate() hangs
        # forever. Push a terminal error output to every context that has not
        # yet been marked finished, so no waiting consumer is left dangling.
        # Pop-only-reap-safe: a live consumer holds its own collector ref, so
        # a put() here is harmless even if the dict entry was already reaped.
        n = 0
        for rid, ctx in list(self._active_contexts.items()):
            if ctx.finished_event.is_set():
                continue
            try:
                ctx.collector.put(
                    RequestOutput(
                        request_id=rid,
                        finished=True,
                        finish_reason="error",
                        error=error_msg,
                    )
                )
            except Exception:
                pass
            self._mark_request_finished(rid)
            n += 1
        return n

    def _mark_request_finished(self, request_id: str) -> None:
        """Stamp finish time and signal the consumer.

        The timestamp lets _reap_orphaned_collectors() drop collectors whose
        consumer never cleaned up (e.g. client disconnected mid-stream).
        """
        self._finished_at.setdefault(request_id, time.monotonic())
        ctx = self._active_contexts.get(request_id)
        if ctx is not None:
            ctx.finished_event.set()

    def _reap_orphaned_collectors(self, now: float, grace: float | None = None) -> int:
        """Drop tracking for finished requests whose consumer never cleaned up.

        Pop-only: never clear() the collector object. A live consumer holds its
        own reference, so dropping the dict entry cannot truncate output.
        grace defaults to _orphan_reap_grace (overridable per engine type, e.g.
        video/long-streaming) so slow consumers are not starved.
        """
        if grace is None:
            grace = getattr(self, "_orphan_reap_grace", 5.0)
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
                len(stale),
                stale,
            )
        return len(stale)

    async def stream_outputs(
        self, request_id: str, timeout: float | None = None
    ) -> AsyncIterator[RequestOutput]:
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
                            # E-5 (#811): closed collector returns None.
                            if output is None:
                                logger.info(
                                    "stream_outputs collector closed for %s, stopping",
                                    request_id,
                                )
                                break
                    else:
                        output = collector.get_nowait() or await collector.get()
                    # E-5 (#811): a reaped/closed collector returns None from
                    # get() — stop the stream instead of yielding None.
                    if output is None:
                        logger.info(
                            "stream_outputs collector closed for %s, stopping",
                            request_id,
                        )
                        break
                    yield output
                    if output.error:
                        _raise_request_output_error(output)
                    if output.finished:
                        logger.info(
                            "stream_outputs done: %s, finish=%s, tokens=%d",
                            request_id,
                            output.finish_reason,
                            output.completion_tokens,
                        )
                        break
                except TimeoutError:
                    logger.warning("Timeout waiting for request %s", request_id)
                    break
        finally:
            # P2-4: if the consumer disconnected before the request finished
            # (generator closed/cancelled mid-stream), abort the scheduler
            # request so decode stops burning tokens for a dead client. Only
            # abort when still active — a finished request is already cleaned.
            ctx_after = self._active_contexts.get(request_id)
            if ctx_after is not None and not ctx_after.finished_event.is_set():
                logger.info(
                    "stream_outputs client disconnect before finish, " "aborting %s",
                    request_id,
                )
                try:
                    await self.abort_request(request_id)
                except Exception as e:
                    logger.warning(
                        "abort_request on stream disconnect %s failed: %s",
                        request_id,
                        e,
                    )
            self._cleanup_request(request_id)

    async def generate(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams | None = None,
        request_id: str | None = None,
        **kwargs,
    ) -> RequestOutput:
        request_id = await self.add_request(
            prompt=prompt,
            sampling_params=sampling_params,
            request_id=request_id,
            **kwargs,
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
        # P3 (#811): this sync path drives scheduler.step() directly,
        # bypassing the AsyncEngineCore continuous-batching loop, the MLX
        # executor, streaming, abort-on-disconnect, and the error handlers.
        # It has no production callers (not in public_api) — it is a
        # bench/test convenience. Prefer generate_batch_async / generate.
        logger.warning(
            "generate_batch_sync bypasses the executor + streaming + "
            "abort/error handling; prefer generate_batch_async for "
            "production paths"
        )
        if sampling_params is None:
            sampling_params = SamplingParams()
        request_ids = []
        for prompt in prompts:
            rid = str(uuid.uuid4())
            req = Request(
                request_id=rid, prompt=prompt, sampling_params=sampling_params
            )
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
            prompt=prompt,
            sampling_params=sampling_params,
            request_id=str(uuid.uuid4()),
        )
        sched = self.scheduler
        if not sched:
            raise RuntimeError("No scheduler for prefill")

        def _prefill_loop():
            # VLM: mlx_vlm.generation_stream 是模块级单例 (import 时绑主线程),
            # executor 线程跑 prefill 前需显式注入线程局部 stream 避 "There is no Stream(gpu,1)" 报错
            import sys as _sys

            _vlm_gen = _sys.modules.get("mlx_vlm.generate")
            if _vlm_gen is not None:
                _vlm_gen.generation_stream = self._mlx_stream
            for _ in range(1000):
                sched.step()
                req = sched.requests.get(request_id)
                if req is None:
                    break
                remaining = (
                    req.remaining_tokens if req.remaining_tokens is not None else []
                )
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
            "running": self._running,
            "uptime_seconds": uptime,
            "steps_executed": self._steps_executed,
            "active_requests": len(self._active_contexts),
            "stream_interval": self.config.stream_interval,
            **scheduler_stats,
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
        for fn in (
            (self.scheduler.shutdown, self.scheduler.deep_reset)
            if self.scheduler
            else ()
        ):
            try:
                self._mlx_executor.submit(fn).result(timeout=FATAL_TEARDOWN_TIMEOUT_S)
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
                    self._mlx_executor.submit(clear_thread_compile_cache).result(
                        timeout=FATAL_TEARDOWN_TIMEOUT_S
                    )
                except concurrent.futures.TimeoutError:
                    fatal_exit(
                        "compile cache clear timed out after "
                        f"{FATAL_TEARDOWN_TIMEOUT_S}s"
                    )
                except RuntimeError:
                    pass
            else:
                # E-10 (#811): the executor must stay alive (its worker thread
                # holds a thread-local MLX Stream + CompilerCache that cannot
                # be torn down without a GIL-free crash). ThreadPoolExecutor
                # has no API to cancel queued futures WITHOUT marking the
                # executor shut down — shutdown(cancel_futures=True) sets
                # _shutdown=True, which would reject future submits and
                # contradict the "immortal, reusable" intent. So leave the
                # executor fully alive. Queued-but-not-started futures carry
                # no MLX state yet and are harmless to let drain on the
                # worker thread; the executor is pinned in
                # _immortal_mlx_executors and never re-submitted to after
                # close (self._mlx_executor is nulled below).
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
        except Exception as e:
            # P3 (#811): bare pass swallowed the release error silently.
            # __del__ runs at GC time so a raised exception is uncatchable
            # and would just print to stderr; log the real cause at debug
            # (release failures during interpreter shutdown are common and
            # benign, but a real bug should still be traceable).
            logger.debug("EngineCore.__del__ model release failed: %s", e)

    @property
    def engine_id(self) -> str:
        return self._engine_id


class AsyncEngineCore:
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        config: EngineConfig | None = None,
        *,
        executor: Any | None = None,
    ):
        # AtomCode fix #113: 补 executor kwarg 对齐 test_batching_deterministic.py:113 (2026-07-19)
        # 原签名缺 executor 致 TypeError: unexpected keyword argument 'executor'
        # executor 透到 EngineCore._mlx_executor (Metal 流调度), None 时 EngineCore 内默自建
        self.engine = EngineCore(model, tokenizer, config, executor=executor)
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

    async def add_request(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams | None = None,
        request_id: str | None = None,
        **kwargs,
    ) -> str:
        return await self.engine.add_request(
            prompt=prompt,
            sampling_params=sampling_params,
            request_id=request_id,
            **kwargs,
        )

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

    async def stream_outputs(
        self, request_id: str, timeout: float | None = None
    ) -> AsyncIterator[RequestOutput]:
        async for output in self.engine.stream_outputs(request_id, timeout):
            yield output

    async def generate(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams | None = None,
        **kwargs,
    ) -> RequestOutput:
        return await self.engine.generate(
            prompt=prompt, sampling_params=sampling_params, **kwargs
        )

    def get_stats(self) -> dict[str, Any]:
        return self.engine.get_stats()

    def get_cache_stats(self) -> dict[str, Any] | None:
        return self.engine.get_cache_stats()
