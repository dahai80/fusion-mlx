# SPDX-License-Identifier: Apache-2.0
"""SmartRouter — phase-aware routing with cross-engine handoff.

Replaces the simple RequestRouter with a decision tree that considers:
- prompt_length:  token count of new (uncached) content
- batch_size:    concurrent request count
- task_tag:      claude_code (realtime) / openclaw (batch) / background
- cache_hit_rate: prefix cache hit ratio
- quant_format:  model quantization format (for benchmark-based routing)

Key innovation: prefill and decode can route to different engines.
E.g., prefill goes to omlx (strong matmul), decode to Rapid-MLX (lightweight KV).
"""

from __future__ import annotations

import enum
import logging
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Optional

logger = logging.getLogger(__name__)


class TaskPriority(enum.Enum):
    """Request priority levels, mapped to Metal command queues."""
    REALTIME = "realtime"      # Claude Code — low latency, high priority
    BATCH = "batch"             # OpenClaw — throughput-oriented
    BACKGROUND = "background"    # Embedding, reranking, offline


class EngineBackend(enum.Enum):
    """Available inference backends."""
    OMLX = "omlx"              # omlx — strong matmul, heavy batching
    RAPID = "rapid"            # Rapid-MLX — lightweight KV, low latency
    AUTO = "auto"              # Let SmartRouter decide
    CLOUD = "cloud"            # Cloud fallback


@dataclass
class RouteDecision:
    """The result of a routing decision."""
    prefill_backend: EngineBackend
    decode_backend: EngineBackend
    reason: str
    split_phases: bool = False  # True when prefill != decode

    @property
    def unified_backend(self) -> bool:
        return self.prefill_backend == self.decode_backend


@dataclass
class BenchmarkResult:
    """Kernel benchmark result for a model + backend combo."""
    model_id: str
    backend: EngineBackend
    quant_format: str
    tps: float                    # tokens per second
    latency_p50: float           # ms
    latency_p99: float           # ms
    memory_peak_bytes: int
    timestamp: float = field(default_factory=time.time)


@dataclass
class RouterConfig:
    """SmartRouter tuning parameters."""

    # Prefill/decode split threshold (tokens)
    # Above this, consider splitting prefill and decode to different engines
    phase_split_threshold: int = 8192

    # Cloud fallback threshold (uncached tokens)
    cloud_fallback_threshold: int = 32768

    # Enable benchmark-based routing (run kernel benchmarks at load time)
    enable_benchmark_routing: bool = True

    # Cache for benchmark results (model_id -> {backend -> result})
    benchmark_cache: dict[str, dict[str, BenchmarkResult]] = field(default_factory=dict)

    # Task tag to priority mapping (header/param -> priority)
    default_priority: TaskPriority = TaskPriority.BATCH

    # EMA smoothing for benchmark scores (0.0-1.0)
    # Higher = more weight on historical data, prevents route oscillation
    ema_alpha: float = 0.7

    # Prefill chunk size for soft-preemption (tokens per chunk)
    prefill_chunk_size: int = 2048

    # Batch sizes to pre-warm compute graphs for
    warmup_batch_sizes: list[int] = field(default_factory=lambda: [1, 4, 8])

    # EMA state: model_id -> {backend -> {tps: float, latency_p50: float}}
    _ema_state: dict[str, dict[str, dict[str, float]]] = field(default_factory=dict)


@dataclass
class PhaseHandoff:
    """KVCache handoff object between prefill and decode engines.

    Contains the block table and Metal buffer pointers from the prefill
    engine, ready to be wrapped (zero-copy) by the decode engine.
    """
    request_id: str
    block_table: Any                    # List of block IDs from prefill engine
    kv_buffers: list[tuple[Any, tuple, str]]  # (mx.array, shape, dtype) list
    meta_states: list[tuple]           # Per-layer meta_state tuples
    model_name: str
    num_tokens: int
    prefill_backend: EngineBackend
    decode_backend: EngineBackend


class SmartRouter:
    """Phase-aware request router with cross-engine handoff.

    Routing decision tree:
    1. Explicit backend override (model config or request param) -> use it
    2. Prompt > phase_split_threshold AND low cache hit -> split phases
       - Prefill -> omlx (matmul optimized)
       - Decode  -> Rapid-MLX (KV lightweight)
    3. Prompt <= threshold OR high cache hit -> unified backend
       - REALTIME tasks -> Rapid-MLX (low latency)
       - BATCH tasks    -> omlx (high throughput)
    4. Uncached tokens > cloud_threshold -> CloudRouter
    5. Benchmark-based: if loaded, use actual TPS data to decide
    """

    def __init__(
        self,
        config: Optional[RouterConfig] = None,
        cloud_router: Any = None,
        llm_engine: Any = None,
        rapid_engine: Any = None,
        vlm_engine: Any = None,
        stt_engine: Any = None,
        tts_engine: Any = None,
        sts_engine: Any = None,
        image_gen_engine: Any = None,
        embedding_engine: Any = None,
        reranker_engine: Any = None,
    ):
        self.config = config or RouterConfig()
        self.cloud_router = cloud_router

        # Engine registry — keep both old names (for compatibility) and new
        self.llm_engine = llm_engine         # omlx BatchedEngine
        self.rapid_engine = rapid_engine     # Rapid-MLX engine
        self.vlm_engine = vlm_engine
        self.stt_engine = stt_engine
        self.tts_engine = tts_engine
        self.sts_engine = sts_engine
        self.image_gen_engine = image_gen_engine
        self.embedding_engine = embedding_engine
        self.reranker_engine = reranker_engine

        # Handoff registry — active phase handoffs
        self._handoffs: dict[str, PhaseHandoff] = {}

        # Routing stats
        self._route_count: dict[str, int] = {}
        self._split_count: int = 0
        self._cloud_count: int = 0

    # ================================================================
    # Public API — route requests
    # ================================================================

    def decide(
        self,
        prompt_length: int,
        task_tag: str = "",
        cache_hit_rate: float = 0.0,
        backend_override: Optional[EngineBackend] = None,
        quant_format: str = "",
        model_id: str = "",
    ) -> RouteDecision:
        """Make a routing decision based on request characteristics.

        Args:
            prompt_length:     Number of tokens in the prompt (new + cached)
            task_tag:          Source tag (claude_code, openclaw, etc.)
            cache_hit_rate:    Fraction of prompt already cached (0.0-1.0)
            backend_override:  If set, force specific backend (bypasses decision tree)
            quant_format:      Quantization format string (e.g., "4bit", "8bit")
            model_id:          Model ID for benchmark lookups

        Returns:
            RouteDecision with prefill_backend, decode_backend, reason
        """
        new_tokens = int(prompt_length * (1.0 - cache_hit_rate))

        # 1. Explicit override
        if backend_override is not None:
            self._record_route(backend_override.value, False)
            return RouteDecision(
                prefill_backend=backend_override,
                decode_backend=backend_override,
                reason=f"explicit override: {backend_override.value}",
            )

        # 2. Cloud fallback for massive uncached context
        if (
            self.cloud_router
            and new_tokens > self.config.cloud_fallback_threshold
        ):
            self._cloud_count += 1
            self._record_route("cloud", False)
            return RouteDecision(
                prefill_backend=EngineBackend.CLOUD,
                decode_backend=EngineBackend.CLOUD,
                reason=f"cloud fallback: {new_tokens} uncached tokens > {self.config.cloud_fallback_threshold}",
                split_phases=False,
            )

        # 3. Benchmark-based routing (if we have data)
        benchmark_decision = self._benchmark_route(model_id, quant_format, new_tokens)
        if benchmark_decision is not None:
            self._record_route(
                benchmark_decision.prefill_backend.value,
                benchmark_decision.split_phases,
            )
            return benchmark_decision

        # 4. Priority-based routing
        priority = self._resolve_priority(task_tag)

        # 5. Phase split: long prompts with low cache coverage
        if (
            new_tokens > self.config.phase_split_threshold
            and cache_hit_rate < 0.5
            and self.llm_engine is not None
            and self.rapid_engine is not None
        ):
            # Split: prefill on omlx (matmul), decode on Rapid (KV)
            self._split_count += 1
            self._record_route("split", True)
            return RouteDecision(
                prefill_backend=EngineBackend.OMLX,
                decode_backend=EngineBackend.RAPID,
                reason=(
                    f"phase split: {new_tokens} new tokens > "
                    f"{self.config.phase_split_threshold}, cache={cache_hit_rate:.0%}"
                ),
                split_phases=True,
            )

        # 6. Unified backend based on priority
        if priority == TaskPriority.REALTIME:
            # Realtime tasks prefer Rapid-MLX (lower latency)
            backend = self._pick_low_latency_engine()
        else:
            # Batch/background tasks prefer omlx (higher throughput)
            backend = self._pick_high_throughput_engine()

        self._record_route(backend.value, False)
        return RouteDecision(
            prefill_backend=backend,
            decode_backend=backend,
            reason=f"priority={priority.value}, unified={backend.value}",
        )

    async def route_chat(
        self,
        messages: list[dict],
        request_data: dict[str, Any],
        prompt_length: int = 0,
        task_tag: str = "",
        cache_hit_rate: float = 0.0,
        **kwargs,
    ) -> Any:
        """Route a chat request with phase-aware decisions."""
        engine, etype, decision = self._select_engine_with_decision(
            messages, request_data, prompt_length, task_tag, cache_hit_rate
        )

        # Cloud routing
        if decision.prefill_backend == EngineBackend.CLOUD and self.cloud_router:
            return await self.cloud_router.completion(messages, **kwargs)

        return await engine.chat(messages, **kwargs)

    async def route_stream_chat(
        self,
        messages: list[dict],
        request_data: dict[str, Any],
        prompt_length: int = 0,
        task_tag: str = "",
        cache_hit_rate: float = 0.0,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Route a streaming chat request. If phase-split, orchestrates handoff."""
        engine, etype, decision = self._select_engine_with_decision(
            messages, request_data, prompt_length, task_tag, cache_hit_rate
        )

        if decision.prefill_backend == EngineBackend.CLOUD and self.cloud_router:
            async for chunk in self.cloud_router.stream_completion(messages, **kwargs):
                yield chunk
            return

        if decision.split_phases:
            yield await self._execute_split_phase(
                engine, messages, request_data, decision, **kwargs
            )
        else:
            async for chunk in engine.stream_chat(messages, **kwargs):
                yield chunk

    def get_stats(self) -> dict[str, Any]:
        return {
            "route_count": dict(self._route_count),
            "split_count": self._split_count,
            "cloud_count": self._cloud_count,
            "active_handoffs": len(self._handoffs),
            "benchmarks_cached": len(self.config.benchmark_cache),
        }

    # ================================================================
    # Phase handoff — execute prefill on one engine, decode on another
    # ================================================================

    async def _execute_split_phase(
        self,
        prefill_engine: Any,
        messages: list[dict],
        request_data: dict[str, Any],
        decision: RouteDecision,
        **kwargs,
    ) -> str:
        """Execute a split-phase request: prefill on omlx, decode on Rapid.

        Flow:
        1. Run prefill on omlx engine -> get KV cache
        2. Extract KV buffers as PhaseHandoff
        3. Transfer to Rapid-MLX engine (zero-copy)
        4. Continue decode from KV state
        """
        decode_engine = self.rapid_engine or self.llm_engine
        request_id = request_data.get("request_id", "")

        # Step 1: Prefill on omlx
        logger.info(
            f"[SmartRouter] Phase split: prefill on "
            f"{decision.prefill_backend.value}, decode on {decision.decode_backend.value}"
        )

        # Run prefill — generate initial tokens and capture KV state
        # The prefill engine processes the full prompt and returns the KV cache
        prefill_result = await prefill_engine.chat(messages, **{
            **kwargs,
            "prefill_only": True,  # Signal to only run prefill, return KV state
        })

        # Step 2: Create handoff
        handoff = PhaseHandoff(
            request_id=request_id,
            block_table=getattr(prefill_result, "block_table", []),
            kv_buffers=getattr(prefill_result, "kv_buffers", []),
            meta_states=getattr(prefill_result, "meta_states", []),
            model_name=getattr(prefill_result, "model_name", ""),
            num_tokens=getattr(prefill_result, "num_tokens", 0),
            prefill_backend=decision.prefill_backend,
            decode_backend=decision.decode_backend,
        )
        self._handoffs[request_id] = handoff

        # Step 3: Decode on Rapid-MLX with transferred KV state
        # The decode engine receives the KV state and continues generation
        decode_result = await decode_engine.chat(messages, **{
            **kwargs,
            "kv_handoff": handoff,  # Inject KV state from prefill
            "continue_from": handoff.num_tokens,
        })

        # Clean up handoff
        self._handoffs.pop(request_id, None)
        return decode_result

    # ================================================================
    # Internal — engine selection, priority resolution, benchmarking
    # ================================================================

    def _select_engine_with_decision(
        self,
        messages: list[dict],
        request_data: dict[str, Any],
        prompt_length: int,
        task_tag: str,
        cache_hit_rate: float,
    ) -> tuple[Any, str, RouteDecision]:
        """Select engine and return (engine, type, decision)."""
        decision = self.decide(
            prompt_length=prompt_length or self._estimate_tokens(messages),
            task_tag=task_tag or request_data.get("task_tag", ""),
            cache_hit_rate=cache_hit_rate,
            backend_override=self._parse_backend_override(request_data),
            quant_format=request_data.get("quant_format", ""),
            model_id=request_data.get("model_id", ""),
        )

        # Content-based routing first (non-LLM tasks)
        engine, etype = self._route_by_content(messages, request_data)

        # For LLM tasks, apply phase-aware backend selection
        if etype == "llm":
            if decision.prefill_backend == EngineBackend.OMLX and self.llm_engine:
                engine, etype = self.llm_engine, "llm"
            elif decision.prefill_backend == EngineBackend.RAPID and self.rapid_engine:
                engine, etype = self.rapid_engine, "llm"
            elif self.llm_engine:
                engine, etype = self.llm_engine, "llm"

        return engine, etype, decision

    def _route_by_content(
        self,
        messages: list[dict],
        request_data: dict[str, Any],
    ) -> tuple[Any, str]:
        """Content-based routing (same as old RequestRouter)."""
        # Explicit task-based
        if self._is_task(request_data, "embedding") and self.embedding_engine:
            return self.embedding_engine, "embedding"
        if self._is_task(request_data, "rerank") and self.reranker_engine:
            return self.reranker_engine, "reranker"
        if self._is_task(request_data, "image_gen") and self.image_gen_engine:
            return self.image_gen_engine, "image_gen"
        if self._is_task(request_data, "tts") and self.tts_engine:
            return self.tts_engine, "tts"
        if self._is_task(request_data, "sts") and self.sts_engine:
            return self.sts_engine, "sts"
        if self._has_audio(request_data) and self.stt_engine:
            return self.stt_engine, "stt"

        # VLM
        if self._has_images(messages):
            if self.vlm_engine:
                return self.vlm_engine, "vlm"
            logger.warning("VLM requested but unavailable, falling back to LLM")

        # LLM default
        if self.rapid_engine:
            return self.rapid_engine, "llm"
        if self.llm_engine:
            return self.llm_engine, "llm"

        raise RuntimeError("No suitable engine available")

    def _resolve_priority(self, task_tag: str) -> TaskPriority:
        """Map task tag to priority level."""
        tag = task_tag.lower().strip()
        if tag in ("claude_code", "claude", "codex", "copilot", "interactive"):
            return TaskPriority.REALTIME
        if tag in ("openclaw", "open-claw", "agent", "batch"):
            return TaskPriority.BATCH
        if tag in ("embedding", "rerank", "offline", "background"):
            return TaskPriority.BACKGROUND
        return self.config.default_priority

    def _parse_backend_override(self, request_data: dict[str, Any]) -> Optional[EngineBackend]:
        """Parse explicit backend override from request."""
        override = request_data.get("backend_override")
        if not override:
            return None
        try:
            return EngineBackend(override)
        except (ValueError, KeyError):
            # Try matching by value
            for be in EngineBackend:
                if be.value == str(override).lower():
                    return be
            return None

    def _estimate_tokens(self, messages: list[dict]) -> int:
        """Rough token estimate with CJK-aware counting."""


        cjk_chars = 0
        ascii_chars = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                text = ""
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text += part.get("text", "")
            else:
                text = ""
            for ch in text:
                cp = ord(ch)
                if (0x4e00 <= cp <= 0x9fff or
                     0x3040 <= cp <= 0x30ff or
                     0xac00 <= cp <= 0xd7af):
                    cjk_chars += 1
                else:
                    ascii_chars += 1
        return max(1, cjk_chars + ascii_chars // 4)
    def _has_images(self, messages: list[dict]) -> bool:
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        return True
        return False

    def _has_audio(self, request_data: dict[str, Any]) -> bool:
        return bool(request_data.get("audio") or request_data.get("audio_path"))

    def _is_task(self, request_data: dict[str, Any], task: str) -> bool:
        return (
            request_data.get("task") == task
            or request_data.get(task)
            or request_data.get(task.replace("-", "_"))
        )

    def _pick_low_latency_engine(self):
        """Pick the engine with lowest latency (prefer Rapid-MLX)."""
        if self.rapid_engine:
            return EngineBackend.RAPID
        if self.llm_engine:
            return EngineBackend.OMLX
        raise RuntimeError("No LLM engine available")

    def _pick_high_throughput_engine(self):
        """Pick the engine with highest throughput (prefer omlx for batching)."""
        if self.llm_engine:
            return EngineBackend.OMLX
        if self.rapid_engine:
            return EngineBackend.RAPID
        raise RuntimeError("No LLM engine available")

    # ================================================================
    # Graph pre-warm — prevent JIT stall on phase handoff
    # ================================================================

    def warmup_graphs(self, model_name: str, prompt_template: str | None = None) -> None:
        """Pre-compile compute graphs on both engines for common batch sizes.

        Must be called at startup or model load time. Without this,
        the first Decode request after a Prefill handoff will stall
        while Rapid-MLX dynamically compiles its decode graph.
        """
        prompt = prompt_template or "warmup"

        for bs in self.config.warmup_batch_sizes:
            if self.llm_engine:
                try:
                    self.llm_engine._warmup(model_name, prompt, batch_size=bs)
                    logger.info(f"[Warmup] omlx: {model_name} bs={bs} done")
                except Exception as e:
                    logger.warning(f"[Warmup] omlx: {model_name} bs={bs} failed: {e}")

            if self.rapid_engine:
                try:
                    self.rapid_engine._warmup(model_name, prompt, batch_size=bs)
                    logger.info(f"[Warmup] Rapid: {model_name} bs={bs} done")
                except Exception as e:
                    logger.warning(f"[Warmup] Rapid: {model_name} bs={bs} failed: {e}")

    # ================================================================
    # Benchmark-based routing
    # ================================================================"

    def store_benchmark(self, result: BenchmarkResult) -> None:
        """Store a benchmark result for future routing decisions."""
        model_id = result.model_id
        if model_id not in self.config.benchmark_cache:
            self.config.benchmark_cache[model_id] = {}
        self.config.benchmark_cache[model_id][result.backend.value] = result
        logger.info(
            f"[Benchmark] {result.model_id} @ {result.backend.value}: "
            f"{result.tps:.1f} tps, p50={result.latency_p50:.0f}ms"
        )

    def _benchmark_route(
        self,
        model_id: str,
        quant_format: str,
        new_tokens: int,
     ) -> Optional[RouteDecision]:
        """Make routing decision based on EMA-smoothed benchmark data."""
        # Uses EMA to prevent route oscillation from GC/Metal jitter
        # formula: score = alpha * history + (1-alpha) * measured
        if not self.config.enable_benchmark_routing:
            return None
        if not model_id or model_id not in self.config.benchmark_cache:
            return None

        benchmarks = self.config.benchmark_cache[model_id]

        # Need at least 2 backends to compare
        if len(benchmarks) < 2:
            return None

        # Apply EMA smoothing to each backend scores
        alpha = self.config.ema_alpha
        ema_state = self.config._ema_state.setdefault(model_id, {})

        smoothed = {}
        for backend_name, br in benchmarks.items():
            prev = ema_state.get(backend_name, {})
            smooth_tps = alpha * prev.get("tps", br.tps) + (1 - alpha) * br.tps
            smooth_lat = alpha * prev.get("latency_p50", br.latency_p50) + (1 - alpha) * br.latency_p50
            ema_state[backend_name] = {"tps": smooth_tps, "latency_p50": smooth_lat}
            smoothed[backend_name] = BenchmarkResult(
                model_id=br.model_id,
                backend=br.backend,
                quant_format=br.quant_format,
                tps=smooth_tps,
                latency_p50=smooth_lat,
                latency_p99=br.latency_p99,
                memory_peak_bytes=br.memory_peak_bytes,
                timestamp=br.timestamp,
             )

        # Find best backend for prefill (highest smoothed TPS)
        best_prefill = max(smoothed.items(), key=lambda x: x[1].tps)
        # Find best backend for decode (lowest smoothed p50 latency)
        best_decode = min(smoothed.items(), key=lambda x: x[1].latency_p50)

        # Only split if the difference is meaningful (>15%)
        min_tps = min(b.tps for b in smoothed.values())
        max_lat = max(b.latency_p50 for b in smoothed.values())
        tps_diff = (best_prefill[1].tps - min_tps) / best_prefill[1].tps if best_prefill[1].tps > 0 else 0
        latency_diff = (max_lat - best_decode[1].latency_p50) / max_lat if max_lat > 0 else 0

        split = tps_diff > 0.15 or latency_diff > 0.15

        # Map backend name back to enum
        prefill_be = EngineBackend(best_prefill[0])
        decode_be = EngineBackend(best_decode[0]) if split else prefill_be

        return RouteDecision(
            prefill_backend=prefill_be,
            decode_backend=decode_be,
            reason=(
                f"benchmark(ema): prefill={prefill_be.value}({best_prefill[1].tps:.0f}tps), "
                f"decode={decode_be.value}({best_decode[1].latency_p50:.0f}ms p50), "
                f"split={split}"
             ),
            split_phases=split,
        )
    # Stats helpers
    # ================================================================

    def _record_route(self, backend: str, is_split: bool) -> None:
        self._route_count[backend] = self._route_count.get(backend, 0) + 1

    def reset_stats(self) -> None:
        self._route_count.clear()
        self._split_count = 0
        self._cloud_count = 0
