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
import threading
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
    prefill_chunk_size: int = 512

    # Batch sizes to pre-warm compute graphs for
    warmup_batch_sizes: list[int] = field(default_factory=lambda: [1, 4, 8])


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
        self._lock = threading.Lock()
        # EMA state — instance-level, not on config (avoids cross-instance pollution)
        self._ema_state: dict[str, dict[str, dict[str, float]]] = {}

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
            with self._lock:
                self._cloud_count += 1
            self._record_route("cloud", False)
            return RouteDecision(
                prefill_backend=EngineBackend.CLOUD,
                decode_backend=EngineBackend.CLOUD,
                reason=f"cloud fallback: {new_tokens} uncached tokens > {self.config.cloud_fallback_threshold}",
                split_phases=False,
            )

        # 3. Priority-based routing (resolved before benchmark to prevent
        # REALTIME requests from being routed to high-TPS/high-latency backends)
        priority = self._resolve_priority(task_tag)

        # 4. Benchmark-based routing — skip for REALTIME (latency > throughput)
        if priority != TaskPriority.REALTIME:
            benchmark_decision = self._benchmark_route(model_id, quant_format, new_tokens)
            if benchmark_decision is not None:
                self._record_route(
                    benchmark_decision.prefill_backend.value,
                    benchmark_decision.split_phases,
                )
                return benchmark_decision

        # 5. Phase split: long prompts with low cache coverage
        if (
            new_tokens > self.config.phase_split_threshold
            and cache_hit_rate < 0.5
            and self.llm_engine is not None
            and self.rapid_engine is not None
        ):
            # Split: prefill on omlx (matmul), decode on Rapid (KV)
            with self._lock:
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

        # Execute local inference with circuit breaker tracking
        try:
            result = await engine.chat(messages, **kwargs)
            if self.cloud_router:
                self.cloud_router.report_local_success()
            return result
        except Exception:
            if self.cloud_router:
                self.cloud_router.report_local_failure()
            raise

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
            async for chunk in self._execute_split_phase(
                engine, messages, request_data, decision, **kwargs
            ):
                yield chunk
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
    ) -> AsyncIterator[str]:
        """Execute split-phase: prefill on omlx, handoff KV to Rapid-MLX for decode."""
        decode_engine = self.rapid_engine or self.llm_engine
        request_id = request_data.get("request_id", "")

        if not (getattr(prefill_engine, "supports_prefill_only", False)
        and getattr(decode_engine, "supports_kv_handoff", False)):
                # Fallback: if engines don't support handoff, use unified streaming
            logger.info(
                "[SmartRouter] Phase split fallback: engines lack handoff support, "
                "streaming on %s", decision.prefill_backend.value,
            )
            async for chunk in prefill_engine.stream_chat(messages, **kwargs):
                yield chunk
            return

        logger.info(
            "[SmartRouter] Phase split: prefill=%s, decode=%s",
            decision.prefill_backend.value, decision.decode_backend.value,
        )

        # Step 1: Prefill on omlx — get KV state
        prefill_result = await prefill_engine.chat(messages, prefill_only=True, **kwargs)
        kv_state = prefill_result.kv_state or {}

        # Step 2: Create PhaseHandoff from KV state
        handoff = PhaseHandoff(
            request_id=request_id,
            block_table=kv_state.get("block_table"),
            kv_buffers=kv_state,
            meta_states=[],
            model_name=getattr(prefill_result, "model_name", ""),
            num_tokens=kv_state.get("num_computed_tokens", 0),
            prefill_backend=decision.prefill_backend,
            decode_backend=decision.decode_backend,
        )
        with self._lock:
            self._handoffs[request_id] = handoff

        # Step 3: Decode on Rapid-MLX with KV handoff — stream SSE chunks
        try:
            async for chunk in decode_engine.stream_chat(messages, kv_handoff=handoff, **kwargs):
                yield chunk
        finally:
            with self._lock:
                self._handoffs.pop(request_id, None)

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
        return max(1, int(cjk_chars * 1.5) + ascii_chars // 3)
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

    def _estimate_tps_from_params(
        self,
        model_id: str,
        quant_format: str,
        backend: EngineBackend,
    ) -> tuple[float, float]:
        """Estimate TPS and latency from model metadata using theoretical throughput.

        Uses parameter count + quant bits to estimate FLOPs per token,
        then divides by backend peak throughput. Values calibrated for M4 Max
        (~300 GFLOPS FP16, ~1200 GFLOPS INT4 via Matmul).

        Returns:
            (estimated_tps, estimated_latency_p50_ms)
        """
        # Get parameter count — check config, model-config.json, or estimate from name
        param_count = self._get_param_estimate(model_id)
        if param_count < 1e6:
            # Unreliable estimate, use defaults
            return (30.0, 50.0)

        # Quant bits from format string
        bits = 4
        if "8" in quant_format:
            bits = 8
        elif "16" in quant_format or "fp16" in quant_format:
            bits = 16

        # FLOPs per token = 2 * param_count * (16 / bits) for decode (1-token input)
        flops_per_token = 2 * param_count * (16.0 / bits)

        # Backend peak FLOPS (conservative estimates for M4 Max)
        peak_gflops = {
            EngineBackend.OMLX: 1400e9,
            EngineBackend.RAPID: 1000e9,
        }
        peak = peak_gflops.get(backend, 800e9)

        # Estimated TPS = peak / flops_per_token, with 30% efficiency factor
        efficiency = 0.3 if backend == EngineBackend.OMLX else 0.35
        tps = (peak * efficiency) / flops_per_token
        tps = max(5.0, min(tps, 200.0))

        # Latency estimate: base overhead + 1/tps
        base_overhead_ms = 15.0 if backend == EngineBackend.RAPID else 25.0
        latency_ms = base_overhead_ms + 1000.0 / tps
        return (tps, latency_ms)

    def _get_param_estimate(self, model_id: str) -> int:
        """Get estimated parameter count for a model.

        Checks model-config.json first, then infers from model name.
        """
        import json
        from pathlib import Path

        config_path = Path(__file__).parent.parent.parent / ".." / "model-config.json"
        try:
            with open(config_path) as f:
                mc = json.load(f)
            model_cfg = mc.get("models", {}).get(model_id, {})
            if model_cfg.get("parameter_count_estimate", 0) > 0:
                return int(model_cfg["parameter_count_estimate"])
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            pass

        # Infer from model name — common patterns
        name_lower = model_id.lower()
        for suffix, params in [
            ("70b", 70_000_000_000), ("72b", 72_000_000_000),
            ("35b", 35_000_000_000), ("32b", 32_000_000_000),
            ("14b", 14_000_000_000), ("13b", 13_000_000_000),
            ("10b", 10_000_000_000),
            ("7b", 7_000_000_000), ("8b", 8_000_000_000),
            ("4b", 4_000_000_000),
            ("3b", 3_000_000_000),
            ("2b", 2_000_000_000),
            ("1b", 1_000_000_000),
            ("760m", 760_000_000), ("460m", 460_000_000),
            ("350m", 350_000_000), ("260m", 260_000_000),
        ]:
            if suffix in name_lower:
                return params
        return 3_000_000_000

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

        # Apply EMA smoothing with cold-start prior
        alpha = self.config.ema_alpha
        ema_state = self._ema_state.setdefault(model_id, {})

        smoothed = {}
        for backend_name, br in benchmarks.items():
            if backend_name in ("auto", "cloud"):
                continue
            prev = ema_state.get(backend_name, {})
            observation_count = prev.get("count", 0)
            # Cold start: blend static prior when few observations
            if observation_count < 20:
                backend_enum = br.backend
                try:
                    backend_enum = EngineBackend(backend_name)
                except (ValueError, KeyError):
                    pass
                prior_tps, prior_lat = self._estimate_tps_from_params(
                    model_id, quant_format, backend_enum
                )
                # Weight prior by (20 - observations) / 20
                prior_weight = max(0.0, (20 - observation_count) / 20.0)
                effective_tps = (
                    prior_weight * prior_tps
                    + (1 - prior_weight) * (alpha * prev.get("tps", prior_tps) + (1 - alpha) * br.tps)
                )
                effective_lat = (
                    prior_weight * prior_lat
                    + (1 - prior_weight) * (alpha * prev.get("latency_p50", prior_lat) + (1 - alpha) * br.latency_p50)
                )
            else:
                effective_tps = alpha * prev.get("tps", br.tps) + (1 - alpha) * br.tps
                effective_lat = alpha * prev.get("latency_p50", br.latency_p50) + (1 - alpha) * br.latency_p50
            ema_state[backend_name] = {
                "tps": effective_tps,
                "latency_p50": effective_lat,
                "count": observation_count + 1,
            }
            smoothed[backend_name] = BenchmarkResult(
                model_id=br.model_id,
                backend=br.backend,
                quant_format=br.quant_format,
                tps=effective_tps,
                latency_p50=effective_lat,
                latency_p99=br.latency_p99,
                memory_peak_bytes=br.memory_peak_bytes,
                timestamp=br.timestamp,
            )

        if len(smoothed) < 2:
            return None

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

        # Map backend name back to enum — skip if invalid
        try:
            prefill_be = EngineBackend(best_prefill[0])
        except (ValueError, KeyError):
            return None
        try:
            decode_be = EngineBackend(best_decode[0]) if split else prefill_be
        except (ValueError, KeyError):
            decode_be = prefill_be

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
        with self._lock:
            self._route_count[backend] = self._route_count.get(backend, 0) + 1

    def reset_stats(self) -> None:
        with self._lock:
            self._route_count.clear()
            self._split_count = 0
            self._cloud_count = 0
