# SPDX-License-Identifier: Apache-2.0
"""
Scheduler for oMLX continuous batching.

This module provides a Scheduler class that manages request scheduling
using mlx-lm's BatchGenerator for efficient continuous batching.

The scheduler follows vLLM's design with:
- Waiting queue for pending requests
- Running set for active requests
- Continuous batching via BatchGenerator
"""

import concurrent.futures
import copy
import logging

logger = logging.getLogger(__name__)

# Import protocol-specific output parser support
try:
    from ..parsers.output_parser import (
        OutputParserFactory,
        OutputParserSession,
        detect_output_parser,
    )

    HAS_OUTPUT_PARSER = True
except ImportError:
    OutputParserFactory = None
    OutputParserSession = None
    detect_output_parser = None
    HAS_OUTPUT_PARSER = False

import os
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from typing import Any

import mlx.core as mx
from mlx_lm.generate import (
    BatchGenerator,
)

from ..cache.boundary_snapshot_store import BoundarySnapshotSSDStore
from ..cache.observability import CacheRateTracker
from ..cache.paged_cache import PagedCacheManager
from ..cache.paged_ssd_cache import PagedSSDCacheManager
from ..cache.prefix_cache import BlockAwarePrefixCache
from ..memory_monitor import MemoryMonitor
from ..prefill_transient_tracker import PrefillTransientTracker
from ..request import Request
from ..speculative.vlm_mtp import VLMMTPDrafter

# Module-level alias so Scheduler.__init__ can fall back to mlx-lm's default
# stream when no per-engine stream is provided.
from .config import SchedulerConfig
from .helpers import (
    _default_generation_stream,
)
from .types import (
    _PrefillState,
    _StoreCacheGate,
    _VLMMTPDecodeState,
)


def __init__(
    self,
    model: Any,
    tokenizer: Any,
    config: SchedulerConfig | None = None,
    stream: Any | None = None,
):
    """
    Initialize the scheduler.

    Args:
        model: The MLX model
        tokenizer: The tokenizer
        config: Scheduler configuration
        stream: Optional mx.Stream for this engine. Falls back to the
            module-level _default_generation_stream when not provided.
    """
    self.model = model
    # Deep-copy the tokenizer so the scheduler owns an independent Rust
    # tokenizer backend.  Without this, concurrent access from the asyncio
    # event loop (encode/apply_chat_template in engine handlers) and the
    # MLX executor thread (scheduler.step) causes
    # "RuntimeError: Already borrowed" from the HuggingFace tokenizers
    # Rust RefCell.  See: https://github.com/huggingface/tokenizers/issues/537
    self.tokenizer = copy.deepcopy(tokenizer)
    self.config = copy.copy(config) if config else SchedulerConfig()
    self._stream = stream if stream is not None else _default_generation_stream

    # Load additional EOS tokens from generation_config.json.
    # Some models (e.g. GLM-4.6V) define multiple EOS tokens there
    # that are not in tokenizer.eos_token_id.
    self._generation_config_eos: set[int] | None = self._load_generation_config_eos()

    # Load suppress_tokens from generation_config.json. Standard HF field
    # forbidding certain tokens during generation (e.g. runaway-output
    # triggers). Empty set when absent; honored as a logits processor.
    self._model_suppress_tokens: set[int] = (
        self._load_generation_config_suppress_tokens()
    )

    # For strict RotatingKVCache reuse, align paged cache block size to
    # the model's rotating window size when paged cache is enabled.
    self._align_block_size_with_rotating_window()
    # For ArraysCache-only models (no RotatingKVCache), use a larger block
    # size to reduce boundary snapshot overhead during prefill.
    self._enlarge_block_size_for_arrays_cache()

    # TurboQuant KV cache (set by engine if model_settings has it enabled)
    self._turboquant_kv_bits: float | None = None
    self._turboquant_skip_last: bool = True
    self._turboquant_kv_mode: str = "v4"

    # Request management - following vLLM's design
    self.waiting: deque[Request] = deque()  # Waiting queue (FCFS)
    self.running: dict[str, Request] = {}  # Running requests by ID
    # Chunked prefill queue: requests whose prefill spans multiple steps.
    # Populated when chunked_prefill=True and prompt exceeds prefill_step_size.
    self.prefilling: deque[Request] = deque()
    self._prefill_states: dict[str, _PrefillState] = {}
    self.requests: dict[str, Request] = {}  # All requests by ID
    self.finished_req_ids: set[str] = set()  # Recently finished

    # Thread-safe set for deferred aborts (main thread → executor thread)
    # CPython GIL guarantees set.add() and `x in set` are atomic.
    self._pending_abort_ids: set[str] = set()

    # Lock-free admin snapshot. Published at the end of each step() while
    # the engine thread is the sole writer of running/waiting; the admin
    # endpoint reads the dict reference atomically (GIL) and never iterates
    # the live mutable structures.
    self._admin_snapshot: dict[str, Any] = {
        "running_by_id": {},
        "waiting": [],
    }

    # Memory limits for inline prefill checking.
    # Set by ProcessMemoryEnforcer; propagated to BatchGenerator.
    self._memory_limit_bytes: int = 0  # soft limit (dynamic, jittery)
    self._memory_hard_limit_bytes: int = 0  # hard limit (dynamic ceiling)
    # Stable physical cap = min(static_ceiling, metal_cap).  Used ONLY to
    # abort an in-flight prefill, so a transient dynamic-ceiling dip can't
    # kill a near-complete request that actually fits.  0 => fall back to
    # _memory_hard_limit_bytes (pre-propagation / old enforcer).
    self._memory_abort_limit_bytes: int = 0
    # Prefill abort margin — fraction of the abort cap we allow a chunk's
    # predicted peak to reach.  Set by enforcer per tier.
    self._prefill_abort_margin: float = 0.90
    # Last mx.get_active_memory() sample taken on this scheduler's MLX
    # executor thread.  The background memory enforcer reads this cached
    # value during active decode instead of touching MLX/Metal directly.
    self._last_mlx_active_memory_bytes: int = 0
    # Component ceilings — propagated alongside the hard limit so the
    # rejection-path error message can identify which constraint is
    # binding and suggest the right remedy (close apps / raise tier /
    # raise iogpu.wired_limit_mb / reduce context).  0 = not set yet.
    self._memory_static_ceiling_bytes: int = 0
    self._memory_dynamic_ceiling_bytes: int = 0
    self._memory_metal_cap_bytes: int = 0
    self._memory_hot_cache_reserved_bytes: int = 0
    self._memory_guard_tier: str = ""
    self._prefill_memory_guard: bool = False  # set by ProcessMemoryEnforcer
    # Set to True by ProcessMemoryEnforcer when phys_footprint crosses
    # soft_threshold. Schedulers stop admitting new prefills while this is
    # set; in-flight requests proceed.
    self._admission_paused: bool = False
    # Adaptive prefill throttle params, propagated from enforcer.
    # Until set, _adaptive_chunk_size is a no-op (returns requested as-is).
    self._prefill_safe_zone_ratio: float = 0.80
    self._prefill_min_chunk_tokens: int = 32
    # EWMA estimator of per-token chunk transient bytes, used by
    # _adaptive_chunk_size in the caution zone. Owned per-scheduler.
    _tracker_model_id = ""
    if config is not None and config.model_name:
        _tracker_model_id = os.path.basename(config.model_name.rstrip("/"))
    self._prefill_transient_tracker = PrefillTransientTracker(
        model_id=_tracker_model_id
    )

    # SpecPrefill: draft model for attention-based sparse prefill
    self._specprefill_draft_model: Any | None = None
    # Track active specprefill request for RoPE cleanup
    self._specprefill_active_request_id: str | None = None

    # VLM MTP: gemma4_assistant drafter attached by VLMBatchedEngine.
    # When set, eligible requests bypass mlx-lm BatchGenerator for decode
    # and run through mlx-vlm's _mtp_rounds round loop instead.
    self._vlm_mtp_drafter: VLMMTPDrafter | None = None
    # Active vlm_mtp decode generators keyed by synthesized negative uid
    # (negative to make collision with BatchGenerator uids impossible).
    self._vlm_mtp_active: dict[int, _VLMMTPDecodeState] = {}
    self._vlm_mtp_next_uid: int = -1
    # Per-request settings snapshot for vlm_mtp routing (block size etc.).
    # Injected by VLMBatchedEngine.set_vlm_mtp_drafter alongside the drafter.
    self._vlm_mtp_draft_block_size: int | None = None

    # Llama 4 serialization: ChunkedKVCache does not support multi-row batching
    from .helpers import _model_declares_llama4

    self._serialize_llama4_requests = _model_declares_llama4(model)
    if self._serialize_llama4_requests and self.config.max_num_seqs > 1:
        logger.info(
            "Llama 4 detected; serializing requests because ChunkedKVCache "
            "does not support multi-row batching yet"
        )
        self.config.max_num_seqs = 1

    # MiniMax M3 detection: requires special cache alignment handling
    from .helpers import _model_declares_minimax_m3

    self._uses_minimax_m3 = _model_declares_minimax_m3(model)

    # Overflow recovery: request IDs that triggered generation overflow
    self._generation_overflow_recovery_ids: set[str] = set()

    # GLM DSA adaptive prefill (Sparse MLA kernels)
    self._glm_dsa_adaptive_prefill = None
    try:
        from ..patches.glm_moe_dsa.generate_patch import (
            _glm_dsa_adaptive_prefill_config,
        )

        self._glm_dsa_adaptive_prefill = _glm_dsa_adaptive_prefill_config(
            model, self.config.prefill_step_size
        )
    except Exception:
        logger.debug("GLM DSA adaptive prefill config unavailable", exc_info=True)
    if self._glm_dsa_adaptive_prefill is not None:
        logger.info(
            "GLM DSA adaptive scheduler prefill enabled: step=%d after=%d "
            "min_remaining=%d",
            self._glm_dsa_adaptive_prefill.step_size,
            self._glm_dsa_adaptive_prefill.after,
            self._glm_dsa_adaptive_prefill.min_remaining,
        )

    # Phase timing instrumentation for cache-on overhead diagnostics.
    self._phase_total_ms: dict[str, float] = defaultdict(float)
    self._phase_count: dict[str, int] = defaultdict(int)

    # Async store_cache executor (G2-async). Offloads the post-finish
    # bulk memcpy (28GB+ per 32k request) off the inference thread so
    # response streaming isn't blocked by it.
    self._store_cache_executor: concurrent.futures.ThreadPoolExecutor | None = None
    # Gate that caps in-flight store-cache submissions. Set only when
    # tiered cache is enabled (alongside _store_cache_executor).
    self._store_cache_gate: _StoreCacheGate | None = None
    # Pending (uid, request_id, future) entries waiting for async store
    # to finish before batch_generator.remove() can safely run. Drained
    # at the start of every step.
    self._pending_async_removes: deque = deque()
    # Track in-flight store futures per request_id for lookup wait /
    # shutdown wait.
    self._inflight_store_futures: dict[str, concurrent.futures.Future] = {}
    self._inflight_store_info: dict[str, Any] = {}

    # Admission control: memory and store-cache stall tracking
    self._memory_admission_blocked_request_id: str | None = None
    self._memory_admission_blocked_since: float = 0.0
    self._store_cache_admission_blocked_request_id: str | None = None
    self._store_cache_admission_blocked_since: float = 0.0
    # Cache freshness waits: per-request deferral for in-flight store_cache
    self._cache_freshness_waits: dict[str, Any] = {}
    self._prefix_cache_prepared: set[str] = set()

    # Mapping between our request IDs and BatchGenerator UIDs
    self.request_id_to_uid: dict[str, int] = {}
    self.uid_to_request_id: dict[int, str] = {}

    # BatchGenerator - the actual batching engine
    self.batch_generator: BatchGenerator | None = None
    self._current_sampler_params: tuple | None = None
    # Boundary cache snapshots for stateful non-sliceable caches (e.g., ArraysCache).
    # request_id -> {token_count -> snapshot_cache_or_None}
    # Multiple snapshots per request to support per-block ArraysCache state storage.
    # Values are None when offloaded to SSD via _boundary_snapshot_store.
    self._boundary_cache_snapshots: dict[str, dict[int, Any]] = {}
    # Lazy detection flag: True/False once determined, None before first check.
    self._boundary_snapshot_required: bool | None = None
    # SSD store for offloading boundary snapshots (initialized in _init_tiered_cache).
    self._boundary_snapshot_store: BoundarySnapshotSSDStore | None = None

    # paged SSD cache for KV state persistence (oMLX only supports paged SSD-based caching)
    self.paged_cache_manager: PagedCacheManager | None = None
    self.block_aware_cache: BlockAwarePrefixCache | None = None
    self.paged_ssd_cache_manager: PagedSSDCacheManager | None = None
    self._cache_rate_tracker = CacheRateTracker()
    self.memory_monitor: MemoryMonitor | None = None

    # Initialize paged SSD cache if paged_ssd_cache_dir is specified
    if self.config.paged_ssd_cache_dir:
        # Calculate max_blocks automatically if not specified
        if self.config.max_cache_blocks is not None:
            max_blocks = self.config.max_cache_blocks
        else:
            max_blocks = self._calculate_max_blocks()

        # Initialize paged cache manager for block metadata
        self.paged_cache_manager = PagedCacheManager(
            block_size=self.config.paged_cache_block_size,
            max_blocks=max_blocks,
            model_name=self.config.model_name,
            initial_blocks=self.config.initial_cache_blocks,
        )
        self.block_aware_cache = BlockAwarePrefixCache(
            model=model,
            paged_cache_manager=self.paged_cache_manager,
        )

        # Auto-init MemoryMonitor BEFORE _init_tiered_cache so the
        # SSD manager can use model-derived KV bytes-per-token instead
        # of its 200 KB default.  Regression guard: PR #1627.
        try:
            max_kv = getattr(self.config, "max_kv_cache_memory", None) or (4 * 1024**3)
            self.memory_monitor = MemoryMonitor(
                max_kv_cache_memory=max_kv,
            )
            if self.paged_cache_manager is not None:
                self.memory_monitor.set_paged_cache_manager(self.paged_cache_manager)
            self._set_model_info_for_monitor()
        except Exception as e:
            logger.debug("Auto-init MemoryMonitor failed: %s", e)
            self.memory_monitor = None

        # Initialize paged SSD cache
        self._init_tiered_cache()

        # Set cold restore callback for prefix cache
        if self.paged_ssd_cache_manager is not None:
            self.block_aware_cache.set_cold_restore_callback(
                self._restore_block_from_cold
            )
            logger.info(
                f"paged SSD cache enabled: {self.config.paged_ssd_cache_dir}, "
                f"block_size={self.config.paged_cache_block_size}, "
                f"max_blocks={max_blocks}"
            )

        # Async store_cache executor: single worker so submissions are
        # serialized (matches the original synchronous order) and we
        # never have two stores racing on the same paged_ssd index.
        self._store_cache_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="omlx-store-cache",
        )
        # Gate caps the post-completion store-cache pipeline so a burst
        # of finishes cannot pile up unbounded KV caches in memory while
        # the single writer drains. Cap starts at max_concurrent_requests
        # and is shrunk by ProcessMemoryEnforcer under pressure (#1383).
        self._store_cache_gate = _StoreCacheGate(cap=self.config.max_num_seqs)
    else:
        logger.info("oMLX cache disabled (mlx-lm BatchGenerator manages KV internally)")

    # Streaming detokenizers for proper UTF-8 handling (one per active request)
    # NOTE: No pooling - each request gets a fresh instance to prevent state contamination
    self._request_detokenizers: dict[str, Any] = {}  # request_id → active detokenizer

    # Protocol-specific output parser support (e.g. Harmony, Gemma 4)
    self._output_parser_factory: OutputParserFactory | None = None
    self._output_parser_kind: str | None = None
    self._output_parser_sessions: dict[str, OutputParserSession] = {}
    self._is_harmony_model: bool = False
    if HAS_OUTPUT_PARSER and detect_output_parser is not None:
        try:
            model_config = None
            if hasattr(model, "config"):
                # model.config may be a Pydantic model or dict
                try:
                    if hasattr(model.config, "model_dump"):
                        model_config = model.config.model_dump()
                    elif hasattr(model.config, "dict"):
                        model_config = model.config.dict()
                    elif isinstance(model.config, dict):
                        model_config = model.config
                    else:
                        # Try to convert to dict via __dict__
                        model_config = getattr(model.config, "__dict__", None)
                except Exception as e:
                    logger.debug(f"Failed to extract model.config: {e}")
            elif hasattr(model, "args"):
                try:
                    if hasattr(model.args, "model_dump"):
                        model_config = model.args.model_dump()
                    elif hasattr(model.args, "__dict__"):
                        model_config = model.args.__dict__
                except Exception as e:
                    logger.debug(f"Failed to extract model.args: {e}")

            self._output_parser_factory = detect_output_parser(
                self.config.model_name,
                self.tokenizer,
                model_config,
            )
            if self._output_parser_factory is not None:
                self._output_parser_kind = self._output_parser_factory.kind
                self._is_harmony_model = self._output_parser_kind == "harmony"
                logger.info(
                    "Output parser detected: %s for %s, stop_tokens=%s",
                    self._output_parser_kind,
                    self.config.model_name,
                    sorted(self._output_parser_factory.stop_token_ids),
                )
        except Exception as e:
            logger.warning(f"Error detecting output parser: {e}, assuming none")
            self._output_parser_factory = None
            self._output_parser_kind = None
            self._is_harmony_model = False

    # Statistics
    self.num_requests_processed = 0
    self.total_prompt_tokens = 0
    self.total_completion_tokens = 0

    # Step counter for periodic cleanup
    self._step_counter = 0
    # Deferred Metal cache cleanup after request completion.
    # Immediate mx.clear_cache() after request completion races with
    # IOKit's asynchronous completeMemory() callbacks, causing
    # 'prepare count underflow' kernel panics. Deferring the clear
    # by a few generation steps gives IOKit time to process callbacks.
    #
    # Stored as the absolute step number at which the clear should fire,
    # rather than a countdown integer.  This avoids the burst-completion
    # bug (#557): with max_num_seqs > 1 two requests can finish in the
    # same batch.  The old "only set if None" guard meant the second
    # completion never extended the window, so the first request's KV
    # cache blocks could be re-allocated before IOKit finished its
    # completeMemory() callbacks.  Using max() ensures the window always
    # covers the *latest* completion.
    # None = no deferred clear pending; int = step at which to fire.
    self._deferred_clear_at: int | None = None
    self._DEFERRED_CLEAR_DELAY = 4
    self._tokens_since_clear_cache = 0

    # GPU contention detection: rolling window of decode step times.
    # When CV (std/mean) exceeds threshold, competing GPU processes
    # are likely causing bimodal latency (3-4x slowdown observed).
    self._step_time_window: list[float] = []
    self._step_time_window_size: int = 20
    # Most recent pure-decode forward time (seconds, GPU-synced). Feeds the
    # n-gram spec dynamic break-even (T_verify / T_decode). Set in
    # _step_pure_decode; None until the first decode step.
    self._last_decode_dt: float | None = None
    self._contention_cv_threshold: float = 0.15
    self._contention_detected: bool = False
    self._contention_log_interval: int = 50  # log every N steps when contended
    self._last_contention_log_step: int = 0

    # Cache XTC special tokens (newline + EOS) — stable per tokenizer.
    # Must be after _is_harmony_model / _generation_config_eos init
    # since _get_xtc_special_tokens() delegates to _get_stop_tokens().
    self._xtc_special_tokens: list[int] = self._get_xtc_special_tokens()

    # Speculative decode state (lazy-initialized in _step_pure_decode)
    self._spec_decode_state = None

    # N-gram speculative decode state (lazy-initialized in _step_pure_decode)
    self._ngram_spec_state = None

    # DFlash block-diffusion speculative decode runtime
    self._dflash_runtime = None

    # DSpark (DeepSeek DeepSpec) speculative decode runtime
    self._dspark_runtime = None


@contextmanager
def _phase_timer(self, phase: str):
    """Lightweight wall-time accumulator for cache-on overhead diagnostics.

    Tracks total ms and invocation count per named phase. Intended for
    boundary capture / store_cache / hot cache eviction hot paths.
    """
    t0 = time.perf_counter()
    try:
        yield
    finally:
        self._phase_total_ms[phase] += (time.perf_counter() - t0) * 1000.0
        self._phase_count[phase] += 1


def get_phase_stats(self) -> dict[str, dict[str, float]]:
    """Return accumulated phase timings for diagnostics.

    Returns dict of phase -> {total_ms, count, avg_ms}.
    """
    result = {}
    for phase, total in self._phase_total_ms.items():
        count = self._phase_count.get(phase, 0)
        result[phase] = {
            "total_ms": total,
            "count": count,
            "avg_ms": total / count if count else 0.0,
        }
    return result


def _periodic_clear_threshold_bytes(self) -> int:
    """Cache-bytes threshold above which the periodic clear runs.

    Defaults to memory_limit/3 when a process memory limit is set,
    otherwise an absolute 2 GiB floor. Each periodic clear releases
    the entire MLX buffer pool in one batch; gating it on accumulated
    bytes avoids producing IOGPUFamily refcount bursts when the pool
    is already small.
    """
    if self._memory_limit_bytes > 0:
        return max(self._memory_limit_bytes // 3, 2 * 1024**3)
    return 2 * 1024**3


def _should_periodic_clear_cache(self) -> bool:
    """Decide whether the per-step periodic clear should fire.

    Returns False unless ``mlx_cache_cleanup_interval`` is configured,
    the step counter just landed on the interval boundary, AND the
    MLX buffer pool exceeds the threshold. See #978 / #1040 for the
    kernel panic class this gating is meant to mitigate.
    """
    interval = self.config.mlx_cache_cleanup_interval
    if interval <= 0 or self._step_counter % interval != 0:
        return False
    return mx.get_cache_memory() > self._periodic_clear_threshold_bytes()
