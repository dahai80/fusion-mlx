from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SchedulingPolicy(Enum):
    """Scheduling policy for request ordering."""

    FCFS = "fcfs"   # First-Come-First-Served
    PRIORITY = "priority"   # Priority-based


@dataclass
class SchedulerConfig:
    """Configuration for the scheduler."""

    # Maximum number of concurrent requests in the batch
    max_num_seqs: int = 256
    # Maximum tokens to process per step (for prefill chunking)
    max_num_batched_tokens: int = 8192
    # Scheduling policy
    policy: SchedulingPolicy = SchedulingPolicy.FCFS
    # BatchGenerator settings (passed directly to mlx-lm)
    completion_batch_size: int = 64
    # Per-forward embedding input chunk size
    embedding_batch_size: int = 32
    prefill_step_size: int = 2048
    # When True, long prefills are processed one chunk per step() call,
    # interleaved with decode steps for already-running requests. This
    # reduces TTFT for concurrent requests but adds per-step overhead.
    chunked_prefill: bool = False

    # Paged cache settings (internal defaults)
    paged_cache_block_size: int = 256   # Tokens per block
    max_cache_blocks: int | None = (
        None   # Auto-calculated from available KV cache memory
    )
    initial_cache_blocks: int = (
        256   # Starting blocks (grows dynamically to max_cache_blocks)
    )

    # paged SSD cache settings (oMLX only supports paged SSD-based caching)
    # When paged_ssd_cache_dir is set, oMLX stores KV cache on paged SSD for prefix reuse.
    # When None, no oMLX caching (mlx-lm BatchGenerator manages KV internally).
    paged_ssd_cache_dir: str | None = (
        None   # Path for paged SSD cache storage (None = disabled)
    )
    hot_cache_only: bool = False
    paged_ssd_cache_max_size: int = 100 * 1024 * 1024 * 1024   # 100GB default
    hot_cache_max_size: int = 0   # In-memory hot cache size in bytes (0 = disabled)

    # Model identification (for cache isolation between different models)
    model_name: str = ""   # OpenAI API model name (e.g., "mlx-community/Llama-3.2-3B")

    # GC/cleanup settings (memory optimization)
    gc_cleanup_interval: int = 0   # Steps between gc.collect() calls (0=disabled)
    mlx_cache_cleanup_interval: int = 2048   # Steps between mx.clear_cache() calls
    memory_check_interval: int = 64     # Steps between memory pressure checks
    admin_snapshot_interval: int = 32     # Steps between admin snapshots
    decode_clear_interval: int = 4096     # Tokens between decode-phase cache clears

    def __post_init__(self):
        """Validate configuration after init."""
        if self.max_num_seqs < 1:
            raise ValueError(f"max_num_seqs must be >= 1, got {self.max_num_seqs}")
        if self.max_num_batched_tokens < self.prefill_step_size:
            raise ValueError(
                f"max_num_batched_tokens ({self.max_num_batched_tokens}) must be >= "
                f"prefill_step_size ({self.prefill_step_size})"
            )
        if self.completion_batch_size > self.max_num_seqs:
            raise ValueError(
                f"completion_batch_size ({self.completion_batch_size}) must be <= "
                f"max_num_seqs ({self.max_num_seqs})"
            )
        if self.paged_cache_block_size < 1:
            raise ValueError("paged_cache_block_size must be >= 1")
        if self.max_cache_blocks is not None and self.max_cache_blocks < self.initial_cache_blocks:
            raise ValueError(
                f"max_cache_blocks must be >= initial_cache_blocks ({self.initial_cache_blocks})"
            )
        if self.gc_cleanup_interval < 0:
            raise ValueError("gc_cleanup_interval must be >= 0")
        if self.mlx_cache_cleanup_interval < 1:
            raise ValueError("mlx_cache_cleanup_interval must be >= 1")


@dataclass
class SchedulerOutput:
    """
    Output from a scheduling step.

    Contains information about what was scheduled and results.
    """

    # Requests scheduled in this step
    scheduled_request_ids: list[str] = field(default_factory=list)
    # Total tokens scheduled
    num_scheduled_tokens: int = 0
    # Requests that finished in this step
    finished_request_ids: set[str] = field(default_factory=set)
    # Request outputs (tokens generated)
    outputs: list = field(default_factory=list)
    # Whether any work was done
    has_work: bool = False
    # Prefill eviction request for memory management (omlx compat)
    prefill_eviction_request: Any = None
