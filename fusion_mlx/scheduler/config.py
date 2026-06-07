from dataclasses import dataclass, field
from enum import Enum


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
    completion_batch_size: int = 32
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
    mlx_cache_cleanup_interval: int = 512   # Steps between mx.clear_cache() calls


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