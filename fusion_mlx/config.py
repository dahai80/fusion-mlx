"""Global configuration for fusion-mlx.

Merged from omlx model_settings + Rapid-MLX SchedulerConfig.
"""

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class MemoryTier(Enum):
    """Memory enforcement tiers."""

    SAFE = "safe"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    CUSTOM = "custom"


class SchedulingPolicy(Enum):
    """Scheduling policy for request ordering."""

    FCFS = "fcfs"
    PRIORITY = "priority"


@dataclass
class SchedulerConfig:
    """Scheduler configuration (merged from omlx + Rapid-MLX)."""

    # Concurrency limits
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 65536
    policy: SchedulingPolicy = SchedulingPolicy.FCFS

    # BatchGenerator settings
    prefill_batch_size: int = 8
    completion_batch_size: int = 32
    prefill_step_size: int = 2048

    # Chunked prefill
    chunked_prefill_tokens: int = 2048  # >0 = enabled, value = chunk size
    # Runtime bool mirror. The rapid-MLX Scheduler reads ``config.chunked_prefill``
    # (bool) while the released CLI sets ``chunked_prefill_tokens`` (int). Kept as
    # a real field so ``server._convert_scheduler_config`` can still pass
    # ``chunked_prefill=<bool>`` explicitly; ``__post_init__`` syncs it from the
    # int knob when the CLI omits it.
    chunked_prefill: bool = False
    # Mid-prefill cache saving every N tokens (from Rapid-MLX)
    mid_prefill_save_interval: int = 8192

    # Prefix cache
    enable_prefix_cache: bool = True
    prefix_cache_size: int = 100

    # Memory-aware cache
    use_memory_aware_cache: bool = True
    cache_memory_mb: int | None = None  # None = auto (20% available RAM)
    cache_memory_percent: float = 0.20

    # KV cache quantization
    kv_cache_quantization: bool = False
    kv_cache_quantization_bits: int = 8
    kv_cache_quantization_group_size: int = 64
    kv_cache_min_quantize_tokens: int = 256

    # TurboQuant V-only compression
    kv_cache_turboquant: bool = False
    kv_cache_turboquant_bits: int | None = None
    kv_cache_turboquant_group_size: int = 32

    # Paged cache
    use_paged_cache: bool = False
    paged_cache_block_size: int = 64
    max_cache_blocks: int = 1000

    # MTP (Multi-Token Prediction)
    enable_mtp: bool = False
    mtp_num_draft_tokens: int = 1
    mtp_optimistic: bool = False

    # SuffixDecoding
    enable_suffix_decoding: bool = False
    suffix_max_draft: int = 8
    suffix_max_suffix_len: int = 4
    suffix_min_confidence: float = 0.3
    suffix_min_draft_len: int = 2

    # Admission control
    max_concurrent_requests: int = 256

    # R15-P1 additions
    prefix_cache_index: str = "radix"
    spec_decode: str = "none"
    dflash_drafter_path: str = ""
    kv_cache_dtype: str = "bf16"
    kv_disk_checkpoint_interval: int = 256
    kv_cache_turboquant_mode: str = "v4"
    pflash_config: Any = None
    gpu_memory_utilization: float = 0.90

    # --- Rapid-MLX runtime fields (read directly by the Scheduler runtime in
    # fusion_mlx/scheduler/core.py + sched_*.py and by the engine layer). Folded
    # in from the minimal scheduler/config.py SchedulerConfig so the single
    # merged class serves BOTH the released CLI knobs above AND the runtime
    # field reads. The rapid-MLX migration had split these into a separate
    # minimal dataclass; ``from .scheduler import SchedulerConfig`` resolved to
    # that minimal one, so the single-model ``serve --model`` / ``bench`` CLI
    # paths (which build the rich config) raised TypeError. One class now.
    # Model identification (cache isolation between different models).
    model_name: str = ""
    # Per-forward embedding input chunk size.
    embedding_batch_size: int = 32
    # Paged cache runtime sizing.
    initial_cache_blocks: int = 256
    # Paged SSD cache (oMLX prefix-reuse layer); None = disabled.
    paged_ssd_cache_dir: str | None = None
    hot_cache_only: bool = False
    paged_ssd_cache_max_size: int = 100 * 1024 * 1024 * 1024  # 100 GiB
    hot_cache_max_size: int = 0  # bytes; 0 = disabled
    # GC / cache-clear cadence (steps between calls).
    gc_cleanup_interval: int = 0  # 0 = disabled
    mlx_cache_cleanup_interval: int = 8192
    memory_check_interval: int = 64
    admin_snapshot_interval: int = 32
    decode_clear_interval: int = 16384


    def __post_init__(self):
         # Validate mutual exclusivity
        if self.chunked_prefill_tokens > 0 and self.use_paged_cache:
            logger.warning(
                 "chunked_prefill_tokens and use_paged_cache both enabled — "
                 "disabling paged cache to avoid conflicts"
             )
            self.use_paged_cache = False
         # Validate ranges
        if self.max_num_seqs < 1:
            self.max_num_seqs = 1
        if self.max_num_batched_tokens < self.max_num_seqs:
            self.max_num_batched_tokens = self.max_num_seqs
        if self.cache_memory_percent < 0 or self.cache_memory_percent > 1:
            self.cache_memory_percent = 0.20
        if self.suffix_min_confidence < 0 or self.suffix_min_confidence > 1:
            self.suffix_min_confidence = 0.3
        if self.kv_cache_quantization_bits not in (4, 8, 16):
            self.kv_cache_quantization_bits = 8
        # Sync the runtime bool from the released int knob. The Scheduler
        # runtime reads ``config.chunked_prefill`` (bool); the released CLI
        # sets ``chunked_prefill_tokens`` (int). Respect an explicit True
        # (the multi-model converter passes one).
        self.chunked_prefill = self.chunked_prefill or (
            self.chunked_prefill_tokens > 0
        )

@dataclass
class MemoryConfig:
    """Process memory enforcement configuration."""

    tier: MemoryTier = MemoryTier.BALANCED
    # Custom memory limit in MB (used when tier=CUSTOM)
    custom_limit_mb: int | None = None
    # Per-engine memory percentage of total budget
    per_engine_pct: float = 0.7
    # Enable SSD cold layer for evicted KV blocks
    ssd_cache_enabled: bool = False
    ssd_cache_dir: str = "~/Library/Caches/fusion-mlx/ssd"
    ssd_cache_max_bytes: int = 20 * 1024 * 1024 * 1024  # 20 GiB
    # Memory enforcer watermark thresholds (fraction of hard ceiling)
    soft_threshold: float = 0.85
    hard_threshold: float = 0.95


    def __post_init__(self):
        if self.per_engine_pct < 0 or self.per_engine_pct > 1:
            self.per_engine_pct = 0.7
        if self.ssd_cache_max_bytes < 1024 * 1024:
             # Min 1 MB for SSD cache
            self.ssd_cache_max_bytes = 1024 * 1024
        if self.custom_limit_mb is not None and self.custom_limit_mb < 100:
             # Min 100 MB for custom limit
            self.custom_limit_mb = 100
        # Clamp thresholds to valid range
        if self.soft_threshold < 0.1 or self.soft_threshold > 0.99:
            self.soft_threshold = 0.85
        if self.hard_threshold < 0.1 or self.hard_threshold > 0.99:
            self.hard_threshold = 0.95
        # Ensure soft < hard
        if self.soft_threshold >= self.hard_threshold:
            self.hard_threshold = min(0.99, self.soft_threshold + 0.1)

@dataclass
class ServerConfig:
    """FastAPI server configuration."""

    host: str = "0.0.0.0"
    port: int = 8000
    # Model directory (defaults to ~/.fusion-mlx/models)
    model_dir: str | None = None
    # Settings directory (defaults to ~/.fusion-mlx)
    settings_dir: str | None = None
    # Memory config
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    # Scheduler config
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    # Model aliases: friendly name -> real model ID
    model_aliases: dict[str, str] = field(default_factory=dict)
    # Enable admin UI
    admin_enabled: bool = True
    # Enable cloud router fallback
    cloud_router_enabled: bool = False
    # Cloud router API key (for fallback to cloud providers)
    cloud_router_api_key: str | None = None
    # Cloud router threshold (tokens) - route uncached requests above this to cloud
    cloud_router_threshold: int = 32768

    def __post_init__(self) -> None:
        if self.model_dir is None:
            self.model_dir = str(Path.home() / ".fusion-mlx" / "models")
        if self.settings_dir is None:
            self.settings_dir = str(Path.home() / ".fusion-mlx")



# Model config loader
def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and k in merged and isinstance(merged[k], dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def _load_model_config() -> dict[str, Any]:
    config = {}
    pkg_config = Path(__file__).parent / "model-config.json"
    if pkg_config.exists():
        try:
            with open(pkg_config) as f:
                config.update(json.load(f))
        except json.JSONDecodeError as e:
            logger.warning(f"pkg model-config.json invalid JSON: {e}. Skipping pkg defaults.")
        except Exception as e:
            logger.warning(f"Failed to read pkg model-config.json: {e}")

    user_config = Path.home() / ".fusion-mlx" / "model-config.json"
    if user_config.exists():
        try:
            with open(user_config) as f:
                user_data = json.load(f)
            config = _deep_merge(config, user_data)
        except json.JSONDecodeError as e:
            logger.warning(f"User model-config.json invalid JSON: {e}. Using pkg defaults only.")
        except Exception as e:
            logger.warning(f"Failed to read user model-config.json: {e}")

    if not config:
        return {"aliases": {}, "models": {}}
    return config


_model_config = _load_model_config()
DEFAULT_ALIASES: dict[str, str] = _model_config.get("aliases", {})

# Global config singleton
_config: ServerConfig | None = None


def get_config() -> ServerConfig:
    global _config
    if _config is None:
        _config = ServerConfig()
    return _config


def reset_config() -> ServerConfig:
    global _config
    _config = ServerConfig()
    return _config
