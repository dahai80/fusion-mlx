"""Global configuration for fusion-mlx.

Merged from omlx model_settings + Rapid-MLX SchedulerConfig.
"""

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

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
    max_num_batched_tokens: int = 8192
    policy: SchedulingPolicy = SchedulingPolicy.FCFS

    # BatchGenerator settings
    prefill_batch_size: int = 8
    completion_batch_size: int = 32
    prefill_step_size: int = 2048

    # Chunked prefill
    chunked_prefill_tokens: int = 0  # 0 = disabled
    # Mid-prefill cache saving every N tokens (from Rapid-MLX)
    mid_prefill_save_interval: int = 8192

    # Prefix cache
    enable_prefix_cache: bool = True
    prefix_cache_size: int = 100

    # Memory-aware cache
    use_memory_aware_cache: bool = True
    cache_memory_mb: Optional[int] = None  # None = auto (20% available RAM)
    cache_memory_percent: float = 0.20

    # KV cache quantization
    kv_cache_quantization: bool = False
    kv_cache_quantization_bits: int = 8
    kv_cache_quantization_group_size: int = 64
    kv_cache_min_quantize_tokens: int = 256

    # TurboQuant V-only compression
    kv_cache_turboquant: bool = False
    kv_cache_turboquant_bits: Optional[int] = None
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


@dataclass
class MemoryConfig:
    """Process memory enforcement configuration."""

    tier: MemoryTier = MemoryTier.BALANCED
    # Custom memory limit in MB (used when tier=CUSTOM)
    custom_limit_mb: Optional[int] = None
    # Per-engine memory percentage of total budget
    per_engine_pct: float = 0.7
    # Enable SSD cold layer for evicted KV blocks
    ssd_cache_enabled: bool = False
    ssd_cache_dir: str = "~/Library/Caches/fusion-mlx/ssd"
    ssd_cache_max_bytes: int = 20 * 1024 * 1024 * 1024  # 20 GiB


@dataclass
class ServerConfig:
    """FastAPI server configuration."""

    host: str = "0.0.0.0"
    port: int = 8000
    # Model directory (defaults to ~/.fusion-mlx/models)
    model_dir: Optional[str] = None
    # Settings directory (defaults to ~/.fusion-mlx)
    settings_dir: Optional[str] = None
    # Memory config
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    # Scheduler config
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    # Model aliases: friendly name -> real model ID
    model_aliases: Dict[str, str] = field(default_factory=dict)
    # Enable admin UI
    admin_enabled: bool = True
    # Enable cloud router fallback
    cloud_router_enabled: bool = False
    # Cloud router API key (for fallback to cloud providers)
    cloud_router_api_key: Optional[str] = None
    # Cloud router threshold (tokens) - route uncached requests above this to cloud
    cloud_router_threshold: int = 32768

    def __post_init__(self) -> None:
        if self.model_dir is None:
            self.model_dir = str(Path.home() / ".fusion-mlx" / "models")
        if self.settings_dir is None:
            self.settings_dir = str(Path.home() / ".fusion-mlx")



# Model config loader
def _load_model_config() -> Dict[str, Any]:
    """Load model config from JSON file."""
    user_config = Path.home() / ".fusion-mlx" / "model-config.json"
    if user_config.exists():
        with open(user_config) as f:
            return json.load(f)
    pkg_config = Path(__file__).parent / "model-config.json"
    if pkg_config.exists():
        with open(pkg_config) as f:
            return json.load(f)
    return {"aliases": {}, "models": {}}


_model_config = _load_model_config()
DEFAULT_ALIASES: Dict[str, str] = _model_config.get("aliases", {})
