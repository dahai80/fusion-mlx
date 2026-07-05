# SPDX-License-Identifier: Apache-2.0
"""mlx-kv-cache — KV cache management layer for fusion-mlx.

Merged from omlx/cache + Rapid-MLX cache implementations.
"""

# Stats
# Boundary snapshots
from .boundary_snapshot_store import BoundarySnapshotSSDStore

# Factory
from .factory import CacheConfig, CacheFactory

# Hybrid cache config
from .hybrid_cache import (
    LayerCacheConfig,
    ModelCacheConfig,
    create_default_kvcache_config,
)

# Interfaces
from .interface import CacheManager

# MLLM cache (from Rapid-MLX)
from .mllm_cache import (
    MLLMCacheStats,
    MLLMPrefixCacheEntry,
    MLLMPrefixCacheManager,
    compute_image_hash,
    compute_images_hash,
)

# Observability
from .observability import CacheRateTracker

# Paged cache implementations
from .paged_cache import (
    BlockHashToBlockMap,
    BlockTable,
    CacheBlock,
    FreeKVCacheBlockQueue,
    PagedCacheManager,
)

# Paged SSD cache implementations
from .paged_ssd_cache import (
    _CACHE_FORMAT_VERSION,
    _MAX_INLINE_UNLINKS_PER_SAVE,
    PagedSSDBlockMetadata,
    PagedSSDCacheIndex,
    PagedSSDCacheManager,
    SharedHotCacheBudget,
    _cache_compat_signature,
    _canonicalize_layer_cache_types,
    _compute_max_pending_writes,
    parse_size,
)
from .paged_ssd_cache import (
    PagedSSDCacheStats as SSDCacheStatsData,
)

# Prefix cache implementations
from .prefix_cache import (
    BlockAwarePrefixCache,
    BlockCacheEntry,
)

# Managers
from .recovery import CacheRecoveryManager
from .stats import (
    BaseCacheStats,
    PagedCacheStats,
    PagedSSDCacheStats,
    VLMCacheStats,
)

# Type handlers
from .type_handlers import (
    ArraysCacheHandler,
    CacheListHandler,
    CacheStateInfo,
    CacheType,
    CacheTypeHandler,
    DefaultCacheHandler,
    KVCacheHandler,
    MiniMaxM3BatchKVCacheHandler,
    MiniMaxM3KVCacheHandler,
    RotatingKVCacheHandler,
    SizedArraysCache,
)

# Type registry
from .type_registry import CacheTypeRegistry

# Vision embedding cache (from Rapid-MLX)
from .vision_embedding_cache import (
    VisionCacheStats,
    VisionEmbeddingCache,
)

# Vision feature cache
from .vision_feature_cache import (
    VisionFeatureSSDCache,
    VisionFeatureSSDEntry,
)

__all__ = [
    # Stats
    "BaseCacheStats",
    "PagedCacheStats",
    "VLMCacheStats",
    "PagedSSDCacheStats",
    # Interfaces
    "CacheManager",
    # Paged cache
    "PagedCacheManager",
    "CacheBlock",
    "BlockTable",
    "FreeKVCacheBlockQueue",
    "BlockHashToBlockMap",
    # Prefix cache
    "BlockAwarePrefixCache",
    "BlockCacheEntry",
    # Paged SSD cache
    "PagedSSDCacheManager",
    "PagedSSDBlockMetadata",
    "PagedSSDCacheIndex",
    "parse_size",
    "_CACHE_FORMAT_VERSION",
    "_MAX_INLINE_UNLINKS_PER_SAVE",
    "_cache_compat_signature",
    "_canonicalize_layer_cache_types",
    "_compute_max_pending_writes",
    "SharedHotCacheBudget",
    # Vision feature cache
    "VisionFeatureSSDCache",
    "VisionFeatureSSDEntry",
    "VisionEmbeddingCache",
    "VisionCacheStats",
    "MLLMPrefixCacheManager",
    "MLLMPrefixCacheEntry",
    "compute_image_hash",
    "compute_images_hash",
    # Managers
    "CacheRecoveryManager",
    # Factory
    "CacheConfig",
    "CacheFactory",
    # Type handlers
    "CacheType",
    "CacheTypeHandler",
    "CacheStateInfo",
    "KVCacheHandler",
    "RotatingKVCacheHandler",
    "ArraysCacheHandler",
    "CacheListHandler",
    "DefaultCacheHandler",
    "SizedArraysCache",
    "MiniMaxM3KVCacheHandler",
    "MiniMaxM3BatchKVCacheHandler",
    # Type registry
    "CacheTypeRegistry",
    # Hybrid cache config
    "LayerCacheConfig",
    "ModelCacheConfig",
    "create_default_kvcache_config",
    # Observability
    "CacheRateTracker",
    # Boundary snapshots
    "BoundarySnapshotSSDStore",
]
