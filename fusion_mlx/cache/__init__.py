# SPDX-License-Identifier: Apache-2.0
"""mlx-kv-cache — KV cache management layer for fusion-mlx.

Merged from omlx/cache + Rapid-MLX cache implementations.
"""

# Stats
from .stats import (
    BaseCacheStats,
    PagedCacheStats,
    VLMCacheStats,
    PagedSSDCacheStats,
)

# Interfaces
from .interface import CacheManager

# Paged cache implementations
from .paged_cache import (
    PagedCacheManager,
    CacheBlock,
    BlockTable,
    FreeKVCacheBlockQueue,
    BlockHashToBlockMap,
)

# Prefix cache implementations
from .prefix_cache import (
    BlockAwarePrefixCache,
    BlockCacheEntry,
)

# Paged SSD cache implementations
from .paged_ssd_cache import (
    PagedSSDCacheManager,
    PagedSSDBlockMetadata,
    PagedSSDCacheIndex,
    parse_size,
)

# Vision feature cache
from .vision_feature_cache import (
    VisionFeatureSSDCache,
    VisionFeatureSSDEntry,
)

# Managers
from .recovery import CacheRecoveryManager

# Factory
from .factory import CacheConfig, CacheFactory

# Type handlers
from .type_handlers import (
    CacheType,
    CacheTypeHandler,
    CacheStateInfo,
    KVCacheHandler,
    RotatingKVCacheHandler,
    ArraysCacheHandler,
    CacheListHandler,
    DefaultCacheHandler,
    SizedArraysCache,
)

# Type registry
from .type_registry import CacheTypeRegistry

# Hybrid cache config
from .hybrid_cache import (
    LayerCacheConfig,
    ModelCacheConfig,
    create_default_kvcache_config,
)

# Observability
from .observability import CacheRateTracker

# Boundary snapshots
from .boundary_snapshot_store import BoundarySnapshotSSDStore

# MLLM cache (from Rapid-MLX)
from .mllm_cache import (
    MLLMCacheStats,
    MLLMPrefixCacheEntry,
    MLLMPrefixCacheManager,
    compute_image_hash,
    compute_images_hash,
)

# Vision embedding cache (from Rapid-MLX)
from .vision_embedding_cache import (
    VisionCacheStats,
    VisionEmbeddingCache,
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
