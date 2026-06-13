"""Model pool - multi-model lifecycle management.

EnginePool with LRU eviction, pinning, TTL auto-unload, and
4-tier ProcessMemoryEnforcer (safe/balanced/aggressive/custom).
Plus UnifiedMemoryPool for cross-engine memory coordination and
PriorityScheduler for Metal multi-queue scheduling.
"""

from .engine_pool import EnginePool
from .memory_enforcer import MemoryProfile, ProcessMemoryEnforcer
from .model_discovery import ModelDiscovery
from .priority_scheduler import (
    PriorityLevel,
    PriorityRequest,
    PriorityScheduler,
    PrioritySchedulerConfig,
    ScheduleDecision,
)
from .unified_memory_pool import (
    BackendQuota,
    FragmentationMonitor,
    KVCacheBridge,
    KVCacheState,
    MetalBufferRegistry,
    UnifiedMemoryPool,
)

__all__ = [
    "EnginePool",
    "ProcessMemoryEnforcer",
    "MemoryProfile",
    "ModelDiscovery",
    "UnifiedMemoryPool",
    "MetalBufferRegistry",
    "KVCacheBridge",
    "KVCacheState",
    "BackendQuota",
    "FragmentationMonitor",
    "PriorityScheduler",
    "PrioritySchedulerConfig",
    "PriorityLevel",
    "PriorityRequest",
    "ScheduleDecision",
]
