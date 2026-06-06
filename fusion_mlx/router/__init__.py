"""Router — request routing by modality and phase.

Routes requests to the correct engine based on content type,
prompt length, task priority, and cache hit rate. Supports
phase-aware routing where prefill and decode can use different engines.
"""

from .router import RequestRouter
from .cloud_router import CloudRouter
from .smart_router import (
    SmartRouter,
    RouterConfig,
    RouteDecision,
    TaskPriority,
    EngineBackend,
    PhaseHandoff,
    BenchmarkResult,
)

__all__ = [
     "RequestRouter",
     "CloudRouter",
     "SmartRouter",
     "RouterConfig",
     "RouteDecision",
     "TaskPriority",
     "EngineBackend",
     "PhaseHandoff",
     "BenchmarkResult",
]
