"""Dispatch — request routing by modality and phase.

Routes requests to the correct engine based on content type,
prompt length, task priority, and cache hit rate. Supports
phase-aware routing where prefill and decode can use different engines.
"""

from .cloud_router import CloudRouter
from .router import RequestRouter
from .smart_router import (
    BenchmarkResult,
    EngineBackend,
    PhaseHandoff,
    RouteDecision,
    RouterConfig,
    SmartRouter,
    TaskPriority,
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
