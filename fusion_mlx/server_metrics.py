"""Server metrics tracking for fusion-mlx."""

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class ServerMetrics:
    """Collects server-level metrics."""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_tokens_generated: int = 0
    total_tokens_prompt: int = 0
    total_cached_tokens: int = 0
    active_requests: int = 0
    # Per-model stats: model_name -> dict
    model_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)


# Global singleton
_metrics = ServerMetrics()


def get_server_metrics() -> ServerMetrics:
    """Get the global server metrics instance."""
    return _metrics
