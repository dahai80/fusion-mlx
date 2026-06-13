"""Server metrics tracking for fusion-mlx."""

import threading
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

    def __post_init__(self):
        self._lock = threading.Lock()

    def inc_tokens(self, generated: int = 0, prompt: int = 0, cached: int = 0) -> None:
        with self._lock:
            self.total_tokens_generated += generated
            self.total_tokens_prompt += prompt
            self.total_cached_tokens += cached

    def update_active_requests(self, delta: int) -> None:
        with self._lock:
            self.active_requests += delta


# Global singleton
_metrics = ServerMetrics()


def get_server_metrics() -> ServerMetrics:
    """Get the global server metrics instance."""
    return _metrics
