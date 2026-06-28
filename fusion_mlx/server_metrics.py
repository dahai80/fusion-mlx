"""Server metrics tracking for fusion-mlx."""

import threading
from dataclasses import dataclass, field
from typing import Any


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
    model_stats: dict[str, dict[str, Any]] = field(default_factory=dict)

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

    def record_request_complete(
        self,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cached_tokens: int = 0,
        prefill_duration: float = 0.0,
        model_id: str | None = None,
    ) -> None:
        with self._lock:
            self.total_requests += 1
            self.successful_requests += 1
            self.total_tokens_prompt += prompt_tokens
            self.total_tokens_generated += completion_tokens
            self.total_cached_tokens += cached_tokens
            if model_id:
                stats = self.model_stats.get(model_id)
                if stats is None:
                    stats = {
                        "requests": 0,
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "avg_prefill_tps": 0.0,
                        "avg_generation_tps": 0.0,
                    }
                    self.model_stats[model_id] = stats
                stats["requests"] += 1
                stats["prompt_tokens"] += prompt_tokens
                stats["completion_tokens"] += completion_tokens
                if prefill_duration > 0 and prompt_tokens > 0:
                    tps = prompt_tokens / prefill_duration
                    old_avg = stats["avg_prefill_tps"]
                    stats["avg_prefill_tps"] = (old_avg * (stats["requests"] - 1) + tps) / stats["requests"]

    def to_dict(self) -> dict:
        """Return a JSON-safe dict, excluding internal lock."""
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "total_tokens_generated": self.total_tokens_generated,
            "total_tokens_prompt": self.total_tokens_prompt,
            "total_cached_tokens": self.total_cached_tokens,
            "active_requests": self.active_requests,
            "model_stats": self.model_stats,
        }


# Global singleton
_metrics = ServerMetrics()


def get_server_metrics() -> ServerMetrics:
    """Get the global server metrics instance."""
    return _metrics
