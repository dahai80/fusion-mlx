"""Server metrics tracking for fusion-mlx."""

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

_KV_CACHE_DTYPE_KNOWN = ("bf16", "int8", "int4")


def _resolve_kv_cache_dtype() -> str:
    # Effective KV cache dtype for /metrics observability. Priority:
    #   1. ServerConfig.kv_cache_dtype stash — set by cli_serve after the
    #      safelist resolves --kv-cache-dtype, or after the legacy
    #      --kv-cache-quantization synthesis. Primary source.
    #   2. Legacy fallback — if the stash is unset / at the bf16 default
    #      but ServerConfig.scheduler carries kv_cache_quantization=True,
    #      derive from kv_cache_quantization_bits (programmatic callers
    #      that bypass the serve CLI).
    #   3. Default "bf16" — the only no-op value, so observability never
    #      lies about quantization status.
    # Adapted from rapid-mlx's _render_kv_cache_dtype_gauge: fusion uses an
    # engine POOL (no single engine on cfg), so rapid-mlx's "engine
    # scheduler_config wins over stale stash" step is dropped — fusion's
    # stash is set post-resolution pre-load, so stash == engine value with
    # no stale-stash race.
    dtype: str | None = None
    try:
        from .config import get_config

        cfg = get_config()
        dtype = getattr(cfg, "kv_cache_dtype", None)
        if dtype in (None, "bf16"):
            scheduler = getattr(cfg, "scheduler", None)
            if scheduler is not None and getattr(
                scheduler, "kv_cache_quantization", False
            ):
                bits = getattr(scheduler, "kv_cache_quantization_bits", None)
                if bits == 4:
                    dtype = "int4"
                elif bits == 8:
                    dtype = "int8"
    except Exception as exc:
        logger.warning("kv_cache_dtype resolution failed: %s", exc)
        dtype = None
    if dtype not in _KV_CACHE_DTYPE_KNOWN:
        return "bf16"
    return dtype


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
        self._start_time = time.monotonic()

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
        generation_duration: float = 0.0,
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
                    stats["avg_prefill_tps"] = (
                        old_avg * (stats["requests"] - 1) + tps
                    ) / stats["requests"]
                if generation_duration > 0 and completion_tokens > 0:
                    tps = completion_tokens / generation_duration
                    old_avg = stats["avg_generation_tps"]
                    stats["avg_generation_tps"] = (
                        old_avg * (stats["requests"] - 1) + tps
                    ) / stats["requests"]

    def uptime_seconds(self) -> float:
        return time.monotonic() - self._start_time

    def clear_metrics(self) -> None:
        with self._lock:
            self.total_requests = 0
            self.successful_requests = 0
            self.failed_requests = 0
            self.total_tokens_generated = 0
            self.total_tokens_prompt = 0
            self.total_cached_tokens = 0
            self.active_requests = 0
            self.model_stats.clear()

    def clear_alltime_metrics(self) -> None:
        self.clear_metrics()

    def to_dict(self) -> dict:
        """Return a JSON-safe dict, excluding internal lock."""
        total_prompt = self.total_tokens_prompt
        total_cached = self.total_cached_tokens
        total_gen = self.total_tokens_generated
        n_models = len(self.model_stats)
        avg_prefill = (
            sum(s.get("avg_prefill_tps", 0.0) for s in self.model_stats.values())
            / n_models
            if n_models
            else 0.0
        )
        avg_gen = (
            sum(s.get("avg_generation_tps", 0.0) for s in self.model_stats.values())
            / n_models
            if n_models
            else 0.0
        )
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "total_tokens_generated": total_gen,
            "total_prompt_tokens": total_prompt,
            "total_tokens_served": total_gen,
            "total_completion_tokens": total_gen,
            "total_cached_tokens": total_cached,
            "cache_efficiency": total_cached / max(1, total_prompt),
            "active_requests": self.active_requests,
            "model_stats": self.model_stats,
            "avg_prefill_tps": avg_prefill,
            "avg_generation_tps": avg_gen,
            "uptime_seconds": self.uptime_seconds(),
            "kv_cache_dtype": _resolve_kv_cache_dtype(),
        }


# Global singleton
_metrics = ServerMetrics()


def get_server_metrics() -> ServerMetrics:
    return _metrics


def record_llm_metrics(
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    cached_tokens: int = 0,
    prefill_duration: float = 0.0,
    generation_duration: float = 0.0,
    model_id: str | None = None,
) -> None:
    try:
        get_server_metrics().record_request_complete(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_tokens,
            prefill_duration=prefill_duration,
            generation_duration=generation_duration,
            model_id=model_id,
        )
    except Exception as exc:
        logger.debug("Failed to record LLM metrics for %s: %s", model_id, exc)
