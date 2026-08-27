"""Server metrics tracking for fusion-mlx."""

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_KV_CACHE_DTYPE_KNOWN = ("bf16", "int8", "int4")
_STATS_JSON = Path.home() / ".fusion-mlx" / "stats.json"
_ALLTIME_SAVE_INTERVAL = 10.0


def _load_alltime_from_disk() -> dict:
    try:
        if _STATS_JSON.exists():
            data = json.loads(_STATS_JSON.read_text())
            if isinstance(data, dict):
                return data
    except Exception as exc:
        logger.debug("Failed to load alltime stats: %s", exc)
    return {}


def _save_alltime_to_disk(data: dict) -> None:
    try:
        _STATS_JSON.parent.mkdir(parents=True, exist_ok=True)
        _STATS_JSON.write_text(json.dumps(data, indent=2))
    except Exception as exc:
        logger.debug("Failed to save alltime stats: %s", exc)


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
    cancelled_requests: int = 0
    # Per-model stats: model_name -> dict
    model_stats: dict[str, dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self):
        self._lock = threading.Lock()
        self._start_time = time.monotonic()
        self._alltime = _load_alltime_from_disk()
        self._alltime_dirty = False
        self._alltime_last_save = time.monotonic()

    def inc_tokens(self, generated: int = 0, prompt: int = 0, cached: int = 0) -> None:
        with self._lock:
            self.total_tokens_generated += generated
            self.total_tokens_prompt += prompt
            self.total_cached_tokens += cached

    def update_active_requests(self, delta: int) -> None:
        with self._lock:
            self.active_requests += delta

    def record_disconnect_cancel(self) -> None:
        with self._lock:
            self.cancelled_requests += 1
            at = self._alltime
            at["total_cancelled_requests"] = at.get("total_cancelled_requests", 0) + 1
            self._alltime_dirty = True
            now = time.monotonic()
            if now - self._alltime_last_save >= _ALLTIME_SAVE_INTERVAL:
                _save_alltime_to_disk(self._alltime)
                self._alltime_dirty = False
                self._alltime_last_save = now

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

            # Update alltime accumulators
            at = self._alltime
            at["total_requests"] = at.get("total_requests", 0) + 1
            at["total_prompt_tokens"] = at.get("total_prompt_tokens", 0) + prompt_tokens
            at["total_completion_tokens"] = (
                at.get("total_completion_tokens", 0) + completion_tokens
            )
            at["total_cached_tokens"] = at.get("total_cached_tokens", 0) + cached_tokens
            if model_id:
                models = at.setdefault("model_stats", {})
                ms = models.get(
                    model_id,
                    {
                        "requests": 0,
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                    },
                )
                ms["requests"] = ms.get("requests", 0) + 1
                ms["prompt_tokens"] = ms.get("prompt_tokens", 0) + prompt_tokens
                ms["completion_tokens"] = (
                    ms.get("completion_tokens", 0) + completion_tokens
                )
                models[model_id] = ms
            self._alltime_dirty = True
            now = time.monotonic()
            if now - self._alltime_last_save >= _ALLTIME_SAVE_INTERVAL:
                _save_alltime_to_disk(self._alltime)
                self._alltime_dirty = False
                self._alltime_last_save = now

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
            self.cancelled_requests = 0
            self.model_stats.clear()

    def clear_alltime_metrics(self) -> None:
        with self._lock:
            self._alltime = {}
            self._alltime_dirty = False
            _save_alltime_to_disk({})
        self.clear_metrics()

    def flush_alltime(self) -> None:
        with self._lock:
            if self._alltime_dirty:
                _save_alltime_to_disk(self._alltime)
                self._alltime_dirty = False
                self._alltime_last_save = time.monotonic()

    def to_alltime_dict(self) -> dict:
        with self._lock:
            at = dict(self._alltime)
        total_prompt = at.get("total_prompt_tokens", 0)
        total_cached = at.get("total_cached_tokens", 0)
        total_gen = at.get("total_completion_tokens", 0)
        model_stats = at.get("model_stats", {})
        avg_prefill = 0.0
        avg_gen = 0.0
        with self._lock:
            n_session = len(self.model_stats)
            if n_session:
                avg_prefill = (
                    sum(
                        s.get("avg_prefill_tps", 0.0) for s in self.model_stats.values()
                    )
                    / n_session
                )
                avg_gen = (
                    sum(
                        s.get("avg_generation_tps", 0.0)
                        for s in self.model_stats.values()
                    )
                    / n_session
                )
        return {
            "total_requests": at.get("total_requests", 0),
            "total_cancelled_requests": at.get("total_cancelled_requests", 0),
            "total_prompt_tokens": total_prompt,
            "total_completion_tokens": total_gen,
            "total_tokens_served": total_gen,
            "total_cached_tokens": total_cached,
            "cache_efficiency": total_cached / max(1, total_prompt),
            "model_stats": model_stats,
            "avg_prefill_tps": avg_prefill,
            "avg_generation_tps": avg_gen,
        }

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
            "cancelled_requests": self.cancelled_requests,
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


def record_llm_disconnect_cancel() -> None:
    try:
        get_server_metrics().record_disconnect_cancel()
    except Exception as exc:
        logger.debug("Failed to record disconnect cancel: %s", exc)
