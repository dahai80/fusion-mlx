# SPDX-License-Identifier: Apache-2.0
import logging
import threading
from typing import Any

from fastapi import APIRouter, Depends
from fastapi.responses import PlainTextResponse

from .._version import __version__
from ..api import response_format_metrics
from ..middleware.auth import verify_management_access
from ..server_metrics import get_server_metrics

logger = logging.getLogger(__name__)

router = APIRouter()

_CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"


class _StickyCounterAccumulator:
    def __init__(self):
        self._state: dict[str, tuple[int, int]] = {}
        self._lock = threading.Lock()

    def advance(self, key: str, raw: int) -> int:
        raw = max(0, int(raw))
        with self._lock:
            last_raw, baseline = self._state.get(key, (0, 0))
            if raw < last_raw:
                baseline = baseline + last_raw
            self._state[key] = (raw, baseline)
            return baseline + raw


_cache_counter_accumulator = _StickyCounterAccumulator()


def _reset_accumulator_for_tests() -> None:
    # Test-only hook: reset the sticky counter accumulator to a fresh
    # (last_raw=0, baseline=0) state so each test sees clean counters.
    # The live server never calls this; reassigning the module global is
    # safe because render_prometheus_metrics reads it at call time.
    global _cache_counter_accumulator
    _cache_counter_accumulator = _StickyCounterAccumulator()
    logger.debug("reset sticky counter accumulator for tests")


def _escape_label_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _fmt_metric(
    name: str,
    metric_type: str,
    help_text: str,
    value: float | int,
    labels: dict[str, str] | None = None,
) -> list[str]:
    out = [
        f"# HELP {name} {help_text}",
        f"# TYPE {name} {metric_type}",
    ]
    if labels:
        label_str = ",".join(
            f'{k}="{_escape_label_value(str(v))}"' for k, v in labels.items()
        )
        out.append(f"{name}{{{label_str}}} {value}")
    else:
        out.append(f"{name} {value}")
    return out


def _coerce_number(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _render_build_info() -> list[str]:
    return _fmt_metric(
        "fusion_mlx_build_info",
        "gauge",
        "Build metadata (version, engine_type). Always 1.",
        1,
        {"version": __version__},
    )


def _render_engine_metrics() -> list[str]:
    lines: list[str] = []
    try:
        from ..server_metrics import get_server_metrics

        m = get_server_metrics().to_dict()
        lines.extend(
            _fmt_metric(
                "fusion_mlx_requests_total",
                "counter",
                "Total inference requests processed.",
                int(m.get("total_requests", 0)),
            )
        )
        lines.extend(
            _fmt_metric(
                "fusion_mlx_prompt_tokens_total",
                "counter",
                "Total prompt tokens across all requests.",
                int(m.get("total_prompt_tokens", 0)),
            )
        )
        lines.extend(
            _fmt_metric(
                "fusion_mlx_completion_tokens_total",
                "counter",
                "Total completion tokens across all requests.",
                int(m.get("total_tokens_generated", 0)),
            )
        )
    except Exception as e:
        logger.debug("metrics render error: %s", e)
    return lines


def _render_pool_metrics() -> list[str]:
    lines: list[str] = []
    try:
        from ..server import _server_state

        pool = _server_state.get("engine_pool")
        if pool is not None:
            lines.extend(
                _fmt_metric(
                    "fusion_mlx_models_discovered",
                    "gauge",
                    "Number of models discovered in model_dir.",
                    pool.model_count,
                )
            )
            lines.extend(
                _fmt_metric(
                    "fusion_mlx_models_loaded",
                    "gauge",
                    "Number of models currently loaded.",
                    pool.loaded_model_count,
                )
            )
            mem = pool.current_model_memory
            lines.extend(
                _fmt_metric(
                    "fusion_mlx_model_memory_bytes",
                    "gauge",
                    "GPU memory used by loaded models.",
                    mem,
                )
            )
    except Exception as e:
        logger.debug("pool metrics render error: %s", e)
    return lines


def _render_kv_cache_dtype_gauge() -> list[str]:
    dtype = "bf16"
    lines: list[str] = [
        "# HELP fusion_mlx_kv_cache_dtype Effective KV cache dtype. One series per dtype label; the value is 1 for the active dtype and 0 for the others.",
        "# TYPE fusion_mlx_kv_cache_dtype gauge",
    ]
    for candidate in ("bf16", "int8", "int4"):
        active = 1 if dtype == candidate else 0
        lines.append(f'fusion_mlx_kv_cache_dtype{{dtype="{candidate}"}} {active}')
    return lines


def _render_response_format_metrics() -> list[str]:
    snap = response_format_metrics.snapshot()
    lines: list[str] = []
    lines.extend(
        _fmt_metric(
            "fusion_mlx_response_format_strict_total",
            "counter",
            "Total strict json_schema response_format requests seen.",
            snap.get("strict_requests_total", 0),
        )
    )
    lines.extend(
        _fmt_metric(
            "fusion_mlx_response_format_strict_violations_total",
            "counter",
            "Strict json_schema outputs that violated the schema.",
            snap.get("strict_violations_total", 0),
        )
    )
    lines.extend(
        _fmt_metric(
            "fusion_mlx_response_format_strict_repairs_attempted_total",
            "counter",
            "Strict json_schema repair retries attempted.",
            snap.get("strict_repairs_attempted_total", 0),
        )
    )
    lines.extend(
        _fmt_metric(
            "fusion_mlx_response_format_strict_repairs_succeeded_total",
            "counter",
            "Strict json_schema repair retries that produced valid output.",
            snap.get("strict_repairs_succeeded_total", 0),
        )
    )
    lines.extend(
        _fmt_metric(
            "fusion_mlx_response_format_strict_repairs_skipped_context_overflow_total",
            "counter",
            "Strict repair retries skipped due to context overflow.",
            snap.get("strict_repairs_skipped_context_overflow_total", 0),
        )
    )
    return lines


def _render_response_cache_metrics() -> list[str]:
    lines: list[str] = []
    try:
        from ..cache.response_cache import get_response_cache

        stats = get_response_cache().stats.to_dict()
        lines.extend(
            _fmt_metric(
                "fusion_mlx_response_cache_hits_total",
                "counter",
                "Response cache hit count.",
                int(stats.get("hits", 0)),
            )
        )
        lines.extend(
            _fmt_metric(
                "fusion_mlx_response_cache_misses_total",
                "counter",
                "Response cache miss count.",
                int(stats.get("misses", 0)),
            )
        )
        lines.extend(
            _fmt_metric(
                "fusion_mlx_response_cache_evictions_total",
                "counter",
                "Response cache eviction count.",
                int(stats.get("evictions", 0)),
            )
        )
        lines.extend(
            _fmt_metric(
                "fusion_mlx_response_cache_hit_rate",
                "gauge",
                "Response cache hit rate (0-1).",
                float(stats.get("hit_rate", 0)),
            )
        )
        lines.extend(
            _fmt_metric(
                "fusion_mlx_response_cache_entries",
                "gauge",
                "Current number of cached responses.",
                int(stats.get("entry_count", 0)),
            )
        )
        lines.extend(
            _fmt_metric(
                "fusion_mlx_response_cache_size_bytes",
                "gauge",
                "Total size of cached responses in bytes.",
                int(stats.get("size_bytes", 0)),
            )
        )
    except Exception as e:
        logger.debug("response cache metrics render error: %s", e)
    return lines


def _render_disconnect_metrics() -> list[str]:
    sm = get_server_metrics().to_dict()
    return _fmt_metric(
        "fusion_mlx_requests_cancelled_total",
        "counter",
        "Client-disconnected requests (streaming + non-stream)",
        sm["cancelled_requests"],
        None,
    )


def _render_queue_metrics() -> list[str]:
    lines: list[str] = []
    try:
        from ..server import _server_state

        pool = _server_state.get("engine_pool")
        running = 0
        waiting = 0
        if pool is not None:
            for _mid, entry in getattr(pool, "_entries", {}).items():
                engine = getattr(entry, "engine", None)
                if engine is None:
                    continue
                get_stats = getattr(engine, "get_stats", None)
                if callable(get_stats):
                    stats = get_stats() or {}
                    running += int(stats.get("num_running", 0))
                    waiting += int(stats.get("num_waiting", 0))
                else:
                    sched = getattr(engine, "scheduler", None)
                    if sched is None:
                        sched = getattr(
                            getattr(engine, "_engine", None), "scheduler", None
                        )
                        if sched is not None:
                            sched = getattr(
                                getattr(sched, "engine", None),
                                "scheduler",
                                None,
                            )
                    if sched is None:
                        continue
                    gs = getattr(sched, "get_stats", None)
                    if callable(gs):
                        stats = gs() or {}
                        running += int(stats.get("num_running", 0))
                        waiting += int(stats.get("num_waiting", 0))
                    else:
                        running += len(getattr(sched, "running", []) or [])
                        waiting += len(getattr(sched, "waiting", []) or [])
        lines.extend(
            _fmt_metric(
                "fusion_mlx_requests_running",
                "gauge",
                "Currently running inference requests.",
                running,
            )
        )
        lines.extend(
            _fmt_metric(
                "fusion_mlx_requests_waiting",
                "gauge",
                "Requests waiting in scheduler queues.",
                waiting,
            )
        )
    except Exception:
        logger.debug("queue metrics render error", exc_info=True)
    return lines


def _render_uptime_metal() -> list[str]:
    lines: list[str] = []
    try:
        m = get_server_metrics().to_dict()
        lines.extend(
            _fmt_metric(
                "fusion_mlx_uptime_seconds",
                "gauge",
                "Process uptime in seconds.",
                float(m.get("uptime_seconds", 0)),
            )
        )
    except Exception:
        logger.debug("uptime metric render error", exc_info=True)
    try:
        import mlx.core as mx

        if mx.metal.is_available():
            lines.extend(
                _fmt_metric(
                    "fusion_mlx_metal_active_bytes",
                    "gauge",
                    "MLX Metal active memory in bytes.",
                    int(mx.get_active_memory() or 0),
                )
            )
            lines.extend(
                _fmt_metric(
                    "fusion_mlx_metal_cache_bytes",
                    "gauge",
                    "MLX Metal cache memory in bytes.",
                    int(mx.get_cache_memory() or 0),
                )
            )
            lines.extend(
                _fmt_metric(
                    "fusion_mlx_metal_peak_bytes",
                    "gauge",
                    "MLX Metal peak memory in bytes.",
                    int(mx.get_peak_memory() or 0),
                )
            )
    except Exception:
        logger.debug("mlx memory metrics unavailable", exc_info=True)
    return lines


def render_prometheus_metrics() -> str:
    lines: list[str] = []
    lines.extend(_render_build_info())
    lines.extend(_render_engine_metrics())
    lines.extend(_render_disconnect_metrics())
    lines.extend(_render_pool_metrics())
    lines.extend(_render_queue_metrics())
    lines.extend(_render_uptime_metal())
    lines.extend(_render_kv_cache_dtype_gauge())
    lines.extend(_render_response_format_metrics())
    lines.extend(_render_response_cache_metrics())
    return "\n".join(lines) + "\n"


@router.get("/metrics")
async def prometheus_metrics(_auth: bool = Depends(verify_management_access)):
    body = render_prometheus_metrics()
    return PlainTextResponse(content=body, media_type=_CONTENT_TYPE)
