# SPDX-License-Identifier: Apache-2.0
import logging
import threading
from typing import Any

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse

from .._version import __version__

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
                int(m.get("total_tokens_prompt", 0)),
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
        from ..service.helpers import _server_state

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


def render_prometheus_metrics() -> str:
    lines: list[str] = []
    lines.extend(_render_build_info())
    lines.extend(_render_engine_metrics())
    lines.extend(_render_pool_metrics())
    lines.extend(_render_kv_cache_dtype_gauge())
    return "\n".join(lines) + "\n"


@router.get("/metrics")
async def prometheus_metrics():
    body = render_prometheus_metrics()
    return PlainTextResponse(content=body, media_type=_CONTENT_TYPE)
