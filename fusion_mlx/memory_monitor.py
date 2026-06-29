# SPDX-License-Identifier: Apache-2.0
"""Memory monitoring for fusion-mlx on Apple Silicon unified memory."""

import logging
import subprocess
from ctypes import CDLL, byref, c_uint32, c_uint64, c_void_p
from typing import Any

from fusion_mlx.utils.formatting import format_bytes

logger = logging.getLogger(__name__)

# macOS libsystem for host_statistics — initialised lazily
_libsystem: CDLL | None = None
_HOST_PAGE_SIZE = 4096

# mach_vm_statistics64 struct fields (13 uint64 fields)
_VM_STAT_FIELDS = 13


def _get_libsystem():
    global _libsystem
    if _libsystem is None:
        _libsystem = CDLL("/usr/lib/libSystem.B.dylib")
    return _libsystem


def _get_host_page_size() -> int:
    try:
        host = c_void_p(0)  # mach_host_self()
        host_page_size = c_uint64(0)
        _get_libsystem().host_page_size(host, byref(host_page_size))
        return host_page_size.value
    except Exception:
        return _HOST_PAGE_SIZE


def _get_vm_stat_fast() -> dict[str, int]:
    """Get vm stat via direct host_statistics64 syscall — zero subprocess overhead.

    Returns the same keys as the old _parse_vm_stat output (in bytes).
    """
    try:
        host = c_void_p(0)  # mach_host_self()
        page_size = _get_host_page_size()
        stat = (c_uint64 * _VM_STAT_FIELDS)()
        count = c_uint32(_VM_STAT_FIELDS)
        ret = _get_libsystem().host_statistics64(
            host, 2,  # HOST_VM_STAT
            c_void_p(stat), byref(count)
        )
        if ret != 0:
            return {}
        return {
            "free": stat[0] * page_size,
            "active": stat[1] * page_size,
            "inactive": stat[2] * page_size,
            "throttled": stat[3] * page_size,
            "wired": stat[4] * page_size,
            "purgeable": stat[5] * page_size,
            "speculative": stat[6] * page_size,
            "compressed": stat[7] * page_size,
        }
    except Exception:
        return {}


class MemoryMonitor:
    """Monitors unified memory, wired memory, and compressed memory on macOS.

    Uses psutil when available, falls back to vm_stat/sysctl for core metrics.
    Tracks MLX cache/GPU memory via mlx.core when importable.
    """

    def __init__(
        self,
        max_kv_cache_memory: int = 4 * 1024**3,
        paged_cache_manager: Any = None,
    ):
        self.max_kv_cache_memory = max_kv_cache_memory
        self._paged_cache_manager = paged_cache_manager
        self._has_psutil = False
        self._ps = None
        self._closed = False
        try:
            import psutil
            self._ps = psutil
            self._has_psutil = True
        except ImportError:
            pass
        # Cache total RAM at init — hardware value, never changes
        if self._has_psutil:
            self._total_memory = self._ps.virtual_memory().total
        else:
            sys_mem = self._get_sysctl_memory()
            self._total_memory = sys_mem.get("total", 0)

    def set_paged_cache_manager(self, paged_cache_manager: Any) -> None:
        self._paged_cache_manager = paged_cache_manager

    def _parse_vm_stat(self) -> dict[str, int]:
        """Parse vm_stat output into page counts.

        Returns pages keyed by metric name. Page size is 4096 on macOS.
        """
        try:
            result = subprocess.run(
                ["vm_stat"], capture_output=True, text=True, timeout=5
            )
            stats = {}
            page_size = 4096
            for line in result.stdout.splitlines():
                line = line.strip()
                if "Pages " not in line:
                    continue
                parts = line.split(":")
                if len(parts) < 2:
                    continue
                key = parts[0].replace("Pages ", "").strip()
                try:
                    val = int(parts[1].replace(",", "").replace(".", "").strip())
                    stats[key] = val * page_size
                except ValueError:
                    pass
            return stats
        except (subprocess.TimeoutExpired, OSError):
            return {}

    def _get_sysctl_memory(self) -> dict[str, int]:
        """Get system memory from sysctl."""
        keys = {"hw.memsize": "total"}
        result = {}
        for k, name in keys.items():
            try:
                r = subprocess.run(
                    ["sysctl", "-n", k], capture_output=True, text=True, timeout=5
                )
                result[name] = int(r.stdout.strip())
            except (subprocess.TimeoutExpired, OSError, ValueError):
                result[name] = 0
        return result

    def get_memory_usage(self) -> dict[str, Any]:
        """Return current memory usage snapshot.

        Returns safe defaults when the monitor has been closed.
        """
        if self._closed:
            return {
                "total": 0, "available": 0, "wired": 0,
                "compressed": 0, "active": 0, "inactive": 0,
                "mlx_cache": 0, "mlx_peak": 0,
                "paged_cache_blocks": 0, "paged_cache_hit_rate": 0.0,
            }

        if self._has_psutil:
            try:
                virt = self._ps.virtual_memory()
                total = self._total_memory
                available = virt.available
                wired = getattr(virt, "wired", 0)
                active = getattr(virt, "active", 0)
                inactive = getattr(virt, "inactive", 0)
                compressed = getattr(virt, "compressed", 0)
                if not compressed:
                    compressed = getattr(virt, "cached", 0)
            except Exception:
                # Fallback to subprocess on psutil failure
                mem = self._parse_vm_stat()
                total = self._total_memory
                inactive = mem.get("inactive", 0)
                free = mem.get("free", 0)
                wired = mem.get("wired", 0)
                active = mem.get("active", 0)
                compressed = 0
                available = inactive + free
        else:
            mem = self._parse_vm_stat()
            total = self._total_memory
            inactive = mem.get("inactive", 0)
            free = mem.get("free", 0)
            wired = mem.get("wired", 0)
            active = mem.get("active", 0)
            compressed = 0
            available = inactive + free

        mlx_cache = 0
        mlx_peak = 0
        try:
            import mlx.core as mx
            mlx_cache = mx.get_cache_memory()
            mlx_peak = mx.get_peak_memory()
        except Exception:
            pass

        paged_blocks = 0
        paged_hit_rate = 0.0
        if self._paged_cache_manager is not None:
            try:
                p = self._paged_cache_manager.get_stats()
                paged_blocks = p.get("blocks_allocated", 0)
                total_lookups = p.get("total_lookups", 0)
                total_hits = p.get("total_hits", 0)
                paged_hit_rate = total_hits / total_lookups if total_lookups > 0 else 0.0
            except Exception:
                pass

        return {
            "total": total,
            "available": available,
            "wired": wired,
            "compressed": compressed,
            "active": active,
            "mlx_cache": mlx_cache,
            "mlx_peak": mlx_peak,
            "paged_cache_blocks": paged_blocks,
            "paged_cache_hit_rate": paged_hit_rate,
        }

    def is_memory_pressure(self) -> bool:
        """Return True when system is under memory pressure.

        Returns False when the monitor has been closed (safe default).
        """
        if self._closed:
            return False

        usage = self.get_memory_usage()
        total = usage.get("total", 0)
        available = usage.get("available", 0)

        if total > 0 and available < total * 0.10:
            return True

        mlx_cache = usage.get("mlx_cache", 0)
        if mlx_cache > self.max_kv_cache_memory:
            return True

        if self._has_psutil:
            try:
                import psutil
                if psutil.virtual_memory().available < psutil.virtual_memory().total * 0.10:
                    return True
            except Exception:
                pass

        return False

    def close(self) -> None:
        """Gracefully shut down the memory monitor.

        Releases references to the paged cache manager and marks the
        monitor as closed so subsequent calls are no-ops.
        """
        if self._closed:
            return
        self._closed = True
        self._paged_cache_manager = None
        self._ps = None
        self._has_psutil = False
        logger.info("MemoryMonitor closed")

    @property
    def is_closed(self) -> bool:
        return self._closed

    # Model info for prefill peak estimation. Set by the scheduler
    # when the model is loaded so we can derive num_layers, head_dim,
    # num_kv_heads, and dtype without holding a model reference.
    _model_num_layers: int = 0
    _model_head_dim: int = 0
    _model_num_kv_heads: int = 0
    _model_num_query_heads: int = 0
    _model_dtype_bytes: int = 2  # float16 by default

    def set_model_info(
        self,
        num_layers: int,
        head_dim: int,
        num_kv_heads: int,
        num_query_heads: int | None = None,
        dtype_bytes: int = 2,
    ) -> None:
        """Cache model architecture params for memory estimation.

        Called by the scheduler once the model is loaded.
        """
        self._model_num_layers = num_layers
        self._model_head_dim = head_dim
        self._model_num_kv_heads = num_kv_heads
        self._model_num_query_heads = num_query_heads or num_kv_heads
        self._model_dtype_bytes = dtype_bytes

    def estimate_prefill_peak_bytes(
        self,
        new_tokens: int,
        prefill_step_size: int,
        cached_tokens: int = 0,
    ) -> int:
        """Estimate worst-case peak memory for a prefill chunk.

        Accounts for:
        - KV cache for the last prefill chunk (K + V, 2 groups)
        - SDPA temporary attention matrix (materialized for head_dim > 128)
        - For head_dim <= 128, MLX uses a fused kernel with minimal temp memory

        Returns 0 if model info is not available (set_model_info not called).
        """
        if self._model_num_layers <= 0 or self._model_head_dim <= 0:
            return 0
        if new_tokens <= 0:
            return 0

        chunk = min(new_tokens, prefill_step_size)
        layers = self._model_num_layers
        hd = self._model_head_dim
        db = self._model_dtype_bytes
        kv_heads = self._model_num_kv_heads
        q_heads = self._model_num_query_heads

        # KV cache: 2 (K+V) * layers * chunk * kv_heads * head_dim * dtype
        kv_bytes = 2 * layers * chunk * kv_heads * hd * db

        # SDPA temp matrix: only materialized when head_dim > 128
        # Shape: [batch=1, n_q_heads, chunk, kv_len] in float32
        # kv_len = cached_tokens + chunk (existing cache + new tokens)
        sdpa_bytes = 0
        if hd > 128:
            kv_len = cached_tokens + chunk
            sdpa_bytes = q_heads * chunk * kv_len * 4  # float32

        return kv_bytes + sdpa_bytes

    def estimate_decode_kv_bytes(self, total_tokens: int) -> int:
        """Estimate KV cache memory for `total_tokens` across all running requests.

        Used by the scheduler to account for existing decode-state memory
        when deciding whether to admit a new prefill.
        """
        if self._model_num_layers <= 0 or self._model_head_dim <= 0:
            return 0
        layers = self._model_num_layers
        hd = self._model_head_dim
        db = self._model_dtype_bytes
        kv_heads = self._model_num_kv_heads
        # 2 (K+V) * layers * total_tokens * kv_heads * head_dim * dtype
        return 2 * layers * total_tokens * kv_heads * hd * db

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


def _cfg_get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _pos_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and v > 0


def set_model_info_from_model(monitor: "MemoryMonitor", model: Any) -> None:
    """Populate ``monitor`` with KV/SDPA dims read from an mlx-lm ``model``.

    Best-effort: on any extraction failure the monitor is left dim-less
    and ``estimate_prefill_peak_bytes`` returns 0, making the guard a no-op
    rather than raising spuriously.
    """
    try:
        import mlx.core as mx

        config = None
        if hasattr(model, "config"):
            config = model.config
        elif hasattr(model, "args"):
            config = model.args

        if config is None:
            logger.debug("Could not extract model config for memory estimation")
            return

        for sub_attr in ("text_config", "language_config", "llm_config"):
            sub = _cfg_get(config, sub_attr)
            if sub is not None and (
                _cfg_get(sub, "num_hidden_layers") or _cfg_get(sub, "n_layer")
            ):
                config = sub
                break

        num_layers = _cfg_get(config, "num_hidden_layers") or _cfg_get(config, "n_layer")
        num_kv_heads = (
            _cfg_get(config, "num_key_value_heads")
            or _cfg_get(config, "num_attention_heads")
            or _cfg_get(config, "n_head")
        )
        head_dim = _cfg_get(config, "head_dim")
        hidden_size = _cfg_get(config, "hidden_size") or _cfg_get(config, "n_embd")

        if head_dim is None and hidden_size and num_kv_heads:
            num_heads = _cfg_get(config, "num_attention_heads") or num_kv_heads
            head_dim = hidden_size // num_heads

        dtype_size = 2
        if hasattr(model, "dtype"):
            if model.dtype == mx.float32:
                dtype_size = 4
            elif model.dtype == mx.bfloat16:
                dtype_size = 2

        num_attention_heads = (
            _cfg_get(config, "num_attention_heads")
            or _cfg_get(config, "n_head")
            or num_kv_heads
        )

        if _pos_int(num_layers) and _pos_int(num_kv_heads) and _pos_int(head_dim):
            monitor.set_model_info(
                num_layers=num_layers,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                num_query_heads=num_attention_heads,
                dtype_bytes=dtype_size,
            )
            logger.debug(
                "Model info for memory estimation: "
                "layers=%s, kv_heads=%s, q_heads=%s, "
                "head_dim=%s, dtype_size=%s",
                num_layers, num_kv_heads, num_attention_heads,
                head_dim, dtype_size,
            )
        else:
            logger.debug(
                "Incomplete model info: layers=%s, kv_heads=%s, head_dim=%s",
                num_layers, num_kv_heads, head_dim,
            )
    except Exception as e:
        logger.debug("Failed to extract model info: %s", e)


def raise_if_prefill_exceeds(
    monitor: "MemoryMonitor | None",
    *,
    prefill_memory_guard: bool,
    hard_limit_bytes: int,
    current_usage_bytes: int,
    prefill_step_size: int,
    num_prompt_tokens: int,
    cached_tokens: int = 0,
    request_id: str | None = None,
) -> None:
    if not prefill_memory_guard:
        return
    if hard_limit_bytes <= 0:
        return
    if monitor is None:
        return

    new_tokens = max(int(num_prompt_tokens) - max(int(cached_tokens), 0), 0)
    if new_tokens == 0:
        return

    peak = monitor.estimate_prefill_peak_bytes(
        new_tokens, prefill_step_size, cached_tokens=cached_tokens
    )
    if peak == 0:
        return

    current = max(0, int(current_usage_bytes))
    if current + peak <= hard_limit_bytes:
        return

    from fusion_mlx.exceptions import PrefillMemoryExceededError

    usage_gb = current / (1024**3)
    ceiling_gb = hard_limit_bytes / (1024**3)
    message = (
        f"Prefill would require ~{format_bytes(current + peak)} peak "
        f"(current {format_bytes(current)} + KV+SDPA {format_bytes(peak)}) "
        f"but ceiling is {format_bytes(hard_limit_bytes)} "
        f"(usage {usage_gb:.1f} GB, ceiling {ceiling_gb:.1f} GB). "
        f"Reduce context length, free system memory, or loosen "
        f"memory_guard_tier (safe → balanced → aggressive)."
    )

    if not request_id:
        import uuid as _uuid

        request_id = f"preflight-{_uuid.uuid4().hex[:8]}"
    logger.warning(
        "Preflight rejected (%d tokens, cached=%d, request_id=%s): %s",
        num_prompt_tokens, cached_tokens, request_id, message,
    )
    raise PrefillMemoryExceededError(
        message=message,
        request_id=request_id,
        estimated_bytes=int(current + peak),
        limit_bytes=int(hard_limit_bytes),
    )
