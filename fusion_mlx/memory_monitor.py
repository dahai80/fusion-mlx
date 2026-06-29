# SPDX-License-Identifier: Apache-2.0
"""Memory monitoring for fusion-mlx on Apple Silicon unified memory."""

import logging
import subprocess
from ctypes import CDLL, byref, c_uint32, c_uint64, c_void_p
from typing import Any

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

    # MLX SDPA fused-kernel head-dim cutoff.  Head dims up to this value
    # dispatch to the fused Metal kernel (minimal temp memory).  Above it,
    # MLX falls back to an unfused path that materialises the full fp32
    # score matrix [q_heads, query_len, kv_len].
    _SDPA_FUSED_MAX_HEAD_DIM: int = 128

    # Dtype size for the unfused SDPA score matrix (always float32).
    _SDPA_SCORE_DTYPE_SIZE: int = 4

    def _uses_fused_sdpa(self, query_tokens: int, kv_len: int) -> bool:
        """Return True when MLX dispatches to the fused SDPA kernel.

        The fused path avoids materialising the full attention score
        matrix.  Currently gated on head_dim only; query/kv lengths do
        not affect dispatch in the MLX backend as of 0.24.
        """
        return self._model_head_dim <= self._SDPA_FUSED_MAX_HEAD_DIM

    def _estimate_sdpa_activation_bytes(
        self, query_tokens: int, kv_len: int
    ) -> int:
        """Estimate SDPA activation peak for one attention layer.

        Fused path: only the weighted-output buffer
        ``[q_heads, query_tokens, head_dim]`` in float32.
        Unfused path: the full score matrix
        ``[q_heads, query_tokens, kv_len]`` in float32 *plus* the output.
        """
        hd = self._model_head_dim or 0
        n_q = self._model_num_query_heads or 0
        if n_q == 0 or hd == 0 or query_tokens <= 0:
            return 0

        query_tokens = int(query_tokens)
        kv_len = max(int(kv_len), 0)

        output = n_q * query_tokens * hd * 4
        if self._uses_fused_sdpa(query_tokens, kv_len):
            return output

        scores = n_q * query_tokens * kv_len * self._SDPA_SCORE_DTYPE_SIZE
        return scores + output

    def estimate_prefill_peak_bytes(
        self,
        new_tokens: int,
        prefill_step_size: int,
        cached_tokens: int = 0,
    ) -> int:
        """Estimate worst-case peak memory for a prefill chunk.

        Accounts for:
        - KV cache for the last prefill chunk (K + V, 2 groups)
        - SDPA activation (fused output or unfused score matrix + output)

        Returns 0 if model info is not available (set_model_info not called).
        """
        if self._model_num_layers <= 0 or self._model_head_dim <= 0:
            return 0
        if new_tokens <= 0:
            return 0

        eff_chunk = min(new_tokens, prefill_step_size)
        layers = self._model_num_layers
        hd = self._model_head_dim
        db = self._model_dtype_bytes
        kv_heads = self._model_num_kv_heads

        # KV cache: 2 (K+V) * layers * chunk * kv_heads * head_dim * dtype
        kv_bytes = 2 * layers * eff_chunk * kv_heads * hd * db

        # SDPA activation: delegate to the proper estimator.
        # kv_len includes cached prefix positions — they participate in
        # attention even though their KV tensors are already resident.
        full_kv_len = eff_chunk + max(cached_tokens, 0)
        sdpa_bytes = self._estimate_sdpa_activation_bytes(eff_chunk, full_kv_len)

        return kv_bytes + sdpa_bytes

    def estimate_prompt_kv_bytes(
        self,
        new_tokens: int,
        cached_tokens: int = 0,
    ) -> tuple[int, int]:
        """Estimate KV cache growth for ``new_tokens`` prompt tokens.

        Same math as the KV portion of ``estimate_prefill_peak_bytes``
        but without the chunk-size cap — used by the safety-rejection
        path to charge the full prompt's KV allocation.

        Returns:
            ``(new_kv_bytes, cached_kv_bytes)`` — the caller needs both
            to distinguish already-resident cache from new allocation.
        """
        if self._model_num_layers <= 0 or self._model_head_dim <= 0:
            return (0, 0)
        layers = self._model_num_layers
        hd = self._model_head_dim
        db = self._model_dtype_bytes
        kv_heads = self._model_num_kv_heads
        new_kv = 0
        if new_tokens > 0:
            new_kv = 2 * layers * new_tokens * kv_heads * hd * db
        cached_kv = 0
        if cached_tokens > 0:
            cached_kv = 2 * layers * cached_tokens * kv_heads * hd * db
        return (new_kv, cached_kv)

    def _predicted_chunk_transient(
        self, chunk_tokens: int, kv_len: int
    ) -> int:
        """Predict the peak transient of a single decode/prefill chunk.

        Returns the SDPA activation peak for the given chunk width and
        context length.  Used by the safety-rejection path and the
        adaptive chunk sizer.
        """
        return self._estimate_sdpa_activation_bytes(chunk_tokens, kv_len)

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
