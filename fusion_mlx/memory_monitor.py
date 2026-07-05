# SPDX-License-Identifier: Apache-2.0
"""Memory monitoring for fusion-mlx on Apple Silicon unified memory.

Provides memory utilities for paged KV cache management, prefill peak
estimation, and SDPA dispatch set modeling on Apple Silicon.

Key enhancements over upstream:
- Per-head-dim SDPA dispatch sets (vector vs full kernel)
- Hybrid model support (num_kv_cache_layers < num_layers)
- MLA-style compressed KV estimation (kv_bytes_per_token_override)
- Fractional dtype_size for TurboQuant KV layouts
- Preflight guard (raise_if_prefill_exceeds)
- set_model_info_from_model() with VLM sub-config resolution
"""

from __future__ import annotations

import logging
import threading
import time
from ctypes import CDLL, byref, c_uint32, c_uint64, c_void_p
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# macOS libsystem for host_statistics — initialised lazily
_libsystem: CDLL | None = None
_HOST_PAGE_SIZE = 4096

# mach_vm_statistics64 struct fields (13 uint64 fields)
_VM_STAT_FIELDS = 13

# SDPA dispatch sets — mirror MLX Metal ScaledDotProductAttention::use_fallback.
# Full prefill and short vector kernels support different head dimensions;
# unsupported cases fall back to an unfused score-matrix allocation.
_SDPA_VECTOR_QUERY_TOKEN_THRESHOLD = 8
_SDPA_FULL_SUPPORTED_HEAD_DIMS = frozenset({64, 80, 128})
_SDPA_VECTOR_SUPPORTED_HEAD_DIMS = frozenset({64, 96, 128, 256})
# Default bytes/elem for the materialized unfused score matrix when the model's
# compute dtype is unknown. MLX softmax accumulates in fp32, but the dominant
# scratch buffer is allocated at the model's compute dtype, not fp32 — measured
# ~2.1-2.2 bytes/elem on MLX 0.31.2 for a head_dim=256 prefill (fp16/bf16),
# ~4.4 for fp32. Callers that know the model dtype pass it via
# set_model_info(compute_dtype_size=...); this default covers the rare
# dim-less path and matches the fp16/bf16 majority of MLX inference models.
_SDPA_FALLBACK_SCORE_DTYPE_SIZE = 2


def _get_libsystem():
    global _libsystem
    if _libsystem is None:
        _libsystem = CDLL("/usr/lib/libSystem.B.dylib")
    return _libsystem


def _get_host_page_size() -> int:
    try:
        host = c_void_p(0)
        host_page_size = c_uint64(0)
        _get_libsystem().host_page_size(host, byref(host_page_size))
        return host_page_size.value
    except Exception:
        return _HOST_PAGE_SIZE


def _get_vm_stat_fast() -> dict[str, int]:
    """Get vm stat via direct host_statistics64 syscall — zero subprocess overhead."""
    try:
        host = c_void_p(0)
        page_size = _get_host_page_size()
        stat = (c_uint64 * _VM_STAT_FIELDS)()
        count = c_uint32(_VM_STAT_FIELDS)
        ret = _get_libsystem().host_statistics64(host, 2, c_void_p(stat), byref(count))
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


@dataclass
class MemoryInfo:
    """Current GPU memory state."""

    total_bytes: int
    used_bytes: int
    available_bytes: int
    utilization: float


def _cfg_get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _pos_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and v > 0


class MemoryMonitor:
    """Monitors unified memory, wired memory, and compressed memory on macOS.

    Uses psutil when available, falls back to vm_stat/sysctl for core metrics.
    Tracks MLX cache/GPU memory via mlx.core when importable.

    Supports hybrid models (num_kv_cache_layers < num_layers), MLA-style
    compressed KV estimation, and per-head-dim SDPA dispatch modeling.
    """

    def __init__(
        self,
        max_kv_cache_memory: int | None = 4 * 1024**3,
        paged_cache_manager: Any = None,
        check_interval: float = 1.0,
        *,
        eviction_enabled: bool = True,
    ):
        self._max_kv_cache_memory = max_kv_cache_memory or 0
        self._eviction_enabled = eviction_enabled
        self._paged_cache_manager = paged_cache_manager
        self._check_interval = check_interval
        self._has_psutil = False
        self._ps = None
        self._closed = False
        try:
            import psutil

            self._ps = psutil
            self._has_psutil = True
        except ImportError:
            pass
        if self._has_psutil:
            self._total_memory = self._ps.virtual_memory().total
        else:
            sys_mem = self._get_sysctl_memory()
            self._total_memory = sys_mem.get("total", 0)

        # Model info for memory estimation (set by scheduler)
        self._num_layers: int | None = None
        self._num_kv_heads: int | None = None
        self._head_dim: int | None = None
        # KV storage width; may be fractional with TurboQuant.
        self._dtype_size: float = 2
        self._kv_bytes_per_token_override: float | None = None
        # SDPA score-matrix width = model compute/activation dtype
        self._score_dtype_size: float = _SDPA_FALLBACK_SCORE_DTYPE_SIZE
        self._num_attention_heads: int | None = None
        self._num_kv_cache_layers: int | None = None

        # Baseline memory (model weights) - set after model load
        self._baseline_memory: int = 0

        # Request stats (set by scheduler for logging)
        self._running_requests: int = 0
        self._waiting_requests: int = 0

        self._last_check_time: float = 0.0
        self._last_memory_info: MemoryInfo | None = None
        self._lock = threading.Lock()

    def set_paged_cache_manager(self, paged_cache_manager: Any) -> None:
        self._paged_cache_manager = paged_cache_manager

    @property
    def max_kv_cache_memory(self) -> int:
        return self._max_kv_cache_memory

    @property
    def eviction_enabled(self) -> bool:
        return self._eviction_enabled

    def _parse_vm_stat(self) -> dict[str, int]:
        """Parse vm_stat output into page counts."""
        import subprocess

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
        except Exception:
            return {}

    def _get_sysctl_memory(self) -> dict[str, int]:
        """Get system memory from sysctl."""
        import subprocess

        keys = {"hw.memsize": "total"}
        result = {}
        for k, name in keys.items():
            try:
                r = subprocess.run(
                    ["sysctl", "-n", k], capture_output=True, text=True, timeout=5
                )
                result[name] = int(r.stdout.strip())
            except Exception:
                result[name] = 0
        return result

    def _get_max_working_set_bytes(self) -> int:
        """Get max_recommended_working_set_size from MLX Metal."""
        try:
            import mlx.core as mx

            if mx.metal.is_available():
                return mx.metal.get_recommended_max_working_set_size()
        except Exception:
            pass
        return self._total_memory

    def set_baseline_memory(self) -> None:
        """Set baseline memory after model load.

        Call this after loading the model to capture the baseline memory usage
        (model weights, etc.). The KV cache memory is calculated as:
        active_memory - baseline_memory
        """
        try:
            import mlx.core as mx

            if mx.metal.is_available():
                self._baseline_memory = mx.get_active_memory()
                logger.info(f"Baseline memory set: {_fmt_bytes(self._baseline_memory)}")
                return
        except Exception:
            pass
        self._baseline_memory = 0
        logger.warning("MLX Metal not available, baseline memory set to 0")

    def has_model_info(self) -> bool:
        """Whether set_model_info has been called with real dims."""
        return (
            self._num_layers is not None
            and self._num_layers > 0
            and self._num_kv_heads is not None
            and self._num_kv_heads > 0
            and self._head_dim is not None
            and self._head_dim > 0
        )

    def set_model_info(
        self,
        num_layers: int,
        head_dim: int,
        num_kv_heads: int,
        num_query_heads: int | None = None,
        dtype_bytes: float = 2,
        num_kv_cache_layers: int | None = None,
        compute_dtype_size: float | None = None,
        kv_bytes_per_token: float | None = None,
    ) -> None:
        """Cache model architecture params for memory estimation.

        Args:
            num_layers: Number of transformer layers.
            head_dim: Dimension per attention head.
            num_kv_heads: Number of KV attention heads.
            num_query_heads: Number of query attention heads (for SDPA
                peak estimation). Defaults to num_kv_heads.
            dtype_bytes: Bytes per element of the stored KV cache. May be
                fractional for quantized (e.g. TurboQuant) KV layouts.
            num_kv_cache_layers: Number of layers that use KVCache
                (full attention). For hybrid models this may be less than
                num_layers. Defaults to num_layers.
            compute_dtype_size: Bytes per element of the model's
                compute/activation dtype (2 for fp16/bf16, 4 for fp32).
            kv_bytes_per_token: Optional exact resident KV-cache bytes added
                per token. Use for compressed-cache architectures such as MLA.
        """
        self._num_layers = num_layers
        self._num_kv_heads = num_kv_heads
        self._head_dim = head_dim
        self._dtype_size = float(dtype_bytes)
        self._score_dtype_size = (
            compute_dtype_size
            if compute_dtype_size and compute_dtype_size > 0
            else _SDPA_FALLBACK_SCORE_DTYPE_SIZE
        )
        self._kv_bytes_per_token_override = (
            float(kv_bytes_per_token)
            if kv_bytes_per_token is not None and kv_bytes_per_token > 0
            else None
        )
        self._num_attention_heads = num_query_heads or num_kv_heads
        self._num_kv_cache_layers = num_kv_cache_layers or num_layers

        logger.info(
            f"Model info set: {num_layers} layers "
            f"({self._num_kv_cache_layers} KVCache), "
            f"{num_kv_heads} KV heads, {self._num_attention_heads} Q heads, "
            f"{head_dim} head_dim, dtype={dtype_bytes}"
        )

    def get_memory_usage(self) -> dict[str, Any]:
        """Return current memory usage snapshot."""
        if self._closed:
            return {
                "total": 0,
                "available": 0,
                "wired": 0,
                "compressed": 0,
                "active": 0,
                "inactive": 0,
                "mlx_cache": 0,
                "mlx_peak": 0,
                "paged_cache_blocks": 0,
                "paged_cache_hit_rate": 0.0,
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
                paged_hit_rate = (
                    total_hits / total_lookups if total_lookups > 0 else 0.0
                )
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

    def get_memory_info(self) -> MemoryInfo:
        """Get current memory state with throttled checks."""
        with self._lock:
            current_time = time.time()
            if (
                self._last_memory_info is not None
                and current_time - self._last_check_time < self._check_interval
            ):
                return self._last_memory_info

            usage = self.get_memory_usage()
            used = usage.get("active", 0) + usage.get("wired", 0)
            max_mem = self._get_max_working_set_bytes()
            available = max(0, max_mem - used)
            utilization = used / max_mem if max_mem > 0 else 0.0

            self._last_memory_info = MemoryInfo(
                total_bytes=max_mem,
                used_bytes=used,
                available_bytes=available,
                utilization=utilization,
            )
            self._last_check_time = current_time
            return self._last_memory_info

    def is_under_pressure(self) -> bool:
        """Return True when system is under memory pressure."""
        return self.is_memory_pressure()

    def bytes_to_free(self) -> int:
        """Calculate bytes needed to free from KV cache.

        Returns the excess when current KV cache usage exceeds
        max_kv_cache_memory, 0 otherwise.
        """
        if not self._eviction_enabled:
            return 0
        try:
            import mlx.core as mx

            active = mx.get_active_memory()
        except Exception:
            active = 0
        kv_usage = max(0, active - self._baseline_memory)
        if kv_usage > self._max_kv_cache_memory:
            return kv_usage - self._max_kv_cache_memory
        return 0

    def estimate_blocks_to_free(self, bytes_to_free: int, block_size: int) -> int:
        """Estimate number of blocks to evict to free the given bytes.

        Args:
            bytes_to_free: Target bytes to free.
            block_size: Tokens per block.

        Returns:
            Number of blocks to evict (minimum 1).
        """
        if not self._eviction_enabled:
            raise RuntimeError(
                "estimate_blocks_to_free called on a MemoryMonitor "
                "constructed with eviction_enabled=False"
            )
        block_mem = self.estimate_block_memory(block_size)
        if block_mem <= 0:
            return 0
        num_blocks = int((bytes_to_free + block_mem - 1) // block_mem)
        return max(1, num_blocks)

    def set_request_stats(self, running: int, waiting: int) -> None:
        """Update request stats for logging.

        Args:
            running: Number of currently running requests.
            waiting: Number of waiting requests.
        """
        self._running_requests = running
        self._waiting_requests = waiting

    def is_memory_pressure(self) -> bool:
        """Return True when system is under memory pressure."""
        if self._closed:
            return False

        usage = self.get_memory_usage()
        total = usage.get("total", 0)
        available = usage.get("available", 0)

        if total > 0 and available < total * 0.10:
            return True

        mlx_cache = usage.get("mlx_cache", 0)
        if mlx_cache > self._max_kv_cache_memory:
            return True

        return False

    def _uses_fused_sdpa(self, query_tokens: int, kv_len: int) -> bool:
        """Return True when MLX dispatches to the fused SDPA kernel.

        Mirrors MLX Metal ScaledDotProductAttention::use_fallback:
        - Short queries (<=_SDPA_VECTOR_QUERY_TOKEN_THRESHOLD) use vector kernel
          if head_dim is in _SDPA_VECTOR_SUPPORTED_HEAD_DIMS
        - Longer queries use full-attention kernel if head_dim is in
          _SDPA_FULL_SUPPORTED_HEAD_DIMS
        - Everything else falls back to unfused score-matrix allocation
        """
        hd = self._head_dim or 0
        n_q = self._num_attention_heads or 0
        n_kv = self._num_kv_heads or n_q
        if n_q <= 0 or n_kv <= 0 or hd <= 0 or query_tokens <= 0:
            return False
        if kv_len < query_tokens:
            return False

        if query_tokens <= _SDPA_VECTOR_QUERY_TOKEN_THRESHOLD:
            gqa_factor = max(1, n_q // n_kv)
            return (
                hd in _SDPA_VECTOR_SUPPORTED_HEAD_DIMS
                and query_tokens * gqa_factor <= 32
            )

        return hd in _SDPA_FULL_SUPPORTED_HEAD_DIMS

    def _estimate_sdpa_activation_bytes(self, query_tokens: int, kv_len: int) -> int:
        """Estimate SDPA activation peak for one attention layer.

        Fused path: only the weighted-output buffer
        [q_heads, query_tokens, head_dim] in float32.
        Unfused path: the full score matrix
        [q_heads, query_tokens, kv_len] in compute dtype *plus* the output.
        """
        hd = self._head_dim or 0
        n_q = self._num_attention_heads or 0
        if n_q == 0 or hd == 0 or query_tokens <= 0:
            return 0

        query_tokens = int(query_tokens)
        kv_len = max(int(kv_len), 0)

        output = n_q * query_tokens * hd * 4
        if self._uses_fused_sdpa(query_tokens, kv_len):
            return output

        scores = n_q * query_tokens * kv_len * self._score_dtype_size
        return scores + output

    def estimate_block_memory(
        self,
        block_size: int,
        num_layers: int | None = None,
        num_kv_heads: int | None = None,
        head_dim: int | None = None,
        dtype_size: float | None = None,
    ) -> float:
        """Estimate memory usage for a KV cache block."""
        layers = num_layers or self._num_layers or 32
        kv_heads = num_kv_heads or self._num_kv_heads or 8
        dim = head_dim or self._head_dim or 128
        dtype = dtype_size or self._dtype_size

        if (
            self._kv_bytes_per_token_override is not None
            and num_layers is None
            and num_kv_heads is None
            and head_dim is None
            and dtype_size is None
        ):
            return block_size * self._kv_bytes_per_token_override

        per_layer = block_size * kv_heads * dim * dtype * 2
        return per_layer * layers

    def estimate_prefill_peak_bytes(
        self,
        new_tokens: int,
        prefill_step_size: int,
        cached_tokens: int = 0,
    ) -> int:
        """Estimate worst-case peak memory for a prefill chunk.

        Accounts for:
        - KV cache for the new tokens (using num_kv_cache_layers for hybrid)
        - SDPA activation (fused output or unfused score matrix + output)

        Returns 0 if model info is not available.
        """
        if not self.has_model_info():
            return 0
        if new_tokens <= 0:
            return 0

        eff_chunk = min(new_tokens, prefill_step_size)
        layers = self._num_kv_cache_layers or self._num_layers or 0
        hd = self._head_dim or 0
        kv_heads = self._num_kv_heads or 0

        # KV cache: use override if set, otherwise standard formula
        if self._kv_bytes_per_token_override is not None:
            kv_bytes = int(new_tokens * self._kv_bytes_per_token_override)
        else:
            kv_bytes = int(2 * layers * new_tokens * kv_heads * hd * self._dtype_size)

        # SDPA activation: kv_len includes cached prefix positions
        full_kv_len = eff_chunk + max(cached_tokens, 0)
        sdpa_bytes = self._estimate_sdpa_activation_bytes(eff_chunk, full_kv_len)

        return kv_bytes + sdpa_bytes

    def estimate_prompt_kv_bytes(
        self,
        new_tokens: int,
        cached_tokens: int = 0,
    ) -> tuple[int, int]:
        """Estimate KV cache growth for new_tokens prompt tokens.

        Returns (new_kv_bytes, cached_kv_bytes).
        """
        if not self.has_model_info():
            return (0, 0)
        layers = self._num_kv_cache_layers or self._num_layers or 0
        hd = self._head_dim or 0
        kv_heads = self._num_kv_heads or 0

        if self._kv_bytes_per_token_override is not None:
            new_kv = (
                int(new_tokens * self._kv_bytes_per_token_override)
                if new_tokens > 0
                else 0
            )
            cached_kv = (
                int(cached_tokens * self._kv_bytes_per_token_override)
                if cached_tokens > 0
                else 0
            )
            return (new_kv, cached_kv)

        new_kv = 0
        if new_tokens > 0:
            new_kv = int(2 * layers * new_tokens * kv_heads * hd * self._dtype_size)
        cached_kv = 0
        if cached_tokens > 0:
            cached_kv = int(
                2 * layers * cached_tokens * kv_heads * hd * self._dtype_size
            )
        return (new_kv, cached_kv)

    def estimate_decode_kv_bytes(self, total_tokens: int) -> int:
        """Estimate KV cache memory for total_tokens across all running requests."""
        if not self.has_model_info():
            return 0
        layers = self._num_kv_cache_layers or self._num_layers or 0
        hd = self._head_dim or 0
        kv_heads = self._num_kv_heads or 0

        if self._kv_bytes_per_token_override is not None:
            return int(total_tokens * self._kv_bytes_per_token_override)

        return int(2 * layers * total_tokens * kv_heads * hd * self._dtype_size)

    def estimate_chunk_transient_bytes(self, n_tokens: int, kv_len: int) -> int:
        """Transient SDPA activation bytes for ONE prefill chunk.

        Isolates the per-chunk attention transient — the spike that drives
        prefill OOM — for a chunk of n_tokens query tokens attending over
        kv_len total context tokens. Unlike estimate_prefill_peak_bytes this
        excludes newly-allocated KV.

        Returns 0 when model info is unavailable.
        """
        return self._estimate_sdpa_activation_bytes(n_tokens, kv_len)

    def get_stats(self) -> dict:
        """Get memory statistics as a dictionary."""
        info = self.get_memory_info()
        return {
            "total_bytes": info.total_bytes,
            "used_bytes": info.used_bytes,
            "available_bytes": info.available_bytes,
            "utilization": info.utilization,
            "max_kv_cache_memory": self._max_kv_cache_memory,
            "baseline_memory": self._baseline_memory,
            "has_model_info": self.has_model_info(),
        }

    def close(self) -> None:
        """Gracefully shut down the memory monitor."""
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

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def __repr__(self) -> str:
        info = self.get_memory_info()
        return (
            f"MemoryMonitor(max_kv_cache={_fmt_bytes(self._max_kv_cache_memory)}, "
            f"used={_fmt_bytes(info.used_bytes)})"
        )


def estimate_mla_kv_bytes_per_token(
    config: Any,
    cache_list: Any,
    dtype_size: float,
) -> float | None:
    """Estimate exact resident KV bytes/token for MLA-style caches.

    GLM/DeepSeek MLA models do not store expanded num_kv_heads * head_dim
    K/V tensors. Their main cache stores a latent key and RoPE value
    (kv_lora_rank + qk_rope_head_dim) with a single KV head. GLM-5.2's DSA
    indexer adds a second cache on full-indexer layers containing only
    index_head_dim keys and zero-width values.
    """
    kv_lora_rank = _cfg_get(config, "kv_lora_rank")
    rope_dim = _cfg_get(config, "qk_rope_head_dim")
    if not (_pos_int(kv_lora_rank) and _pos_int(rope_dim)):
        return None

    if cache_list is None:
        return None

    main_cache_layers = 0
    indexer_cache_layers = 0
    try:
        for layer_cache in cache_list:
            caches = getattr(layer_cache, "caches", None)
            if caches is None:
                continue
            n_caches = len(caches)
            if n_caches >= 1:
                main_cache_layers += 1
            if n_caches >= 2:
                indexer_cache_layers += 1
    except Exception:
        return None

    if main_cache_layers <= 0:
        return None

    index_head_dim = _cfg_get(config, "index_head_dim", 0) or 0
    if not _pos_int(index_head_dim):
        index_head_dim = 0

    elems_per_token = (
        main_cache_layers * (kv_lora_rank + rope_dim)
        + indexer_cache_layers * index_head_dim
    )
    return float(elems_per_token) * float(dtype_size)


def set_model_info_from_model(monitor: MemoryMonitor, model: Any) -> None:
    """Populate monitor with KV/SDPA dims read from an mlx-lm model.

    VLM / multimodal configs (e.g. Qwen3.6-VL, Gemma-4) nest the
    language-model dimensions under a sub-config. Prefer text_config /
    language_config / llm_config when ANY of them exposes the LM layer count.
    """
    try:
        config = None
        if hasattr(model, "config"):
            config = model.config
        elif hasattr(model, "args"):
            config = model.args

        if config is None:
            logger.debug("Could not extract model config for memory estimation")
            return

        # VLM sub-config resolution
        for sub_attr in ("text_config", "language_config", "llm_config"):
            sub = _cfg_get(config, sub_attr)
            if sub is not None and (
                _cfg_get(sub, "num_hidden_layers") or _cfg_get(sub, "n_layer")
            ):
                config = sub
                break

        num_layers = _cfg_get(config, "num_hidden_layers") or _cfg_get(
            config, "n_layer"
        )
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
        try:
            import mlx.core as mx

            if hasattr(model, "dtype"):
                if model.dtype == mx.float32:
                    dtype_size = 4
                elif model.dtype == mx.bfloat16:
                    dtype_size = 2
        except ImportError:
            pass

        num_attention_heads = (
            _cfg_get(config, "num_attention_heads")
            or _cfg_get(config, "n_head")
            or num_kv_heads
        )

        # Count KVCache layers for hybrid models
        cache_list = None
        num_kv_cache_layers = num_layers
        if hasattr(model, "make_cache"):
            try:
                cache_list = model.make_cache()
                from mlx_lm.models.cache import CacheList, KVCache

                def _count_kv(c: Any) -> int:
                    if type(c) is KVCache:
                        return 1
                    if isinstance(c, CacheList):
                        return sum(_count_kv(inner) for inner in c.caches)
                    return 0

                num_kv_cache_layers = sum(_count_kv(c) for c in cache_list)
                if num_kv_cache_layers == 0:
                    num_kv_cache_layers = num_layers
            except Exception:
                pass

        kv_bytes_per_token = estimate_mla_kv_bytes_per_token(
            config, cache_list, dtype_size
        )

        if _pos_int(num_layers) and _pos_int(num_kv_heads) and _pos_int(head_dim):
            monitor.set_model_info(
                num_layers=num_layers,
                head_dim=head_dim,
                num_kv_heads=num_kv_heads,
                num_query_heads=num_attention_heads,
                dtype_bytes=dtype_size,
                num_kv_cache_layers=num_kv_cache_layers,
                compute_dtype_size=dtype_size,
                kv_bytes_per_token=kv_bytes_per_token,
            )
            logger.debug(
                f"Model info for memory estimation: "
                f"layers={num_layers} ({num_kv_cache_layers} KVCache), "
                f"kv_heads={num_kv_heads}, q_heads={num_attention_heads}, "
                f"head_dim={head_dim}, dtype_size={dtype_size}"
            )
        else:
            logger.debug(
                f"Incomplete model info: layers={num_layers}, "
                f"kv_heads={num_kv_heads}, head_dim={head_dim}"
            )

    except Exception as e:
        logger.debug(f"Failed to extract model info: {e}")


def raise_if_prefill_exceeds(
    monitor: MemoryMonitor | None,
    *,
    prefill_memory_guard: bool,
    hard_limit_bytes: int,
    current_usage_bytes: int,
    prefill_step_size: int,
    num_prompt_tokens: int,
    cached_tokens: int = 0,
    request_id: str | None = None,
) -> None:
    """Raise PrefillMemoryExceededError if a prompt's prefill peak would
    push memory past hard_limit_bytes.

    The shared front-door guard. No-op when the guard is disabled, no limit
    is set, the monitor is missing, or the request fits.
    """
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
        f"Prefill would require ~{_fmt_bytes(current + peak)} peak "
        f"(current {_fmt_bytes(current)} + KV+SDPA {_fmt_bytes(peak)}) "
        f"but ceiling is {_fmt_bytes(hard_limit_bytes)} "
        f"(usage {usage_gb:.1f} GB, ceiling {ceiling_gb:.1f} GB). "
        f"Reduce context length, free system memory, or loosen "
        f"memory_guard_tier (safe → balanced → aggressive)."
    )

    if not request_id:
        import uuid as _uuid

        request_id = f"preflight-{_uuid.uuid4().hex[:8]}"

    logger.warning(
        "Preflight rejected (%d tokens, cached=%d, request_id=%s): %s",
        num_prompt_tokens,
        cached_tokens,
        request_id,
        message,
    )
    raise PrefillMemoryExceededError(
        message=message,
        request_id=request_id,
        estimated_bytes=int(current + peak),
        limit_bytes=int(hard_limit_bytes),
    )


def _fmt_bytes(n: int) -> str:
    """Format bytes as human-readable string."""
    if n < 0:
        return f"-{_fmt_bytes(-n)}"
    if n == 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    f = float(n)
    while f >= 1024 and i < len(units) - 1:
        f /= 1024
        i += 1
    return f"{f:.1f} {units[i]}"
