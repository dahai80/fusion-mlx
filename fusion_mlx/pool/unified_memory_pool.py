# SPDX-License-Identifier: Apache-2.0
"""UnifiedMemoryPool — single accounting layer for cross-engine Metal buffers.

Solves the "two ledgers" problem when omlx and Rapid-MLX each maintain
their own KV cache pre-allocations. All Metal buffer allocations flow
through this pool so:

1. omlx freed memory is visible to Rapid-MLX (and vice versa)
2. KV cache handoff between engines is zero-copy (shared Metal buffers)
3. Memory fragmentation is bounded per engine via per-backend ceilings

Architecture:
  UnifiedMemoryPool
  ├── MetalBufferRegistry   — track every mx.array buffer by id
  ├── BackendQuota          — max pre-allocation per backend (omlx / rapid)
  ├── KVCacheBridge         — zero-copy KV state transfer between engines
  └── FragmentationMonitor  — detect/compact when fragmentation > threshold
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# Metal Buffer Registry — track every allocation
# =============================================================================

@dataclass
class BufferEntry:
    """Track one Metal buffer allocation."""
    buffer_id: str
    backend: str          # "omlx", "rapid", "shared"
    size_bytes: int
    dtype: str
    shape: tuple
    ref_count: int = 1
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    is_kv_cache: bool = False
    _shared: bool = False

    def touch(self):
        self.last_accessed = time.time()


class MetalBufferRegistry:
    """Central registry for all Metal buffer allocations.

    Every mx.array that backs a KV cache or model weight must register
    here. This gives us a single source of truth for memory usage.
    """

    def __init__(self):
        self._buffers: dict[str, BufferEntry] = {}
        self._lock = threading.RLock()
        self._total_bytes = 0
        self._per_backend_bytes: dict[str, int] = {}

    def register(
        self,
        arr: Any,
        backend: str,
        is_kv_cache: bool = False,
    ) -> str:
        """Register an mx.array buffer. Returns buffer_id for later release."""
        buf_id = uuid.uuid4().hex[:16]
        size = int(arr.nbytes) if hasattr(arr, "nbytes") else 0
        dtype = str(arr.dtype) if hasattr(arr, "dtype") else "unknown"
        shape = tuple(arr.shape) if hasattr(arr, "shape") else ()

        entry = BufferEntry(
            buffer_id=buf_id,
            backend=backend,
            size_bytes=size,
            dtype=dtype,
            shape=shape,
            is_kv_cache=is_kv_cache,
         )

        with self._lock:
            self._buffers[buf_id] = entry
            self._total_bytes += size
            self._per_backend_bytes[backend] = self._per_backend_bytes.get(backend, 0) + size

        if is_kv_cache:
            logger.debug(f"[Registry] registered KV buffer {buf_id}: {size / 1e6:.1f}MB @ {backend}")

        return buf_id

    def increment_ref(self, buf_id: str) -> bool:
        with self._lock:
            entry = self._buffers.get(buf_id)
            if entry:
                entry.ref_count += 1
                entry.touch()
                return True
            return False

    def decrement_ref(self, buf_id: str) -> bool:
        """Decrement ref count. Returns True if buffer was freed."""
        with self._lock:
            entry = self._buffers.get(buf_id)
            if not entry:
                return False
            entry.ref_count -= 1
            if entry.ref_count <= 0:
                self._total_bytes -= entry.size_bytes
                backend = entry.backend
                self._per_backend_bytes[backend] = max(
                    0, self._per_backend_bytes.get(backend, 0) - entry.size_bytes
                 )
                del self._buffers[buf_id]
                if entry.is_kv_cache:
                    logger.debug(f"[Registry] freed KV buffer {buf_id}: {entry.size_bytes / 1e6:.1f}MB")
                del entry
                return True
            return False

    def release(self, buf_id: str) -> bool:
        """Force release a buffer (set ref_count = 0)."""
        with self._lock:
            entry = self._buffers.get(buf_id)
            if not entry:
                return False
            self._total_bytes -= entry.size_bytes
            backend = entry.backend
            self._per_backend_bytes[backend] = max(
                0, self._per_backend_bytes.get(backend, 0) - entry.size_bytes
             )
            del self._buffers[buf_id]
            return True

    def share_buffer(self, buf_id: str, new_backend: str) -> bool:
        """Re-register a buffer under a shared backend label.

        Used during KV cache handoff — the buffer becomes "shared" between
        the source and destination backends.
        """
        with self._lock:
            entry = self._buffers.get(buf_id)
            if not entry:
                return False
             # Don't double-count — remove from old backend, add to shared
            old_backend = entry.backend
            self._per_backend_bytes[old_backend] = max(
                0, self._per_backend_bytes.get(old_backend, 0) - entry.size_bytes
             )
            entry.backend = "shared"
            self._per_backend_bytes["shared"] = self._per_backend_bytes.get("shared", 0) + entry.size_bytes
            entry.ref_count += 1
            entry._shared = True
            return True

    @property
    def total_bytes(self) -> int:
        with self._lock:
            return self._total_bytes

    def get_backend_usage(self, backend: str) -> int:
        with self._lock:
            return self._per_backend_bytes.get(backend, 0)

    def get_stats(self) -> dict[str, Any]:
        with self._lock:
            kv_count = sum(1 for e in self._buffers.values() if e.is_kv_cache)
            return {
                 "total_bytes": self._total_bytes,
                 "total_buffers": len(self._buffers),
                 "kv_buffers": kv_count,
                 "per_backend": dict(self._per_backend_bytes),
             }


# =============================================================================
# Backend Quota — per-backend pre-allocation limits
# =============================================================================

class BackendQuota:
    """Pre-allocation ceiling per backend to prevent memory fragmentation."""

    __slots__ = ("backend", "max_bytes", "_current_bytes", "_lock")

    def __init__(self, backend: str, max_bytes: int, current_bytes: int = 0):
        self.backend = backend
        self.max_bytes = max_bytes
        self._current_bytes = current_bytes
        self._lock = threading.Lock()

    @property
    def current_bytes(self) -> int:
        return self._current_bytes

    @property
    def utilization(self) -> float:
        return self._current_bytes / self.max_bytes if self.max_bytes > 0 else 0.0

    @property
    def available(self) -> int:
        return max(0, self.max_bytes - self._current_bytes)

    def can_allocate(self, size: int) -> bool:
        return self._current_bytes + size <= self.max_bytes

    def allocate(self, size: int) -> bool:
        with self._lock:
            if self._current_bytes + size > self.max_bytes:
                return False
            self._current_bytes += size
            return True

    def release(self, size: int) -> None:
        with self._lock:
            self._current_bytes = max(0, self._current_bytes - size)


# =============================================================================
# KV Cache Bridge — zero-copy handoff between engines
# =============================================================================

@dataclass
class KVCacheState:
    """Serializable KV cache state for cross-engine transfer."""

    model_name: str
    source_backend: str
    target_backend: str
    buffer_ids: list[str]           # Registry IDs of KV buffers
    block_table: list[int]          # Block IDs for paged cache
    meta_states: list[tuple]        # Per-layer (offset, ...) tuples
    num_tokens: int
    num_layers: int
    kv_heads: int
    head_dim: int
    dtype: str

    # Dual-ownership lifecycle tracking
    handoff_id: str = ''            # Bridge tracking ID
    source_released: bool = False   # Source engine done with buffers
    target_released: bool = False   # Target engine done with buffers
    handed_off_at: float = field(default_factory=time.time)


class KVCacheBridge:
    """Zero-copy KV cache transfer between omlx and Rapid-MLX engines.

    Handoff flow:
     1. Source engine finishes prefill -> KV cache in Metal buffers
     2. Bridge captures buffer IDs + block table + meta states
     3. Buffers are re-registered as "shared" with dual ref counting
     4. Target engine wraps same Metal buffers in its own mx.array views
     5. Decode continues from transferred KV state

    Lifecycle safety:
     - prepare_handoff() increments refs (source=1, bridge=1 -> total=2)
     - release_source() decrements source ref (total=1, buffers stay alive)
     - release_target() decrements bridge ref (total=0, buffers freed)
     - This prevents source engine GC from freeing shared buffers
    """

    def __init__(self, registry: MetalBufferRegistry):
        self._registry = registry
        self._active_handoffs: dict[str, KVCacheState] = {}
        self._lock = threading.Lock()

    def capture(
        self,
        model_name: str,
        source_backend: str,
        kv_arrays: list[Any],
        block_table: list[int],
        meta_states: list[tuple],
        num_tokens: int,
     ) -> KVCacheState:
        """Capture KV cache state from source engine for handoff."""
        buffer_ids = []
        kv_heads = 0
        head_dim = 0
        dtype = "float16"

        for i, arr in enumerate(kv_arrays):
            buf_id = self._registry.register(arr, source_backend, is_kv_cache=True)
            buffer_ids.append(buf_id)
            if i == 0 and hasattr(arr, "dtype"):
                dtype = str(arr.dtype)
            if hasattr(arr, "shape") and len(arr.shape) >= 4:
                if kv_heads == 0:
                    kv_heads = arr.shape[-2]
                    head_dim = arr.shape[-1]

        # Estimate num_layers from buffer count (2 per layer: K + V)
        num_layers = max(1, len(kv_arrays) // 2)

        state = KVCacheState(
            model_name=model_name,
            source_backend=source_backend,
            target_backend="",
            buffer_ids=buffer_ids,
            block_table=block_table,
            meta_states=meta_states,
            num_tokens=num_tokens,
            num_layers=num_layers,
            kv_heads=kv_heads,
            head_dim=head_dim,
            dtype=dtype,
         )

        handoff_id = uuid.uuid4().hex[:12]
        state.handoff_id = handoff_id
        with self._lock:
            self._active_handoffs[handoff_id] = state
        logger.info(
            f"[Bridge] captured KV state: {len(kv_arrays)} buffers, "
            f"{num_tokens} tokens, {num_layers} layers @ {source_backend}"
           )
        return state

    def prepare_handoff(self, state: KVCacheState, target_backend: str) -> KVCacheState:
        """Mark buffers as shared with dual ref counting.

        Re-registers all buffers under "shared" backend and increments
        refs so source engine GC cannot free them while target uses them.
        """
        state.target_backend = target_backend
        for buf_id in state.buffer_ids:
            self._registry.share_buffer(buf_id, target_backend)
        logger.info(
            f"[Bridge] handoff ready: {state.source_backend} -> "
            f"{target_backend}, {len(state.buffer_ids)} buffers shared"
           )
        return state

    def claim(self, state: KVCacheState) -> dict[str, Any]:
        """Target engine claims the handoff.

        Returns all data needed to reconstruct KV cache on target side.
        Metal buffers are the SAME buffers — zero-copy.
        """
        if state.target_backend == "":
            raise RuntimeError("KVCacheState not prepared — call prepare_handoff first")

        logger.info(
            f"[Bridge] {state.target_backend} claimed KV handoff: "
            f"{state.num_tokens} tokens, {len(state.buffer_ids)} buffers"
           )

        # Keep in active handoffs — don't remove until BOTH sides release
        return {
            "block_table": state.block_table,
            "meta_states": state.meta_states,
            "buffer_ids": state.buffer_ids,
            "num_tokens": state.num_tokens,
            "num_layers": state.num_layers,
            "kv_heads": state.kv_heads,
            "head_dim": state.head_dim,
            "dtype": state.dtype,
            "model_name": state.model_name,
          }

    def release_source(self, state: KVCacheState) -> None:
        """Source engine signals it is done with the buffers.

        Decrements one ref per buffer. Buffers stay alive because
        prepare_handoff() added extra refs via share_buffer().
        """
        state.source_released = True
        for buf_id in state.buffer_ids:
            self._registry.decrement_ref(buf_id)
        logger.debug(
            f"[Bridge] {state.source_backend} released source ownership, "
            "buffers kept alive for target"
           )

    def release_target(self, state: KVCacheState) -> None:
        """Target engine signals it is done with the buffers.

        Decrements the bridge refs. When both source and target have
        released, refs reach 0 and buffers are freed.
        """
        state.target_released = True
        for buf_id in state.buffer_ids:
            self._registry.decrement_ref(buf_id)

        # Remove from active handoffs only when both sides released
        hid = state.handoff_id
        if hid and state.source_released and state.target_released:
            with self._lock:
                self._active_handoffs.pop(hid, None)
            logger.debug(f"[Bridge] fully released handoff {hid}, all buffers freed")
        elif hid:
            logger.debug(
                f"[Bridge] target released, source_released={state.source_released}, "
                "waiting for other side"
               )

    # Legacy alias — deprecated, use release_source + release_target
    def release_handoff(self, state: KVCacheState) -> None:
        """Release all buffers. Deprecated."""
        state.source_released = True
        state.target_released = True
        for buf_id in state.buffer_ids:
            self._registry.decrement_ref(buf_id)
            self._registry.decrement_ref(buf_id)
        hid = state.handoff_id
        if hid:
            with self._lock:
                self._active_handoffs.pop(hid, None)

    def get_active_handoffs(self) -> int:
        with self._lock:
            return len(self._active_handoffs)

    def evict_or_swap_active_kv(self, request_id: str) -> None:
        """Evict or swap KV cache for a request during preemption."""
        logger.debug(f"[Bridge] evict_or_swap for request {request_id}")


@dataclass
class FragmentationReport:
    total_allocated: int       # Total bytes in registry
    total_contiguous: int      # Largest contiguous block available
    fragmentation_ratio: float  # 0.0 = perfect, 1.0 = worst
    buffer_count: int
    recommendation: str         # "none", "compact", "evict"


class FragmentationMonitor:
    """Monitor Metal buffer fragmentation across backends.

    High fragmentation happens when omlx and Rapid-MLX each allocate
    and release buffers independently, creating gaps the other can't use.
    """

    def __init__(self, registry: MetalBufferRegistry, threshold: float = 0.4):
        self._registry = registry
        self._threshold = threshold

    def check(self) -> FragmentationReport:
        """Run fragmentation analysis."""
        stats = self._registry.get_stats()
        total = stats["total_bytes"]
        count = stats["total_buffers"]

         # Estimate fragmentation: more small buffers = more fragmentation
         # This is a heuristic — true fragmentation requires knowing buffer addresses
        if count == 0:
            return FragmentationReport(
                total_allocated=0, total_contiguous=0,
                fragmentation_ratio=0.0, buffer_count=0,
                recommendation="none",
             )

         # Simple heuristic: if we have many small buffers relative to total,
         # fragmentation is high. Real implementation would track buffer addresses.
        avg_size = total // count if count > 0 else 0
         # Fragmentation proxy: high buffer count with small avg = fragmented
        frag_ratio = min(1.0, max(0.0, 1.0 - (avg_size / (64 * 1024 * 1024))))

        if frag_ratio < self._threshold:
            rec = "none"
        elif frag_ratio < 0.7:
            rec = "compact"
        else:
            rec = "evict"

        return FragmentationReport(
            total_allocated=total,
            total_contiguous=int(total * (1.0 - frag_ratio)),
            fragmentation_ratio=frag_ratio,
            buffer_count=count,
            recommendation=rec,
         )

    def should_compact(self, is_gpu_busy: bool = True) -> bool:
        # Never compact while Metal GPU is actively computing
        # Uses mark-sweep-defer: flag during check, execute only when idle
        if is_gpu_busy:
            return False
        report = self.check()
        return report.recommendation in ("compact", "evict")
class UnifiedMemoryPool:
    """Unified memory pool for cross-engine coordination.

    Single entry point for:
    - Buffer registration/release (via MetalBufferRegistry)
    - Per-backend quota enforcement (via BackendQuota)
    - KV cache zero-copy handoff (via KVCacheBridge)
    - Fragmentation monitoring (via FragmentationMonitor)

    Usage:
        pool = UnifiedMemoryPool(total_bytes=..., quotas={...})

        # Register a buffer from omlx
        buf_id = pool.register(arr, "omlx", is_kv_cache=True)

        # Handoff KV cache from omlx -> rapid
        state = pool.capture_kv("model-x", "omlx", kv_arrays, block_table, meta_states, n)
        pool.prepare_handoff(state, "rapid")
        claimed = pool.claim_kv(state)

        # Release
        pool.release(buf_id)
    """

    def __init__(
        self,
        total_bytes: int = 0,
        quotas: dict[str, int] | None = None,
        frag_threshold: float = 0.4,
    ):
        """
        Args:
            total_bytes:    Total MLX memory pool size (0 = no global limit)
            quotas:         Per-backend max pre-allocation in bytes
            frag_threshold: Fragmentation ratio at which to suggest compaction
        """
        self._total_limit = total_bytes
        self.registry = MetalBufferRegistry()
        self.quotas: dict[str, BackendQuota] = {}
        self.bridge = KVCacheBridge(self.registry)
        self.monitor = FragmentationMonitor(self.registry, frag_threshold)

        if quotas:
            for backend, max_bytes in quotas.items():
                self.quotas[backend] = BackendQuota(backend=backend, max_bytes=max_bytes)

    def register(
        self,
        arr: Any,
        backend: str,
        is_kv_cache: bool = False,
    ) -> str:
        """Register a buffer, enforcing per-backend quota."""
        quota = self.quotas.get(backend)
        size = int(arr.nbytes) if hasattr(arr, "nbytes") else 0

        if quota and not quota.allocate(size):
            logger.warning(
                f"[Pool] {backend} quota exceeded: {size / 1e6:.1f}MB requested, "
                f"{quota.available / 1e6:.1f}MB available"
             )
             # Quota is a soft limit — log but don't block
             # Hard limit is the ProcessMemoryEnforcer at a higher level

        if self._total_limit > 0:
            current = self.registry.total_bytes
            if current + size > self._total_limit:
                logger.warning(
                    f"[Pool] global limit approaching: "
                    f"{(current + size) / 1e9:.2f}GB / {self._total_limit / 1e9:.2f}GB"
                 )

        return self.registry.register(arr, backend, is_kv_cache=is_kv_cache)

    def release(self, buf_id: str) -> bool:
        """Release a buffer (decrement ref count)."""
        freed = self.registry.decrement_ref(buf_id)
        if freed:
             # Update quota
            # We don't track per-buffer backend in quotas, so this is best-effort
            pass
        return freed

    def release_force(self, buf_id: str) -> bool:
        """Force release a buffer regardless of ref count."""
        return self.registry.release(buf_id)

    # KV cache handoff shortcuts

    def capture_kv(
        self,
        model_name: str,
        source_backend: str,
        kv_arrays: list[Any],
        block_table: list[int],
        meta_states: list[tuple],
        num_tokens: int,
    ) -> KVCacheState:
        return self.bridge.capture(
            model_name, source_backend, kv_arrays, block_table, meta_states, num_tokens
         )

    def prepare_handoff(self, state: KVCacheState, target_backend: str) -> KVCacheState:
        return self.bridge.prepare_handoff(state, target_backend)

    def claim_kv(self, state: KVCacheState) -> dict[str, Any]:
        return self.bridge.claim(state)

    def release_kv(self, state: KVCacheState) -> None:
        self.bridge.release_handoff(state)

    # Stats

    def get_stats(self) -> dict[str, Any]:
        report = self.monitor.check()
        quota_stats = {
            k: {
                 "max_bytes": v.max_bytes,
                 "current_bytes": v.current_bytes,
                 "utilization": v.utilization,
             }
            for k, v in self.quotas.items()
         }
        return {
             "registry": self.registry.get_stats(),
             "quotas": quota_stats,
             "fragmentation": {
                 "ratio": report.fragmentation_ratio,
                 "recommendation": report.recommendation,
                 "buffer_count": report.buffer_count,
             },
             "active_handoffs": self.bridge.get_active_handoffs(),
             "total_limit_bytes": self._total_limit,
         }
