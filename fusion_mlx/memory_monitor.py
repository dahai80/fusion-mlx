# SPDX-License-Identifier: Apache-2.0
"""Memory monitoring for fusion-mlx on Apple Silicon unified memory."""

import logging
import platform
import subprocess
from typing import Any, Optional

logger = logging.getLogger(__name__)


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
        try:
            import psutil
            self._ps = psutil
            self._has_psutil = True
        except ImportError:
            pass

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

        Keys:
            total: Total system RAM (bytes)
            available: Free + reclaimable memory (bytes)
            wired: Wired (unpageable) memory (bytes)
            compressed: Compressed memory (bytes)
            active: Active (in-use) memory (bytes)
            inactive: Inactive (cacheable) memory (bytes)
            mlx_cache: MLX Metal cache memory (bytes)
            mlx_peak: MLX peak memory (bytes)
            paged_cache_blocks: Active paged cache block count
            paged_cache_hit_rate: Paged cache hit rate (0.0-1.0)
        """
        mem = self._parse_vm_stat()
        sys_mem = self._get_sysctl_memory()

        total = sys_mem.get("total", 0)
        inactive = mem.get("inactive", 0)
        free = mem.get("free", 0)
        wired = mem.get("wired", 0)
        active = mem.get("active", 0)

        compressed = 0
        if self._has_psutil:
            try:
                virt = self._ps.virtual_memory()
                compressed = getattr(virt, "cached", 0)
            except Exception:
                pass

        mlx_cache = 0
        mlx_peak = 0
        try:
            import mlx.core as mx
            mlx_cache = mx.get_cache_memory()
            mlx_peak = mx.get_cache_max_memory()
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

        available = inactive + free

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

        Triggers when:
        - Available memory < 10% of total RAM
        - MLX cache > configured max_kv_cache_memory
        - macOS reports memory pressure via psutil
        """
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
