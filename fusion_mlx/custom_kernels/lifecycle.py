# SPDX-License-Identifier: Apache-2.0
"""Custom kernel memory lifecycle scope.

P0底座 (ANE/Metal扩容): wraps custom Metal/ANE kernel execution so transient
GPU allocations are released deterministically between requests. Prevents the
cross-request activation accumulation that drove four-view image gen into
EXC_BAD_ACCESS (wired exhaustion).

Usage:
    with with_kernel_scope("fused_gemv"):
        out = my_custom_kernel(...)
    # mx.clear_cache() called on exit, baseline/peak logged.
"""

import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)


def _safe_get_active_memory() -> int:
    try:
        import mlx.core as mx

        return int(mx.get_active_memory() or 0)
    except Exception:
        return 0


def _safe_clear_cache() -> None:
    try:
        import mlx.core as mx

        mx.clear_cache()
    except Exception as exc:  # noqa: BLE001
        logger.debug("with_kernel_scope: mx.clear_cache failed: %s", exc)


@contextmanager
def with_kernel_scope(label: str = "kernel"):
    """Context manager that clears MLX cache on exit and logs memory delta.

    On exception inside the scope the cache is still cleared (finally), then
    the exception re-raises. A failing mx.clear_cache is swallowed so a
    secondary cleanup error never masks the original failure.
    """
    baseline = _safe_get_active_memory()
    try:
        yield
    finally:
        _safe_clear_cache()
        peak = _safe_get_active_memory()
        delta = peak - baseline
        logger.debug(
            "kernel_scope[%s] baseline=%s peak=%s delta=%s",
            label,
            baseline,
            peak,
            delta,
        )
