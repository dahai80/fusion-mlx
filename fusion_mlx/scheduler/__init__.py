# SPDX-License-Identifier: Apache-2.0
"""Scheduler subpackage."""

import mlx.core as mx  # noqa: F401  (backward-compat: tests patch scheduler.mx)

# Install the M5 single-stream compat shim (#404/#617) before any submodule
# that transitively imports mlx_lm at module load — mlx_lm/__init__.py
# captures mx.new_thread_local_stream at import time and the shim must wrap
# it first. Central-boot install here covers every entry path through
# fusion_mlx (the package __init__ imports .scheduler), so library callers
# like `import fusion_mlx.oq` are guarded too. Idempotent; no-op on Linux
# CI and on mlx builds lacking new_thread_local_stream (#408).
from .. import _mlx_compat

_mlx_compat.install()

from ..cache.paged_ssd_cache import PagedSSDCacheManager  # noqa: F401
from ..speculative.vlm_mtp import run_vlm_mtp_decode  # noqa: F401
from . import monkeypatches  # noqa: F401
from ._mtp_vendored import _install_mtp_vendored  # noqa: F401
from .config import SchedulerConfig, SchedulerOutput, SchedulingPolicy
from .core import Scheduler

# Backward-compat re-exports: the old monolithic fusion_mlx.scheduler exposed these
# helpers/constants at package level. Tests (and any external callers) still
# reference them via fusion_mlx.scheduler.X after the submodule split.
from .helpers import (
    _advance_vlm_extra,
    _safe_sync_stream,
    _slice_vlm_extra,
    _sync_and_clear_cache,
)  # noqa: F401
from .monkeypatches import _default_generation_stream  # noqa: F401
from .sched_misc import HAS_TIERED_CACHE  # noqa: F401
from .types import (
    _BoundarySnapshotProvider,
    _InflightStoreInfo,
    _PrefillAbortedError,
    _PrefillState,
    _StoreCacheGate,
    _VLMMTPDecodeState,
)


class BackpressureError(Exception):
    pass
