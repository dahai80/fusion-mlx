# SPDX-License-Identifier: Apache-2.0
"""Scheduler subpackage."""

import mlx.core as mx  # noqa: F401  (backward-compat: tests patch scheduler.mx)

from . import monkeypatches  # noqa: F401
from .config import SchedulerConfig, SchedulerOutput, SchedulingPolicy
from .core import Scheduler
from .types import (
    _BoundarySnapshotProvider,
    _InflightStoreInfo,
    _PrefillAbortedError,
    _PrefillState,
    _StoreCacheGate,
    _VLMMTPDecodeState,
)

# Backward-compat re-exports: the old monolithic omlx.scheduler exposed these
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
from ..cache.paged_ssd_cache import PagedSSDCacheManager  # noqa: F401
from ..speculative.vlm_mtp import run_vlm_mtp_decode  # noqa: F401


class BackpressureError(Exception):
    pass
