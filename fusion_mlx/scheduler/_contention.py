# SPDX-License-Identifier: Apache-2.0
#
# Cross-scheduler decode-activity registry (#contention-pool).
#
# In pool mode several engines decode on the same GPU. An engine's own
# decode step times then jitter from the co-resident engines' interleave,
# so the per-engine contention detector's step-time CV spikes even though
# no external GPU process exists. Each scheduler marks its decode steps
# here; the detector consults the registry and suppresses the contention
# flag/warning while a sibling engine is actively decoding.
from __future__ import annotations

import time

_ACTIVE_WINDOW_S = 2.0
_STALE_S = 60.0
_last_decode: dict[int, float] = {}
_pool_shared = False


def set_pool_shared(flag: bool = True) -> None:
    global _pool_shared
    _pool_shared = flag


def pool_shared() -> bool:
    return _pool_shared


def mark_decode(scheduler_id: int, now: float | None = None) -> None:
    now = time.monotonic() if now is None else now
    _last_decode[scheduler_id] = now
    if len(_last_decode) > 16:
        stale = [k for k, ts in _last_decode.items() if now - ts > _STALE_S]
        for k in stale:
            del _last_decode[k]


def others_decoding(
    scheduler_id: int, window_s: float = _ACTIVE_WINDOW_S, now: float | None = None
) -> int:
    now = time.monotonic() if now is None else now
    return sum(
        1 for k, ts in _last_decode.items() if k != scheduler_id and now - ts < window_s
    )


def reset() -> None:
    _last_decode.clear()
