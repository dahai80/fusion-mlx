# SPDX-License-Identifier: Apache-2.0
# #1296: swap telemetry for the monitor capacity gauge. psutil is the
# authoritative source (vm.swapusage / sysinfo). On macOS `used` is parsed
# from vm.swapusage and can legitimately read 0 on a swap-free machine
# while sin/sout still carry lifetime page-in/page-out counts, so the swap
# RATE is derived from the sin/sout deltas instead of used-byte deltas.
import logging
import threading
import time

logger = logging.getLogger(__name__)

_MIN_RATE_INTERVAL_S = 1.0

_lock = threading.Lock()
_last_at: float | None = None
_last_sin = 0
_last_sout = 0
_last_rate: float | None = None

_monotonic = time.monotonic


def _read_swap_counters() -> dict[str, int | float]:
    import psutil

    sm = psutil.swap_memory()
    return {
        "swap_total_bytes": int(getattr(sm, "total", 0) or 0),
        "swap_used_bytes": int(getattr(sm, "used", 0) or 0),
        "swap_free_bytes": int(getattr(sm, "free", 0) or 0),
        "swap_percent": float(getattr(sm, "percent", 0.0) or 0.0),
        "_sin": int(getattr(sm, "sin", 0) or 0),
        "_sout": int(getattr(sm, "sout", 0) or 0),
    }


def sample_swap() -> dict[str, float | None]:
    global _last_at, _last_sin, _last_sout, _last_rate

    try:
        raw = _read_swap_counters()
    except Exception:
        logger.warning("swap stats unavailable; reporting nulls", exc_info=True)
        return {
            "swap_total_bytes": None,
            "swap_used_bytes": None,
            "swap_free_bytes": None,
            "swap_percent": None,
            "swap_rate_bytes_per_second": None,
        }

    now = _monotonic()
    sin = int(raw.pop("_sin"))
    sout = int(raw.pop("_sout"))
    with _lock:
        if _last_at is None:
            # First sample establishes the baseline; a rate needs two points,
            # so report null instead of fabricating 0.
            _last_at = now
            _last_sin = sin
            _last_sout = sout
            _last_rate = None
        elif now - _last_at >= _MIN_RATE_INTERVAL_S:
            elapsed = now - _last_at
            delta = max(0, sin - _last_sin) + max(0, sout - _last_sout)
            _last_rate = delta / elapsed
            _last_at = now
            _last_sin = sin
            _last_sout = sout
        # Within the min interval return the previous rate without moving the
        # baseline: /metrics and /v1/health poll back to back and would
        # otherwise compute a near-zero rate over a millisecond-wide window.
        rate = _last_rate

    return {
        "swap_total_bytes": raw["swap_total_bytes"],
        "swap_used_bytes": raw["swap_used_bytes"],
        "swap_free_bytes": raw["swap_free_bytes"],
        "swap_percent": raw["swap_percent"],
        "swap_rate_bytes_per_second": rate,
    }


def _reset_for_tests() -> None:
    global _last_at, _last_sin, _last_sout, _last_rate
    with _lock:
        _last_at = None
        _last_sin = 0
        _last_sout = 0
        _last_rate = None
    logger.debug("swap stats baseline reset for tests")
