"""fusion-mlx C++ Shim: native fallback layer.

Mirrors the glm_moe_dsa/fast.py degrade pattern: ``from . import _ext``
with a graceful Python fallback when the native extension is not built.
PR-A exports the Tier-1 safety base (hardware_probe, memory_sentinel,
C-ABI last-error). Custom Metal ops land in later PRs and follow the same
``if _ext is not None: _ext.op(...) else: python_fallback(...)`` shape.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

try:
    from . import _ext
except Exception as exc:  # pragma: no cover - depends on local native build
    _ext = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


# Symbols exported by the native _ext module. Updated as PRs add ops.
NATIVE_SYMBOLS = (
    "hardware_probe",
    "hardware_probe_dict",
    "start_memory_sentinel",
    "stop_memory_sentinel",
    "last_memory_pressure",
    "last_error_code",
    "last_error_message",
)


def is_native_available() -> bool:
    return _ext is not None


def import_error() -> Exception | None:
    return _IMPORT_ERROR


def native_symbols() -> tuple[str, ...]:
    if _ext is None:
        return ()
    return tuple(name for name in NATIVE_SYMBOLS if hasattr(_ext, name))


def missing_symbols(required: tuple[str, ...]) -> list[str]:
    return [name for name in required if not hasattr(_ext, name)]


# ---------------------------------------------------------------------------
# hardware_probe — Python fallback reuses utils/hardware.py chip detection.
# ---------------------------------------------------------------------------


def _python_hardware_probe() -> dict[str, Any]:
    # Derive BF16/FP8 MMA capability from the chip generation string.
    # M3+ has hardware BF16 simdgroup matrix MMA; M4+ adds FP8 throughput.
    # On non-Apple/headless hosts this returns a conservative all-false probe.
    # FUSION_SHIM_FORCE_CHIP overrides sysctl (used by tests + the C++
    # probe's own fallback), so the two paths agree on a forced chip.
    forced = os.environ.get("FUSION_SHIM_FORCE_CHIP")
    if forced:
        chip = forced
    else:
        try:
            from ..utils.hardware import get_chip_name, get_mlx_device_name

            chip = get_chip_name() or "Apple Silicon"
        except Exception:
            chip = "Apple Silicon"

    gen = 0
    for i, ch in enumerate(chip):
        if ch == "M" and i + 1 < len(chip) and chip[i + 1].isdigit():
            gen = int(chip[i + 1])
            break

    has_bf16 = gen >= 3
    has_fp8 = gen >= 4
    device_name = chip
    architecture = chip
    if not forced:
        try:
            from ..utils.hardware import get_mlx_device_name

            mlx_name = get_mlx_device_name()
            if mlx_name:
                architecture = mlx_name
        except Exception:
            pass

    return {
        "architecture": architecture,
        "gen": gen,
        "has_bf16_mma": has_bf16,
        "has_fp8_mma": has_fp8,
        "gpu_core_count": 0,
        "device_name": device_name,
    }


def hardware_probe() -> dict[str, Any]:
    if _ext is not None and hasattr(_ext, "hardware_probe_dict"):
        try:
            return _ext.hardware_probe_dict()
        except Exception as exc:
            logger.warning("shim hardware_probe native failed: %s; using fallback", exc)
            _record_native_error(exc)
    return _python_hardware_probe()


# ---------------------------------------------------------------------------
# memory_sentinel — Python fallback is a no-op (ProcessMemoryEnforcer
# polling stays authoritative). The C++ dispatch_source is strictly
# additive: faster reactive signal on top of the existing 1s poll.
# ---------------------------------------------------------------------------


def start_memory_sentinel(callback=None) -> bool:
    if _ext is not None and hasattr(_ext, "start_memory_sentinel"):
        try:
            return bool(_ext.start_memory_sentinel(callback))
        except Exception as exc:
            logger.warning("shim start_memory_sentinel native failed: %s", exc)
            _record_native_error(exc)
            return False
    logger.info(
        "shim memory_sentinel native unavailable; ProcessMemoryEnforcer "
        "polling path stays authoritative"
    )
    return False


def stop_memory_sentinel() -> None:
    if _ext is not None and hasattr(_ext, "stop_memory_sentinel"):
        try:
            _ext.stop_memory_sentinel()
        except Exception as exc:
            logger.debug("shim stop_memory_sentinel: %s", exc)


def last_memory_pressure() -> int:
    if _ext is not None and hasattr(_ext, "last_memory_pressure"):
        try:
            return int(_ext.last_memory_pressure())
        except Exception:
            return 0
    return 0


# ---------------------------------------------------------------------------
# C-ABI last-error accessor
# ---------------------------------------------------------------------------


def last_error_code() -> int:
    if _ext is not None and hasattr(_ext, "last_error_code"):
        try:
            return int(_ext.last_error_code())
        except Exception:
            return 0
    return 0


def last_error_message() -> str:
    if _ext is not None and hasattr(_ext, "last_error_message"):
        try:
            return str(_ext.last_error_message())
        except Exception:
            return ""
    return ""


_last_native_error: Exception | None = None


def _record_native_error(exc: Exception) -> None:
    global _last_native_error
    _last_native_error = exc


def last_native_error() -> Exception | None:
    return _last_native_error


def __getattr__(name: str) -> Any:
    # Transparent proxy: unknown attribute resolves against _ext first so
    # future native ops are reachable without a fast.py shim entry.
    if _ext is not None and hasattr(_ext, name):
        return getattr(_ext, name)
    raise AttributeError(f"module 'fusion_mlx.shim.fast' has no attribute {name!r}")


def __dir__() -> list[str]:
    names = set(globals())
    names.update(NATIVE_SYMBOLS)
    if _ext is not None:
        names.update(dir(_ext))
    return sorted(names)
