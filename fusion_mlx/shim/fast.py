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
    # Single source of truth: utils/hardware.py get_chip_generation +
    # get_mma_capability. FUSION_SHIM_FORCE_CHIP overrides sysctl (used by
    # tests + the C++ probe's own fallback), so the two paths agree on a
    # forced chip. On non-Apple/headless hosts this returns a conservative
    # all-false probe.
    forced = os.environ.get("FUSION_SHIM_FORCE_CHIP")
    if forced:
        chip = forced
        architecture = forced
    else:
        try:
            from ..utils.hardware import get_chip_name, get_mlx_device_name

            chip = get_chip_name() or "Apple Silicon"
            architecture = chip
            mlx_name = get_mlx_device_name()
            if mlx_name:
                architecture = mlx_name
        except Exception:
            chip = "Apple Silicon"
            architecture = "Apple Silicon"

    try:
        from ..utils.hardware import get_chip_generation, get_mma_capability

        gen = get_chip_generation(chip)
        mma = get_mma_capability(chip)
    except Exception:
        gen = 0
        mma = {"has_bf16_mma": False, "has_fp8_mma": False}

    # GPU core count is a physical host property (IORegistry
    # AGXAccelerator "gpu-core-count") — NOT faked under FORCE_CHIP.
    gpu_cores = 0
    try:
        from ..utils.hardware import get_gpu_core_count

        n = get_gpu_core_count()
        if n:
            gpu_cores = int(n)
    except Exception:
        gpu_cores = 0

    return {
        "architecture": architecture,
        "gen": gen,
        "has_bf16_mma": mma["has_bf16_mma"],
        "has_fp8_mma": mma["has_fp8_mma"],
        "gpu_core_count": gpu_cores,
        "device_name": chip,
    }


_PROBE_CACHE: tuple[tuple[str, str], dict[str, Any]] | None = None


def hardware_probe() -> dict[str, Any]:
    # Cached at module level, keyed on availability mode ("native" vs
    # "fallback") so tests / degrade paths that swap _ext to None get the
    # fallback probe, not a stale native result. The native probe shells
    # out to sysctl per call and the fallback re-runs chip detection —
    # neither is free on the status/admin surfaces that poll this.
    global _PROBE_CACHE
    mode = "native" if _ext is not None else "fallback"
    # fallback probe honors FUSION_SHIM_FORCE_CHIP — key on it too, else a
    # forced-chip degrade test can be served a stale real-machine fallback
    force_chip = os.environ.get("FUSION_SHIM_FORCE_CHIP", "")
    key = (mode, force_chip)
    if _PROBE_CACHE is not None and _PROBE_CACHE[0] == key:
        return _PROBE_CACHE[1]
    value = None
    if mode == "native" and hasattr(_ext, "hardware_probe_dict"):
        try:
            value = _ext.hardware_probe_dict()
        except Exception as exc:
            logger.warning("shim hardware_probe native failed: %s; using fallback", exc)
            _record_native_error(exc)
    if value is None:
        mode = "fallback"
        value = _python_hardware_probe()
    _PROBE_CACHE = (key, value)
    return value


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
# EngineRunner (PR-J) — dedicated decode thread + P-core QoS binding.
# Python fallback: an inline runner that executes submitted callables on the
# calling thread (same semantics as the native runner when its thread is
# not started). Switch: FUSION_ENGINE_RUNNER (default OFF).
# ---------------------------------------------------------------------------


class _InlineEngineRunner:
    # Degraded EngineRunner: no dedicated thread, every submit runs inline
    # and never fails (exceptions from the callable propagate to the caller
    # exactly as they would without the runner).

    def __init__(self, qos: int = 0):
        self.qos = qos

    def start(self) -> bool:
        return False

    def stop(self) -> None:
        return None

    def is_running(self) -> bool:
        return False

    def submit(self, fn):
        fn()
        return 0, ""

    def stats(self) -> dict[str, int]:
        return {
            "submitted": 0,
            "completed": 0,
            "failed": 0,
            "thread_started": 0,
            "thread_stopped": 0,
            "qos_class": self.qos,
        }


def is_engine_runner_enabled() -> bool:
    return os.environ.get("FUSION_ENGINE_RUNNER", "0") == "1"


def engine_runner(qos: int = 0):
    # Returns the native runner (dedicated thread + QoS) when the shim
    # extension is built AND FUSION_ENGINE_RUNNER=1; otherwise the inline
    # fallback — callers get a duck-typed runner either way.
    if (
        _ext is not None
        and hasattr(_ext, "EngineRunner")
        and is_engine_runner_enabled()
    ):
        return _ext.EngineRunner(qos)
    logger.debug(
        "shim engine_runner: using inline fallback (native=%s, enabled=%s)",
        _ext is not None,
        is_engine_runner_enabled(),
    )
    return _InlineEngineRunner(qos)


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
