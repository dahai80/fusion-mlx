"""Apple Silicon GPU detection via system_profiler (macOS only)."""

from __future__ import annotations

import json
import logging
import subprocess

from .types import GPUInfo

logger = logging.getLogger(__name__)

# Apple Silicon unified memory bandwidth in GB/s (theoretical peak)
APPLE_GPU_BANDWIDTH: dict[str, float] = {
    "M1 Ultra": 800.0,
    "M1 Max": 400.0,
    "M1 Pro": 200.0,
    "M1": 68.25,
    "M2 Ultra": 800.0,
    "M2 Max": 400.0,
    "M2 Pro": 200.0,
    "M2": 100.0,
    "M3 Ultra": 800.0,
    "M3 Max": 400.0,
    "M3 Pro": 150.0,
    "M3": 100.0,
    "M4 Ultra": 819.2,
    "M4 Max": 546.0,
    "M4 Pro": 273.0,
    "M4": 120.0,
    "M5 Max": 614.0,
    "M5 Pro": 307.0,
    "M5": 153.0,
}


def _lookup_bandwidth(chip_name: str) -> float | None:
    chip_upper = chip_name.upper()
    for key in sorted(APPLE_GPU_BANDWIDTH, key=len, reverse=True):
        if key.upper() in chip_upper:
            return APPLE_GPU_BANDWIDTH[key]
    return None


def detect_apple_gpu() -> list[GPUInfo]:
    """Detect Apple Silicon GPU. Returns empty list on non-macOS or failure."""
    try:
        result = subprocess.run(
            ["system_profiler", "SPHardwareDataType", "-json"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return []
        data = json.loads(result.stdout)
    except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError):
        logger.debug("system_profiler not available (not macOS)")
        return []

    try:
        hw_items = data["SPHardwareDataType"]
        hw = hw_items[0]
        chip_name = hw.get("chip_type", "")
        if not chip_name:
            return []

        # Parse physical memory string like "32 GB" -> bytes
        memory_str = hw.get("physical_memory", "0 GB")
        parts = memory_str.split()
        mem_value = int(parts[0])
        mem_unit = parts[1].upper() if len(parts) > 1 else "GB"
        multiplier = {"GB": 1024**3, "TB": 1024**4, "MB": 1024**2}.get(
            mem_unit, 1024**3
        )
        unified_memory = mem_value * multiplier

        return [
            GPUInfo(
                name=chip_name,
                vendor="apple",
                vram_bytes=unified_memory,
                memory_bandwidth_gbps=_lookup_bandwidth(chip_name),
                shared_memory=True,
            )
        ]
    except (KeyError, IndexError, ValueError) as e:
        logger.debug(f"Failed to parse Apple hardware info: {e}")
        return []
