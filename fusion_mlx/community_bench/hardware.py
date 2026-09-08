# SPDX-License-Identifier: Apache-2.0
"""Community benchmark hardware fingerprint collection."""

from __future__ import annotations

import logging
import platform
import subprocess
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class HardwareInfo:
    chip: str
    ram_gb: float
    cpu_cores: int
    gpu_cores: int


@dataclass
class SoftwareInfo:
    macos: str
    fusion_mlx: str
    mlx: str
    python: str


def _get_chip_name() -> str:
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() or platform.machine()
    except Exception:
        return platform.machine()


def _get_total_memory_gb() -> float:
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True, text=True, timeout=5,
        )
        return round(int(result.stdout.strip()) / (1024 ** 3), 1)
    except Exception:
        return 0.0


def _get_gpu_cores() -> int:
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.ncpu"],
            capture_output=True, text=True, timeout=5,
        )
        return int(result.stdout.strip())
    except Exception:
        return 0


def _get_fusion_mlx_version() -> str:
    try:
        from .._version import __version__
        return __version__
    except Exception:
        return "unknown"


def _get_mlx_version() -> str:
    try:
        import mlx.core as mx
        return getattr(mx, "__version__", "unknown")
    except Exception:
        return "unknown"


def is_apple_silicon() -> bool:
    return platform.machine() == "arm64"


def collect() -> tuple[HardwareInfo, SoftwareInfo]:
    hardware = HardwareInfo(
        chip=_get_chip_name(),
        ram_gb=_get_total_memory_gb(),
        cpu_cores=subprocess.run(
            ["sysctl", "-n", "hw.ncpu"], capture_output=True, text=True, timeout=5
        ).stdout.strip() if True else 0,
        gpu_cores=_get_gpu_cores(),
    )
    try:
        hardware.cpu_cores = int(
            subprocess.run(
                ["sysctl", "-n", "hw.ncpu"],
                capture_output=True, text=True, timeout=5,
            ).stdout.strip()
        )
    except Exception:
        hardware.cpu_cores = 0

    software = SoftwareInfo(
        macos=platform.mac_ver()[0] if hasattr(platform, "mac_ver") else platform.platform(),
        fusion_mlx=_get_fusion_mlx_version(),
        mlx=_get_mlx_version(),
        python=platform.python_version(),
    )
    return hardware, software


def detect_hardware() -> dict:
    hw, sw = collect()
    return {
        "chip": hw.chip,
        "ram_gb": hw.ram_gb,
        "cpu_cores": hw.cpu_cores,
        "gpu_cores": hw.gpu_cores,
        "macos": sw.macos,
        "fusion_mlx": sw.fusion_mlx,
        "mlx": sw.mlx,
        "python": sw.python,
    }
