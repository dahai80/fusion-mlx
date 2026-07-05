"""Unified hardware detection orchestrator for fusion-mlx (macOS/Apple Silicon focused)."""

from __future__ import annotations

import logging
import platform

from .apple import detect_apple_gpu
from .cpu import detect_cpu_cores, detect_cpu_name
from .memory import detect_disk_free_bytes, detect_ram_bytes
from .types import HardwareInfo

logger = logging.getLogger(__name__)


def detect_hardware() -> HardwareInfo:
    os_name = platform.system().lower()
    if os_name not in ("linux", "darwin", "windows"):
        os_name = "darwin"

    gpus = []
    if os_name == "darwin":
        gpus.extend(detect_apple_gpu())

    cpu_name = detect_cpu_name()
    cpu_cores = detect_cpu_cores()

    ram_bytes = detect_ram_bytes()
    disk_free = detect_disk_free_bytes()

    return HardwareInfo(
        gpus=gpus,
        cpu_name=cpu_name,
        cpu_cores=cpu_cores,
        ram_bytes=ram_bytes,
        disk_free_bytes=disk_free,
        os=os_name,
    )
