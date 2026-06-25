"""CPU detection for macOS."""

from __future__ import annotations

import logging
import platform
import subprocess

logger = logging.getLogger(__name__)


def detect_cpu_name() -> str:
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=5,
         )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return platform.processor() or "Unknown"


def detect_cpu_cores() -> int:
    try:
        import psutil
        return psutil.cpu_count(logical=True) or 0
    except Exception:
        return 0
