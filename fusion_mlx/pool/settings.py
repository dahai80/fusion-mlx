# SPDX-License-Identifier: Apache-2.0
"""Settings stub for fusion-mlx pool."""

from dataclasses import dataclass
import psutil


@dataclass
class GlobalSettings:
    idle_timeout: int = 3600
    custom_ceiling_bytes: int = 0


def get_system_memory() -> int:
    return psutil.virtual_memory().total
