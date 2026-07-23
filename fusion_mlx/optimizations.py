# SPDX-License-Identifier: Apache-2.0
"""
Hardware detection and optimization status for fusion-mlx.

Re-exports from utils.hardware for backward compatibility with fusion-mlx API.
"""

import logging

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

from fusion_mlx.utils.hardware import (
    HardwareInfo,
    detect_hardware,
)
from fusion_mlx.utils.hardware import (
    get_total_memory_gb as get_system_memory_gb,
)

logger = logging.getLogger(__name__)

__all__ = [
    "HardwareInfo",
    "detect_hardware",
    "get_system_memory_gb",
    "get_optimization_status",
]


def get_optimization_status() -> dict:
    hw = detect_hardware()
    result = {
        "hardware": {
            "chip": hw.chip_name,
            "total_memory_gb": hw.total_memory_gb,
        },
        "mlx_lm_features": {
            "flash_attention": "built-in",
            "metal_kernels": "optimized for Apple Silicon",
            "kv_cache": "managed by mlx-lm",
            "quantization": "4-bit and 8-bit supported",
        },
    }
    if HAS_MLX:
        try:
            device_info = mx.device_info()
            result["hardware"]["device_name"] = device_info.get(
                "device_name", "Unknown"
            )
            result["mlx_memory"] = {
                "active_bytes": mx.get_active_memory(),
                "cache_bytes": mx.get_cache_memory(),
                "peak_bytes": mx.get_peak_memory(),
            }
            flash_available = hasattr(mx, "fast") and hasattr(
                mx.fast, "scaled_dot_product_attention"
            )
            result["mlx_lm_features"]["flash_attention"] = (
                "built-in" if flash_available else "not available"
            )
        except Exception:
            pass
    return result
