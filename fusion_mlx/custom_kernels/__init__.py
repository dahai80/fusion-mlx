# SPDX-License-Identifier: Apache-2.0
"""Optional native custom kernels bundled with FusionMLX."""

# AtomCode fix #121: 导出 fp8_linear + quantize 模块 (2026-07-19)
# m5_optimizer.py:116 引用 fusion_mlx.custom_kernels.fp8_linear,
# 原 __init__.py 空导出致 ImportError, 补导让 FP8 量化路径真生效
from . import fp8_linear, quantize  # noqa: F401

__all__ = ["fp8_linear", "quantize"]
