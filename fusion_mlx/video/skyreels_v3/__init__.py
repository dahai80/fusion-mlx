# SPDX-License-Identifier: Apache-2.0
"""SkyReels-V3 MLX port - fusion-mlx video backend.

Ports the three SkyReels-V3 backbones (R2V 14B, V2V 14B, A2V 19B) from
PyTorch/CUDA to pure MLX on Apple Silicon, reusing the fusion-mlx
dFlash (mlx-mfa) attention, xfuser step-level strategy, TurboQuant KV,
and the existing wan2/ltx2 DiT ports as blueprints.
"""

from . import _device  # noqa: F401 - ensures device/stream init on import
from .config import (
    BRANCH_CONFIGS,
    SkyReelsBranchConfig,
    get_branch_config,
    list_models,
)
from .generate import TASK_TO_MODEL, generate_video

__all__ = [
    "_device",
    "SkyReelsBranchConfig",
    "BRANCH_CONFIGS",
    "get_branch_config",
    "list_models",
    "generate_video",
    "TASK_TO_MODEL",
]
