# SPDX-License-Identifier: Apache-2.0
# Pure-MLX port of Wan2.2 video model (vendored from mlx-video).
# Phase 5 Wan2 direct-MLX port. Replaces mlx_video.models.wan_2.
from ..ltx2.utils import get_model_path
from .generate import generate_video

__all__ = ["generate_video", "get_model_path"]
