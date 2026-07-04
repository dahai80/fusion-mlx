# SPDX-License-Identifier: Apache-2.0
"""Multi-modal LLM utilities for video/image processing."""

import sys
from typing import Any

MAX_FRAMES = 16
DEFAULT_FPS = 0.5


def require_mlx_vlm_or_exit():
    print("VLM (vision-language model) is not available in this build", file=sys.stderr)
    sys.exit(1)


def process_image_input(image_data: Any) -> Any | None:
    """Process image input for VLM models."""
    return image_data


def process_video_input(video_path: str, max_frames: int = MAX_FRAMES) -> list[str]:
    """Extract frames from video for VLM models."""
    return []


def extract_video_frames_smart(video_path: str, fps: float = DEFAULT_FPS) -> list[str]:
    """Intelligently extract key frames from video."""
    return []


def save_frames_to_temp(frames: list[Any], prefix: str = "frame") -> list[str]:
    """Save extracted frames to temporary files."""
    return []
