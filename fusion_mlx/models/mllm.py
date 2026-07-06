# SPDX-License-Identifier: Apache-2.0
"""Multi-modal LLM utilities — re-exports from utils/video and utils/image."""

import sys

from ..utils.video import (
    DEFAULT_FPS,
    MAX_FRAMES,
    FileSizeExceededError,
    cleanup_all_temp_files,
    cleanup_temp_file,
    decode_base64_video,
    describe_video,
    download_video,
    extract_video_frames_smart,
    is_base64_video,
    is_url,
    process_image_input,
    process_video_input,
    save_frames_to_temp,
    smart_nframes,
)


def require_mlx_vlm_or_exit():
    print("VLM (vision-language model) is not available in this build", file=sys.stderr)
    sys.exit(1)
