# SPDX-License-Identifier: Apache-2.0
"""Multi-modal LLM utilities — re-exports from utils/video and utils/image."""

from ..utils.video import (  # noqa: F401 - re-exports for backward compat
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
