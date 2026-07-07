# SPDX-License-Identifier: Apache-2.0
"""Multi-modal LLM utilities — re-exports from utils/video and utils/image."""

import sys


def require_mlx_vlm_or_exit():
    print("VLM (vision-language model) is not available in this build", file=sys.stderr)
    sys.exit(1)
