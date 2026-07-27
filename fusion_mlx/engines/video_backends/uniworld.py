# SPDX-License-Identifier: Apache-2.0
# Backend bridge: re-exports UniWorldBackend from the video/uniworld package
# so the video_backends registry can import it by submodule path.

from fusion_mlx.video.uniworld.backend import UniWorldBackend

__all__ = ["UniWorldBackend"]
