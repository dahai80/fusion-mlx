# SPDX-License-Identifier: Apache-2.0
# Stub backends for video models with no MLX port.
#
# CogVideoX has no MLX reference implementation and is not shipped by mlx-video
# (which provides only ltx_2 and wan_2). A from-scratch MLX port of a video
# diffusion model (VAE + DiT + scheduler) is multi-thousand-line and unverifiable
# without the model weights and compute. Rather than ship unverified code, this
# backend auto-detects the model family and fails loudly with an actionable
# message pointing to upstream support and the working alternatives.
#
# Upstream feature request filed 2026-07-09 (per the file-issue-first flow):
#   CogVideoX: https://github.com/Blaizzy/mlx-video/issues/42
#
# Legacy LTX-Video (0.9.x) previously lived here as a stub; Phase 3 replaced it
# with a pure-MLX port in ltx_video_legacy.py (upstream issue #43 referenced for
# context). When an MLX port of CogVideoX lands upstream, replace the
# NotImplementedError in start()/generate() with a real delegator - the registry
# and engine wiring already handle the rest.

from __future__ import annotations

import logging
from typing import Any

from .base import VideoBackend, VideoConstraints, VideoGenParams

logger = logging.getLogger(__name__)

_UPSTREAM_ISSUES = "https://github.com/Blaizzy/mlx-video/issues"
_ALTERNATIVES = "ltx2 (LTX-2 / LTX-2.3) or wan2 (Wan2.1 / Wan2.2)"


class UnimplementedBackend(VideoBackend):
    name: str = ""
    supports_i2v: bool = False
    _family: str = ""

    def __init__(self, model_name: str, **kwargs: Any) -> None:
        self._model_name = model_name
        self._loaded = False

    @classmethod
    def detect(cls, model_path: str) -> bool:
        return False

    def _not_implemented(self) -> NotImplementedError:
        return NotImplementedError(
            f"{self._family} has no MLX port and is not shipped by mlx-video; "
            f"cannot generate on backend '{self.name}'. Request/track upstream "
            f"support: {_UPSTREAM_ISSUES}. Use {_ALTERNATIVES} instead."
        )

    async def start(self, model_path: str, **kwargs: Any) -> None:
        logger.warning(
            "%s requested (%s) but has no MLX implementation",
            self.name,
            model_path,
        )
        raise self._not_implemented()

    async def stop(self) -> None:
        # Nothing was started; stop is a no-op so EnginePool teardown is clean.
        return

    async def generate(self, params: VideoGenParams) -> list[bytes]:
        raise self._not_implemented()

    def constraints(self) -> VideoConstraints:
        # Permissive: let the request pass API-layer validation so the user
        # hits the clear NotImplementedError from start()/generate() instead of
        # a confusing 422 about dimensions for a backend that cannot run.
        return VideoConstraints(
            supports_i2v=self.supports_i2v,
            max_n=4,
            dim_divisibility=1,
            num_frames_validator=None,
        )


class CogVideoBackend(UnimplementedBackend):
    name = "cogvideo"
    supports_i2v = True
    _family = "CogVideoX"

    @classmethod
    def detect(cls, model_path: str) -> bool:
        p = model_path.lower()
        return "cogvideo" in p or "cog_video" in p
