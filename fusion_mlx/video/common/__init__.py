# SPDX-License-Identifier: Apache-2.0
"""Unified video common base layer (PRD v1 §2.2/§3.5).

Pure public base classes shared by every video backend. NO model-specific
operators here — LTX/H3 differential kernels stay in their own packages
(`fusion_mlx.video.ltx2_5`, `fusion_mlx.video.minimax_h3`). The common
layer only defines abstract VAE / upsampler / scheduler interfaces so
the unified scheduler + router can treat both backends polymorphically.

Design rules (PRD v1 §2.2 enforced here):
  1. Public base absolute purity — no model-specific ops in this package.
  2. Runtime zero-dependency — MLX-native only, no Diffusers/PyTorch.
  3. Memory safety first — bases expose release() for immediate teardown.
"""

from __future__ import annotations

from .scheduler_base import NoiseSchedulerBase
from .upsample_base import UpsampleBase
from .video_vae_base import VideoVAEBase

__all__ = ["VideoVAEBase", "UpsampleBase", "NoiseSchedulerBase"]
