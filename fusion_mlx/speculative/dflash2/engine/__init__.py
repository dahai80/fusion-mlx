# SPDX-License-Identifier: Apache-2.0
"""DFlash2 engine — thin bridge over the official dflash pkg.

Unlike DSpark's vendored engine (12 files, 200K+), DFlash2 delegates the
entire propose->verify->rollback loop to ``dflash.stream_generate``
(model_mlx.py:713). This package is a single generator module that loads
target + draft, binds, and yields tokens from stream_generate's
GenerationResponse chunks. Hidden-state capture (_patch_model),
CandidateSelector path selection, cache rollback, and per-accept hidden
trim all run inside stream_generate — we surface only the accepted
tokens to the scheduler.
"""

from .generator import DFlash2Generator

__all__ = ["DFlash2Generator"]
