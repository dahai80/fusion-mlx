# SPDX-License-Identifier: Apache-2.0
"""DFlash2 engine — thin bridge over the official dflash pkg.

In-target pattern: DFlash2InTargetDrafter loads ONLY the draft model and
binds to the scheduler's already-loaded target. The propose->verify->rollback
loop runs in dflash2_spec_step (scheduler/spec_decode.py), reusing the
scheduler's model + prompt_cache.
"""

from .generator import DFlash2InTargetDrafter

__all__ = ["DFlash2InTargetDrafter"]
