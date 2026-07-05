# SPDX-License-Identifier: Apache-2.0
# DSpark speculative decoding — DeepSeek DeepSpec block drafter.
#
# DSpark (DeepSeek's open-source DeepSpec) is a lossless block speculative
# decoder: a 5-layer context-injected drafter reads hidden states tapped
# from the target's own layers, proposes a 7-token block per round, and
# the target verifies in a single forward pass. Acceptance uses
# distribution-preserving rejection sampling so greedy output is
# bit-for-bit identical to vanilla decoding (0 divergences vs the
# DeepSeek reference over 6,614 tokens — see dspark-metal PROOF.md).
#
# This package mirrors the DFlash integration shape (eligibility + runtime
# + dedicated FastAPI server) but is SIMPLER: dspark-metal's
# DSparkGenerator is self-contained (loads its own target + draft), so
# the server wraps a single generator rather than tapping an existing
# BatchedEngine. Drafter checkpoints must be converted to MLX format
# locally via dspark-metal-convert (the draft is a local path, not an HF
# repo), so the drafter is operator-supplied via --dspark-drafter-path
# rather than a per-alias registry field.

from .eligibility import (
    DSparkUnavailable,
    check,
    eligible_aliases,
    have_runtime,
    report,
)
from .runtime import DSparkRuntime, load_runtime

__all__ = [
    "DSparkUnavailable",
    "check",
    "eligible_aliases",
    "have_runtime",
    "report",
    "DSparkRuntime",
    "load_runtime",
]
