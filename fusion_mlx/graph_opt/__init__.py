# SPDX-License-Identifier: Apache-2.0
"""graph_opt — mx.compile-injected graph-rewrite passes for vision models (#911).

MLX does not expose a public graph IR for arbitrary rewriting. This module
provides a *pattern-matched functional fusion* layer that intercepts common
op sequences (Conv -> GroupNorm -> SiLU) and replaces them with a single fused
module, registered before mx.compile. The registry is DFS-memoized so a repeated
pattern in a loop is matched once.
"""

from .passes import (
    GraphPass,
    compile_with_custom_pass,
    fused_conv_groupnorm_silu,
    register_pattern,
)
from .patterns import ConvGroupNormSiLU

__all__ = [
    "GraphPass",
    "compile_with_custom_pass",
    "register_pattern",
    "fused_conv_groupnorm_silu",
    "ConvGroupNormSiLU",
]
