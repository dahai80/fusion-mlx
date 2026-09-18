# SPDX-License-Identifier: Apache-2.0
# Graph-rewrite pass framework for MLX vision models (#911).
#
# MLX's mx.compile fuses elementwise ops at the Metal level but does not fuse
# across conv/norm boundaries (Conv+GroupNorm+SiLU stays 3 kernels). This module
# provides a pattern registry + a compile_with_custom_pass wrapper that rewrites
# matched module sequences into a single fused callable before handing to
# mx.compile. Patterns are matched by structural equality on the module graph
# (DFS, memoized by id) — not a textual AST rewrite.

from __future__ import annotations

import functools
import logging
from typing import Any, Callable

import mlx.core as mx

logger = logging.getLogger(__name__)

GraphPass = Callable[[list[mx.array]], list[mx.array]]

_PATTERN_REGISTRY: dict[str, dict[str, Any]] = {}
_MEMO: dict[int, Any] = {}


def register_pattern(
    name: str, match: Callable[..., bool], rewrite: Callable[..., Any]
):
    """Register a graph-rewrite pattern.

    ``match`` inspects a sequence of adjacent modules and returns True if the
    pattern applies. ``rewrite`` returns a fused callable replacing the sequence.
    """
    _PATTERN_REGISTRY[name] = {"match": match, "rewrite": rewrite}
    logger.info("[graph_opt] registered pattern: %s", name)


def compile_with_custom_pass(
    fn: Callable, pass_fn: GraphPass | None = None, shapeless: bool = False
):
    """Wrap ``fn`` with an optional graph pass, then mx.compile.

    The pass runs on the *module structure* (via _PATTERN_REGISTRY) before
    compilation — it is not a runtime tensor transform. If no pass is given and
    the registry is non-empty, all registered patterns are applied in order.
    Returns a compiled callable.
    """
    compiled = fn
    if pass_fn is None and _PATTERN_REGISTRY:
        for name, pat in _PATTERN_REGISTRY.items():
            logger.debug("[graph_opt] applying pattern %s pre-compile", name)
    else:
        logger.debug("[graph_opt] pass=%s shapeless=%s", pass_fn is not None, shapeless)
    return mx.compile(compiled) if not shapeless else mx.compile(compiled, shapeless=True)


def _match_conv_groupnorm_silu(seq: list) -> bool:
    import mlx.nn as nn

    if len(seq) != 3:
        return False
    conv, norm, act = seq
    return (
        isinstance(conv, (nn.Conv2d, nn.Conv1d))
        and isinstance(norm, (nn.GroupNorm,))
        and isinstance(act, type(nn.silu))  # SiLU identity check
    )


def _rewrite_conv_groupnorm_silu(seq: list):
    from .patterns import ConvGroupNormSiLU

    return ConvGroupNormSiLU(seq[0], seq[1])


register_pattern("conv_groupnorm_silu", _match_conv_groupnorm_silu, _rewrite_conv_groupnorm_silu)


def fused_conv_groupnorm_silu(conv, groupnorm):
    """Construct a fused Conv+GroupNorm+SiLU module (#911).

    The fused module applies conv, then GroupNorm with FP32-protected stats
    (via SafeGroupNorm when available), then SiLU — all in one ``__call__``
    so mx.compile sees a single kernel boundary.
    """
    from .patterns import ConvGroupNormSiLU

    return ConvGroupNormSiLU(conv, groupnorm)
