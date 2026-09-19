# SPDX-License-Identifier: Apache-2.0
# Graph-rewrite pass framework for MLX vision models (#911, #918).
#
# MLX's mx.compile fuses elementwise ops at the Metal level but does not fuse
# across conv/norm boundaries (Conv+GroupNorm+SiLU stays 3 kernels). This module
# provides a pattern registry + two entry points:
#
#   apply_patterns(root)         — REAL structural rewrite: walks nn.Sequential
#                                  containers and plain module lists under root
#                                  and replaces consecutive
#                                  Conv2d/Conv1d -> GroupNorm -> SiLU triples
#                                  with the fused ConvGroupNormSiLU module
#                                  (weights shared, not copied). Run AFTER
#                                  weight loading — the swap changes parameter
#                                  paths (.conv/.groupnorm under the fused
#                                  module), so a strict reload would mismatch.
#   compile_with_custom_pass(fn) — mx.compile wrapper. NOTE (#918): a bare
#                                  wrapper does NOT rewrite anything — module
#                                  structure is Python code, not a static
#                                  graph, so matching only happens through
#                                  apply_patterns. When no explicit pass is
#                                  given this logs INFO that it is a plain
#                                  mx.compile (never silently claim a pass).
#
# Matching is structural equality on the module graph (DFS, memoized by id) —
# not a textual AST rewrite. Container order is assumed to be execution order
# (nn.Sequential / ordered lists).

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import mlx.core as mx

logger = logging.getLogger(__name__)

GraphPass = Callable[[list[mx.array]], list[mx.array]]

_PATTERN_REGISTRY: dict[str, dict[str, Any]] = {}


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

    ``pass_fn`` is a TENSOR-level transform (list[mx.array] -> list[mx.array])
    applied around ``fn``. Module-level fusion must go through ``apply_patterns``
    BEFORE calling this — a bare wrapper is plain mx.compile (#918).
    """
    if pass_fn is None:
        if _PATTERN_REGISTRY:
            logger.info(
                "[graph_opt] %d pattern(s) registered but not applied — "
                "call apply_patterns(root) for module-level fusion; "
                "returning plain mx.compile (#918)",
                len(_PATTERN_REGISTRY),
            )
        else:
            logger.debug("[graph_opt] no pass, no patterns — plain mx.compile")
        return mx.compile(fn, shapeless=shapeless) if shapeless else mx.compile(fn)

    def _wrapped(*args, **kwargs):
        outs = pass_fn(list(args))
        return fn(*outs, **kwargs)

    return mx.compile(_wrapped, shapeless=True) if shapeless else mx.compile(_wrapped)


def _match_conv_groupnorm_silu(seq: list) -> bool:
    import mlx.nn as nn

    if len(seq) != 3:
        return False
    conv, norm, act = seq
    return (
        isinstance(conv, (nn.Conv2d, nn.Conv1d))
        and isinstance(norm, nn.GroupNorm)
        and isinstance(act, nn.SiLU)
    )


def _rewrite_conv_groupnorm_silu(seq: list):
    from .patterns import ConvGroupNormSiLU

    return ConvGroupNormSiLU(seq[0], seq[1])


register_pattern(
    "conv_groupnorm_silu", _match_conv_groupnorm_silu, _rewrite_conv_groupnorm_silu
)


def apply_patterns(root, _seen=None) -> int:
    """Apply registered patterns to the module tree under ``root`` (#918).

    Walks container children (nn.Sequential .layers, plain module lists, module
    dicts) and replaces consecutive matched triples in place. Returns the
    number of rewrites. Assumes container order == execution order; code-level
    call sequences (e.g. ``self.conv_out(silu(self.norm(x)))`` inside a custom
    ``__call__``) are NOT visible here — rewrite those by using the fused
    module directly. Call AFTER weight loading (weights are shared refs, but
    parameter paths change under the fused module).
    """
    import mlx.nn as nn

    if _seen is None:
        _seen = set()
    if id(root) in _seen:
        return 0
    _seen.add(id(root))
    applied = 0

    def _match_window(window):
        for pat in _PATTERN_REGISTRY.values():
            if pat["match"](window):
                return pat["rewrite"](window)
        return None

    def _rewrite(mods, keys=None, container=None):
        # ``keys``/``container`` set: dict-backed — rewrite mirrored by key.
        # ``keys`` and ``mods`` stay index-aligned via lockstep slicing.
        nonlocal applied
        i = 0
        while i <= len(mods) - 3:
            fused = _match_window(mods[i : i + 3])
            if fused is None:
                i += 1
                continue
            if container is not None:
                container[keys[i]] = fused
                for k in keys[i + 1 : i + 3]:
                    del container[k]
                keys[i : i + 3] = [keys[i]]
            mods[i : i + 3] = [fused]
            applied += 1
            logger.info("[graph_opt] fused 3-module sequence at index %d", i)
            # keep i: the fused module may start the next match

    # MLX nn.Module is a dict subclass: mx.array/list/dict/Module attributes
    # are stored via self[key] and root[key] returns the LIVE stored value.
    # children() rebuilds container copies (mutations would be lost), so
    # traversal uses the dict interface. Lists/dicts are mutated in place so
    # parameter paths stay live on the parent.
    if isinstance(root, nn.Module):
        for key in list(root.keys()):
            val = root[key]
            if isinstance(val, list):
                if val and all(isinstance(c, nn.Module) for c in val):
                    _rewrite(val)
                for m in val:
                    if isinstance(m, nn.Module):
                        applied += apply_patterns(m, _seen)
            elif isinstance(val, dict):
                keys = list(val.keys())
                mods = [val[k] for k in keys]
                if mods and all(isinstance(v, nn.Module) for v in mods):
                    _rewrite(mods, keys=keys, container=val)
                for v in mods:
                    applied += apply_patterns(v, _seen)
            elif isinstance(val, nn.Module):
                applied += apply_patterns(val, _seen)
        return applied
    if isinstance(root, list):
        if root and all(isinstance(c, nn.Module) for c in root):
            _rewrite(root)
        for m in root:
            if isinstance(m, nn.Module):
                applied += apply_patterns(m, _seen)
        return applied
    if isinstance(root, dict):
        keys = list(root.keys())
        mods = [root[k] for k in keys]
        if mods and all(isinstance(v, nn.Module) for v in mods):
            _rewrite(mods, keys=keys, container=root)
        for v in mods:
            applied += apply_patterns(v, _seen)
        return applied
    return applied


def fused_conv_groupnorm_silu(conv, groupnorm):
    """Construct a fused Conv+GroupNorm+SiLU module (#911).

    The fused module applies conv, then GroupNorm with FP32-protected stats
    (via SafeGroupNorm when available), then SiLU — all in one ``__call__``
    so mx.compile sees a single kernel boundary.
    """
    from .patterns import ConvGroupNormSiLU

    return ConvGroupNormSiLU(conv, groupnorm)
