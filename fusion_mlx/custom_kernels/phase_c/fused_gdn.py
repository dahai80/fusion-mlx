# SPDX-License-Identifier: Apache-2.0
"""Fused GDN (Generalized Divisive Normalization) megakernel.

GDN is a activation normalization used in neural audio/image codecs
(SoundStream, EnCodec, JPEG-style autoencoders): y_i = x_i / sqrt(c_i + sum_j
gamma_ij * x_j^2). The fused form collapses the square, weighted sum, add
epsilon, rsqrt, and divide into a single op graph so MLX fuses them into one
Metal kernel (no intermediate buffers).

No model in this repo currently consumes a standalone GDN layer — the only
GDN references are dflash_mlx.engine.target_qwen_gdn (upstream-vendored) and
prefix_cache comments. This module is the registered Phase C kernel so a
future audio/diffusion codec can wire it without re-implementing the math.
A native Metal megakernel (metal/fused_gdn.metal) would fuse the same graph
into a single threadgroup pass; the pure-MLX graph below already lets
mx.compile emit a single fused kernel.

NOTE: this is the forward-only inference path. Training GDN (with the
backward through the divisive normalization) is out of scope.
"""

from __future__ import annotations

import logging

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)


class FusedGDN(nn.Module):
    # y = x / sqrt(gamma @ x^2 + beta + eps), the standard GDN parameterization
    # (Ballé 2016). gamma is (C, C), beta is (C,). For the common diagonal
    # approximation gamma is (C,) and the matmul collapses to an elementwise
    # weighted square — supported via the `diagonal` flag.

    def __init__(
        self,
        channels: int,
        diagonal: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.channels = channels
        self.diagonal = diagonal
        self.eps = eps
        if diagonal:
            self.gamma = mx.ones((channels,), dtype=mx.float16)
            self.beta = mx.zeros((channels,), dtype=mx.float16)
        else:
            self.gamma = mx.eye(channels, dtype=mx.float16)
            self.beta = mx.zeros((channels,), dtype=mx.float16)

    def __call__(self, x: mx.array) -> mx.array:
        # x: (..., C) or (C, ...). Operate along channel dim = last axis if
        # 2D, else axis 0 (codec convention: channel-first).
        if x.ndim == 2:
            sq = x * x
            if self.diagonal:
                denom = self.gamma * sq
            else:
                denom = sq @ self.gamma
            denom = denom + self.beta + mx.array(self.eps, dtype=x.dtype)
            return x * mx.rsqrt(denom.astype(mx.float32)).astype(x.dtype)
        # channel-first (C, ...): move C to last for the weighted sum.
        c = x.shape[0]
        x_perm = mx.swapaxes(x, 0, -1)
        sq = x_perm * x_perm
        if self.diagonal:
            denom = self.gamma * sq
        else:
            denom = sq @ self.gamma
        denom = denom + self.beta + mx.array(self.eps, dtype=x.dtype)
        out = x_perm * mx.rsqrt(denom.astype(mx.float32)).astype(x.dtype)
        return mx.swapaxes(out, 0, -1)


def apply_fused_gdn(model: nn.Module) -> nn.Module:
    # No-op converter stub: scans for a 'gdn' / '_gdn' submodule attribute and
    # replaces it with FusedGDN if its shape is inferable. Logs when no GDN
    # submodules are found (the expected case in this repo today).
    from ..fp8_linear import _iter_submodules

    n = 0
    for parent, key, name, module, container_kind in _iter_submodules(model):
        if isinstance(module, FusedGDN):
            continue
        # Only convert modules that explicitly declare a gdn interface.
        if not getattr(module, "_is_gdn", False):
            continue
        channels = getattr(module, "channels", None)
        if channels is None:
            continue
        new = FusedGDN(channels, diagonal=getattr(module, "diagonal", True))
        if container_kind == "list":
            parent[int(key)] = new
        else:
            setattr(parent, key, new)
        n += 1
        logger.info("apply_fused_gdn: %s -> FusedGDN(%d)", name, channels)
    if n == 0:
        logger.debug(
            "apply_fused_gdn: no GDN submodules found in model "
            "(expected — no in-repo model consumes standalone GDN today)"
        )
    return model


__all__ = ["FusedGDN", "apply_fused_gdn"]
