# SPDX-License-Identifier: Apache-2.0
# Fused Conv+GroupNorm+SiLU pattern for graph_opt (#911).
#
# Replaces the 3-kernel sequence Conv2D -> GroupNorm -> SiLU with a single
# module whose __call__ chains all three, so mx.compile can fuse the elementwise
# parts (SiLU, affine) into the conv/norm boundary. GroupNorm uses SafeGroupNorm
# (FP32 stats) to avoid FP16 variance drift — PRD V2 cosine >= 0.98.

from __future__ import annotations

import os

import mlx.nn as nn

from ..nn_ext.safe_group_norm import SafeGroupNorm


def _fused_conv_gn_silu_enabled() -> bool:
    # Env-gated (default OFF): FUSION_FUSED_CONV_GN_SILU=1 opts into the MSL
    # fused kernel (#924). Off -> the 3-op chain (mx.compile elementwise only).
    return os.environ.get("FUSION_FUSED_CONV_GN_SILU", "0") == "1"


class ConvGroupNormSiLU(nn.Module):
    """Fused Conv2D + GroupNorm + SiLU (#911).

    Wraps an existing conv and groupnorm (weights reused, not copied) so the
    sequence executes as one __call__ — mx.compile then fuses the SiLU + affine
    into the conv boundary. If the passed GroupNorm is a stock nn.GroupNorm it is
    swapped for a SafeGroupNorm (same weight/bias, FP32 stats) on construction.
    """

    def __init__(self, conv: nn.Module, groupnorm: nn.Module):
        super().__init__()
        self.conv = conv
        # Upgrade stock GroupNorm to SafeGroupNorm (FP32-protected) if needed.
        if isinstance(groupnorm, nn.GroupNorm) and not isinstance(
            groupnorm, SafeGroupNorm
        ):
            groups = getattr(groupnorm, "num_groups", None) or getattr(
                groupnorm, "groups", 4
            )
            dims = (
                groupnorm.dims
                if hasattr(groupnorm, "dims")
                else groupnorm.weight.shape[0]
            )
            eps = getattr(groupnorm, "eps", 1e-6)
            # Preserve the source GroupNorm's grouping semantics — MLX
            # nn.GroupNorm(pytorch_compatible=False) groups differently from
            # the PyTorch layout; mixing them silently breaks parity.
            compatible = bool(getattr(groupnorm, "pytorch_compatible", False))
            safe = SafeGroupNorm(groups, dims, eps=eps, pytorch_compatible=compatible)
            safe.weight = groupnorm.weight
            safe.bias = groupnorm.bias
            self.groupnorm = safe
        else:
            self.groupnorm = groupnorm

    def __call__(self, x):
        if _fused_conv_gn_silu_enabled():
            try:
                from ..custom_kernels.fused_conv_gn_silu import (
                    fused_conv_gn_silu as _fused,
                )

                w = self.conv.weight
                b = getattr(self.conv, "bias", None)
                gn = self.groupnorm
                out = _fused(x, w, b, gn.weight, gn.bias, gn.num_groups, eps=gn.eps)
                if out is not None:
                    return out
            except Exception as e:
                import logging

                logging.getLogger(__name__).debug(
                    "[graph_opt] fused_conv_gn_silu unavailable (%s); 3-op chain", e
                )
        x = self.conv(x)
        x = self.groupnorm(x)
        return nn.silu(x)
