# SPDX-License-Identifier: Apache-2.0
"""Force dense-rotation TurboQuant codecs for power-of-two head dims.

#1230: the vendored mlx_vlm 0.5.0 RHT (Hadamard) codec is broken end to end —
multiple fused Metal kernels (fused KV quantize, decode attention) rotate or
dequantize with the DENSE matrix regardless of codec.use_rht, so any
power-of-two head_dim (32/64/128/256) gets 134-143% roundtrip error and
garbled generation (Qwen2.5-32B: rel_err 4.5 at hd=128, bits=4).

Until the pinned mlx_vlm ships the upstream fix, force use_rht=False on
_TurboQuantMSECodec: power-of-two dims then use the dense rotation path that
non-power-of-two dims (hd=80/96) already exercise in production — measured
roundtrip error drops from 134% to the normal ~9% 4-bit quantization noise.
Dense rotation costs one extra D×D matmul per quantize/dequantize; the fused
single-token kernel (dense rotation inside) still applies.
"""

import logging

logger = logging.getLogger(__name__)

_PATCHED = False


def apply_turboquant_dense_codec_guard() -> bool:
    global _PATCHED
    if _PATCHED:
        return False
    from mlx_vlm.turboquant import _TurboQuantMSECodec

    original_init = _TurboQuantMSECodec.__init__
    if getattr(original_init, "_fusion_mlx_dense_guard", False):
        return False

    def _dense_guard_init(self, dim, bits, seed):
        original_init(self, dim, bits, seed)
        if getattr(self, "use_rht", False):
            self.use_rht = False

    _dense_guard_init._fusion_mlx_dense_guard = True
    _TurboQuantMSECodec.__init__ = _dense_guard_init
    _PATCHED = True
    logger.info(
        "TurboQuant dense-codec guard installed: RHT (Hadamard) rotation "
        "disabled for power-of-two head dims — vendored fused kernels "
        "dequantize with the dense matrix (broken roundtrip, #1230)"
    )
    return True
