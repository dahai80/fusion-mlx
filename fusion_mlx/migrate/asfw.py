# SPDX-License-Identifier: Apache-2.0
"""ASFW (Apple Silicon Friendly Weight) layout converter.

Transforms GGUF native interleaved quantized weight blocks into a
SIMD32-aligned layout suitable for Metal simdgroup_matrix operations.

Problem (v2 doc §2.1, §5.7): GGUF stores quantized blocks contiguously
[block_0(scale, packed), block_1(scale, packed), ...]. On Apple Silicon,
a Metal simdgroup (32 threads) issuing simdgroup_load on this layout
hits bank conflicts — the scale and packed data are interleaved at an
18-byte stride (Q4_0), which does not align to the 4-byte SIMD lane
boundaries.

ASFW transform: split the per-block metadata (scales) from the packed
data, and group blocks in chunks of SIMD32_WIDTH (32) so a simdgroup
can coalesce one contiguous scale read + one contiguous packed read.

Block layouts supported (v2 doc §3.3 — production-first):
  Q4_0, Q8_0 — block_size=32, 1 f16 scale + packed data
  Q4_K       — super_block=256, 2 f16 (d,min) + 12 f8 scales + 128 packed

Degradation: when ASFW is not needed (stock MLX path), the converter
also provides dequantize_to_f16() which expands quantized blocks to
FP16 for standard mlx.core matmul. This is the Tier-2 fallback — the
C++ Metal dequant-GEMV kernel (PR-K) will consume the ASFW layout
directly without dequantizing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from .gguf_reader import TensorInfo

logger = logging.getLogger(__name__)

# Apple Silicon simdgroup width — 32 threads per group.
SIMD32_WIDTH = 32


@dataclass
class ASFWLayout:
    """Result of an ASFW layout conversion for one tensor."""

    tensor_name: str
    dtype_name: str
    shape: tuple[int, ...]
    # Raw bytes in ASFW layout (scales split from packed, SIMD32 grouped).
    asfw_bytes: bytes
    # Dequantized FP16 array (degrade path for stock MLX matmul).
    # None when dequantization was not requested or unsupported.
    f16_array: np.ndarray | None
    # Number of quant blocks.
    n_blocks: int
    # Block size in elements.
    block_size: int
    # Layout description for the Metal kernel (PR-K).
    layout_desc: str


class ASFWConverter:
    """Convert GGUF quantized tensor data to ASFW layout.

    Each dtype has a dedicated _convert_<dtype> method. Unsupported
    dtypes raise UnsupportedQuantError so the loader can fall back.
    """

    def __init__(self, simd_width: int = SIMD32_WIDTH):
        self.simd_width = simd_width

    def convert(
        self,
        info: TensorInfo,
        raw_bytes: bytes,
        dequantize: bool = True,
    ) -> ASFWLayout:
        """Convert raw GGUF quantized bytes to ASFW layout.

        Args:
            info: Tensor descriptor from GGUFReader.
            raw_bytes: Raw quantized weight bytes.
            dequantize: If True, also produce an FP16 dequantized array
                for the degrade path (stock MLX matmul without a custom
                Metal kernel).

        Raises:
            UnsupportedQuantError: dtype has no ASFW converter.
        """
        dtype = info.dtype_name
        shape = tuple(info.dims)
        n_elements = 1
        for d in shape:
            n_elements *= d

        converter = self._dispatchers.get(dtype)
        if converter is None:
            raise UnsupportedQuantError(
                f"ASFW converter not implemented for dtype {dtype} "
                f"(tensor {info.name}). Falling back to stock MLX."
            )

        asfw_bytes, f16_array, block_size, n_blocks, layout_desc = converter(
            self, raw_bytes, n_elements, dequantize
        )

        logger.debug(
            "ASFW convert %s: dtype=%s shape=%s blocks=%d asfw_bytes=%d f16=%s",
            info.name,
            dtype,
            shape,
            n_blocks,
            len(asfw_bytes),
            "yes" if f16_array is not None else "no",
        )

        return ASFWLayout(
            tensor_name=info.name,
            dtype_name=dtype,
            shape=shape,
            asfw_bytes=asfw_bytes,
            f16_array=f16_array,
            n_blocks=n_blocks,
            block_size=block_size,
            layout_desc=layout_desc,
        )

    # -- Q4_0: block_size=32, 1 f16 scale + 16 uint8 (32×4bit) = 18 bytes --

    def _convert_q4_0(
        self, raw: bytes, n_elements: int, dequantize: bool
    ) -> tuple[bytes, np.ndarray | None, int, int, str]:
        block_size = 32
        block_bytes = 18
        n_blocks = n_elements // block_size

        # Parse: [f16 scale, 16×uint8 packed] per block.
        dtype = np.dtype(
            [("scale", np.float16), ("packed", np.uint8, 16)],
            align=False,
        )
        blocks = np.frombuffer(raw, dtype=dtype, count=n_blocks)

        # ASFW: split scales from packed, group by SIMD32_WIDTH.
        scales = blocks["scale"]  # shape (n_blocks,)
        packed = blocks["packed"]  # shape (n_blocks, 16)

        # Reshape to (n_groups, simd_width, ...) for coalesced simdgroup load.
        # Pad last group if not divisible.
        n_groups = (n_blocks + self.simd_width - 1) // self.simd_width
        padded = n_groups * self.simd_width
        if padded != n_blocks:
            scales = np.pad(scales, (0, padded - n_blocks), constant_values=0)
            packed = np.pad(packed, ((0, padded - n_blocks), (0, 0)), constant_values=0)

        scales_grouped = scales.reshape(n_groups, self.simd_width)
        packed_grouped = packed.reshape(n_groups, self.simd_width, 16)

        # ASFW layout: [scales_grouped (contiguous)] [packed_grouped (contiguous)]
        asfw = scales_grouped.tobytes() + packed_grouped.tobytes()

        f16 = None
        if dequantize:
            f16 = self._dequant_q4_0(scales[:n_blocks], packed[:n_blocks])

        return asfw, f16, block_size, n_blocks, "q4_0:split_scales_packed_simd32"

    @staticmethod
    def _dequant_q4_0(scales: np.ndarray, packed: np.ndarray) -> np.ndarray:
        """Dequantize Q4_0 blocks to FP16. Degrade path for stock MLX."""
        n_blocks = scales.shape[0]
        # Each block: 32 values. Low 4 bits from packed[16] (2 per byte),
        # high bit = sign (Q4_0 uses symmetric: value = (val - 8) * scale).
        out = np.zeros((n_blocks, 32), dtype=np.float16)
        for i in range(16):
            lo = (packed[:, i] & 0x0F).astype(np.float16) - 8.0
            hi = (packed[:, i] >> 4).astype(np.float16) - 8.0
            out[:, 2 * i] = lo * scales
            out[:, 2 * i + 1] = hi * scales
        return out.reshape(-1)

    # -- Q8_0: block_size=32, 1 f16 scale + 32 int8 = 34 bytes --

    def _convert_q8_0(
        self, raw: bytes, n_elements: int, dequantize: bool
    ) -> tuple[bytes, np.ndarray | None, int, int, str]:
        block_size = 32
        n_blocks = n_elements // block_size

        dtype = np.dtype(
            [("scale", np.float16), ("qs", np.int8, 32)],
            align=False,
        )
        blocks = np.frombuffer(raw, dtype=dtype, count=n_blocks)

        scales = blocks["scale"]
        qs = blocks["qs"]

        n_groups = (n_blocks + self.simd_width - 1) // self.simd_width
        padded = n_groups * self.simd_width
        if padded != n_blocks:
            scales = np.pad(scales, (0, padded - n_blocks), constant_values=0)
            qs = np.pad(qs, ((0, padded - n_blocks), (0, 0)), constant_values=0)

        scales_grouped = scales.reshape(n_groups, self.simd_width)
        qs_grouped = qs.reshape(n_groups, self.simd_width, 32)

        asfw = scales_grouped.tobytes() + qs_grouped.tobytes()

        f16 = None
        if dequantize:
            f16 = (
                (qs[:n_blocks].astype(np.float32) * scales[:n_blocks, None])
                .astype(np.float16)
                .reshape(-1)
            )

        return asfw, f16, block_size, n_blocks, "q8_0:split_scales_qs_simd32"

    # -- Q4_K: super_block=256, complex layout --
    # Production-first per v2 doc. ASFW groups super-blocks for simdgroup.
    # Dequantization is deferred to the C++ Metal kernel (PR-K) — the
    # degrade path loads via ggml reference (not reimplemented here).

    def _convert_q4_k(
        self, raw: bytes, n_elements: int, dequantize: bool
    ) -> tuple[bytes, np.ndarray | None, int, int, str]:
        block_size = 256
        block_bytes = 144
        n_blocks = n_elements // block_size

        # Q4_K layout per super-block (144 bytes):
        #   2× f16 (d, dmin) + 12× uint8 (scales) + 128× uint8 (packed)
        # ASFW: split the 2 f16 scales + 12 scale bytes from the 128 packed,
        # group by simd_width.
        header_dtype = np.dtype(
            [("d", np.float16), ("dmin", np.float16), ("scales", np.uint8, 12)],
            align=False,
        )
        headers = np.frombuffer(raw, dtype=header_dtype, count=n_blocks)
        packed = np.frombuffer(raw, dtype=np.uint8, count=n_blocks * 128).reshape(
            n_blocks, 128
        )

        n_groups = (n_blocks + self.simd_width - 1) // self.simd_width
        padded = n_groups * self.simd_width
        if padded != n_blocks:
            headers = np.pad(
                headers,
                (0, padded - n_blocks),
                constant_values=0,
            )
            packed = np.pad(packed, ((0, padded - n_blocks), (0, 0)), constant_values=0)

        headers_grouped = headers.reshape(n_groups, self.simd_width)
        packed_grouped = packed.reshape(n_groups, self.simd_width, 128)

        asfw = headers_grouped.tobytes() + packed_grouped.tobytes()

        # Q4_K dequant requires 6-bit scale interpolation — deferred to PR-K
        # Metal kernel. Degrade path: caller falls back to mlx_lm native.
        f16 = None
        if dequantize:
            logger.debug(
                "Q4_K dequant deferred to PR-K Metal kernel; degrade path "
                "uses native mlx_lm loading"
            )

        return asfw, f16, block_size, n_blocks, "q4_k:split_headers_packed_simd32"

    # Dispatch table — populated after method definitions.
    _dispatchers: dict[str, Any] = {}


class UnsupportedQuantError(Exception):
    """Raised when ASFW has no converter for a quant dtype. Triggers fallback."""


# Populate dispatch table.
ASFWConverter._dispatchers = {
    "q4_0": ASFWConverter._convert_q4_0,
    "q8_0": ASFWConverter._convert_q8_0,
    "q4_k": ASFWConverter._convert_q4_k,
}
