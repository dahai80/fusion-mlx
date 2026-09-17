# SPDX-License-Identifier: Apache-2.0
"""Minimal GGUF header reader.

Reads GGUF magic, version, tensor count, KV metadata, and tensor info
without loading weights. Used by the llama.cpp backend bridge to detect
GGUF files and extract model architecture metadata (arch, context length,
quantization level) for routing decisions.

GGUF binary format (v3):
  - magic: uint32 "GGUF"
  - version: uint32 (3)
  - n_tensors: uint64
  - n_kv: uint64
  - n_kv × KV pairs (key_string, value_type, value)
  - n_tensors × tensor_info (name_string, n_dims, dims[], dtype, offset)

Reference: llama.cpp ggml/include/gguf.h + ggml/src/gguf.cpp
"""

from __future__ import annotations

import logging
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

GGUF_MAGIC = 0x46554747  # "GGUF" little-endian
GGUF_VERSION = 3

# gguf_type enum (gguf.h:53-68)
_GGUF_TYPE_NAMES = {
    0: "uint8",
    1: "int8",
    2: "uint16",
    3: "int16",
    4: "uint32",
    5: "int32",
    6: "float32",
    7: "bool",
    8: "string",
    9: "array",
    10: "uint64",
    11: "int64",
    12: "float64",
}

# ggml_dtype enum (subset)
_DTYPE_NAMES = {
    0: "f32",
    1: "f16",
    2: "q4_0",
    3: "q4_1",
    6: "q5_0",
    7: "q5_1",
    8: "q8_0",
    9: "q2_K",
    10: "q3_K",
    11: "q4_K",
    12: "q5_K",
    13: "q6_K",
    14: "iq2_xxs",
    15: "iq2_xs",
    16: "iq3_xxs",
    17: "iq3_s",
    18: "iq2_s",
    19: "iq4_nl",
    20: "iq1_s",
    24: "iq4_xs",
    28: "bf16",
    30: "q8_1",
    31: "mxfp4",
}


@dataclass
class GGUFMetadata:
    arch: str = ""
    context_length: int = 0
    quant_level: str = ""
    n_params: int = 0
    raw_kv: dict[str, Any] = field(default_factory=dict)

    @property
    def is_low_bit(self) -> bool:
        """Q3 and below = ultra-low-bit, candidate for llama.cpp routing."""
        ql = self.quant_level.lower()
        return any(q in ql for q in ("q2", "q3", "iq2", "iq3", "iq1"))


@dataclass
class TensorInfo:
    name: str
    n_dims: int
    dims: list[int]
    dtype_id: int
    dtype_name: str
    data_offset: int
    raw_offset: int


# Quant block sizes (elements per block) + bytes per block. Used by the
# ASFW converter to compute tensor data size and stride. Source: ggml.h.
QUANT_BLOCK: dict[str, tuple[int, int]] = {
    # dtype_name: (block_size_elements, block_bytes)
    "f32": (1, 4),
    "f16": (1, 2),
    "bf16": (1, 2),
    "q4_0": (32, 18),  # 1 f16 scale + 16 uint8 (32×4bit)
    "q4_1": (32, 20),  # 1 f16 scale + 1 f16 min + 16 uint8
    "q5_0": (32, 22),  # 1 f16 scale + 4 uint8 (5th bits) + 16 uint8
    "q5_1": (32, 24),  # 1 f16 scale + 1 f16 min + 4 uint8 + 16 uint8
    "q8_0": (32, 34),  # 1 f16 scale + 32 int8
    "q8_1": (32, 36),  # 1 f16 scale + 1 f16 min + 32 uint8 (packed)
    "q2_K": (256, 84),
    "q3_K": (256, 110),
    "q4_K": (256, 144),
    "q5_K": (256, 176),
    "q6_K": (256, 210),
    "iq2_xxs": (256, 66),
    "iq2_xs": (256, 74),
    "iq3_xxs": (256, 98),
    "iq3_s": (256, 110),
    "iq2_s": (256, 82),
    "iq4_nl": (32, 18),
    "iq1_s": (256, 66),
    "iq4_xs": (256, 136),
    "mxfp4": (32, 18),  # 1 f8 scale + 16 uint8 (32×4bit)
}


def quant_block_bytes(dtype_name: str, n_elements: int) -> int:
    """Compute the byte size of a quantized tensor given its dtype + element count."""
    blk = QUANT_BLOCK.get(dtype_name)
    if blk is None:
        raise ValueError(f"Unsupported GGUF quant dtype: {dtype_name}")
    block_size, block_bytes = blk
    if n_elements % block_size != 0:
        raise ValueError(
            f"Tensor element count {n_elements} not divisible by block size "
            f"{block_size} for dtype {dtype_name}"
        )
    n_blocks = n_elements // block_size
    return n_blocks * block_bytes


class GGUFReader:
    """Stream-parse a GGUF file header (no weight loading)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._offset = 0

    def _read(self, f, fmt: str) -> Any:
        size = struct.calcsize(fmt)
        data = f.read(size)
        if len(data) < size:
            raise EOFError(f"GGUF truncated at offset {self._offset}")
        self._offset += size
        return struct.unpack(fmt, data)[0]

    def _read_string(self, f) -> str:
        n = self._read(f, "<Q")
        raw = f.read(n)
        self._offset += n
        return raw.decode("utf-8", errors="replace")

    def _read_value(self, f, vtype: int) -> Any:
        if vtype == 8:  # string
            return self._read_string(f)
        elif vtype == 9:  # array
            inner_type = self._read(f, "<I")
            n = self._read(f, "<Q")
            return [self._read_value(f, inner_type) for _ in range(n)]
        elif vtype == 0:
            return self._read(f, "<B")
        elif vtype == 1:
            return self._read(f, "<b")
        elif vtype == 2:
            return self._read(f, "<H")
        elif vtype == 3:
            return self._read(f, "<h")
        elif vtype == 4:
            return self._read(f, "<I")
        elif vtype == 5:
            return self._read(f, "<i")
        elif vtype == 6:
            return self._read(f, "<f")
        elif vtype == 7:
            return self._read(f, "<?")
        elif vtype == 10:
            return self._read(f, "<Q")
        elif vtype == 11:
            return self._read(f, "<q")
        elif vtype == 12:
            return self._read(f, "<d")
        else:
            raise ValueError(f"Unknown GGUF value type {vtype}")

    def read_header(self) -> GGUFMetadata:
        """Parse GGUF header, return extracted metadata.

        Reads KV metadata + all tensor descriptors. Stores tensor info in
        self.tensors for later weight loading by read_tensor_data().
        """
        if not self.path.exists():
            raise FileNotFoundError(f"GGUF file not found: {self.path}")

        self.tensors: list[TensorInfo] = []

        with open(self.path, "rb") as f:
            magic = self._read(f, "<I")
            if magic != GGUF_MAGIC:
                raise ValueError(
                    f"Not a GGUF file: magic 0x{magic:08X} != 0x{GGUF_MAGIC:08X}"
                )
            version = self._read(f, "<I")
            if version != GGUF_VERSION:
                logger.warning("GGUF version %d (expected %d)", version, GGUF_VERSION)
            n_tensors = self._read(f, "<Q")
            n_kv = self._read(f, "<Q")

            kv: dict[str, Any] = {}
            for _ in range(n_kv):
                key = self._read_string(f)
                vtype = self._read(f, "<I")
                val = self._read_value(f, vtype)
                kv[key] = val

            # Extract key metadata
            meta = GGUFMetadata(
                arch=str(kv.get("general.architecture", "")),
                context_length=int(
                    kv.get(f"{kv.get('general.architecture', '')}.context_length", 0)
                ),
                n_params=int(kv.get("general.parameter_count", 0)),
                raw_kv=kv,
            )

            # Read ALL tensor descriptors (not just the first).
            header_data_offset = 0
            for i in range(n_tensors):
                name = self._read_string(f)
                n_dims = self._read(f, "<I")
                dims = [self._read(f, "<Q") for _ in range(n_dims)]
                dtype = self._read(f, "<I")
                raw_offset = self._read(f, "<Q")
                dtype_name = _DTYPE_NAMES.get(dtype, f"dtype_{dtype}")
                info = TensorInfo(
                    name=name,
                    n_dims=n_dims,
                    dims=list(dims),
                    dtype_id=dtype,
                    dtype_name=dtype_name,
                    data_offset=0,
                    raw_offset=raw_offset,
                )
                self.tensors.append(info)
                if i == 0:
                    meta.quant_level = dtype_name

            # GGUF tensor data starts after the header, aligned to 32 bytes.
            # The raw_offset in each tensor descriptor is relative to the
            # start of the data section, not the file. We record the data
            # section start (= current file offset, aligned) for read_tensor_data.
            self._data_section_offset = self._offset
            # Align to 32 bytes (GGUF alignment is typically 32).
            alignment = kv.get("general.alignment", 32)
            if alignment > 0:
                pad = (alignment - (self._offset % alignment)) % alignment
                if pad:
                    f.read(pad)
                    self._offset += pad
                self._data_section_offset = self._offset

            # Patch data_offset for each tensor: data_section_start + raw_offset.
            for info in self.tensors:
                info.data_offset = self._data_section_offset + info.raw_offset

            if self.tensors:
                logger.info(
                    "GGUF %s: arch=%s ctx=%d quant=%s params=%.1fB tensors=%d",
                    self.path.name,
                    meta.arch,
                    meta.context_length,
                    meta.quant_level,
                    meta.n_params / 1e9,
                    len(self.tensors),
                )

        return meta

    def read_tensor_data(self, info: TensorInfo) -> bytes:
        """Read raw weight bytes for a single tensor.

        The caller must have called read_header() first to populate
        self.tensors + self._data_section_offset. Returns the raw
        quantized bytes (not dequantized).
        """
        if not hasattr(self, "_data_section_offset"):
            raise RuntimeError("read_header() must be called before read_tensor_data()")
        n_elements = 1
        for d in info.dims:
            n_elements *= d
        size = quant_block_bytes(info.dtype_name, n_elements)
        with open(self.path, "rb") as f:
            f.seek(info.data_offset)
            data = f.read(size)
        if len(data) < size:
            raise EOFError(
                f"GGUF tensor {info.name} truncated: expected {size} bytes, "
                f"got {len(data)}"
            )
        return data


def read_gguf_metadata(path: str | Path) -> GGUFMetadata:
    """Convenience: read GGUF header, return metadata."""
    return GGUFReader(path).read_header()
