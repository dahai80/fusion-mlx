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
        """Parse GGUF header, return extracted metadata."""
        if not self.path.exists():
            raise FileNotFoundError(f"GGUF file not found: {self.path}")

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

            # Read first tensor dtype to infer quant level
            if n_tensors > 0:
                name = self._read_string(f)
                n_dims = self._read(f, "<I")
                for _ in range(n_dims):
                    self._read(f, "<Q")
                dtype = self._read(f, "<I")
                meta.quant_level = _DTYPE_NAMES.get(dtype, f"dtype_{dtype}")
                logger.info(
                    "GGUF %s: arch=%s ctx=%d quant=%s params=%.1fB",
                    self.path.name,
                    meta.arch,
                    meta.context_length,
                    meta.quant_level,
                    meta.n_params / 1e9,
                )

        return meta


def read_gguf_metadata(path: str | Path) -> GGUFMetadata:
    """Convenience: read GGUF header, return metadata."""
    return GGUFReader(path).read_header()
