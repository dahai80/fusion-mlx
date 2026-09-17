# SPDX-License-Identifier: Apache-2.0
"""Tests for GGUF loader + ASFW layout converter (PR-C)."""

from __future__ import annotations

import struct
from pathlib import Path

import pytest


def _write_gguf_with_weights(
    path: Path,
    arch: str = "llama",
    tensors: list[tuple[str, list[int], int, bytes]] | None = None,
) -> None:
    """Write a GGUF v3 file with real tensor weight data.

    Args:
        tensors: list of (name, dims, dtype_id, raw_bytes). The raw_bytes
            must match the dtype's block size × element count.
    """
    if tensors is None:
        # Default: a small Q4_0 tensor, 32 elements = 1 block = 18 bytes.
        raw = struct.pack("<e", 1.5) + bytes([0x12, 0x34] * 8)
        tensors = [("token_embd.weight", [4, 32], 2, raw)]

    kv_pairs = [
        ("general.architecture", 8, arch),
        (f"{arch}.context_length", 5, 4096),
        ("general.parameter_count", 10, 1000),
        ("general.alignment", 5, 32),
    ]

    with open(path, "wb") as f:
        f.write(struct.pack("<I", 0x46554747))  # magic
        f.write(struct.pack("<I", 3))  # version
        f.write(struct.pack("<Q", len(tensors)))
        f.write(struct.pack("<Q", len(kv_pairs)))

        for key, vtype, val in kv_pairs:
            key_b = key.encode()
            f.write(struct.pack("<Q", len(key_b)))
            f.write(key_b)
            f.write(struct.pack("<I", vtype))
            if vtype == 8:
                vb = str(val).encode()
                f.write(struct.pack("<Q", len(vb)))
                f.write(vb)
            elif vtype == 5:
                f.write(struct.pack("<i", val))
            elif vtype == 10:
                f.write(struct.pack("<Q", val))

        for name, dims, dtype, _raw in tensors:
            nb = name.encode()
            f.write(struct.pack("<Q", len(nb)))
            f.write(nb)
            f.write(struct.pack("<I", len(dims)))
            for d in dims:
                f.write(struct.pack("<Q", d))
            f.write(struct.pack("<I", dtype))
            f.write(struct.pack("<Q", 0))  # offset (patched below)

        # Align data section to 32 bytes.
        pos = f.tell()
        pad = (32 - (pos % 32)) % 32
        f.write(b"\x00" * pad)

        # Write tensor data, recording offsets.
        base = f.tell()
        for i, (_name, _dims, _dtype, raw) in enumerate(tensors):
            offset = f.tell() - base
            # Patch the offset in the tensor descriptor.
            old_pos = f.tell()
            # Offset was written as Q at: descriptor_start + name_len + dims + dtype
            # Easier: rewrite the whole file is overkill; the reader uses
            # raw_offset relative to data section, and we wrote 0. So we
            # need to patch. Seek back to each descriptor's offset field.
            f.seek(0)
            # Re-read to find descriptor positions is complex; instead
            # restructure: write offsets as we go is not possible in one pass.
            # Simplification: for the test, all tensors start at offset 0
            # sequentially — we'll write them contiguously and the reader
            # uses raw_offset=0 for all (only works for 1 tensor).
            f.seek(old_pos)
            f.write(raw)
            break  # Only support single-tensor test files for offset simplicity


def _write_single_tensor_gguf(
    path: Path,
    name: str,
    dims: list[int],
    dtype_id: int,
    raw: bytes,
    arch: str = "llama",
) -> None:
    """Write a GGUF v3 with exactly one tensor at offset 0."""
    kv_pairs = [
        ("general.architecture", 8, arch),
        (f"{arch}.context_length", 5, 4096),
        ("general.parameter_count", 10, 1000),
        ("general.alignment", 5, 32),
    ]
    with open(path, "wb") as f:
        f.write(struct.pack("<I", 0x46554747))
        f.write(struct.pack("<I", 3))
        f.write(struct.pack("<Q", 1))  # 1 tensor
        f.write(struct.pack("<Q", len(kv_pairs)))
        for key, vtype, val in kv_pairs:
            kb = key.encode()
            f.write(struct.pack("<Q", len(kb)))
            f.write(kb)
            f.write(struct.pack("<I", vtype))
            if vtype == 8:
                vb = str(val).encode()
                f.write(struct.pack("<Q", len(vb)))
                f.write(vb)
            elif vtype == 5:
                f.write(struct.pack("<i", val))
            elif vtype == 10:
                f.write(struct.pack("<Q", val))
        nb = name.encode()
        f.write(struct.pack("<Q", len(nb)))
        f.write(nb)
        f.write(struct.pack("<I", len(dims)))
        for d in dims:
            f.write(struct.pack("<Q", d))
        f.write(struct.pack("<I", dtype_id))
        f.write(struct.pack("<Q", 0))  # raw offset = 0
        # Align to 32.
        pos = f.tell()
        pad = (32 - (pos % 32)) % 32
        f.write(b"\x00" * pad)
        f.write(raw)


class TestGGUFReaderTensors:
    def test_read_all_tensor_descriptors(self, tmp_path):
        from fusion_mlx.migrate.gguf_reader import GGUFReader

        p = tmp_path / "test.gguf"
        raw = struct.pack("<e", 1.5) + bytes([0x12] * 16)
        _write_single_tensor_gguf(p, "token_embd.weight", [1, 32], 2, raw)
        reader = GGUFReader(p)
        meta = reader.read_header()
        assert len(reader.tensors) == 1
        info = reader.tensors[0]
        assert info.name == "token_embd.weight"
        assert info.dims == [1, 32]
        assert info.dtype_name == "q4_0"

    def test_read_tensor_data_q4_0(self, tmp_path):
        from fusion_mlx.migrate.gguf_reader import GGUFReader

        p = tmp_path / "test.gguf"
        scale = 1.5
        packed = bytes([0x12] * 16)
        raw = struct.pack("<e", scale) + packed
        _write_single_tensor_gguf(p, "token_embd.weight", [1, 32], 2, raw)
        reader = GGUFReader(p)
        reader.read_header()
        data = reader.read_tensor_data(reader.tensors[0])
        assert len(data) == 18
        assert struct.unpack("<e", data[:2])[0] == scale

    def test_quant_block_bytes_q4_0(self):
        from fusion_mlx.migrate.gguf_reader import quant_block_bytes

        assert quant_block_bytes("q4_0", 32) == 18
        assert quant_block_bytes("q4_0", 64) == 36
        assert quant_block_bytes("q8_0", 32) == 34

    def test_quant_block_bytes_unsupported(self):
        from fusion_mlx.migrate.gguf_reader import quant_block_bytes

        with pytest.raises(ValueError, match="Unsupported"):
            quant_block_bytes("unknown_dtype", 32)

    def test_quant_block_bytes_not_divisible(self):
        from fusion_mlx.migrate.gguf_reader import quant_block_bytes

        with pytest.raises(ValueError, match="not divisible"):
            quant_block_bytes("q4_0", 33)


class TestASFWConverter:
    def _make_q4_0_raw(self, n_blocks: int, scale: float = 2.0) -> bytes:
        """Build n_blocks of Q4_0 data with a known scale."""
        raw = b""
        for _ in range(n_blocks):
            raw += struct.pack("<e", scale) + bytes([0x18] * 16)
        return raw

    def test_convert_q4_0_produces_asfw_bytes(self, tmp_path):
        from fusion_mlx.migrate.asfw import ASFWConverter
        from fusion_mlx.migrate.gguf_reader import TensorInfo

        n_blocks = 32
        raw = self._make_q4_0_raw(n_blocks)
        info = TensorInfo(
            name="test.weight",
            n_dims=2,
            dims=[n_blocks, 32],
            dtype_id=2,
            dtype_name="q4_0",
            data_offset=0,
            raw_offset=0,
        )
        conv = ASFWConverter()
        layout = conv.convert(info, raw, dequantize=True)
        assert layout.dtype_name == "q4_0"
        assert layout.n_blocks == n_blocks
        assert layout.block_size == 32
        assert len(layout.asfw_bytes) > 0
        # ASFW splits scales from packed: scales (32 f16 = 64 bytes) + packed
        scales_bytes = n_blocks * 2  # 32 f16 scales
        packed_bytes = n_blocks * 16
        assert len(layout.asfw_bytes) == scales_bytes + packed_bytes

    def test_convert_q4_0_dequant_correct(self, tmp_path):
        from fusion_mlx.migrate.asfw import ASFWConverter
        from fusion_mlx.migrate.gguf_reader import TensorInfo

        n_blocks = 4
        scale = 3.0
        # packed = 0x18 → low nibble=8, high nibble=1
        # val = (nibble - 8) * scale → low: 0*3=0, high: -7*3=-21
        raw = b""
        for _ in range(n_blocks):
            raw += struct.pack("<e", scale) + bytes([0x18] * 16)
        info = TensorInfo(
            name="test.weight",
            n_dims=2,
            dims=[n_blocks, 32],
            dtype_id=2,
            dtype_name="q4_0",
            data_offset=0,
            raw_offset=0,
        )
        conv = ASFWConverter()
        layout = conv.convert(info, raw, dequantize=True)
        assert layout.f16_array is not None
        arr = layout.f16_array
        assert arr.shape == (n_blocks * 32,)
        # First element: low nibble of byte 0 = 8, (8-8)*3 = 0
        assert arr[0] == 0.0
        # Second element: high nibble of byte 0 = 1, (1-8)*3 = -21
        assert arr[1] == -21.0

    def test_convert_q8_0_dequant_correct(self):
        from fusion_mlx.migrate.asfw import ASFWConverter
        from fusion_mlx.migrate.gguf_reader import TensorInfo

        n_blocks = 4
        scale = 2.0
        raw = b""
        for _ in range(n_blocks):
            raw += struct.pack("<e", scale) + bytes([10] * 32)
        info = TensorInfo(
            name="test.q8",
            n_dims=2,
            dims=[n_blocks, 32],
            dtype_id=8,
            dtype_name="q8_0",
            data_offset=0,
            raw_offset=0,
        )
        conv = ASFWConverter()
        layout = conv.convert(info, raw, dequantize=True)
        assert layout.f16_array is not None
        arr = layout.f16_array
        assert arr.shape == (n_blocks * 32,)
        # Each value: int8(10) * scale(2.0) = 20.0
        assert arr[0] == 20.0

    def test_convert_unsupported_dtype_raises(self):
        from fusion_mlx.migrate.asfw import ASFWConverter, UnsupportedQuantError
        from fusion_mlx.migrate.gguf_reader import TensorInfo

        info = TensorInfo(
            name="test.iq2",
            n_dims=2,
            dims=[1, 256],
            dtype_id=14,
            dtype_name="iq2_xxs",
            data_offset=0,
            raw_offset=0,
        )
        conv = ASFWConverter()
        with pytest.raises(UnsupportedQuantError, match="iq2_xxs"):
            conv.convert(info, b"\x00" * 66, dequantize=False)

    def test_asfw_groups_by_simd32(self):
        from fusion_mlx.migrate.asfw import ASFWConverter
        from fusion_mlx.migrate.gguf_reader import TensorInfo

        # 40 blocks → 2 groups (32 + 8 padded to 32)
        n_blocks = 40
        raw = self._make_q4_0_raw(n_blocks)
        info = TensorInfo(
            name="test.pad",
            n_dims=2,
            dims=[n_blocks, 32],
            dtype_id=2,
            dtype_name="q4_0",
            data_offset=0,
            raw_offset=0,
        )
        conv = ASFWConverter()
        layout = conv.convert(info, raw, dequantize=False)
        # Padded to 64 blocks → scales = 64*2, packed = 64*16
        expected = 64 * 2 + 64 * 16
        assert len(layout.asfw_bytes) == expected


class TestGGUFLoader:
    def test_load_header_only_when_asfw_off(self, tmp_path):
        from fusion_mlx.migrate.gguf_loader import load_gguf

        p = tmp_path / "test.gguf"
        raw = struct.pack("<e", 1.5) + bytes([0x12] * 16)
        _write_single_tensor_gguf(p, "token_embd.weight", [1, 32], 2, raw)
        result = load_gguf(p, convert_asfw=False)
        assert result.metadata.arch == "llama"
        assert len(result.tensors) == 1
        assert len(result.asfw_layouts) == 0

    def test_load_with_asfw_conversion(self, tmp_path, monkeypatch):
        from fusion_mlx.migrate.gguf_loader import load_gguf

        p = tmp_path / "test.gguf"
        raw = struct.pack("<e", 2.0) + bytes([0x18] * 16)
        _write_single_tensor_gguf(p, "token_embd.weight", [1, 32], 2, raw)
        result = load_gguf(p, convert_asfw=True, dequantize=True)
        assert len(result.asfw_layouts) == 1
        assert "token_embd.weight" in result.asfw_layouts
        assert result.all_converted
        assert result.f16_arrays.get("token_embd.weight") is not None

    def test_load_unsupported_quant_falls_back(self, tmp_path):
        from fusion_mlx.migrate.gguf_loader import load_gguf

        p = tmp_path / "test.gguf"
        # iq2_xxs (dtype_id=14), 256 elements = 1 block = 66 bytes
        raw = b"\x00" * 66
        _write_single_tensor_gguf(p, "test.weight", [1, 256], 14, raw)
        result = load_gguf(p, convert_asfw=True, dequantize=True)
        assert len(result.fallback_tensors) == 1
        assert "test.weight" in result.fallback_tensors
        assert "unsupported_quant" in result.fallback_tensors["test.weight"]
        assert not result.all_converted

    def test_validate_layer_formats(self):
        from fusion_mlx.migrate.gguf_loader import validate_layer_formats
        from fusion_mlx.migrate.gguf_reader import TensorInfo

        tensors = [
            TensorInfo("a", 2, [4, 32], 2, "q4_0", 0, 0),
            TensorInfo("b", 2, [4, 32], 8, "q8_0", 0, 0),
            TensorInfo("c", 2, [1, 256], 14, "iq2_xxs", 0, 0),
        ]
        result = validate_layer_formats(tensors)
        assert "a" in result["supported"]
        assert "b" in result["supported"]
        assert "c" in result["unsupported"]

    def test_is_asfw_enabled_default_off(self, monkeypatch):
        from fusion_mlx.migrate.gguf_loader import is_asfw_enabled

        monkeypatch.delenv("FUSION_SHIM_ASFW", raising=False)
        assert is_asfw_enabled() is False

    def test_is_asfw_enabled_on(self, monkeypatch):
        from fusion_mlx.migrate.gguf_loader import is_asfw_enabled

        monkeypatch.setenv("FUSION_SHIM_ASFW", "1")
        assert is_asfw_enabled() is True
