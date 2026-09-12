# SPDX-License-Identifier: Apache-2.0
"""Tests for GGUF reader + llama.cpp backend bridge."""

from __future__ import annotations

import struct
from pathlib import Path

import pytest


def _write_minimal_gguf(path: Path, arch: str = "llama", quant_dtype: int = 10) -> None:
    """Write a minimal valid GGUF v3 file for testing."""
    kv_pairs = [
        ("general.architecture", 8, arch),  # type 8 = string
        (f"{arch}.context_length", 5, 4096),  # type 5 = int32
        ("general.parameter_count", 10, 2700000000),  # type 10 = uint64
    ]
    # Tensor: name="token_embd.weight", 2 dims, dtype=q3_K(10)
    tensors = [("token_embd.weight", [4096, 32000], quant_dtype)]

    with open(path, "wb") as f:
        f.write(struct.pack("<I", 0x46554747))  # magic GGUF
        f.write(struct.pack("<I", 3))  # version
        f.write(struct.pack("<Q", len(tensors)))  # n_tensors
        f.write(struct.pack("<Q", len(kv_pairs)))  # n_kv

        for key, vtype, val in kv_pairs:
            key_b = key.encode()
            f.write(struct.pack("<Q", len(key_b)))
            f.write(key_b)
            f.write(struct.pack("<I", vtype))
            if vtype == 8:  # string
                vb = str(val).encode()
                f.write(struct.pack("<Q", len(vb)))
                f.write(vb)
            elif vtype == 5:
                f.write(struct.pack("<i", val))
            elif vtype == 10:
                f.write(struct.pack("<Q", val))

        for name, dims, dtype in tensors:
            nb = name.encode() if isinstance(name, str) else name
            f.write(struct.pack("<Q", len(nb)))
            f.write(nb)
            f.write(struct.pack("<I", len(dims)))
            for d in dims:
                f.write(struct.pack("<Q", d))
            f.write(struct.pack("<I", dtype))
            f.write(struct.pack("<Q", 0))  # offset


def test_gguf_reader_parses_header(tmp_path):
    from fusion_mlx.migrate.gguf_reader import read_gguf_metadata

    p = tmp_path / "test.gguf"
    _write_minimal_gguf(p, arch="llama", quant_dtype=10)  # q3_K
    meta = read_gguf_metadata(p)
    assert meta.arch == "llama"
    assert meta.context_length == 4096
    assert meta.n_params == 2700000000
    assert meta.quant_level == "q3_K"
    assert meta.is_low_bit is True


def test_gguf_reader_detects_high_bit(tmp_path):
    from fusion_mlx.migrate.gguf_reader import read_gguf_metadata

    p = tmp_path / "test.gguf"
    _write_minimal_gguf(p, arch="qwen2", quant_dtype=11)  # q4_K
    meta = read_gguf_metadata(p)
    assert meta.arch == "qwen2"
    assert meta.quant_level == "q4_K"
    assert meta.is_low_bit is False


def test_gguf_reader_rejects_non_gguf(tmp_path):
    from fusion_mlx.migrate.gguf_reader import GGUFReader

    p = tmp_path / "not_gguf.bin"
    p.write_bytes(b"not a gguf file content here")
    with pytest.raises(ValueError, match="Not a GGUF file"):
        GGUFReader(p).read_header()


def test_gguf_reader_missing_file(tmp_path):
    from fusion_mlx.migrate.gguf_reader import GGUFReader

    with pytest.raises(FileNotFoundError):
        GGUFReader(tmp_path / "nonexistent.gguf").read_header()


def test_gguf_reader_detects_iq2_quant(tmp_path):
    from fusion_mlx.migrate.gguf_reader import read_gguf_metadata

    p = tmp_path / "iq2.gguf"
    _write_minimal_gguf(p, arch="llama", quant_dtype=14)  # iq2_xxs
    meta = read_gguf_metadata(p)
    assert "iq2" in meta.quant_level
    assert meta.is_low_bit is True


def test_llama_cpp_config_to_args():
    from fusion_mlx.engines.llama_cpp_backend import LlamaCppConfig

    config = LlamaCppConfig(
        model_path="/models/test.gguf",
        n_gpu_layers=-1,
        ctx_size=8192,
        port=12345,
    )
    args = config.to_args()
    assert "--model" in args
    assert "/models/test.gguf" in args
    assert "--port" in args
    assert "12345" in args
    assert "--n-gpu-layers" in args
    assert "all" in args
    assert "--ctx-size" in args
    assert "8192" in args
    assert "--no-ui" in args


def test_llama_cpp_config_cpu_only():
    from fusion_mlx.engines.llama_cpp_backend import LlamaCppConfig

    config = LlamaCppConfig(model_path="/m.gguf", n_gpu_layers=0, port=8080)
    args = config.to_args()
    assert "--n-gpu-layers" not in args


def test_llama_cpp_backend_allocates_port():
    from fusion_mlx.engines.llama_cpp_backend import LlamaCppConfig, _allocate_port

    port = _allocate_port()
    assert 1024 < port < 65536
    config = LlamaCppConfig(model_path="/m.gguf")
    assert config.port == 0  # before allocation
