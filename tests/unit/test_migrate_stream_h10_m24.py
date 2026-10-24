# SPDX-License-Identifier: Apache-2.0
# #1354 audit H10/M24: streaming weight conversion (no fp32-for-all),
# writable-fd durability, and no forced trust_remote_code in validation.

import fcntl
import json
import logging
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import pytest

from fusion_mlx.cli_convert import _convert_pytorch_to_safetensors
from fusion_mlx.migrate import converter as converter_mod
from fusion_mlx.migrate.architectures import KNOWN_TEMPLATES
from fusion_mlx.migrate.converter import _SourceTensor, convert_model
from fusion_mlx.migrate.validator import validate_model
from fusion_mlx.migrate.weight_mapper import build_weight_map

# The converter loads source weights via safetensors' "pt" framework and
# converts through torch.Tensor (tensor_to_mx), so every test in this file
# requires a real torch install. CI shards run without torch — skip the
# whole module there instead of failing (same treatment as the M26
# pytorch-fallback test below).
pytest.importorskip("torch")

_LLAMA = KNOWN_TEMPLATES["llama"]


def _config(num_layers: int = 1) -> dict:
    return {
        "num_hidden_layers": num_layers,
        "hidden_size": 64,
        "num_attention_heads": 2,
        "intermediate_size": 128,
        "vocab_size": 100,
    }


def _load_outputs(output_dir: str) -> dict[str, mx.array]:
    weights: dict[str, mx.array] = {}
    for path in sorted(Path(output_dir).glob("model*.safetensors")):
        weights.update(mx.load(str(path)))
    return weights


# ---------------------------------------------------------------------------
# H10: dtype preservation + bounded shards, real safetensors round-trip
# ---------------------------------------------------------------------------


def test_dtype_preserved_and_output_sharded(tmp_path):
    hf_dir = tmp_path / "hf"
    hf_dir.mkdir()
    mx.save_safetensors(
        str(hf_dir / "a.safetensors"),
        {
            "model.embed_tokens.weight": mx.zeros((8, 8), dtype=mx.bfloat16),
            "model.layers.0.self_attn.q_proj.weight": mx.zeros(
                (128, 128), dtype=mx.float16
            ),
        },
    )
    mx.save_safetensors(
        str(hf_dir / "b.safetensors"),
        {
            "model.norm.weight": mx.zeros((8,), dtype=mx.float32),
            "model.layers.0.mlp.gate_proj.weight": mx.zeros(
                (128, 128), dtype=mx.float16
            ),
        },
    )

    out_dir = str(tmp_path / "out")
    max_shard_bytes = 40000
    progress: list[tuple[float, str]] = []

    result = convert_model(
        str(hf_dir),
        out_dir,
        _config(1),
        _LLAMA,
        progress_cb=lambda frac, msg: progress.append((frac, msg)),
        max_shard_bytes=max_shard_bytes,
    )

    assert result.error is None, result.error
    assert result.num_weights == 4

    shards = sorted(Path(out_dir).glob("model-*-of-*.safetensors"))
    assert len(shards) == 2, [p.name for p in shards]
    for shard in shards:
        assert shard.stat().st_size <= max_shard_bytes + 128 * 128 * 2

    index_path = Path(out_dir) / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    shard_names = {p.name for p in shards}
    assert set(index["weight_map"].values()) == shard_names
    assert "layers.0.self_attn.q_proj.weight" in index["weight_map"]

    weights = _load_outputs(out_dir)
    # Source dtypes survive: bf16 stays bf16, fp16 stays fp16, fp32 stays fp32.
    assert weights["embed_tokens.weight"].dtype == mx.bfloat16
    assert weights["layers.0.self_attn.q_proj.weight"].dtype == mx.float16
    assert weights["layers.0.mlp.gate_proj.weight"].dtype == mx.float16
    assert weights["norm.weight"].dtype == mx.float32
    # No fp32 blow-up of the half-precision tensors.
    assert weights["layers.0.self_attn.q_proj.weight"].nbytes == 128 * 128 * 2

    assert progress[-1][0] == 1.0


def test_operator_requested_dtype_casts_every_weight(tmp_path):
    hf_dir = tmp_path / "hf"
    hf_dir.mkdir()
    mx.save_safetensors(
        str(hf_dir / "a.safetensors"),
        {
            "model.embed_tokens.weight": mx.zeros((8, 8), dtype=mx.bfloat16),
            "model.norm.weight": mx.zeros((8,), dtype=mx.float16),
        },
    )

    out_dir = str(tmp_path / "out")
    result = convert_model(str(hf_dir), out_dir, _config(1), _LLAMA, dtype="fp32")

    assert result.error is None, result.error
    weights = _load_outputs(out_dir)
    assert weights["embed_tokens.weight"].dtype == mx.float32
    assert weights["norm.weight"].dtype == mx.float32


# ---------------------------------------------------------------------------
# H10: conversion is streamed — read/write interleave and the pending batch
# never grows to the whole model
# ---------------------------------------------------------------------------


def test_reads_and_writes_interleave_with_bounded_batch(tmp_path, monkeypatch):
    config = _config(2)
    names = list(build_weight_map(config, _LLAMA))
    tensor_bytes = 512

    manifest = [_SourceTensor("fake.safetensors", name, tensor_bytes) for name in names]
    monkeypatch.setattr(
        converter_mod,
        "_scan_source",
        lambda hf_dir: (manifest, len(names) * tensor_bytes),
    )

    events: list[tuple[str, object]] = []
    tensors = {name: mx.zeros((256,), dtype=mx.float16) for name in names}

    def fake_iter(_manifest):
        for name in names:
            events.append(("read", name))
            yield name, tensors[name]

    monkeypatch.setattr(converter_mod, "_iter_source_tensors", fake_iter)

    saved_batches: list[int] = []
    orig_save = mx.save_safetensors

    def spy_save(path, arrays, *args, **kwargs):
        events.append(("save", len(arrays)))
        saved_batches.append(len(arrays))
        return orig_save(path, arrays, *args, **kwargs)

    monkeypatch.setattr(mx, "save_safetensors", spy_save)

    out_dir = str(tmp_path / "out")
    max_shard_bytes = tensor_bytes * 3
    result = convert_model(
        str(tmp_path / "hf"),
        out_dir,
        config,
        _LLAMA,
        max_shard_bytes=max_shard_bytes,
    )

    assert result.error is None, result.error
    assert result.num_weights == len(names)

    read_count = sum(1 for kind, _ in events if kind == "read")
    save_events = [value for kind, value in events if kind == "save"]
    assert read_count == len(names)
    assert len(save_events) >= 2, "expected multiple shard writes"

    first_save_index = next(
        idx for idx, (kind, _) in enumerate(events) if kind == "save"
    )
    assert first_save_index < len(
        names
    ), "all tensors were read before the first write — conversion is not streamed"
    assert max(save_events) <= 3, f"pending batch exceeded shard cap: {save_events}"

    shards = sorted(Path(out_dir).glob("model-*-of-*.safetensors"))
    assert len(shards) == len(save_events)
    assert (Path(out_dir) / "model.safetensors.index.json").exists()
    assert set(_load_outputs(out_dir)) == {
        build_weight_map(config, _LLAMA)[name] for name in names
    }


# ---------------------------------------------------------------------------
# M24: fsync must target a writable descriptor
# ---------------------------------------------------------------------------


def test_fsync_uses_writable_fd(tmp_path, monkeypatch):
    hf_dir = tmp_path / "hf"
    hf_dir.mkdir()
    mx.save_safetensors(
        str(hf_dir / "a.safetensors"),
        {"model.embed_tokens.weight": mx.zeros((8, 8), dtype=mx.float16)},
    )

    access_modes: list[int] = []
    orig_fsync = os.fsync

    def spy_fsync(fd):
        flags = fcntl.fcntl(fd, fcntl.F_GETFL)
        access_modes.append(flags & os.O_ACCMODE)
        return orig_fsync(fd)

    monkeypatch.setattr(os, "fsync", spy_fsync)

    out_dir = str(tmp_path / "out")
    result = convert_model(str(hf_dir), out_dir, _config(1), _LLAMA)

    assert result.error is None, result.error
    assert access_modes, "conversion did not fsync any file"
    assert all(mode == os.O_RDWR for mode in access_modes), access_modes


# ---------------------------------------------------------------------------
# H10 + #1059: quantize failure mid-stream re-streams unquantized
# ---------------------------------------------------------------------------


def test_quantize_failure_mid_stream_falls_back_to_unquantized(
    tmp_path, monkeypatch, caplog
):
    hf_dir = tmp_path / "hf"
    hf_dir.mkdir()
    mx.save_safetensors(
        str(hf_dir / "a.safetensors"),
        {
            "model.layers.0.self_attn.q_proj.weight": mx.zeros(
                (128, 128), dtype=mx.float16
            ),
            "model.layers.0.self_attn.k_proj.weight": mx.zeros(
                (128, 128), dtype=mx.float16
            ),
        },
    )

    calls = {"n": 0}
    orig_quantize = mx.quantize

    def flaky_quantize(tensor, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("simulated quantize failure")
        return orig_quantize(tensor, **kwargs)

    monkeypatch.setattr(mx, "quantize", flaky_quantize)

    out_dir = str(tmp_path / "out")
    with caplog.at_level(logging.WARNING, logger="fusion_mlx.migrate.converter"):
        result = convert_model(
            str(hf_dir),
            out_dir,
            _config(1),
            _LLAMA,
            quant_bits=4,
            max_shard_bytes=1024,
        )

    assert result.error is None, result.error
    weights = _load_outputs(out_dir)
    assert set(weights) == {
        "layers.0.self_attn.q_proj.weight",
        "layers.0.self_attn.k_proj.weight",
    }
    assert all("scales" not in name and "biases" not in name for name in weights)
    assert weights["layers.0.self_attn.q_proj.weight"].dtype == mx.float16
    assert "dropping quantization entirely" in caplog.text
    # The fallback re-streams without quantization: no further quantize calls.
    assert calls["n"] == 2
    leftovers = [p.name for p in Path(out_dir).iterdir() if ".tmp." in p.name]
    assert leftovers == []


# ---------------------------------------------------------------------------
# M24: validation does not force trust_remote_code
# ---------------------------------------------------------------------------


class _FakeTokenizer:
    def encode(self, text):
        return [1, 2, 3]


def _patch_mlx_lm(monkeypatch):
    calls: dict = {}

    def fake_load(path, tokenizer_config=None, model_config=None, **kwargs):
        calls["path"] = path
        calls["tokenizer_config"] = tokenizer_config
        calls["model_config"] = model_config
        return SimpleNamespace(), _FakeTokenizer()

    monkeypatch.setattr("mlx_lm.load", fake_load)
    monkeypatch.setattr("mlx_lm.generate", lambda *a, **kw: "hello")
    return calls


def test_validate_defaults_trust_remote_code_false(tmp_path, monkeypatch):
    calls = _patch_mlx_lm(monkeypatch)

    result = validate_model(str(tmp_path), prompt="hi", max_tokens=2)

    assert result.success, result.error
    assert calls["tokenizer_config"] == {"trust_remote_code": False}
    assert calls["model_config"] is None


def test_validate_opt_in_trust_remote_code_logs_repo(tmp_path, monkeypatch, caplog):
    calls = _patch_mlx_lm(monkeypatch)
    model_dir = str(tmp_path / "untrusted-repo")

    with caplog.at_level(logging.WARNING, logger="fusion_mlx.migrate.validator"):
        result = validate_model(model_dir, trust_remote_code=True)

    assert result.success, result.error
    assert calls["tokenizer_config"] == {"trust_remote_code": True}
    assert "trust_remote_code=True" in caplog.text
    assert model_dir in caplog.text


# ---------------------------------------------------------------------------
# M26 (shared path only): pytorch .bin fallback keeps dtype and does not
# build one fp32 dict for the whole model
# ---------------------------------------------------------------------------


def test_pytorch_fallback_preserves_dtype(tmp_path):
    torch = pytest.importorskip("torch")
    model_dir = tmp_path / "hf"
    model_dir.mkdir()
    torch.save(
        {
            "model.embed_tokens.weight": torch.zeros((4, 4), dtype=torch.bfloat16),
            "model.norm.weight": torch.zeros((4,), dtype=torch.float16),
        },
        model_dir / "pytorch_model.bin",
    )
    (model_dir / "config.json").write_text("{}")

    out_dir = _convert_pytorch_to_safetensors(model_dir)
    try:
        weights = mx.load(str(Path(out_dir) / "model.safetensors"))
        assert weights["model.embed_tokens.weight"].dtype == mx.bfloat16
        assert weights["model.norm.weight"].dtype == mx.float16
        assert (Path(out_dir) / "config.json").exists()
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)
