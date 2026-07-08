# SPDX-License-Identifier: Apache-2.0
"""Tests for the HF->MLX convert CLI (wraps mlx-lm convert)."""

import argparse
from pathlib import Path

from fusion_mlx.cli_convert import (
    _build_convert_kwargs,
    _resolve_output_path,
    convert_command,
)


def _args(**kw):
    base = dict(
        model="org/model-name",
        out=None,
        quant_bits=None,
        quant_group_size=64,
        quant_mode="affine",
        dtype=None,
        dequantize=False,
        upload_repo=None,
        trust_remote_code=False,
    )
    base.update(kw)
    return argparse.Namespace(**base)


class TestBuildKwargs:
    def test_plain_convert_no_quantize(self):
        kw = _build_convert_kwargs(_args(), "org/model-name")
        assert kw["quantize"] is False
        assert kw["q_bits"] is None
        assert kw["q_group_size"] == 64
        assert kw["q_mode"] == "affine"

    def test_quant_bits_enables_quantize(self):
        kw = _build_convert_kwargs(_args(quant_bits=4), "org/model-name")
        assert kw["quantize"] is True
        assert kw["q_bits"] == 4

    def test_dtype_and_dequantize_passed_through(self):
        kw = _build_convert_kwargs(_args(dtype="bf16", dequantize=True), "org/m")
        assert kw["dtype"] == "bf16"
        assert kw["dequantize"] is True

    def test_upload_repo_and_trust_remote_code_passed_through(self):
        kw = _build_convert_kwargs(
            _args(upload_repo="me/m", trust_remote_code=True), "org/m"
        )
        assert kw["upload_repo"] == "me/m"
        assert kw["trust_remote_code"] is True

    def test_nvfp4_mode_enables_quantize_without_bits(self):
        # fp-quant modes ignore --quant-bits; mlx-lm's defaults_for_mode fills
        # the per-mode (group_size, bits). We must pass None so a stale
        # --quant-group-size=64 does not override nvfp4's required 16.
        kw = _build_convert_kwargs(_args(quant_mode="nvfp4"), "org/m")
        assert kw["quantize"] is True
        assert kw["q_mode"] == "nvfp4"
        assert kw["q_bits"] is None
        assert kw["q_group_size"] is None

    def test_mxfp8_mode_ignores_quant_bits_and_group_size(self):
        kw = _build_convert_kwargs(
            _args(quant_mode="mxfp8", quant_bits=4, quant_group_size=32), "org/m"
        )
        assert kw["quantize"] is True
        assert kw["q_bits"] is None
        assert kw["q_group_size"] is None

    def test_affine_mode_still_requires_bits(self):
        kw = _build_convert_kwargs(_args(quant_mode="affine"), "org/m")
        assert kw["quantize"] is False
        assert kw["q_group_size"] == 64


class TestOutputPath:
    def test_default_is_cwd_basename(self):
        assert _resolve_output_path("org/model-name", None) == str(
            Path.cwd() / "model-name"
        )

    def test_custom_out_respected(self):
        assert _resolve_output_path("org/m", "/tmp/out") == "/tmp/out"


class TestConvertCommand:
    def test_calls_mlx_convert_with_hf_path_and_kwargs(self, monkeypatch, capsys):
        calls = []

        def fake_convert(hf_path, **kw):
            calls.append((hf_path, kw))

        monkeypatch.setattr("mlx_lm.convert", fake_convert)
        rc = convert_command(_args(model="org/model-name"))
        assert rc == 0
        assert len(calls) == 1
        hf_path, kw = calls[0]
        assert hf_path == "org/model-name"
        assert kw["quantize"] is False
        assert "Converted model written to:" in capsys.readouterr().out

    def test_alias_resolved_before_convert(self, monkeypatch):
        calls = []
        monkeypatch.setattr("mlx_lm.convert", lambda hf, **kw: calls.append(hf))
        monkeypatch.setattr(
            "fusion_mlx.model_aliases.resolve_model",
            lambda m: "mlx-community/Qwen3.5-9B" if m == "qwen3.5-9b" else m,
        )
        rc = convert_command(_args(model="qwen3.5-9b", quant_bits=4))
        assert rc == 0
        assert calls == ["mlx-community/Qwen3.5-9B"]

    def test_quant_bits_routed_to_mlx_convert(self, monkeypatch):
        seen = {}
        monkeypatch.setattr("mlx_lm.convert", lambda hf, **kw: seen.update(kw))
        convert_command(_args(model="org/m", quant_bits=4, quant_group_size=32))
        assert seen["quantize"] is True
        assert seen["q_bits"] == 4
        assert seen["q_group_size"] == 32

    def test_failure_returns_1_and_prints_error(self, monkeypatch, capsys):
        def boom(hf, **kw):
            raise RuntimeError("disk full")

        monkeypatch.setattr("mlx_lm.convert", boom)
        rc = convert_command(_args(model="org/m"))
        assert rc == 1
        assert "convert failed" in capsys.readouterr().err

    def test_custom_out_passed_through(self, monkeypatch):
        seen = {}
        monkeypatch.setattr("mlx_lm.convert", lambda hf, **kw: seen.update(kw))
        convert_command(_args(model="org/m", out="/tmp/myout"))
        assert seen["mlx_path"] == "/tmp/myout"
