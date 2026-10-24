# SPDX-License-Identifier: Apache-2.0
"""#1057/#1058/#1059: migrate atomic write, backup-swap, and hard-fail
on 0-layer / quantize-failure."""

import os
from unittest.mock import patch

import mlx.core as mx

from fusion_mlx.migrate.architectures import KNOWN_TEMPLATES
from fusion_mlx.migrate.codegen import generate_model_code
from fusion_mlx.migrate.converter import _quantize_tensor, convert_model

_LLAMA = KNOWN_TEMPLATES["llama"]


# ---------------------------------------------------------------------------
# #1059 part 1: num_hidden_layers=0 → hard fail
# ---------------------------------------------------------------------------


def test_convert_model_zero_layers_hard_fails(tmp_path):
    """num_hidden_layers=0 must produce an error result and NOT write
    any files to disk. Before #1059 it warned + wrote a 0-layer model
    that mlx_lm.load accepted as valid but generated garbage."""
    hf_dir = tmp_path / "hf"
    hf_dir.mkdir()
    out_dir = str(tmp_path / "out")

    config = {
        "num_hidden_layers": 0,
        "hidden_size": 64,
        "num_attention_heads": 2,
        "vocab_size": 100,
    }

    template = _LLAMA

    result = convert_model(str(hf_dir), out_dir, config, template)

    assert result.error is not None
    assert "num_hidden_layers" in result.error
    assert not os.path.exists(out_dir), "0-layer model must not be written to disk"


# ---------------------------------------------------------------------------
# #1059 part 2: per-tensor quantize failure signals the caller (H10 keeps
# the all-or-nothing fallback in _write_weights_streamed)
# ---------------------------------------------------------------------------


def test_quantize_tensor_failure_returns_none():
    """A tensor that fails to quantize returns None so the streaming
    converter can drop quantization for the whole model (#1059)."""
    call_count = {"n": 0}
    orig_quantize = mx.quantize

    def _flaky_quantize(tensor, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("simulated quantize failure")
        return orig_quantize(tensor, **kwargs)

    with patch.object(mx, "quantize", side_effect=_flaky_quantize):
        result = _quantize_tensor("layers.0.linear.weight", mx.zeros((128, 128)), 4, 64)

    assert result is None


def test_quantize_tensor_success_returns_quantized_keys():
    """When a weight quantizes, the .scales/.biases keys are produced."""
    result = _quantize_tensor("layers.0.linear.weight", mx.zeros((128, 128)), 4, 64)
    assert result is not None
    assert "layers.0.linear.weight" in result
    assert "layers.0.linear.scales" in result
    assert "layers.0.linear.biases" in result


def test_quantize_tensor_skips_norm_and_embedding():
    for name in ("model.norm.weight", "embed_tokens.weight", "lm_head.weight"):
        tensor = mx.zeros((128, 128))
        result = _quantize_tensor(name, tensor, 4, 64)
        assert result is not None
        assert list(result) == [name]


# ---------------------------------------------------------------------------
# #1058: backup-swap preserves old output on rename failure
# ---------------------------------------------------------------------------


def test_convert_model_rename_failure_preserves_old_output(tmp_path):
    """If os.rename(tmp_dir, output_dir) fails, the previously-available
    output_dir must be restored from backup. Before #1058 the old code
    did rmtree(output_dir) then rename — a failed rename after rmtree
    destroyed the old model with no recovery."""
    hf_dir = tmp_path / "hf"
    hf_dir.mkdir()
    out_dir = str(tmp_path / "out")

    # Pre-create output_dir with a sentinel file representing the old model.
    os.makedirs(out_dir)
    sentinel = os.path.join(out_dir, "old_model.safetensors")
    with open(sentinel, "w") as f:
        f.write("old")

    config = {
        "num_hidden_layers": 2,
        "hidden_size": 64,
        "num_attention_heads": 2,
        "intermediate_size": 128,
        "vocab_size": 100,
    }

    template = _LLAMA

    rename_calls = {"n": 0}
    orig_rename = os.rename

    def _failing_rename(src, dst):
        rename_calls["n"] += 1
        # First rename: output_dir → backup_dir (succeeds)
        # Second rename: tmp_dir → output_dir (FAILS)
        if rename_calls["n"] == 2:
            raise OSError("simulated cross-device rename failure")
        return orig_rename(src, dst)

    with (
        patch(
            "fusion_mlx.migrate.converter._scan_source",
            return_value=([], 0),
        ),
        patch("os.rename", side_effect=_failing_rename),
    ):
        result = convert_model(str(hf_dir), out_dir, config, template)

    assert result.error is not None
    # The old sentinel file must survive — output_dir was restored from backup.
    assert os.path.exists(sentinel), (
        "old output_dir was not restored after rename failure — "
        "the previous model was destroyed with no recovery"
    )


# ---------------------------------------------------------------------------
# #1057: codegen uses tmp_dir + rename (atomic write)
# ---------------------------------------------------------------------------


def test_codegen_atomic_write_success(tmp_path):
    """generate_model_code builds in a temp dir then swaps — on success
    the output_dir contains both .py and config.json, and no temp dir
    litters the sibling path."""
    template = _LLAMA
    config = {
        "num_hidden_layers": 2,
        "hidden_size": 64,
        "num_attention_heads": 2,
        "intermediate_size": 128,
        "vocab_size": 100,
    }
    out_dir = str(tmp_path / "codegen_out")

    result = generate_model_code(template, config, out_dir)

    assert result.error is None
    assert os.path.isfile(os.path.join(out_dir, "llama.py"))
    assert os.path.isfile(os.path.join(out_dir, "config.json"))
    # No temp dir left behind
    leftovers = [
        p for p in os.listdir(tmp_path) if ".codegen_tmp." in p or ".old." in p
    ]
    assert leftovers == [], f"temp dirs left behind: {leftovers}"


def test_codegen_failure_leaves_no_partial_output(tmp_path):
    """If codegen crashes mid-write, the output_dir must NOT contain a
    half-written state. Before #1057 it wrote directly to output_dir —
    a crash between .py and config.json left a broken model dir."""
    template = _LLAMA
    config = {"num_hidden_layers": 2, "hidden_size": 64, "vocab_size": 100}
    out_dir = str(tmp_path / "codegen_out")

    # Pre-create output_dir with a sentinel.
    os.makedirs(out_dir)
    sentinel = os.path.join(out_dir, "old_config.json")
    with open(sentinel, "w") as f:
        f.write("old")

    # Make json.dump raise to simulate a crash mid-write.
    with patch("json.dump", side_effect=RuntimeError("simulated crash")):
        result = generate_model_code(template, config, out_dir)

    assert result.error is not None
    # The old sentinel must survive — output_dir was not half-overwritten.
    assert os.path.exists(
        sentinel
    ), "codegen crash destroyed the old output_dir — atomic write failed"
    # No temp dir left behind.
    leftovers = [
        p for p in os.listdir(tmp_path) if ".codegen_tmp." in p or ".old." in p
    ]
    assert leftovers == [], f"temp dirs left behind: {leftovers}"
