# SPDX-License-Identifier: Apache-2.0
"""G20: DFlash2 model-aware preset resolver tests.

Validates that block_size/draft_bits/warmup/circuit-breaker are tuned
per model family (Qwen3.8 dense vs Qwen3.5 dense vs generic) and
adjusted for quant bits (8bit target → draft_bits=8).
"""

from fusion_mlx.speculative.dflash2.presets import (
    DFlash2Preset,
    _adjust_for_quant,
    _detect_family,
    resolve_preset,
)


def test_detect_family_qwen38():
    assert _detect_family("mlx-community/Qwen3.8-27B-4bit") == "qwen3_8"
    assert _detect_family("Qwen3.8-8B") == "qwen3_8"


def test_detect_family_qwen35():
    assert _detect_family("mlx-community/Qwen3.5-9B-4bit") == "qwen3_5"
    assert _detect_family("Qwen3.5-3B") == "qwen3_5"


def test_detect_family_qwen3_generic():
    assert _detect_family("Qwen3-32B") == "qwen3"
    assert _detect_family("qwen3-14b") == "qwen3"


def test_detect_family_default():
    assert _detect_family("llama-3-8b") == "default"
    assert _detect_family("") == "default"


def test_detect_family_via_model_args():
    class FakeArgs:
        model_type = "qwen3_8_moe"

    class FakeModel:
        args = FakeArgs()

    # MoE model_type → qwen3 generic (DFlash2 excluded for MoE anyway)
    assert _detect_family("unknown-model", FakeModel()) == "qwen3"


def test_detect_family_via_model_args_dense():
    class FakeArgs:
        model_type = "qwen3_8"

    class FakeModel:
        args = FakeArgs()

    assert _detect_family("unknown-model", FakeModel()) == "qwen3_8"


def test_preset_qwen38_defaults():
    p = resolve_preset("Qwen3.8-27B-4bit", quant_bits=4)
    assert p.block_size == 5
    assert p.draft_bits == 4
    assert p.warmup_steps == 3
    assert p.circuit_breaker_threshold == 0.20


def test_preset_qwen35_tighter_window():
    p = resolve_preset("Qwen3.5-9B-4bit", quant_bits=4)
    assert p.block_size == 4
    assert p.draft_bits == 4


def test_preset_8bit_upgrades_draft_bits():
    p = resolve_preset("Qwen3.8-27B-8bit", quant_bits=8)
    assert p.draft_bits == 8
    assert p.block_size == 5
    assert "8bit" in p.reason


def test_preset_4bit_keeps_draft_bits_4():
    p = resolve_preset("Qwen3.8-27B-4bit", quant_bits=4)
    assert p.draft_bits == 4


def test_preset_none_quant_uses_family_default():
    p = resolve_preset("Qwen3.8-27B", quant_bits=None)
    assert p.draft_bits == 4
    assert p.block_size == 5


def test_preset_default_for_unknown():
    p = resolve_preset("llama-3-8b", quant_bits=4)
    assert p.block_size == 5
    assert p.draft_bits == 4


def test_preset_qwen3_generic_conservative():
    p = resolve_preset("Qwen3-32B", quant_bits=4)
    assert p.block_size == 4
    assert p.warmup_steps == 2
    assert p.circuit_breaker_threshold == 0.25


def test_adjust_for_quant_8bit_upgrades():
    base = DFlash2Preset(block_size=5, draft_bits=4, reason="test")
    adjusted = _adjust_for_quant(base, 8)
    assert adjusted.draft_bits == 8
    assert adjusted.block_size == 5


def test_adjust_for_quant_4bit_no_change():
    base = DFlash2Preset(block_size=5, draft_bits=4, reason="test")
    adjusted = _adjust_for_quant(base, 4)
    assert adjusted.draft_bits == 4


def test_adjust_for_quant_none_no_change():
    base = DFlash2Preset(block_size=5, draft_bits=4, reason="test")
    adjusted = _adjust_for_quant(base, None)
    assert adjusted.draft_bits == 4
