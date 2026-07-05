# SPDX-License-Identifier: Apache-2.0
"""Tests for fusion_mlx.kv_cache_dtype safelist + resolution policy.

Covers the three-way dtype enum, the --reasoning pin, and the
auto-downgrade safelist (sliding-window + MLA) ported from rapid-mlx.
"""

from __future__ import annotations

import pytest

from fusion_mlx.kv_cache_dtype import (
    DEFAULT_KV_CACHE_DTYPE,
    KV_CACHE_DTYPES,
    REASONING_KV_CACHE_DTYPE,
    KVCacheDtypeDecision,
    dtype_to_quantization_bits,
    log_kv_cache_decision,
    resolve_kv_cache_dtype,
)

# ---------------------------------------------------------------------------
# Pass-through + defaults
# ---------------------------------------------------------------------------


def test_bf16_passes_through_unchanged():
    d = resolve_kv_cache_dtype("bf16", model_name="qwen3-8b")
    assert d.dtype == "bf16"
    assert d.downgraded is False
    assert d.requested == "bf16"


def test_int4_default_for_non_safelisted_model():
    d = resolve_kv_cache_dtype("int4", model_name="qwen3-8b")
    assert d.dtype == "int4"
    assert d.downgraded is False
    assert d.requested == "int4"


def test_int8_operator_override_passes_through():
    d = resolve_kv_cache_dtype("int8", model_name="qwen3-8b")
    assert d.dtype == "int8"
    assert d.downgraded is False
    assert d.requested == "int8"


def test_invalid_dtype_raises():
    with pytest.raises(ValueError):
        resolve_kv_cache_dtype("fp8", model_name="x")


# ---------------------------------------------------------------------------
# --reasoning pin
# ---------------------------------------------------------------------------


def test_reasoning_pins_int4_to_int8():
    d = resolve_kv_cache_dtype("int4", reasoning=True, model_name="qwen3-8b")
    assert d.dtype == REASONING_KV_CACHE_DTYPE == "int8"
    assert d.downgraded is True
    assert d.requested == "int4"


def test_reasoning_pins_bf16_to_int8():
    d = resolve_kv_cache_dtype("bf16", reasoning=True, model_name="qwen3-8b")
    assert d.dtype == "int8"
    assert d.downgraded is True


def test_reasoning_with_int8_not_marked_downgraded():
    d = resolve_kv_cache_dtype("int8", reasoning=True, model_name="qwen3-8b")
    assert d.dtype == "int8"
    assert d.downgraded is False


def test_reasoning_wins_over_safelist():
    # Gemma 3 is sliding-window (would normally downgrade to bf16), but
    # --reasoning forces int8 regardless.
    d = resolve_kv_cache_dtype("int4", reasoning=True, model_name="gemma-3-27b")
    assert d.dtype == "int8"
    assert d.downgraded is True


# ---------------------------------------------------------------------------
# Sliding-window safelist
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["gemma-3-27b", "gemma3", "gpt-oss-20b", "gpt_oss_7b"])
def test_sliding_window_name_downgrades_int4_to_bf16(name):
    d = resolve_kv_cache_dtype("int4", model_name=name)
    assert d.dtype == "bf16"
    assert d.downgraded is True
    assert d.requested == "int4"


def test_sliding_window_hf_config_field_downgrades():
    d = resolve_kv_cache_dtype(
        "int4",
        model_name="some-anon-model",
        hf_config={"sliding_window": 1024},
    )
    assert d.dtype == "bf16"
    assert d.downgraded is True


def test_sliding_window_hf_config_zero_does_not_downgrade():
    # sliding_window=0 or absent means no sliding window.
    d = resolve_kv_cache_dtype(
        "int4",
        model_name="some-anon-model",
        hf_config={"sliding_window": 0},
    )
    assert d.dtype == "int4"
    assert d.downgraded is False


def test_sliding_window_hf_config_model_type_downgrades():
    d = resolve_kv_cache_dtype(
        "int4",
        model_name="anon",
        hf_config={"model_type": "gemma3_text"},
    )
    assert d.dtype == "bf16"
    assert d.downgraded is True


def test_sliding_window_alias_metadata_overrides():
    d = resolve_kv_cache_dtype(
        "int4",
        model_name="anon-no-name-hit",
        alias_metadata={"sliding_window": True},
    )
    assert d.dtype == "bf16"
    assert d.downgraded is True


def test_bf16_not_downgraded_even_for_sliding_window():
    # Explicit bf16 is never silently moved.
    d = resolve_kv_cache_dtype("bf16", model_name="gemma-3-27b")
    assert d.dtype == "bf16"
    assert d.downgraded is False


# ---------------------------------------------------------------------------
# MLA safelist
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["deepseek-v3", "deepseek_v3", "kimi-k2", "kimi_k2"])
def test_mla_name_downgrades_int4_to_bf16(name):
    d = resolve_kv_cache_dtype("int4", model_name=name)
    assert d.dtype == "bf16"
    assert d.downgraded is True


def test_mla_hf_config_model_type_downgrades():
    d = resolve_kv_cache_dtype(
        "int4",
        model_name="anon",
        hf_config={"model_type": "deepseek_v3"},
    )
    assert d.dtype == "bf16"
    assert d.downgraded is True


def test_mla_alias_metadata_overrides():
    d = resolve_kv_cache_dtype(
        "int4",
        model_name="anon-no-name-hit",
        alias_metadata={"is_mla": True},
    )
    assert d.dtype == "bf16"
    assert d.downgraded is True


def test_mla_rank_pair_alone_does_not_downgrade():
    # BLOCKING #3: rank fields without a family signal must NOT trigger —
    # a non-DeepSeek/Kimi model could ship both ranks for unrelated
    # reasons (e.g. LoRA adapter metadata).
    d = resolve_kv_cache_dtype(
        "int4",
        model_name="anon-unknown-vendor",
        hf_config={"q_lora_rank": 512, "kv_lora_rank": 256},
    )
    assert d.dtype == "int4"
    assert d.downgraded is False


def test_mla_rank_pair_with_name_hit_downgrades():
    # Rank pair + family signal (name match) together → downgrade.
    d = resolve_kv_cache_dtype(
        "int4",
        model_name="deepseek-v3-fp8",
        hf_config={"q_lora_rank": 512, "kv_lora_rank": 256},
    )
    assert d.dtype == "bf16"
    assert d.downgraded is True


# ---------------------------------------------------------------------------
# dtype_to_quantization_bits
# ---------------------------------------------------------------------------


def test_bits_mapping_bf16_disables_quantization():
    quant, bits = dtype_to_quantization_bits("bf16")
    assert quant is False


def test_bits_mapping_int8():
    quant, bits = dtype_to_quantization_bits("int8")
    assert quant is True
    assert bits == 8


def test_bits_mapping_int4():
    quant, bits = dtype_to_quantization_bits("int4")
    assert quant is True
    assert bits == 4


def test_bits_mapping_invalid_raises():
    with pytest.raises(ValueError):
        dtype_to_quantization_bits("fp8")


# ---------------------------------------------------------------------------
# log_kv_cache_decision — smoke (must not raise, must populate msg)
# ---------------------------------------------------------------------------


def test_log_decision_runs_without_error(capsys):
    d = resolve_kv_cache_dtype("int4", model_name="qwen3-8b")
    log_kv_cache_decision(d, model_name="qwen3-8b")
    out = capsys.readouterr().out
    assert "KV cache dtype" in out
    assert "int4" in out


# ---------------------------------------------------------------------------
# Decision dataclass shape — locks the contract cli_serve.py depends on.
# ---------------------------------------------------------------------------


def test_decision_has_required_fields():
    d = resolve_kv_cache_dtype("int4", model_name="qwen3-8b")
    # cli_serve.py constructs KVCacheDtypeDecision(dtype=, reason=,
    # downgraded=, requested=) and reads .dtype. Lock both directions.
    assert hasattr(d, "dtype")
    assert hasattr(d, "reason")
    assert hasattr(d, "downgraded")
    assert hasattr(d, "requested")


def test_decision_is_constructable_with_rapid_mlx_kwargs():
    # The legacy CLI path in cli_serve.py builds the decision by hand
    # with these exact kwargs — must not TypeError.
    d = KVCacheDtypeDecision(
        dtype="int8", reason="legacy flag", downgraded=False, requested="int8"
    )
    assert d.dtype == "int8"
    assert d.downgraded is False


def test_dtype_enum_order():
    # Order is load-bearing for the metrics gauge label set.
    assert KV_CACHE_DTYPES == ("bf16", "int8", "int4")
    assert DEFAULT_KV_CACHE_DTYPE == "int4"
    assert REASONING_KV_CACHE_DTYPE == "int8"
