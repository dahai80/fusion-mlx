# SPDX-License-Identifier: Apache-2.0
# Migrated from Rapid-MLX test_mtp_gemma4_assistant_inject.py
# vllm_mlx.spec_decode.mtp.* -> fusion_mlx.speculative.* (module does not yet exist)

from __future__ import annotations

import json

import pytest

mx = pytest.importorskip("mlx.core")


try:
    from fusion_mlx.speculative.mtp.accept_counter import (
        reset_global_counter_for_tests,
    )
    from fusion_mlx.speculative.mtp.cache_patch import _unpatch_for_tests

    _HAS_MTP = True
except ImportError:
    _HAS_MTP = False


def _require_mtp():
    if not _HAS_MTP:
        pytest.skip("fusion_mlx.speculative.mtp not migrated yet")


@pytest.fixture(autouse=True)
def _reset_mtp_state():
    import sys

    if not _HAS_MTP:
        yield
        return
    _unpatch_for_tests()
    reset_global_counter_for_tests()
    import mlx_lm.generate

    sys.modules["mlx_lm.generate"].generation_stream = mx.default_stream(
        mx.default_device()
    )
    yield
    _unpatch_for_tests()
    reset_global_counter_for_tests()
    sys.modules["mlx_lm.generate"].generation_stream = mx.default_stream(
        mx.default_device()
    )


def _google_shaped_assistant_config(hidden=64, backbone=128, n_layers=4):
    return {
        "architectures": ["Gemma4UnifiedAssistantForCausalLM"],
        "model_type": "gemma4_unified_assistant",
        "backbone_hidden_size": backbone,
        "num_centroids": 2048,
        "centroid_intermediate_top_k": 32,
        "tie_word_embeddings": True,
        "text_config": {
            "model_type": "gemma4_unified_text",
            "hidden_size": hidden,
            "num_hidden_layers": n_layers,
            "intermediate_size": hidden * 2,
            "num_attention_heads": 4,
            "head_dim": 16,
            "global_head_dim": 32,
            "num_key_value_heads": 1,
            "num_global_key_value_heads": 1,
            "num_kv_shared_layers": n_layers,
            "hidden_size_per_layer_input": 0,
            "sliding_window": 64,
            "layer_types": ["sliding_attention"] * (n_layers - 1) + ["full_attention"],
            "vocab_size": 128,
            "vocab_size_per_layer_input": 0,
            "rms_norm_eps": 1e-6,
            "attention_k_eq_v": True,
            "tie_word_embeddings": True,
            "final_logit_softcapping": None,
            "use_double_wide_mlp": False,
            "enable_moe_block": False,
            "max_position_embeddings": 128,
            "rope_parameters": {
                "full_attention": {
                    "partial_rotary_factor": 0.25,
                    "rope_theta": 1000000.0,
                    "rope_type": "proportional",
                },
                "sliding_attention": {
                    "rope_theta": 10000.0,
                    "rope_type": "default",
                },
            },
        },
    }


def _tiny_gemma4_target_args(hidden=128):
    from mlx_lm.models.gemma4_text import ModelArgs

    args = ModelArgs(
        model_type="gemma4_text",
        hidden_size=hidden,
        intermediate_size=hidden * 2,
        num_hidden_layers=6,
        num_attention_heads=4,
        head_dim=16,
        global_head_dim=32,
        num_key_value_heads=1,
        num_global_key_value_heads=1,
        rms_norm_eps=1e-6,
        vocab_size=128,
        vocab_size_per_layer_input=0,
        num_kv_shared_layers=0,
        hidden_size_per_layer_input=0,
        sliding_window=64,
        sliding_window_pattern=6,
        max_position_embeddings=128,
        final_logit_softcapping=None,
        enable_moe_block=False,
        use_double_wide_mlp=False,
        tie_word_embeddings=True,
        layer_types=[
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
        ],
    )
    return args


def _build_tiny_gemma4_target_model():
    from mlx_lm.models.gemma4_text import Model

    return Model(_tiny_gemma4_target_args())


# ---------------------------------------------------------------------------
# 1. Config parse + module build
# ---------------------------------------------------------------------------


def test_build_assistant_model_args_parses_google_shape():
    pytest.skip("fusion_mlx.speculative.mtp.gemma4_inject not migrated yet")


def test_build_assistant_model_args_rejects_mismatched_backbone_hidden():
    pytest.skip("fusion_mlx.speculative.mtp.gemma4_inject not migrated yet")


def test_build_assistant_model_matches_google_weight_tree():
    pytest.skip("fusion_mlx.speculative.mtp.gemma4_inject not migrated yet")


# ---------------------------------------------------------------------------
# 2. Wiring probe
# ---------------------------------------------------------------------------


def test_inject_attaches_four_surfaces_under_random_init():
    pytest.skip("fusion_mlx.speculative.mtp.gemma4_inject not migrated yet")


# ---------------------------------------------------------------------------
# 3. Weight-loading smoke
# ---------------------------------------------------------------------------


def test_inject_loads_synthetic_google_shaped_sidecar(tmp_path):
    pytest.skip("fusion_mlx.speculative.mtp.gemma4_inject not migrated yet")


def test_inject_refuses_sidecar_missing_tensor(tmp_path):
    pytest.skip("fusion_mlx.speculative.mtp.gemma4_inject not migrated yet")


# ---------------------------------------------------------------------------
# 4. Sidecar refusal
# ---------------------------------------------------------------------------


def test_inject_refuses_no_sidecar_by_default():
    pytest.skip("fusion_mlx.speculative.mtp.gemma4_inject not migrated yet")


# ---------------------------------------------------------------------------
# 5. Architecture guard
# ---------------------------------------------------------------------------


def test_inject_refuses_non_assistant_model_type(tmp_path):
    pytest.skip("fusion_mlx.speculative.mtp.gemma4_inject not migrated yet")


def test_build_assistant_model_args_rejects_layer_types_length_mismatch():
    pytest.skip("fusion_mlx.speculative.mtp.gemma4_inject not migrated yet")


# ---------------------------------------------------------------------------
# 5b. mtp_cache safety
# ---------------------------------------------------------------------------


def test_make_mtp_cache_slots_are_generator_safe():
    pytest.skip("fusion_mlx.speculative.mtp.gemma4_inject not migrated yet")


# ---------------------------------------------------------------------------
# 6. Dispatcher routing
# ---------------------------------------------------------------------------


def test_dispatcher_routes_gemma4_families_to_this_module():
    pytest.skip("fusion_mlx.speculative.mtp.dispatch not migrated yet")


def test_dispatcher_still_routes_qwen3_5():
    pytest.skip("fusion_mlx.speculative.mtp.dispatch not migrated yet")


def test_dispatcher_returns_false_for_unknown_model_type():
    pytest.skip("fusion_mlx.speculative.mtp.dispatch not migrated yet")


def test_dispatcher_swallows_family_exceptions(monkeypatch):
    pytest.skip("fusion_mlx.speculative.mtp.dispatch not migrated yet")


def test_dispatcher_validate_swallows_family_exceptions(monkeypatch):
    pytest.skip("fusion_mlx.speculative.mtp.dispatch not migrated yet")


def test_gemma4_text_modelargs_carries_fields_this_module_reads():
    from mlx_lm.models.gemma4_text import ModelArgs

    required_fields = {
        "model_type",
        "hidden_size",
        "num_hidden_layers",
        "intermediate_size",
        "num_attention_heads",
        "head_dim",
        "global_head_dim",
        "rms_norm_eps",
        "vocab_size",
        "num_key_value_heads",
        "num_global_key_value_heads",
        "num_kv_shared_layers",
        "hidden_size_per_layer_input",
        "rope_parameters",
        "sliding_window",
        "sliding_window_pattern",
        "max_position_embeddings",
        "attention_k_eq_v",
        "final_logit_softcapping",
        "use_double_wide_mlp",
        "enable_moe_block",
        "tie_word_embeddings",
        "layer_types",
    }
    dataclass_fields = set(ModelArgs.__dataclass_fields__.keys())
    missing = required_fields - dataclass_fields
    assert not missing, (
        f"mlx-lm gemma4_text.ModelArgs dropped fields the Gemma 4 inject "
        f"depends on: {sorted(missing)}. Update _build_assistant_model_args."
    )


def test_dispatcher_routes_gemma4_unified_to_gemma4_inject(monkeypatch):
    pytest.skip("fusion_mlx.speculative.mtp.dispatch not migrated yet")


# ---------------------------------------------------------------------------
# 7. Outer-wrapper delegation
# ---------------------------------------------------------------------------


def test_inject_delegates_surfaces_to_outer_wrapper():
    pytest.skip("fusion_mlx.speculative.mtp.gemma4_inject not migrated yet")


# ---------------------------------------------------------------------------
# 8. Codex round-6 fail-closed coverage
# ---------------------------------------------------------------------------


def test_inject_refuses_sidecar_with_shape_mismatched_tensor(tmp_path):
    pytest.skip("fusion_mlx.speculative.mtp.gemma4_inject not migrated yet")


def test_validate_refuses_when_outer_wrapper_missing_delegated_surface():
    pytest.skip("fusion_mlx.speculative.mtp.gemma4_inject not migrated yet")


# ---------------------------------------------------------------------------
# 9. Codex round-7 fail-closed coverage
# ---------------------------------------------------------------------------


def test_inject_refuses_sidecar_with_vocab_size_mismatch(tmp_path):
    pytest.skip("fusion_mlx.speculative.mtp.gemma4_inject not migrated yet")


def test_mtp_forward_rejects_batch_greater_than_one():
    pytest.skip("fusion_mlx.speculative.mtp.gemma4_inject not migrated yet")


# ---------------------------------------------------------------------------
# 10. Codex round-8/9 fail-closed coverage
# ---------------------------------------------------------------------------


def test_injected_class_exposes_mtp_max_batch_size_static_gate():
    pytest.skip("fusion_mlx.speculative.mtp.gemma4_inject not migrated yet")


def test_resolve_sidecar_refuses_non_hf_shape_local_typo(tmp_path):
    pytest.skip("fusion_mlx.speculative.mtp.gemma4_inject not migrated yet")


def test_inject_random_init_refuses_when_target_has_no_vocab_size():
    pytest.skip("fusion_mlx.speculative.mtp.gemma4_inject not migrated yet")


def test_inject_refuses_when_target_tail_layer_types_mismatch(tmp_path):
    pytest.skip("fusion_mlx.speculative.mtp.gemma4_inject not migrated yet")


def test_mtp_forward_rejects_populated_mtp_cache():
    pytest.skip("fusion_mlx.speculative.mtp.gemma4_inject not migrated yet")


def test_mtp_forward_rejects_negative_row_offset():
    pytest.skip("fusion_mlx.speculative.mtp.gemma4_inject not migrated yet")


def test_inject_refuses_when_target_layer_types_shorter_than_assistant(tmp_path):
    pytest.skip("fusion_mlx.speculative.mtp.gemma4_inject not migrated yet")


def test_find_safetensors_refuses_multi_file_even_with_model_safetensors(tmp_path):
    pytest.skip("fusion_mlx.speculative.mtp.gemma4_inject not migrated yet")


def test_validate_refuses_when_outer_mtp_is_none():
    pytest.skip("fusion_mlx.speculative.mtp.gemma4_inject not migrated yet")


def test_validate_refuses_when_outer_mtp_max_batch_size_wrong_value():
    pytest.skip("fusion_mlx.speculative.mtp.gemma4_inject not migrated yet")


def test_mtp_forward_returns_per_position_shape():
    pytest.skip("fusion_mlx.speculative.mtp.gemma4_inject not migrated yet")


def test_inject_refuses_sidecar_with_nonpositive_vocab_size(tmp_path):
    pytest.skip("fusion_mlx.speculative.mtp.gemma4_inject not migrated yet")


def test_dispatcher_swallows_family_import_exception(monkeypatch):
    pytest.skip("fusion_mlx.speculative.mtp.dispatch not migrated yet")
