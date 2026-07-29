# SPDX-License-Identifier: Apache-2.0
# Migrated from Rapid-MLX test_mtp_gemma4_assistant_inject.py
# vllm_mlx.spec_decode.mtp.* -> fusion_mlx.speculative.* (module does not yet exist)

from __future__ import annotations

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
    from fusion_mlx.speculative.mtp.gemma4_inject import _build_assistant_model_args

    cfg = _google_shaped_assistant_config(hidden=64, backbone=128, n_layers=4)
    args = _build_assistant_model_args(cfg, target_backbone_hidden=128)
    assert args is not None
    assert args.hidden_size == 64
    assert args.num_hidden_layers == 4
    assert args.vocab_size == 128
    assert list(args.layer_types) == [
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
    ]
    assert getattr(args, "backbone_hidden_size", None) == 128


def test_build_assistant_model_args_rejects_mismatched_backbone_hidden():
    from fusion_mlx.speculative.mtp.gemma4_inject import _build_assistant_model_args

    cfg = _google_shaped_assistant_config(hidden=64, backbone=256, n_layers=4)
    args = _build_assistant_model_args(cfg, target_backbone_hidden=128)
    assert args is None


def test_build_assistant_model_matches_google_weight_tree():
    from mlx.utils import tree_flatten

    from fusion_mlx.speculative.mtp.gemma4_inject import (
        _build_assistant_model,
        _build_assistant_model_args,
    )

    cfg = _google_shaped_assistant_config(hidden=64, backbone=128, n_layers=4)
    args = _build_assistant_model_args(cfg, target_backbone_hidden=128)
    assert args is not None
    backbone_hidden = int(getattr(args, "backbone_hidden_size", 128))
    assistant = _build_assistant_model(args, backbone_hidden)
    keys = {k for k, _ in tree_flatten(assistant.parameters())}
    assert "model.embed_tokens.weight" in keys
    assert "model.norm.weight" in keys
    assert "pre_projection.weight" in keys
    assert "post_projection.weight" in keys
    assert any(k.startswith("model.layers.0.") for k in keys)


# ---------------------------------------------------------------------------
# 2. Wiring probe
# ---------------------------------------------------------------------------


def test_inject_attaches_four_surfaces_under_random_init():
    from fusion_mlx.speculative.mtp.gemma4_inject import inject_mtp_support

    model = _build_tiny_gemma4_target_model()
    assert inject_mtp_support(model, allow_random_init=True) is True
    assert getattr(model, "mtp", None) is not None
    assert callable(getattr(model, "mtp_forward", None))
    assert callable(getattr(model, "make_mtp_cache", None))
    assert getattr(model, "mtp_max_batch_size", None) == 1
    assert type(model).__name__ == "_Gemma4WithMTP"


# ---------------------------------------------------------------------------
# 3. Weight-loading smoke
# ---------------------------------------------------------------------------


def test_inject_loads_synthetic_google_shaped_sidecar(tmp_path):
    import json

    import mlx.core as mx
    from mlx.utils import tree_flatten

    from fusion_mlx.speculative.mtp.gemma4_inject import (
        _build_assistant_model,
        _build_assistant_model_args,
        inject_mtp_support,
    )

    cfg = _google_shaped_assistant_config(hidden=64, backbone=128, n_layers=4)
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    args = _build_assistant_model_args(cfg, target_backbone_hidden=128)
    assert args is not None
    backbone_hidden = int(getattr(args, "backbone_hidden_size", 128))
    assistant = _build_assistant_model(args, backbone_hidden)
    sd = {k: v for k, v in tree_flatten(assistant.parameters())}
    mx.save_safetensors(str(tmp_path / "assistant.safetensors"), sd)

    model = _build_tiny_gemma4_target_model()
    assert inject_mtp_support(model, str(tmp_path)) is True
    assert getattr(model, "mtp", None) is not None
    assert callable(getattr(model, "mtp_forward", None))
    assert callable(getattr(model, "make_mtp_cache", None))
    assert getattr(model, "mtp_max_batch_size", None) == 1


def test_inject_refuses_sidecar_missing_tensor(tmp_path):
    import json

    import mlx.core as mx
    from mlx.utils import tree_flatten

    from fusion_mlx.speculative.mtp.gemma4_inject import (
        _build_assistant_model,
        _build_assistant_model_args,
        inject_mtp_support,
    )

    cfg = _google_shaped_assistant_config(hidden=64, backbone=128, n_layers=4)
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    args = _build_assistant_model_args(cfg, target_backbone_hidden=128)
    assert args is not None
    backbone_hidden = int(getattr(args, "backbone_hidden_size", 128))
    assistant = _build_assistant_model(args, backbone_hidden)
    sd = {k: v for k, v in tree_flatten(assistant.parameters())}
    dropped = next(iter(sd))
    del sd[dropped]
    mx.save_safetensors(str(tmp_path / "assistant.safetensors"), sd)

    model = _build_tiny_gemma4_target_model()
    assert inject_mtp_support(model, str(tmp_path)) is False
    assert not hasattr(model, "mtp")


# ---------------------------------------------------------------------------
# 4. Sidecar refusal
# ---------------------------------------------------------------------------


def test_inject_refuses_no_sidecar_by_default():
    from fusion_mlx.speculative.mtp.gemma4_inject import inject_mtp_support

    model = _build_tiny_gemma4_target_model()
    assert inject_mtp_support(model) is False
    assert not hasattr(model, "mtp")


# ---------------------------------------------------------------------------
# 5. Architecture guard
# ---------------------------------------------------------------------------


def test_inject_refuses_non_assistant_model_type(tmp_path):
    import json

    from fusion_mlx.speculative.mtp.gemma4_inject import inject_mtp_support

    cfg = _google_shaped_assistant_config()
    cfg["model_type"] = "gemma4_text"
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    model = _build_tiny_gemma4_target_model()
    assert inject_mtp_support(model, str(tmp_path)) is False


def test_build_assistant_model_args_rejects_layer_types_length_mismatch():
    from fusion_mlx.speculative.mtp.gemma4_inject import _build_assistant_model_args

    cfg = _google_shaped_assistant_config(hidden=64, backbone=128, n_layers=4)
    cfg["text_config"]["layer_types"] = [
        "sliding_attention",
        "sliding_attention",
        "full_attention",
    ]
    args = _build_assistant_model_args(cfg, target_backbone_hidden=128)
    assert args is None


# ---------------------------------------------------------------------------
# 5b. mtp_cache safety
# ---------------------------------------------------------------------------


def test_make_mtp_cache_slots_are_generator_safe():
    from mlx_lm.models.cache import KVCache

    from fusion_mlx.speculative.mtp.gemma4_inject import inject_mtp_support

    model = _build_tiny_gemma4_target_model()
    assert inject_mtp_support(model, allow_random_init=True) is True
    cache = model.make_mtp_cache()
    assert isinstance(cache, list)
    assert len(cache) == len(model.mtp.model.layers)
    assert all(isinstance(slot, KVCache) for slot in cache)
    assert all(int(getattr(slot, "offset", 0)) == 0 for slot in cache)


# ---------------------------------------------------------------------------
# 6. Dispatcher routing
# ---------------------------------------------------------------------------


def test_dispatcher_routes_gemma4_families_to_this_module():
    from fusion_mlx.speculative.mtp.dispatch import _MTP_INJECT_DISPATCH

    gemma_keys = [k for k in _MTP_INJECT_DISPATCH if k.startswith("gemma4")]
    assert gemma_keys, "no gemma4 family keys registered in inject dispatch"
    for k in gemma_keys:
        mod_path, func_name = _MTP_INJECT_DISPATCH[k]
        assert mod_path == "fusion_mlx.speculative.mtp.gemma4_inject", k
        assert func_name == "inject_mtp_support", k


def test_dispatcher_still_routes_qwen3_5():
    from fusion_mlx.speculative.mtp.dispatch import _MTP_INJECT_DISPATCH

    for k in ("qwen3_5", "qwen3_5_moe"):
        assert k in _MTP_INJECT_DISPATCH, k
        mod_path, func_name = _MTP_INJECT_DISPATCH[k]
        assert mod_path == "fusion_mlx.speculative.mtp.qwen3_5_inject", k
        assert func_name == "inject_mtp_support", k


def test_dispatcher_returns_false_for_unknown_model_type():
    from fusion_mlx.speculative.mtp.dispatch import (
        dispatch_mtp_inject,
        dispatch_mtp_validate,
    )

    assert dispatch_mtp_inject(object(), "definitely_unknown_type_xyz") is False
    assert dispatch_mtp_validate(object(), "definitely_unknown_type_xyz") is False


def test_dispatcher_swallows_family_exceptions(monkeypatch):
    import fusion_mlx.speculative.mtp.gemma4_inject as gi

    def _boom(model, mtp_sidecar=None, allow_random_init=False):
        raise RuntimeError("synthetic inject failure")

    monkeypatch.setattr(gi, "inject_mtp_support", _boom)
    from fusion_mlx.speculative.mtp.dispatch import dispatch_mtp_inject

    assert dispatch_mtp_inject(object(), "gemma4_unified") is False


def test_dispatcher_validate_swallows_family_exceptions(monkeypatch):
    import fusion_mlx.speculative.mtp.qwen3_5_inject as qi

    def _boom(model):
        raise RuntimeError("synthetic validate failure")

    monkeypatch.setattr(qi, "validate_mtp_support", _boom)
    from fusion_mlx.speculative.mtp.dispatch import dispatch_mtp_validate

    assert dispatch_mtp_validate(object(), "qwen3_5") is False


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
    import fusion_mlx.speculative.mtp.gemma4_inject as gi
    import fusion_mlx.speculative.mtp.qwen3_5_inject as qi

    called = {}

    def _gemma(model, mtp_sidecar=None, allow_random_init=False):
        called["gemma4"] = True
        return True

    def _qwen(model, mtp_sidecar=None, allow_random_init=False):
        called["qwen3_5"] = True
        return True

    monkeypatch.setattr(gi, "inject_mtp_support", _gemma)
    monkeypatch.setattr(qi, "inject_mtp_support", _qwen)
    from fusion_mlx.speculative.mtp.dispatch import dispatch_mtp_inject

    assert dispatch_mtp_inject(object(), "gemma4_unified") is True
    assert called.get("gemma4") is True
    assert "qwen3_5" not in called


# ---------------------------------------------------------------------------
# 7. Outer-wrapper delegation
# ---------------------------------------------------------------------------


def test_inject_delegates_surfaces_to_outer_wrapper():
    from types import SimpleNamespace

    from fusion_mlx.speculative.mtp.gemma4_inject import inject_mtp_support

    inner = _build_tiny_gemma4_target_model()
    outer = SimpleNamespace(language_model=inner)
    assert inject_mtp_support(outer, allow_random_init=True) is True

    assert getattr(outer, "mtp", None) is not None
    assert callable(getattr(outer, "mtp_forward", None))
    assert callable(getattr(outer, "make_mtp_cache", None))
    assert getattr(outer, "mtp_max_batch_size", None) == 1

    assert getattr(inner, "mtp", None) is outer.mtp
    delegated_cache = outer.make_mtp_cache()
    assert isinstance(delegated_cache, list)
    assert len(delegated_cache) == len(inner.mtp.model.layers)


# ---------------------------------------------------------------------------
# 8. Codex round-6 fail-closed coverage
# ---------------------------------------------------------------------------


def test_inject_refuses_sidecar_with_shape_mismatched_tensor(tmp_path):
    import json

    import mlx.core as mx
    from mlx.utils import tree_flatten

    from fusion_mlx.speculative.mtp.gemma4_inject import (
        _build_assistant_model,
        _build_assistant_model_args,
        inject_mtp_support,
    )

    cfg = _google_shaped_assistant_config(hidden=64, backbone=128, n_layers=4)
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    args = _build_assistant_model_args(cfg, target_backbone_hidden=128)
    assert args is not None
    backbone_hidden = int(getattr(args, "backbone_hidden_size", 128))
    assistant = _build_assistant_model(args, backbone_hidden)
    sd = {k: v for k, v in tree_flatten(assistant.parameters())}
    mismatch_key = next(iter(sd))
    original = tuple(sd[mismatch_key].shape)
    wrong = (original[0] + 1,) + original[1:] if original else (2,)
    sd[mismatch_key] = mx.zeros(wrong, dtype=sd[mismatch_key].dtype)
    mx.save_safetensors(str(tmp_path / "assistant.safetensors"), sd)

    model = _build_tiny_gemma4_target_model()
    assert inject_mtp_support(model, str(tmp_path)) is False
    assert not hasattr(model, "mtp")


def test_validate_refuses_when_outer_wrapper_missing_delegated_surface():
    from types import SimpleNamespace

    from fusion_mlx.speculative.mtp.gemma4_inject import (
        inject_mtp_support,
        validate_mtp_support,
    )

    inner = _build_tiny_gemma4_target_model()
    assert inject_mtp_support(inner, allow_random_init=True) is True

    outer = SimpleNamespace(
        language_model=inner,
        mtp=inner.mtp,
        make_mtp_cache=inner.make_mtp_cache,
        mtp_max_batch_size=1,
    )
    assert validate_mtp_support(outer) is False


# ---------------------------------------------------------------------------
# 9. Codex round-7 fail-closed coverage
# ---------------------------------------------------------------------------


def test_inject_refuses_sidecar_with_vocab_size_mismatch(tmp_path):
    import json

    from fusion_mlx.speculative.mtp.gemma4_inject import inject_mtp_support

    cfg = _google_shaped_assistant_config()
    cfg["text_config"]["vocab_size"] = 256
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    (tmp_path / "assistant.safetensors").write_bytes(b"")
    model = _build_tiny_gemma4_target_model()
    assert inject_mtp_support(model, str(tmp_path)) is False


def test_mtp_forward_rejects_batch_greater_than_one():
    import mlx.core as mx

    from fusion_mlx.speculative.mtp.gemma4_inject import inject_mtp_support

    model = _build_tiny_gemma4_target_model()
    assert inject_mtp_support(model, allow_random_init=True) is True

    target_cache = [object() for _ in range(len(model.model.layers))]
    model._mtp_target_cache = target_cache
    hidden_states = mx.zeros((2, 4, 128))
    next_token_ids = mx.zeros((1, 4), dtype=mx.int32)
    with pytest.raises(ValueError):
        model.mtp_forward(hidden_states, next_token_ids, None)


# ---------------------------------------------------------------------------
# 10. Codex round-8/9 fail-closed coverage
# ---------------------------------------------------------------------------


def test_injected_class_exposes_mtp_max_batch_size_static_gate():
    from fusion_mlx.speculative.mtp.gemma4_inject import inject_mtp_support

    model = _build_tiny_gemma4_target_model()
    assert inject_mtp_support(model, allow_random_init=True) is True
    cls = type(model)
    assert cls.__dict__.get("mtp_max_batch_size") == 1
    assert model.mtp_max_batch_size == 1


def test_resolve_sidecar_refuses_non_hf_shape_local_typo(tmp_path):
    from fusion_mlx.speculative.mtp.gemma4_inject import _resolve_sidecar_dir

    assert _resolve_sidecar_dir("nonexistent-local-typo-no-slash") is None


def test_inject_random_init_refuses_when_target_has_no_vocab_size():
    from fusion_mlx.speculative.mtp.gemma4_inject import inject_mtp_support

    model = _build_tiny_gemma4_target_model()
    object.__setattr__(model.args, "vocab_size", 0)
    assert inject_mtp_support(model, allow_random_init=True) is False


def test_inject_refuses_when_target_tail_layer_types_mismatch(tmp_path):
    import json

    from fusion_mlx.speculative.mtp.gemma4_inject import inject_mtp_support

    cfg = _google_shaped_assistant_config(hidden=64, backbone=128, n_layers=4)
    cfg["text_config"]["layer_types"] = [
        "full_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
    ]
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    (tmp_path / "assistant.safetensors").write_bytes(b"")
    model = _build_tiny_gemma4_target_model()
    assert inject_mtp_support(model, str(tmp_path)) is False


def test_mtp_forward_rejects_populated_mtp_cache():
    import mlx.core as mx
    from types import SimpleNamespace

    from fusion_mlx.speculative.mtp.gemma4_inject import inject_mtp_support

    model = _build_tiny_gemma4_target_model()
    assert inject_mtp_support(model, allow_random_init=True) is True

    target_cache = [object() for _ in range(len(model.model.layers))]
    model._mtp_target_cache = target_cache
    hidden_states = mx.zeros((1, 4, 128))
    next_token_ids = mx.zeros((1, 4), dtype=mx.int32)
    mtp_cache = [SimpleNamespace(offset=5)]
    with pytest.raises(ValueError):
        model.mtp_forward(hidden_states, next_token_ids, mtp_cache)


def test_mtp_forward_rejects_negative_row_offset():
    import mlx.core as mx
    from types import SimpleNamespace

    from fusion_mlx.speculative.mtp.gemma4_inject import inject_mtp_support

    model = _build_tiny_gemma4_target_model()
    assert inject_mtp_support(model, allow_random_init=True) is True

    fake_slot = SimpleNamespace(
        state=(mx.zeros((1, 1, 1, 16)), mx.zeros((1, 1, 1, 16))),
        offset=1,
    )
    target_cache = [fake_slot for _ in range(len(model.model.layers))]
    model._mtp_target_cache = target_cache
    hidden_states = mx.zeros((1, 4, 128))
    next_token_ids = mx.zeros((1, 4), dtype=mx.int32)
    with pytest.raises(ValueError):
        model.mtp_forward(hidden_states, next_token_ids, None)


def test_inject_refuses_when_target_layer_types_shorter_than_assistant(tmp_path):
    import json

    from fusion_mlx.speculative.mtp.gemma4_inject import inject_mtp_support

    cfg = _google_shaped_assistant_config(hidden=64, backbone=128, n_layers=4)
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    (tmp_path / "assistant.safetensors").write_bytes(b"")
    model = _build_tiny_gemma4_target_model()
    object.__setattr__(
        model.args,
        "layer_types",
        ["sliding_attention", "full_attention"],
    )
    assert inject_mtp_support(model, str(tmp_path)) is False


def test_find_safetensors_refuses_multi_file_even_with_model_safetensors(tmp_path):
    from fusion_mlx.speculative.mtp.gemma4_inject import _find_safetensors

    (tmp_path / "model.safetensors").write_bytes(b"")
    (tmp_path / "model-00001-of-00002.safetensors").write_bytes(b"")
    assert _find_safetensors(tmp_path) is None


def test_validate_refuses_when_outer_mtp_is_none():
    from types import SimpleNamespace

    from fusion_mlx.speculative.mtp.gemma4_inject import (
        inject_mtp_support,
        validate_mtp_support,
    )

    inner = _build_tiny_gemma4_target_model()
    assert inject_mtp_support(inner, allow_random_init=True) is True

    outer = SimpleNamespace(
        language_model=inner,
        mtp=None,
        mtp_forward=inner.mtp_forward,
        make_mtp_cache=inner.make_mtp_cache,
        mtp_max_batch_size=1,
    )
    assert validate_mtp_support(outer) is False


def test_validate_refuses_when_outer_mtp_max_batch_size_wrong_value():
    from types import SimpleNamespace

    from fusion_mlx.speculative.mtp.gemma4_inject import (
        inject_mtp_support,
        validate_mtp_support,
    )

    inner = _build_tiny_gemma4_target_model()
    assert inject_mtp_support(inner, allow_random_init=True) is True

    outer = SimpleNamespace(
        language_model=inner,
        mtp=inner.mtp,
        mtp_forward=inner.mtp_forward,
        make_mtp_cache=inner.make_mtp_cache,
        mtp_max_batch_size=2,
    )
    assert validate_mtp_support(outer) is False


def test_mtp_forward_returns_per_position_shape():
    import mlx.core as mx
    from mlx_lm.models.cache import KVCache

    from fusion_mlx.speculative.mtp.gemma4_inject import inject_mtp_support

    model = _build_tiny_gemma4_target_model()
    assert inject_mtp_support(model, allow_random_init=True) is True

    target_cache = [KVCache() for _ in range(len(model.model.layers))]
    input_ids = mx.array([[1, 2, 3, 4]])
    _out, hidden = model(input_ids, cache=target_cache, return_hidden=True)
    next_token_ids = mx.array([[5, 6, 7, 8]])
    logits = model.mtp_forward(hidden, next_token_ids, None)

    n_positions = hidden.shape[1]
    assert logits.shape == (1, n_positions, 128)


def test_inject_refuses_sidecar_with_nonpositive_vocab_size(tmp_path):
    import json

    from fusion_mlx.speculative.mtp.gemma4_inject import inject_mtp_support

    cfg = _google_shaped_assistant_config(hidden=64, backbone=128, n_layers=4)
    cfg["text_config"]["vocab_size"] = 0
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    (tmp_path / "assistant.safetensors").write_bytes(b"")
    model = _build_tiny_gemma4_target_model()
    assert inject_mtp_support(model, str(tmp_path)) is False


def test_dispatcher_swallows_family_import_exception(monkeypatch):
    import fusion_mlx.speculative.mtp.dispatch as disp

    real_import = disp.importlib.import_module

    def _fail(module_path):
        if "gemma4_inject" in module_path:
            raise ImportError("synthetic import failure")
        return real_import(module_path)

    monkeypatch.setattr(disp.importlib, "import_module", _fail)
    assert disp.dispatch_mtp_inject(object(), "gemma4_unified") is False
