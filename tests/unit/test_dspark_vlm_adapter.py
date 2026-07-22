# SPDX-License-Identifier: Apache-2.0
"""Qwen3VLTargetAdapter unit tests (no model weights required).

Ported from upstream dspark-metal PR#2 (tests/test_vlm_adapter.py) when the
engine was vendored under fusion_mlx.speculative.dspark.engine. Covers the
VLM-specific pieces added for DSpark->VLM:
  - adapter registration / family detection
  - vision side-channel set/clear + LoadedTargetModel delegation
  - _text_only_taps: vision positions are masked out of ctx_taps so the
    draft (trained on qwen3 TEXT hidden states) never sees OOD vision
    columns and context_len stays text-only
  - _decode_position_ids: MRoPE 3-dim position ids, rope_delta aware,
    works for both mlx_vlm KVCache (_idx) and mlx_lm (offset)
  - forward_verifier_states routing: vision prefill vs text decode
  - text adapter regression: text family rejects vision inputs
"""

from __future__ import annotations

import types

import mlx.core as mx
import numpy as np
import pytest

from fusion_mlx.speculative.dspark.engine.adapters import (
    ADAPTERS,
    LoadedTargetModel,
    MLXTargetAdapter,
    Qwen3TargetAdapter,
    Qwen3VLTargetAdapter,
    adapter_for_model_type,
)


def test_vlm_adapter_registered():
    assert adapter_for_model_type("qwen3_vl") is Qwen3VLTargetAdapter
    assert ADAPTERS["qwen3_vl"] is Qwen3VLTargetAdapter
    assert issubclass(Qwen3VLTargetAdapter, MLXTargetAdapter)
    assert Qwen3VLTargetAdapter.family == "qwen3_vl"


def test_text_adapter_still_registered_and_unaffected():
    assert adapter_for_model_type("qwen3") is Qwen3TargetAdapter
    assert Qwen3TargetAdapter.family == "qwen3"
    assert adapter_for_model_type("unknown") is None


# ---------------------------------------------------------------- vision side-channel


def test_vlm_set_clear_vision_inputs():
    a = Qwen3VLTargetAdapter()
    assert a._vision_inputs is None
    a.set_vision_inputs(pixel_values="PX", image_grid_thw="THW", video_grid_thw=None)
    assert a._vision_inputs == {
        "pixel_values": "PX",
        "image_grid_thw": "THW",
        "video_grid_thw": None,
    }
    a.clear_vision_inputs()
    assert a._vision_inputs is None


def test_vlm_set_vision_inputs_empty_clears():
    a = Qwen3VLTargetAdapter()
    a.set_vision_inputs(pixel_values="PX")
    assert a._vision_inputs is not None
    a.set_vision_inputs()
    assert a._vision_inputs is None


def test_text_adapter_rejects_vision():
    t = Qwen3TargetAdapter()
    with pytest.raises(NotImplementedError) as exc:
        t.set_vision_inputs(pixel_values="PX")
    assert "does not support vision" in str(exc.value)
    assert "qwen3" in str(exc.value)


def test_loaded_target_model_delegates_vision():
    captured = {}

    class _SpyAdapter(Qwen3VLTargetAdapter):
        def set_vision_inputs(self, **kwargs):
            captured["set"] = kwargs

        def clear_vision_inputs(self):
            captured["cleared"] = True

    spy = _SpyAdapter()
    target = LoadedTargetModel(
        requested_model="x",
        resolved_model_path=None,
        model=None,
        tokenizer=None,
        adapter=spy,
    )
    target.set_vision_inputs(pixel_values="PX", image_grid_thw="THW")
    assert captured["set"] == {"pixel_values": "PX", "image_grid_thw": "THW"}
    target.clear_vision_inputs()
    assert captured["cleared"] is True


# ---------------------------------------------------------------- _text_only_taps


def _make_hidden(seq, dim):
    vals = np.arange(seq * dim, dtype=np.float32).reshape(1, seq, dim)
    return mx.array(vals)


def test_text_only_taps_excludes_vision_columns_2d_mask():
    seq, dim = 5, 4
    hidden = _make_hidden(seq, dim)
    vpm = mx.array([[False, True, True, False, False]])
    out = Qwen3VLTargetAdapter._text_only_taps(hidden, vpm)
    assert out.shape == (1, 3, dim)
    kept = np.asarray(out).reshape(3, dim)
    source = np.asarray(hidden).reshape(seq, dim)
    np.testing.assert_array_equal(kept, source[[0, 3, 4]])


def test_text_only_taps_uses_first_submask_for_3d():
    seq, dim = 5, 4
    hidden = _make_hidden(seq, dim)
    col0 = np.array(
        [[[False, True], [True, False], [True, True], [False, False], [False, True]]]
    )
    vpm = mx.array(col0)
    out = Qwen3VLTargetAdapter._text_only_taps(hidden, vpm)
    assert out.shape == (1, 3, dim)
    kept = np.asarray(out).reshape(3, dim)
    source = np.asarray(hidden).reshape(seq, dim)
    np.testing.assert_array_equal(kept, source[[0, 3, 4]])


def test_text_only_taps_none_mask_returns_input():
    hidden = _make_hidden(5, 4)
    out = Qwen3VLTargetAdapter._text_only_taps(hidden, None)
    assert out.shape == hidden.shape


def test_text_only_taps_all_vision_falls_back_to_last():
    seq, dim = 4, 3
    hidden = _make_hidden(seq, dim)
    vpm = mx.array([[True, True, True, True]])
    out = Qwen3VLTargetAdapter._text_only_taps(hidden, vpm)
    assert out.shape == (1, 1, dim)
    kept = np.asarray(out).reshape(1, dim)
    source = np.asarray(hidden).reshape(seq, dim)
    np.testing.assert_array_equal(kept, source[-1:])


# ---------------------------------------------------------------- _decode_position_ids


class _VLMMockModel:
    def __init__(self, rope_delta=None):
        self.language_model = types.SimpleNamespace(_rope_deltas=rope_delta)


class _MLXVLMCache:
    def __init__(self, idx):
        self._idx = idx


class _MLXLMCache:
    def __init__(self, offset):
        self.offset = offset


def test_decode_position_ids_mrope_with_rope_delta_vlm_cache():
    model = _VLMMockModel(rope_delta=mx.array([5]))
    cache = [_MLXVLMCache(10)]
    pos = Qwen3VLTargetAdapter()._decode_position_ids(model, 3, cache)
    assert pos.shape == (3, 1, 3)
    arr = np.asarray(pos)
    for row in range(3):
        np.testing.assert_array_equal(arr[row, 0], [15, 16, 17])


def test_decode_position_ids_no_rope_delta():
    model = _VLMMockModel(rope_delta=None)
    cache = [_MLXVLMCache(10)]
    pos = Qwen3VLTargetAdapter()._decode_position_ids(model, 3, cache)
    arr = np.asarray(pos)
    np.testing.assert_array_equal(arr[0, 0], [10, 11, 12])


def test_decode_position_ids_mlx_lm_offset_cache():
    model = _VLMMockModel(rope_delta=mx.array([2]))
    cache = [_MLXLMCache(7)]
    pos = Qwen3VLTargetAdapter()._decode_position_ids(model, 2, cache)
    arr = np.asarray(pos)
    np.testing.assert_array_equal(arr[0, 0], [9, 10])


def test_decode_position_ids_empty_cache():
    model = _VLMMockModel(rope_delta=None)
    pos = Qwen3VLTargetAdapter()._decode_position_ids(model, 2, [])
    arr = np.asarray(pos)
    np.testing.assert_array_equal(arr[0, 0], [0, 1])


# ---------------------------------------------------------------- routing


def _route_inputs():
    return mx.zeros((1, 1), dtype=mx.uint32)


def test_forward_verifier_states_routes_to_vision_prefill():
    a = Qwen3VLTargetAdapter()
    a.set_vision_inputs(pixel_values="PX", image_grid_thw="THW")
    calls = []

    def fake_prefill(model, inputs, cache, layer_ids):
        calls.append("prefill")
        return mx.zeros((1, 1, 1)), mx.zeros((1, 1, 1))

    def fake_decode(model, inputs, cache, layer_ids):
        calls.append("decode")
        return mx.zeros((1, 1, 1)), mx.zeros((1, 1, 1))

    a._prefill_with_vision = fake_prefill
    a._decode_text = fake_decode
    a.forward_verifier_states(
        model=None, inputs=_route_inputs(), cache=[], layer_ids=[0]
    )
    assert calls == ["prefill"]


def test_forward_verifier_states_routes_to_text_decode_when_no_vision():
    a = Qwen3VLTargetAdapter()
    a.clear_vision_inputs()
    calls = []

    def fake_prefill(model, inputs, cache, layer_ids):
        calls.append("prefill")

    def fake_decode(model, inputs, cache, layer_ids):
        calls.append("decode")
        return mx.zeros((1, 1, 1)), mx.zeros((1, 1, 1))

    a._prefill_with_vision = fake_prefill
    a._decode_text = fake_decode
    a.forward_verifier_states(
        model=None, inputs=_route_inputs(), cache=[], layer_ids=[0]
    )
    assert calls == ["decode"]


def test_forward_verifier_states_routes_to_text_when_pixel_values_none():
    a = Qwen3VLTargetAdapter()
    a.set_vision_inputs(pixel_values=None, image_grid_thw="THW")
    calls = []

    def fake_prefill(model, inputs, cache, layer_ids):
        calls.append("prefill")

    def fake_decode(model, inputs, cache, layer_ids):
        calls.append("decode")
        return mx.zeros((1, 1, 1)), mx.zeros((1, 1, 1))

    a._prefill_with_vision = fake_prefill
    a._decode_text = fake_decode
    a.forward_verifier_states(
        model=None, inputs=_route_inputs(), cache=[], layer_ids=[0]
    )
    assert calls == ["decode"]


# ---------------------------------------------------------------- rewind / summary


def test_rewind_kv_caches_ducktyped_trim():
    trimmed = []

    class _C:
        def trim(self, n):
            trimmed.append(n)

    class _NoTrim:
        offset = 3

    Qwen3VLTargetAdapter().rewind_kv_caches([_C(), _C(), _NoTrim(), None], 2)
    assert trimmed == [2, 2]


def test_cache_summary_ducktyped():
    class _VLM:
        _idx = 5

    class _LM:
        offset = 9

    s = Qwen3VLTargetAdapter().cache_summary([_VLM(), None, _LM()])
    assert "0:kv=5" in s
    assert "2:kv=9" in s


# ---------------------------------------------------------------- tokenizer unwrap


class _InnerTok:
    eos_token_id = 151645

    def apply_chat_template(self, messages, **kwargs):
        return "TPL:" + messages[-1]["content"]

    def encode(self, text, add_special_tokens=False):
        return [1, 2, 3]


class _Processor:
    def __init__(self):
        self.tokenizer = _InnerTok()


def test_tokenizer_of_unwraps_processor():
    proc = _Processor()
    assert Qwen3VLTargetAdapter._tokenizer_of(proc) is proc.tokenizer
    inner = _InnerTok()
    assert Qwen3VLTargetAdapter._tokenizer_of(inner) is inner


def test_vlm_build_prompt_uses_inner_tokenizer():
    a = Qwen3VLTargetAdapter()
    tokens = a.build_prompt(_Processor(), "hi", enable_thinking=False)
    assert tokens.dtype == mx.uint32
    assert list(np.asarray(tokens)) == [1, 2, 3]


def test_vlm_stop_token_ids_via_processor():
    a = Qwen3VLTargetAdapter()
    assert a.stop_token_ids(_Processor()) == {151645}
