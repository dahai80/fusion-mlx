# Early TurboQuant conversion for chunked prefill: converting at the
# first chunk boundary (instead of the post-prefill epilogue) caps the
# fp16 prefill KV at one chunk. The 1M-token run aborted at the hard
# tier with active memory growing 64 kB/token - the fp16 prefill KV
# itself, because k8v4 only took effect after prefill completed.
import types

import mlx.core as mx
from mlx_lm.models.cache import KVCache

from fusion_mlx.scheduler.sched_batch import _maybe_early_tq_convert
from fusion_mlx.scheduler.sched_token import _TQ_KV_OUTLIER_KEY_MAX
from fusion_mlx.turboquant_kv import K8V4TurboQuantKVCache


def _stub_scheduler(bits=4.0, mode="k8v4", skip_last=True):
    from fusion_mlx.scheduler.sched_cache import _turboquant_eligible
    from fusion_mlx.scheduler.sched_token import (
        _apply_turboquant_kv_convert,
        _sync_expected_layer_signature,
        _turboquant_kv_outlier_layers,
    )

    s = types.SimpleNamespace()
    s._turboquant_kv_bits = bits
    s._turboquant_kv_mode = mode
    s._turboquant_skip_last = skip_last
    s._model_uses_mla = lambda: False
    s._model_uses_attention_sinks = lambda: False
    s._stream = None
    s._turboquant_eligible = _turboquant_eligible.__get__(s)
    s._apply_turboquant_kv_convert = _apply_turboquant_kv_convert.__get__(s)
    s._turboquant_kv_outlier_layers = _turboquant_kv_outlier_layers.__get__(s)
    s._sync_expected_layer_signature = _sync_expected_layer_signature.__get__(s)
    return s


def _stub_state(cache):
    return types.SimpleNamespace(
        cache=cache,
        request=types.SimpleNamespace(request_id="test-req"),
    )


def _kv_cache(tokens, heads=2, head_dim=8, scale=1.0):
    c = KVCache()
    keys = mx.random.normal((1, heads, tokens, head_dim)) * scale
    values = mx.random.normal((1, heads, tokens, head_dim))
    c.update_and_fetch(keys, values)
    mx.eval(c.state)
    return c


def test_early_convert_turns_kv_layers_k8v4_after_first_chunk():
    s = _stub_scheduler()
    state = _stub_state([_kv_cache(64), _kv_cache(64), _kv_cache(64)])
    _maybe_early_tq_convert(s, state)
    assert state._tq_early_converted is True
    assert isinstance(state.cache[0], K8V4TurboQuantKVCache)
    assert isinstance(state.cache[1], K8V4TurboQuantKVCache)
    # skip_last keeps the final layer fp16
    assert type(state.cache[2]) is KVCache
    assert int(state.cache[0].offset) == 64


def test_early_convert_is_idempotent():
    s = _stub_scheduler()
    state = _stub_state([_kv_cache(64)])
    _maybe_early_tq_convert(s, state)
    converted = state.cache[0]
    _maybe_early_tq_convert(s, state)
    assert state.cache[0] is converted


def test_early_convert_outlier_gate_keeps_fp16():
    s = _stub_scheduler()
    hot = _kv_cache(64, scale=(_TQ_KV_OUTLIER_KEY_MAX * 10))
    state = _stub_state([hot, _kv_cache(64)])
    _maybe_early_tq_convert(s, state)
    assert state._tq_early_converted is True
    assert type(state.cache[0]) is KVCache
    assert type(state.cache[1]) is KVCache


def test_early_convert_noop_without_tq_bits():
    s = _stub_scheduler(bits=None)
    state = _stub_state([_kv_cache(64)])
    _maybe_early_tq_convert(s, state)
    assert type(state.cache[0]) is KVCache
    assert getattr(state, "_tq_early_converted", False) is False
