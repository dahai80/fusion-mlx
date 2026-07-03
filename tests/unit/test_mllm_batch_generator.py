# SPDX-License-Identifier: Apache-2.0
"""Regression tests for MLLMBatchGenerator model-call kwargs."""

import mlx.core as mx
import pytest

from fusion_mlx.mllm_batch_generator import MLLMBatchGenerator, MLLMBatchRequest


class _RecordingModel:

    def __init__(self):
        self.last_call_kwargs = None
        self.last_input_ids = None
        self.language_model = object()

    def __call__(self, input_ids, cache=None, **kwargs):
        self.last_input_ids = input_ids
        self.last_call_kwargs = kwargs
        return mx.zeros((1, 1, 8))


def _make_generator(model: _RecordingModel) -> MLLMBatchGenerator:
    return MLLMBatchGenerator(
        model=model,
        processor=object(),
        mm_processor=None,
        enable_vision_cache=False,
    )


def _make_request(*, pixel_values, extra_kwargs=None) -> MLLMBatchRequest:
    return MLLMBatchRequest(
        uid=0,
        request_id="r0",
        prompt="hello",
        max_tokens=8,
        input_ids=mx.array([1, 2, 3], dtype=mx.int32),
        pixel_values=pixel_values,
        extra_kwargs=extra_kwargs or {},
    )


def test_run_vision_encoding_passes_pixel_values_none_for_text_only_request():
    model = _RecordingModel()
    gen = _make_generator(model)
    request = _make_request(pixel_values=None)
    gen._run_vision_encoding(request, cache=None)
    assert "pixel_values" in model.last_call_kwargs
    assert model.last_call_kwargs["pixel_values"] is None


def test_run_vision_encoding_forwards_pixel_values_when_set():
    model = _RecordingModel()
    gen = _make_generator(model)
    pixels = mx.zeros((1, 3, 4, 4))
    request = _make_request(pixel_values=pixels)
    gen._run_vision_encoding(request, cache=None)
    assert "pixel_values" in model.last_call_kwargs
    assert model.last_call_kwargs["pixel_values"] is pixels


def test_run_vision_encoding_preserves_extra_kwargs_alongside_pixel_values():
    model = _RecordingModel()
    gen = _make_generator(model)
    request = _make_request(
        pixel_values=None,
        extra_kwargs={"token_type_ids": mx.array([0, 0, 1])},
    )
    gen._run_vision_encoding(request, cache=None)
    assert "pixel_values" in model.last_call_kwargs
    assert model.last_call_kwargs["pixel_values"] is None
    assert "token_type_ids" in model.last_call_kwargs


def test_close_swallows_synchronize_thread_error(monkeypatch):
    import mlx.core as mx

    gen = _make_generator(_RecordingModel())
    gen._old_wired_limit = 1234

    sync_calls: list[object] = []
    set_limit_calls: list[int] = []

    def _raising_sync(stream):
        sync_calls.append(stream)
        raise RuntimeError("There is no Stream(gpu, 2) in current thread")

    def _record_set_limit(value):
        set_limit_calls.append(value)
        return value

    monkeypatch.setattr(mx, "synchronize", _raising_sync)
    monkeypatch.setattr(mx, "set_wired_limit", _record_set_limit)

    gen.close()

    assert len(sync_calls) == 1
    assert set_limit_calls == [1234]
    assert gen._old_wired_limit is None


def test_close_propagates_non_runtime_errors_from_set_wired_limit(monkeypatch):
    import mlx.core as mx

    gen = _make_generator(_RecordingModel())
    gen._old_wired_limit = 999

    monkeypatch.setattr(mx, "synchronize", lambda _s: None)

    def _boom(value):
        raise OSError("metal API call failed")

    monkeypatch.setattr(mx, "set_wired_limit", _boom)

    with pytest.raises(OSError, match="metal API call failed"):
        gen.close()


def _make_step_stub_generator():
    gen = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
    gen._shared_batch_sampler = None

    def _language_model(input_tokens, cache=None):
        B = input_tokens.shape[0]
        return mx.zeros((B, 1, 4))

    gen.language_model = _language_model
    gen.sampler = lambda x: mx.zeros((x.shape[0],), dtype=mx.uint32)
    return gen


def _make_sampling_request(uid: int, temperature: float, top_p: float):
    return MLLMBatchRequest(
        uid=uid,
        request_id=f"r{uid}",
        prompt="hi",
        max_tokens=8,
        temperature=temperature,
        top_p=top_p,
    )


def test_step_homogeneous_requests_calls_fused_sampler_once(monkeypatch):
    from fusion_mlx.scheduler.sampler_fast_path import make_fused_sampler

    make_sampler_calls = []
    shared_sampler_invocations = []

    def shared_sampler(logprobs):
        shared_sampler_invocations.append(logprobs.shape)
        return mx.zeros((logprobs.shape[0],), dtype=mx.uint32)

    def fake_make_fused_sampler(**kwargs):
        make_sampler_calls.append(kwargs)
        return shared_sampler

    monkeypatch.setattr(
        "fusion_mlx.mllm_batch_generator.make_fused_sampler",
        fake_make_fused_sampler,
    )

    gen = _make_step_stub_generator()
    requests = [
        _make_sampling_request(0, 0.7, 0.95),
        _make_sampling_request(1, 0.7, 0.95),
        _make_sampling_request(2, 0.7, 0.95),
        _make_sampling_request(3, 0.7, 0.95),
    ]

    input_tokens = mx.array([[1], [2], [3], [4]], dtype=mx.uint32)
    sampled, _ = MLLMBatchGenerator._step(
        gen, input_tokens, cache=[], requests=requests
    )

    assert len(make_sampler_calls) == 1
    assert make_sampler_calls[0]["temperature"] == 0.7
    assert make_sampler_calls[0]["top_p"] == 0.95
    assert len(shared_sampler_invocations) == 1
    assert sampled.shape == (4,)


def test_step_caches_shared_sampler_across_calls(monkeypatch):
    make_sampler_calls = []

    def fake_make_fused_sampler(**kwargs):
        make_sampler_calls.append(kwargs)
        return lambda x: mx.zeros((x.shape[0],), dtype=mx.uint32)

    monkeypatch.setattr(
        "fusion_mlx.mllm_batch_generator.make_fused_sampler",
        fake_make_fused_sampler,
    )

    gen = _make_step_stub_generator()
    requests = [
        _make_sampling_request(0, 0.7, 0.95),
        _make_sampling_request(1, 0.7, 0.95),
    ]

    for _ in range(5):
        MLLMBatchGenerator._step(
            gen,
            mx.array([[1], [2]], dtype=mx.uint32),
            cache=[],
            requests=requests,
        )

    assert len(make_sampler_calls) == 1


def test_step_param_change_invalidates_cached_sampler(monkeypatch):
    make_sampler_calls = []

    def fake_make_fused_sampler(**kwargs):
        make_sampler_calls.append(kwargs)
        return lambda x: mx.zeros((x.shape[0],), dtype=mx.uint32)

    monkeypatch.setattr(
        "fusion_mlx.mllm_batch_generator.make_fused_sampler",
        fake_make_fused_sampler,
    )

    gen = _make_step_stub_generator()

    MLLMBatchGenerator._step(
        gen,
        mx.array([[1], [2]], dtype=mx.uint32),
        cache=[],
        requests=[
            _make_sampling_request(0, 0.7, 0.95),
            _make_sampling_request(1, 0.7, 0.95),
        ],
    )
    MLLMBatchGenerator._step(
        gen,
        mx.array([[1], [2]], dtype=mx.uint32),
        cache=[],
        requests=[
            _make_sampling_request(0, 0.3, 0.95),
            _make_sampling_request(1, 0.3, 0.95),
        ],
    )

    assert len(make_sampler_calls) == 2
    assert make_sampler_calls[0]["temperature"] == 0.7
    assert make_sampler_calls[0]["top_p"] == 0.95
    assert make_sampler_calls[1]["temperature"] == 0.3
    assert make_sampler_calls[1]["top_p"] == 0.95


def test_step_heterogeneous_requests_use_per_row_loop(monkeypatch):
    make_sampler_calls = []

    def fake_make_fused_sampler(**kwargs):
        make_sampler_calls.append(kwargs)
        return lambda x: mx.zeros((x.shape[0],), dtype=mx.uint32)

    monkeypatch.setattr(
        "fusion_mlx.mllm_batch_generator.make_fused_sampler",
        fake_make_fused_sampler,
    )

    gen = _make_step_stub_generator()
    req_a = _make_sampling_request(0, 0.7, 0.95)
    req_b = _make_sampling_request(1, 0.3, 0.80)

    MLLMBatchGenerator._step(
        gen,
        mx.array([[1], [2]], dtype=mx.uint32),
        cache=[],
        requests=[req_a, req_b],
    )
    assert len(make_sampler_calls) == 2
    assert make_sampler_calls[0]["temperature"] == 0.7
    assert make_sampler_calls[0]["top_p"] == 0.95
    assert make_sampler_calls[1]["temperature"] == 0.3
    assert make_sampler_calls[1]["top_p"] == 0.80
    cached_a = getattr(req_a, "_cached_sampler", None)
    cached_b = getattr(req_b, "_cached_sampler", None)
    assert cached_a is not None and cached_a[0] == (0.7, 0.95, 0, 0.0)
    assert cached_b is not None and cached_b[0] == (0.3, 0.80, 0, 0.0)
    assert gen._shared_batch_sampler is None


def test_step_b1_homogeneous_still_uses_shared_sampler(monkeypatch):
    make_sampler_calls = []

    def fake_make_fused_sampler(**kwargs):
        make_sampler_calls.append(kwargs)
        return lambda x: mx.zeros((x.shape[0],), dtype=mx.uint32)

    monkeypatch.setattr(
        "fusion_mlx.mllm_batch_generator.make_fused_sampler",
        fake_make_fused_sampler,
    )

    gen = _make_step_stub_generator()
    MLLMBatchGenerator._step(
        gen,
        mx.array([[1]], dtype=mx.uint32),
        cache=[],
        requests=[_make_sampling_request(0, 0.7, 0.95)],
    )

    assert len(make_sampler_calls) == 1
    assert gen._shared_batch_sampler is not None
    assert gen._shared_batch_sampler[0] == (0.7, 0.95, 0, 0.0)


def test_step_batch_uses_dataclass_defaults(monkeypatch):
    make_sampler_calls = []

    def fake_make_fused_sampler(**kwargs):
        make_sampler_calls.append(kwargs)
        return lambda x: mx.zeros((x.shape[0],), dtype=mx.uint32)

    monkeypatch.setattr(
        "fusion_mlx.mllm_batch_generator.make_fused_sampler",
        fake_make_fused_sampler,
    )

    gen = _make_step_stub_generator()
    requests = [
        MLLMBatchRequest(uid=i, request_id=f"d{i}", prompt="hi") for i in range(4)
    ]

    MLLMBatchGenerator._step(
        gen,
        mx.array([[1], [2], [3], [4]], dtype=mx.uint32),
        cache=[],
        requests=requests,
    )

    assert len(make_sampler_calls) == 1
    assert make_sampler_calls[0]["temperature"] == 0.7
    assert make_sampler_calls[0]["top_p"] == 0.9


def test_step_heterogeneous_then_homogeneous_populates_shared(monkeypatch):
    make_sampler_calls = []

    def fake_make_fused_sampler(**kwargs):
        make_sampler_calls.append(kwargs)
        return lambda x: mx.zeros((x.shape[0],), dtype=mx.uint32)

    monkeypatch.setattr(
        "fusion_mlx.mllm_batch_generator.make_fused_sampler",
        fake_make_fused_sampler,
    )

    gen = _make_step_stub_generator()

    MLLMBatchGenerator._step(
        gen,
        mx.array([[1], [2]], dtype=mx.uint32),
        cache=[],
        requests=[
            _make_sampling_request(0, 0.7, 0.95),
            _make_sampling_request(1, 0.3, 0.80),
        ],
    )
    assert gen._shared_batch_sampler is None
    assert len(make_sampler_calls) == 2

    MLLMBatchGenerator._step(
        gen,
        mx.array([[3], [4]], dtype=mx.uint32),
        cache=[],
        requests=[
            _make_sampling_request(2, 0.5, 0.85),
            _make_sampling_request(3, 0.5, 0.85),
        ],
    )
    assert gen._shared_batch_sampler is not None
    assert gen._shared_batch_sampler[0] == (0.5, 0.85, 0, 0.0)
    assert len(make_sampler_calls) == 3


# ---------------------------------------------------------------------------
# Per-batch cap regression -- issue #682
# ---------------------------------------------------------------------------


def _make_cap_request(uid: int, token_count: int) -> MLLMBatchRequest:
    return MLLMBatchRequest(
        uid=uid,
        request_id=f"r{uid}",
        prompt="x",
        max_tokens=8,
        input_ids=mx.zeros((token_count,), dtype=mx.int32),
    )


def _gen_with_prefill_cap(prefill_step_size: int) -> MLLMBatchGenerator:
    gen = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
    gen.prefill_step_size = prefill_step_size
    gen.vision_cache = None
    gen.model = object()
    gen.language_model = object()
    gen.processor = object()
    gen.mm_processor = None

    class _Stats:
        prompt_tokens = 0
        prompt_time = 0.0
        num_images_processed = 0
        vision_encoding_time = 0.0

    gen._stats = _Stats()
    return gen


def test_mllm_scheduler_config_default_prefill_step_size_is_positive():
    from fusion_mlx.mllm_scheduler import MLLMSchedulerConfig

    cfg = MLLMSchedulerConfig()
    assert cfg.prefill_step_size > 0, (
        f"MLLMSchedulerConfig.prefill_step_size default ({cfg.prefill_step_size}) "
        f"must be a positive integer."
    )


def test_resolve_mllm_prefill_step_size_bumps_text_default_to_mllm_default():
    from types import SimpleNamespace

    from fusion_mlx.engine.batched import _resolve_mllm_prefill_step_size
    from fusion_mlx.mllm_scheduler import MLLMSchedulerConfig
    from fusion_mlx.scheduler import SchedulerConfig

    text_default = SchedulerConfig.__dataclass_fields__["prefill_step_size"].default
    mllm_default = MLLMSchedulerConfig.__dataclass_fields__["prefill_step_size"].default

    assert mllm_default != text_default, (
        f"MLLM default ({mllm_default}) must differ from text default "
        f"({text_default}); otherwise the bump is a no-op."
    )

    def _resolved(user_value):
        return _resolve_mllm_prefill_step_size(
            user_value,
            text_default=text_default,
            mllm_default=mllm_default,
        )

    assert _resolved(text_default) == mllm_default, (
        f"text-LLM default ({text_default}) must bump to MLLM default "
        f"({mllm_default}) -- this is the #682 fix for Desktop sidecars."
    )

    for explicit_val in [256, 512, 1024, 1500]:
        if explicit_val != text_default:
            assert _resolved(explicit_val) == explicit_val, (
                f"explicit prefill_step_size={explicit_val} must be "
                f"honored as-is (codex r2 MAJOR); got {_resolved(explicit_val)}"
            )

    for explicit_val in [4096, 8192, 16384, 65536]:
        assert _resolved(explicit_val) == explicit_val, (
            f"explicit prefill_step_size={explicit_val} must be "
            f"honored as-is; got {_resolved(explicit_val)}"
        )

    assert _resolved(None) == mllm_default, (
        "missing attribute / no scheduler_config must default to MLLM-tuned"
    )

    cfg_without_attr = SimpleNamespace()
    resolved_missing = _resolve_mllm_prefill_step_size(
        getattr(cfg_without_attr, "prefill_step_size", None),
        text_default=text_default,
        mllm_default=mllm_default,
    )
    assert resolved_missing == mllm_default

    assert _resolved(text_default) == mllm_default


def test_per_batch_cap_fires_on_oversized_batch_with_actionable_message(
    monkeypatch,
):
    gen = _gen_with_prefill_cap(prefill_step_size=100)

    def _noop_preprocess(req):
        pass

    monkeypatch.setattr(gen, "_preprocess_request", _noop_preprocess)

    request = _make_cap_request(uid=0, token_count=500)

    with pytest.raises(ValueError) as excinfo:
        MLLMBatchGenerator._process_prompts(gen, [request])

    msg = str(excinfo.value)
    assert "exceeds the per-batch cap" in msg, (
        f"cap error must keep the marker substring; got: {msg}"
    )
    assert "downscale the image" in msg, (
        f"cap error must suggest image downscale; got: {msg}"
    )
    assert "--prefill-step-size" in msg, (
        f"cap error must mention --prefill-step-size for the text path; got: {msg}"
    )


def test_per_batch_cap_does_not_fail_at_default_on_typical_screenshot(
    monkeypatch,
):
    gen = _gen_with_prefill_cap(prefill_step_size=8192)

    def _noop_preprocess(req):
        pass

    monkeypatch.setattr(gen, "_preprocess_request", _noop_preprocess)

    request = _make_cap_request(uid=0, token_count=2292)

    try:
        MLLMBatchGenerator._process_prompts(gen, [request])
        err_msg = ""
    except Exception as exc:
        err_msg = str(exc)

    assert "exceeds the per-batch cap" not in err_msg, (
        f"with a 8192 prefill_step_size, a 2292-token "
        f"single-request batch must pass the cap; got: {err_msg}"
    )
