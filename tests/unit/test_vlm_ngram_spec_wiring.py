# SPDX-License-Identifier: Apache-2.0
# Regression: the VLM engine never loaded _ngram_spec_state (only the text
# engine batched.py did), so the per-request router found no loaded method and
# VLM text decode never speculated. _apply_ngram_spec() ports the text engine's
# loader so n-gram self-spec decode (the draft-model-free spec method, safe for
# VLM since it matches the generated TEXT token stream, not image placeholders)
# can be enabled per-VLM-model via model_settings.ngram_spec_enabled.
from types import SimpleNamespace

from fusion_mlx.engines.vlm import VLMBatchedEngine


def _make_engine(model_settings, scheduler=None):
    engine = VLMBatchedEngine.__new__(VLMBatchedEngine)
    engine._model_settings = model_settings
    engine._model_name = "test_vlm"
    sched = scheduler if scheduler is not None else SimpleNamespace()
    engine._engine = SimpleNamespace(engine=SimpleNamespace(scheduler=sched))
    return engine


def test_enabled_loads_ngram_spec_state():
    sched = SimpleNamespace()
    engine = _make_engine(
        SimpleNamespace(
            ngram_spec_enabled=True,
            ngram_spec_order=3,
            ngram_spec_num_draft=4,
            ngram_spec_break_even=0.5,
        ),
        scheduler=sched,
    )
    engine._apply_ngram_spec()
    assert sched._ngram_spec_state is not None
    # order/num_draft are forwarded onto the predictor.
    assert sched._ngram_spec_state.predictor.order == 3
    assert sched._ngram_spec_state.predictor.num_draft == 4


def test_disabled_clears_ngram_spec_state():
    sched = SimpleNamespace()
    sched._ngram_spec_state = "preexisting"
    engine = _make_engine(
        SimpleNamespace(ngram_spec_enabled=False),
        scheduler=sched,
    )
    engine._apply_ngram_spec()
    assert sched._ngram_spec_state is None


def test_absent_setting_is_noop():
    # ngram_spec_enabled defaults None -> do not touch the scheduler (mirrors
    # the text engine, so default VLM loads are unaffected).
    sched = SimpleNamespace()
    engine = _make_engine(
        SimpleNamespace(ngram_spec_enabled=None),
        scheduler=sched,
    )
    engine._apply_ngram_spec()
    assert not hasattr(sched, "_ngram_spec_state")


def test_none_model_settings_is_noop():
    sched = SimpleNamespace()
    engine = _make_engine(None, scheduler=sched)
    engine._apply_ngram_spec()
    assert not hasattr(sched, "_ngram_spec_state")


def test_defaults_when_order_num_draft_unset():
    sched = SimpleNamespace()
    engine = _make_engine(
        SimpleNamespace(
            ngram_spec_enabled=True,
            ngram_spec_order=None,
            ngram_spec_num_draft=None,
            ngram_spec_break_even=None,
        ),
        scheduler=sched,
    )
    engine._apply_ngram_spec()
    assert sched._ngram_spec_state is not None


def test_scheduler_missing_is_swallowed_and_logged(caplog):
    engine = VLMBatchedEngine.__new__(VLMBatchedEngine)
    engine._model_settings = SimpleNamespace(ngram_spec_enabled=True)
    engine._model_name = "test_vlm"
    engine._engine = SimpleNamespace(engine=SimpleNamespace())
    engine._apply_ngram_spec()
    assert any("N-gram spec init failed" in r.message for r in caplog.records)


def test_router_assigns_ngram_to_request_without_vlm_guard():
    # The dispatch path is engine-type-agnostic: given ngram is the only loaded
    # method, the router must return METHOD_NGRAM for ANY request (including a
    # VLM-shaped one). This is what makes VLM spec decode work once the runtime
    # is loaded - there is no is_vlm guard to remove.
    from fusion_mlx.speculative.per_request_route import (
        loaded_methods,
        select_active_method,
    )

    loaded = loaded_methods(suffix=True)
    method = select_active_method(
        prompt_token_count=128,
        loaded=loaded,
        has_mtp=False,
    )
    from fusion_mlx.speculative.auto_router import METHOD_NGRAM

    assert method == METHOD_NGRAM
