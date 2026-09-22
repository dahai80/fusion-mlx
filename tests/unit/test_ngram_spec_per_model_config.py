# SPDX-License-Identifier: Apache-2.0
"""Per-model n-gram spec config: NGramSpecState ctor override + break-even default."""

import logging

from fusion_mlx.scheduler.ngram_spec import (
    NGRAM_SPEC_DEFAULT_BREAK_EVEN,
    NGRAM_SPEC_NUM_DRAFT,
    NGRAM_SPEC_ORDER,
    NGramSpecState,
    SuffixDrafterAdapter,
)

logger = logging.getLogger(__name__)


def test_default_ctor_uses_suffix_backend():
    state = NGramSpecState()
    assert isinstance(state.predictor, SuffixDrafterAdapter)
    assert state.predictor.num_draft == NGRAM_SPEC_NUM_DRAFT
    assert state._break_even_default == NGRAM_SPEC_DEFAULT_BREAK_EVEN


def test_ngram_ctor_uses_module_defaults(monkeypatch):
    monkeypatch.setenv("FUSION_NGRAM_SPEC_PREDICTOR", "ngram")
    state = NGramSpecState()
    assert state.predictor.order == NGRAM_SPEC_ORDER
    assert state.predictor.num_draft == NGRAM_SPEC_NUM_DRAFT
    assert state._break_even_default == NGRAM_SPEC_DEFAULT_BREAK_EVEN


def test_ctor_kwargs_override_predictor_and_break_even(monkeypatch):
    monkeypatch.setenv("FUSION_NGRAM_SPEC_PREDICTOR", "ngram")
    state = NGramSpecState(order=3, num_draft=2, break_even=0.7)
    assert state.predictor.order == 3
    assert state.predictor.num_draft == 2
    assert state._break_even_default == 0.7


def test_ctor_none_falls_back_to_defaults(monkeypatch):
    monkeypatch.setenv("FUSION_NGRAM_SPEC_PREDICTOR", "ngram")
    state = NGramSpecState(order=None, num_draft=None, break_even=None)
    assert state.predictor.order == NGRAM_SPEC_ORDER
    assert state.predictor.num_draft == NGRAM_SPEC_NUM_DRAFT
    assert state._break_even_default == NGRAM_SPEC_DEFAULT_BREAK_EVEN


def test_suffix_ctor_honors_explicit_order():
    state = NGramSpecState(order=3, num_draft=2)
    assert isinstance(state.predictor, SuffixDrafterAdapter)
    assert state.predictor.order == 3
    assert state.predictor.num_draft == 2


def test_break_even_returns_custom_default_before_timings():
    state = NGramSpecState(break_even=0.65)
    assert state._verify_dt_ema is None
    assert state._decode_dt_ema is None
    assert state._break_even() == 0.65


def test_break_even_returns_module_default_when_not_overridden():
    state = NGramSpecState()
    assert state._break_even() == NGRAM_SPEC_DEFAULT_BREAK_EVEN


def test_explicit_break_even_zero_is_respected():
    state = NGramSpecState(break_even=0.0)
    assert state._break_even_default == 0.0
    assert state._break_even() == 0.0
