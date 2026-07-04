# SPDX-License-Identifier: Apache-2.0
"""Recurrent-cache gate for speculative decode.

Hybrid models (GatedDeltaNet / Mamba / ArraysCache layers) hold sequential
state that the batched spec-verify forward computes in parallel, derailing
generation into repetition. ``model_has_recurrent_cache`` is the single
probe used by both the config-level gate (``enrich_model_config``) and the
boot-level gate (``engine_core``) to disable spec decode for these models.
"""
from fusion_mlx.model_auto_config import (
    ModelConfig,
    enrich_model_config,
    model_has_recurrent_cache,
)
from mlx_lm.models.cache import ArraysCache, KVCache


class _RecurrentModel:
    """Stand-in for a hybrid model whose make_cache() returns recurrent state."""

    def make_cache(self):
        return [ArraysCache(48), ArraysCache(48), KVCache()]


class _AttentionModel:
    """Stand-in for a pure-attention model (all trimmable KV caches)."""

    def make_cache(self):
        return [KVCache() for _ in range(28)]


class _NoCacheModel:
    """Model without make_cache (e.g. an embedding model)."""

    pass


class _BrokenModel:
    """Model whose make_cache() raises (must not crash the probe)."""

    def make_cache(self):
        raise RuntimeError("boom")


def test_recurrent_model_detected():
    assert model_has_recurrent_cache(_RecurrentModel()) is True


def test_attention_model_not_detected():
    assert model_has_recurrent_cache(_AttentionModel()) is False


def test_no_make_cache_is_false():
    assert model_has_recurrent_cache(_NoCacheModel()) is False


def test_probe_failure_is_false_not_raise():
    assert model_has_recurrent_cache(_BrokenModel()) is False


def test_enrich_disables_spec_for_recurrent():
    cfg = ModelConfig(is_hybrid=False, supports_spec_decode=True)
    out = enrich_model_config(cfg, _RecurrentModel())
    assert out.supports_spec_decode is False


def test_enrich_keeps_spec_for_attention():
    cfg = ModelConfig(is_hybrid=False, supports_spec_decode=True)
    out = enrich_model_config(cfg, _AttentionModel())
    assert out.supports_spec_decode is True


def test_enrich_promotes_recurrent_to_hybrid():
    cfg = ModelConfig(is_hybrid=False, supports_spec_decode=True)
    out = enrich_model_config(cfg, _RecurrentModel())
    assert out.is_hybrid is True
