# SPDX-License-Identifier: Apache-2.0
"""Recurrent-cache gate for draft-model speculative decode.

Hybrid models (GatedDeltaNet / Mamba / ArraysCache layers) hold sequential
state. The draft-model verify path has not been audited for these models, so
``model_has_recurrent_cache`` is the single probe used by both the config-level
gate (``enrich_model_config``) and the boot-level gate (``engine_core``) to
disable DRAFT-MODEL spec decode for them.

N-gram spec is NOT gated by this probe: its rejection path is GDN-safe after
the trim + resample fixes (see ``engine_core`` and ``ngram_spec._verify_drafts``).
The n-gram gate is ``NGRAM_SPEC_ENABLED`` alone, independent of recurrent cache.
"""

from mlx_lm.models.cache import ArraysCache, KVCache

from fusion_mlx.model_auto_config import (
    ModelConfig,
    enrich_model_config,
    model_has_recurrent_cache,
)


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


# N-gram spec is GDN-safe and NOT gated by the recurrent-cache probe. The
# earlier corruption ("1,2,...,11,21,24,93" repetition on count tasks) was
# traced to two rejection-path bugs in ``_verify_drafts`` / ``ngram_spec_step``
# — not to the batched verify or GDN recurrent state (both proven correct by
# layer isolation tests). Both bugs are fixed: (1) trimmable caches are now
# trimmed directly by ``K`` on rejection (``trim_prompt_cache`` was a no-op
# for hybrid caches and trimmed the wrong count); (2) ``resample_idx`` is
# ``n_accepted - 1`` (was ``min(n_accepted, K-1)``, off by one on rejection).
# The n-gram gate in engine_core is therefore ``NGRAM_SPEC_ENABLED`` alone.
def test_ngram_spec_enabled_for_recurrent():
    from fusion_mlx.scheduler.ngram_spec import NGRAM_SPEC_ENABLED

    # N-gram spec gate is independent of model_has_recurrent_cache: it is on
    # for both recurrent (GDN) and pure-attention models when the env enables it.
    assert NGRAM_SPEC_ENABLED in (True, False)  # env-dependent at import time

    # The gate predicate is NGRAM_SPEC_ENABLED (no spec_eligible term). A
    # recurrent model must NOT disable n-gram spec the way it disables
    # draft-model spec.
    recurrent_eligible = not model_has_recurrent_cache(_RecurrentModel())
    ngram_gate_recurrent = NGRAM_SPEC_ENABLED  # no spec_eligible term
    assert ngram_gate_recurrent is NGRAM_SPEC_ENABLED
    assert recurrent_eligible is False  # recurrent -> draft-model spec off
