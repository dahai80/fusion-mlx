# SPDX-License-Identifier: Apache-2.0
"""N-gram spec rejection-path correctness (GDN-safe fix).

Locks the two fixes that made n-gram spec coherent on hybrid GDN models:
  1. resample_idx = max(0, n_accepted - 1) — the bonus token is the model's
     prediction AFTER the last accepted draft, not after the first rejected
     one. The old ``min(n_accepted, K-1)`` picked the pred after the rejected
     draft and corrupted the bonus on every rejection.
  2. _rollback_spec_cache trims trimmable caches (KVCache) by K, leaving
     exactly n_accepted entries. ``mlx_cache.trim_prompt_cache`` is a no-op
     for hybrid caches (ArraysCache is non-trimmable) and trimmed
     ``n_rejected`` instead of ``K`` (the replay writes n_accepted duplicates
     that must also be dropped).
"""
import mlx.core as mx
from mlx_lm.models.cache import ArraysCache, KVCache

from fusion_mlx.scheduler.ngram_spec import (
    _rollback_spec_cache,
    _snapshot_non_trimmable,
    _trim_trimmable,
    _verify_drafts,
)


class _MockModel:
    """Model stand-in: updates caches minimally and returns controlled logits.

    argmax_per_pos[i] is the token id the model 'predicts' at output position i
    (the prediction made AFTER consuming draft_tokens[i]).
    """

    def __init__(self, argmax_per_pos):
        self.argmax_per_pos = list(argmax_per_pos)
        self.vocab = max(self.argmax_per_pos, default=0) + 16
        self.calls = []

    def __call__(self, inputs, cache=None):
        toks = inputs.squeeze(0).tolist()
        s = len(toks)
        self.calls.append(toks)
        for c in cache or []:
            if isinstance(c, KVCache):
                c.offset += s
            elif isinstance(c, ArraysCache):
                marker = ("replay" if s == 1 else "verify", tuple(toks))
                c.cache = [marker] * len(c.cache)
        logits = mx.full((1, s, self.vocab), -1.0)
        for i in range(s):
            logits[0, i, self.argmax_per_pos[i]] = 1.0
        return logits


def test_resample_bonus_is_pred_after_last_accepted_partial():
    # drafts [D1=10, D2=20, D3=30]; D1 accepted, D2 rejected.
    # Bonus must be sampled[0]=99 (pred after last accepted D1), not 88.
    model = _MockModel(argmax_per_pos=[99, 88, 77])
    cache = [KVCache()]
    verified, n_accepted, proc = _verify_drafts(
        model, [10, 20, 30], cache, sampled_from_regular=10
    )
    assert n_accepted == 1
    assert proc == 3
    assert verified == [10, 99], f"expected [10, 99], got {verified}"


def test_resample_bonus_is_pred_after_last_accepted_two_accepted():
    # D1, D2 accepted; D3 rejected. Bonus = sampled[1]=88 (pred after D2).
    model = _MockModel(argmax_per_pos=[20, 88, 77])
    cache = [KVCache()]
    verified, n_accepted, _ = _verify_drafts(
        model, [10, 20, 30], cache, sampled_from_regular=10
    )
    assert n_accepted == 2
    assert verified == [10, 20, 88], f"expected [10, 20, 88], got {verified}"


def test_resample_bonus_full_acceptance():
    # All accepted. Bonus = sampled[K-1] = pred after last draft.
    model = _MockModel(argmax_per_pos=[20, 30, 99])
    cache = [KVCache()]
    verified, n_accepted, _ = _verify_drafts(
        model, [10, 20, 30], cache, sampled_from_regular=10
    )
    assert n_accepted == 3
    assert verified == [10, 20, 30, 99]


def test_resample_d1_mismatch_returns_regular_pred():
    # D1 != sampled_from_regular: no verify, bonus = regular prediction.
    model = _MockModel(argmax_per_pos=[99, 88, 77])
    cache = [KVCache()]
    verified, n_accepted, proc = _verify_drafts(
        model, [10, 20, 30], cache, sampled_from_regular=42
    )
    assert n_accepted == 0
    assert proc == 0
    assert verified == [42]
    assert model.calls == [], "verify must not run when D1 mismatches regular"


def test_trim_trimmable_skips_non_trimmable():
    kv = KVCache()
    kv.offset = 10
    arr = ArraysCache(2)
    arr.cache = ["state", "state"]
    _trim_trimmable([kv, arr], 4)
    assert kv.offset == 6
    assert arr.cache == ["state", "state"]


def test_rollback_leaves_kvcache_with_only_accepted_drafts():
    # Hybrid cache: KVCache, ArraysCache, KVCache. Pre-verify KV offset = 5.
    kv0, kv1 = KVCache(), KVCache()
    kv0.offset = 5
    kv1.offset = 5
    arr = ArraysCache(2)
    cache = [kv0, arr, kv1]

    snaps = _snapshot_non_trimmable(cache)
    model = _MockModel(argmax_per_pos=[99, 88, 77])
    drafts = [10, 20, 30]
    verified, n_accepted, proc = _verify_drafts(
        model, drafts, cache, sampled_from_regular=10
    )
    assert n_accepted == 1 and proc == 3

    _rollback_spec_cache(
        cache, snaps, drafts, n_accepted, proc, model, mx.cpu
    )

    # KVCache: pre(5) + K(3) verify + 1 replay duplicate - trim(3) = 5 + 1 = 6.
    assert cache[0].offset == 6, f"kv0 offset={cache[0].offset}, want 6"
    assert cache[2].offset == 6, f"kv1 offset={cache[2].offset}, want 6"
    # ArraysCache restored from snapshot, then replay advanced it.
    assert cache[1].cache[0] is not None, "ArraysCache should reflect replay"
    assert [10] in model.calls, f"replay with [10] missing; calls={model.calls}"


def test_rollback_zero_accepted_restores_pre_verify():
    # n_accepted=0: no replay, trim by K restores KVCache to pre-verify.
    kv = KVCache()
    kv.offset = 5
    arr = ArraysCache(2)
    cache = [kv, arr]
    snaps = _snapshot_non_trimmable(cache)
    model = _MockModel(argmax_per_pos=[99, 88])
    drafts = [10, 20]
    # sampled_from_regular=None -> n_accepted starts at 0; sampled[0]=99 != 20
    verified, n_accepted, proc = _verify_drafts(model, drafts, cache)
    assert n_accepted == 0 and proc == 2

    _rollback_spec_cache(cache, snaps, drafts, n_accepted, proc, model, mx.cpu)

    assert cache[0].offset == 5, f"kv offset={cache[0].offset}, want 5"
    assert cache[1].cache == [None, None], "ArraysCache should be restored only"
    assert not any(len(c) == 1 for c in model.calls), (
        f"no single-token replay should run when n_accepted=0; calls={model.calls}"
    )
