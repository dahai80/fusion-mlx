# SPDX-License-Identifier: Apache-2.0
"""AWSD Phase A: MTP chain-of-K partial-accept rollback unit tests.

Verifies the per-position GDN snapshot selection (``rollback_state_list``)
and the chain-aware ``_restore_or_trim_caches`` trim math that K>1
partial-accept rollback depends on. Pure logic — no model loaded.
"""

from fusion_mlx.patches.mlx_lm_mtp.batch_generator import (
    _has_rollback_for,
    _restore_or_trim_caches,
    _rollback_snapshot_for,
)


class _FakeSSMCache:
    """Stand-in for ArraysCache (GatedDeltaNet recurrent state)."""

    def __init__(self):
        self.rollback_state = None
        self.rollback_state_list = None
        self[0] = "conv_init"
        self[1] = "ssm_init"

    def __getitem__(self, i):
        return getattr(self, f"_slot{i}")

    def __setitem__(self, i, v):
        setattr(self, f"_slot{i}", v)

    def advance(self, n):
        pass


class _FakeKVCache:
    """Stand-in for a trimmable full-attention KV cache."""

    def __init__(self):
        self.trimmed = 0

    def is_trimmable(self):
        return True

    def trim(self, n):
        self.trimmed += n
        return n


def test_snapshot_pre_draft_for_accepted_zero():
    c = _FakeSSMCache()
    c.rollback_state = ("conv_pre", "ssm_pre")
    assert _rollback_snapshot_for(c, accepted=0) == ("conv_pre", "ssm_pre")


def test_snapshot_list_index_for_partial_accept():
    c = _FakeSSMCache()
    c.rollback_state = ("conv_pre", "ssm_pre")
    c.rollback_state_list = [
        ("conv_after_d0", "ssm_after_d0"),
        ("conv_after_d1", "ssm_after_d1"),
    ]
    # accepted=1 → restore state after d0 (index 0)
    assert _rollback_snapshot_for(c, accepted=1) == ("conv_after_d0", "ssm_after_d0")
    # accepted=2 → restore state after d1 (index 1)
    assert _rollback_snapshot_for(c, accepted=2) == ("conv_after_d1", "ssm_after_d1")


def test_snapshot_falls_back_to_pre_draft_when_list_missing():
    c = _FakeSSMCache()
    c.rollback_state = ("conv_pre", "ssm_pre")
    c.rollback_state_list = None
    # K=1 back-compat: accepted>=1 with no list → pre-draft snapshot
    assert _rollback_snapshot_for(c, accepted=1) == ("conv_pre", "ssm_pre")


def test_snapshot_out_of_range_returns_pre_draft():
    c = _FakeSSMCache()
    c.rollback_state = ("conv_pre", "ssm_pre")
    c.rollback_state_list = [("conv_after_d0", "ssm_after_d0")]
    # accepted=2 but list only has 1 entry → fall back to pre-draft
    assert _rollback_snapshot_for(c, accepted=2) == ("conv_pre", "ssm_pre")


def test_has_rollback_for_dispatch():
    c = _FakeSSMCache()
    c.rollback_state = ("conv_pre", "ssm_pre")
    assert _has_rollback_for(c, accepted=0)
    assert _has_rollback_for(c, accepted=1)  # falls back to rollback_state


def test_restore_k1_zero_accept_trims_one():
    ssm = _FakeSSMCache()
    ssm.rollback_state = ("conv_pre", "ssm_pre")
    kv = _FakeKVCache()
    ok = _restore_or_trim_caches([ssm, kv], accepted=0, block_size=2)
    assert ok
    assert ssm[0] == "conv_pre"
    assert ssm[1] == "ssm_pre"
    assert ssm.rollback_state is None
    assert kv.trimmed == 1  # block_size - (accepted+1) = 2 - 1


def test_restore_k2_full_reject_trims_two():
    ssm = _FakeSSMCache()
    ssm.rollback_state = ("conv_pre", "ssm_pre")
    kv = _FakeKVCache()
    ok = _restore_or_trim_caches([ssm, kv], accepted=0, block_size=3)
    assert ok
    assert ssm[0] == "conv_pre"
    assert kv.trimmed == 2  # 3 - 1


def test_restore_k2_partial_accept_trims_one():
    ssm = _FakeSSMCache()
    ssm.rollback_state = ("conv_pre", "ssm_pre")
    ssm.rollback_state_list = [
        ("conv_after_d0", "ssm_after_d0"),
        ("conv_after_d1", "ssm_after_d1"),
    ]
    kv = _FakeKVCache()
    # K=2, draft 0 accepted, draft 1 rejected → accepted=1
    ok = _restore_or_trim_caches([ssm, kv], accepted=1, block_size=3)
    assert ok
    assert ssm[0] == "conv_after_d0"
    assert ssm[1] == "ssm_after_d0"
    assert ssm.rollback_state is None
    assert ssm.rollback_state_list is None
    assert kv.trimmed == 1  # 3 - (1+1)


def test_restore_k2_all_accepted_not_a_rollback_case():
    # accepted=2 (all accepted) is handled by _clear_rollback in the accept
    # path, NOT _restore_or_trim_caches. But if called, n_to_trim = 3-3 = 0
    # → KV trim 0, SSM restore snapshot_for(accepted=2) = list[1] = final.
    ssm = _FakeSSMCache()
    ssm.rollback_state = ("conv_pre", "ssm_pre")
    ssm.rollback_state_list = [
        ("conv_after_d0", "ssm_after_d0"),
        ("conv_after_d1", "ssm_after_d1"),
    ]
    kv = _FakeKVCache()
    ok = _restore_or_trim_caches([ssm, kv], accepted=2, block_size=3)
    assert ok
    assert kv.trimmed == 0


def test_restore_rejects_when_n_to_trim_negative():
    ssm = _FakeSSMCache()
    kv = _FakeKVCache()
    # accepted=3, block_size=3 → n_to_trim = -1 → invalid
    ok = _restore_or_trim_caches([ssm, kv], accepted=3, block_size=3)
    assert not ok


def test_restore_rejects_when_layer_supports_neither():
    class _Bare:
        pass

    ssm = _FakeSSMCache()
    ssm.rollback_state = ("conv_pre", "ssm_pre")
    bare = _Bare()
    ok = _restore_or_trim_caches([ssm, bare], accepted=0, block_size=2)
    assert not ok
