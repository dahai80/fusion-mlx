# SPDX-License-Identifier: Apache-2.0
"""Tests for PR-I: Tree-Mask verify + Draft Virtual Append Offset.

Validates tree causal mask, tree verify (longest accepted path), chain
degenerates to linear first-mismatch (golden), and virtual append offset
commit/rollback semantics.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from fusion_mlx.custom_kernels.paged_kv_cache import FusionPagedKVCache
from fusion_mlx.speculative.dflash.verifier import _decide_accepted_prefix
from fusion_mlx.speculative.tree_mask import (
    DraftTree,
    DraftVirtualAppendOffset,
    apply_tree_mask,
    build_tree_attention_mask,
    is_tree_mask_enabled,
    verify_tree_logits,
)

_ROOT = 99


def _enable(monkeypatch):
    monkeypatch.setenv("FUSION_SHIM_TREE_MASK", "1")


class TestDraftTree:
    def test_chain_tree_topology(self):
        tree = DraftTree.from_chain(_ROOT, [10, 20, 30])
        assert tree.n_nodes == 4
        assert tree.nodes[0].token == _ROOT
        assert tree.nodes[0].parent == -1
        assert tree.nodes[1].parent == 0
        assert tree.nodes[2].parent == 1
        assert tree.nodes[3].parent == 2
        assert tree.tokens() == [_ROOT, 10, 20, 30]

    def test_branch_tree_topology(self):
        tree = DraftTree.from_branches(_ROOT, [1, 2], [[10, 11], [20, 21]])
        # 0=99(root), 1=1, 2=2, 3=10, 4=11, 5=20, 6=21
        assert tree.n_nodes == 7
        assert tree.nodes[0].token == _ROOT
        assert tree.nodes[1].token == 1
        assert tree.nodes[2].token == 2
        assert tree.nodes[3].token == 10
        assert tree.nodes[3].parent == 2
        assert tree.nodes[4].token == 11
        assert tree.nodes[4].parent == 3
        assert tree.nodes[5].token == 20
        assert tree.nodes[5].parent == 2
        assert tree.nodes[6].token == 21
        assert tree.nodes[6].parent == 5

    def test_ancestors(self):
        tree = DraftTree.from_branches(_ROOT, [1, 2], [[10, 11]])
        # 0=99, 1=1, 2=2, 3=10, 4=11
        assert tree.ancestors_of(0) == [0]
        assert tree.ancestors_of(1) == [0, 1]
        assert tree.ancestors_of(4) == [0, 1, 2, 3, 4]

    def test_empty_chain(self):
        tree = DraftTree.from_chain(_ROOT, [])
        assert tree.n_nodes == 1  # root only
        assert tree.tokens() == [_ROOT]


class TestTreeMask:
    def test_chain_mask_is_lower_triangular(self):
        tree = DraftTree.from_chain(_ROOT, [1, 2, 3])
        mask = build_tree_attention_mask(tree)
        mx.eval(mask)
        np_mask = np.array(mask.tolist())
        expected = np.tril(np.ones((4, 4)))
        assert np.allclose(np_mask, expected)

    def test_branch_mask_blocks_siblings(self):
        tree = DraftTree.from_branches(_ROOT, [1, 2], [[10], [20]])
        mask = build_tree_attention_mask(tree)
        mx.eval(mask)
        np_mask = np.array(mask.tolist())
        n = tree.n_nodes
        for i in range(n):
            anc = set(tree.ancestors_of(i))
            for j in range(n):
                assert np_mask[i, j] == (1.0 if j in anc else 0.0)

    def test_apply_mask_sets_neg_inf(self):
        tree = DraftTree.from_chain(_ROOT, [1, 2])
        mask = build_tree_attention_mask(tree)
        scores = mx.zeros((1, 1, 3, 3), dtype=mx.float32)
        masked = apply_tree_mask(scores, mask)
        mx.eval(masked)
        np_masked = np.array(masked.tolist())
        assert np_masked[0, 0, 0, 1] == -float("inf")
        assert np_masked[0, 0, 1, 0] == 0.0
        assert np_masked[0, 0, 1, 1] == 0.0

    def test_root_only_mask(self):
        tree = DraftTree.from_chain(_ROOT, [])
        mask = build_tree_attention_mask(tree)
        assert mask.shape == (1, 1)
        mx.eval(mask)
        assert float(mask[0, 0]) == 1.0


class TestVerifyTreeLogits:
    def test_root_only(self):
        res = verify_tree_logits(DraftTree.from_chain(_ROOT, []), [55])
        assert res.accepted_len == 0
        assert res.bonus_token == 55
        assert res.accepted_tokens == ()

    def test_chain_all_accepted(self):
        # tree: 0=99, 1=10, 2=20, 3=30. argmax[i]=pred after node i.
        # node 1 (10) iff argmax[0]==10; node 2 (20) iff argmax[1]==20;
        # node 3 (30) iff argmax[2]==30.
        tree = DraftTree.from_chain(_ROOT, [10, 20, 30])
        res = verify_tree_logits(tree, [10, 20, 30, 99])
        assert res.accepted_len == 3
        assert res.accepted_tokens == (10, 20, 30)
        assert res.bonus_token == 99

    def test_chain_first_mismatch(self):
        tree = DraftTree.from_chain(_ROOT, [10, 20, 30])
        # node 1 (10) accepted (argmax[0]==10); node 2 (20) iff argmax[1]==20 -> yes;
        # node 3 (30) iff argmax[2]==30 -> no (99).
        res = verify_tree_logits(tree, [10, 20, 99, 77])
        assert res.accepted_len == 2
        assert res.accepted_tokens == (10, 20)
        assert res.bonus_token == 99

    def test_chain_none_accepted(self):
        tree = DraftTree.from_chain(_ROOT, [10, 20])
        # 3 nodes; node 1 (10) iff argmax[0]==10 -> no (99).
        res = verify_tree_logits(tree, [99, 88, 77])
        assert res.accepted_len == 0
        assert res.bonus_token == 99

    def test_branch_picks_longest(self):
        tree = DraftTree.from_branches(_ROOT, [1, 2], [[10, 11], [20, 21]])
        # nodes: 0=99,1=1,2=2,3=10,4=11,5=20,6=21
        # node 1 (1) iff argmax[0]==1; node 2 (2) iff argmax[1]==2;
        # branch A: node 3 (10) iff argmax[2]==10; node 4 (11) iff argmax[3]==11
        # branch B: node 5 (20) iff argmax[2]==20; node 6 (21) iff argmax[5]==21
        # Make B win: argmax[2]=20 (reject A, accept B), argmax[5]=21 (accept 6).
        argmax = [1, 2, 20, 99, 99, 21, 99]
        res = verify_tree_logits(tree, argmax)
        assert res.accepted_len == 4  # shared 1,2 + branch 20,21
        assert res.accepted_tokens == (1, 2, 20, 21)
        assert res.bonus_token == 99

    def test_branch_both_reject_at_first(self):
        tree = DraftTree.from_branches(_ROOT, [1, 2], [[10], [20]])
        # nodes: 0=99,1=1,2=2,3=10,4=20 (5 nodes)
        # shared prefix 1,2 accepted; both branch heads reject (argmax[2]=99).
        argmax = [1, 2, 99, 99, 99]
        res = verify_tree_logits(tree, argmax)
        assert res.accepted_len == 2  # shared prefix 1,2
        assert res.accepted_tokens == (1, 2)
        assert res.bonus_token == 99

    def test_length_mismatch_raises(self):
        tree = DraftTree.from_chain(_ROOT, [1, 2])
        with pytest.raises(ValueError, match="must equal"):
            verify_tree_logits(tree, [1])

    def test_chain_matches_linear_first_mismatch(self):
        # Golden: tree verify on a chain tree must return the same accepted
        # length + bonus as the stock linear _decide_accepted_prefix, for
        # every mismatch case.
        #
        # full_argmax[i] = target prediction after position i, where position 0
        # = last_confirmed (root), position i (i>=1) = draft[i-1].
        # Length = len(draft)+1 == n_tree_nodes (root + n draft).
        # Linear _decide_accepted_prefix receives full_argmax[:n] (drops the
        # post-last-draft pred); its bonus on mismatch = full_argmax[accepted_len].
        # Tree target_argmax = full_argmax (length n+1 == n_nodes). Tree bonus
        # on mismatch = target_argmax[leaf] = full_argmax[accepted_len] ==
        # linear bonus.
        draft = [10, 20, 30, 40]
        # mismatch at index 2: draft[2]=30 but full_argmax[2]=99.
        full_argmax = [10, 20, 99, 40, 77]
        lin_len, lin_bonus = _decide_accepted_prefix(full_argmax[:4], draft)
        assert lin_len == 2
        assert lin_bonus == 99
        tree = DraftTree.from_chain(_ROOT, draft)
        res = verify_tree_logits(tree, full_argmax)
        assert res.accepted_len == lin_len
        assert res.bonus_token == lin_bonus

    def test_chain_matches_linear_first_mismatch_at_zero(self):
        draft = [10, 20, 30]
        full_argmax = [99, 20, 30, 77]
        lin_len, lin_bonus = _decide_accepted_prefix(full_argmax[:3], draft)
        assert lin_len == 0
        assert lin_bonus == 99
        tree = DraftTree.from_chain(_ROOT, draft)
        res = verify_tree_logits(tree, full_argmax)
        assert res.accepted_len == lin_len
        assert res.bonus_token == lin_bonus

    def test_chain_matches_linear_mid_mismatch(self):
        draft = [10, 20, 30, 40, 50]
        full_argmax = [10, 20, 30, 99, 50, 88]
        lin_len, lin_bonus = _decide_accepted_prefix(full_argmax[:5], draft)
        tree = DraftTree.from_chain(_ROOT, draft)
        res = verify_tree_logits(tree, full_argmax)
        assert res.accepted_len == lin_len
        assert res.bonus_token == lin_bonus


class TestDraftVirtualAppendOffset:
    def _kv(self, steps=4, dtype=mx.float32):
        k = mx.array(np.random.randn(1, 2, steps, 8).astype(np.float32)).astype(dtype)
        v = mx.array(np.random.randn(1, 2, steps, 8).astype(np.float32)).astype(dtype)
        return k, v

    def test_disabled_propose_writes_real(self):
        cache = FusionPagedKVCache(block_size=4, num_blocks=8)
        voff = DraftVirtualAppendOffset(cache)
        assert is_tree_mask_enabled() is False
        k, v = self._kv(steps=4)
        voff.propose(k, v)
        mx.eval([cache.keys_pool, cache.values_pool])
        assert cache.offset == 4
        assert voff.virtual_offset == 4

    def test_enabled_propose_does_not_move_real(self, monkeypatch):
        _enable(monkeypatch)
        cache = FusionPagedKVCache(block_size=4, num_blocks=8)
        voff = DraftVirtualAppendOffset(cache)
        k, v = self._kv(steps=4)
        virt = voff.propose(k, v)
        assert cache.offset == 0
        assert virt == 4
        assert voff.staged_steps == 4

    def test_commit_advances_real(self, monkeypatch):
        _enable(monkeypatch)
        cache = FusionPagedKVCache(block_size=4, num_blocks=8)
        voff = DraftVirtualAppendOffset(cache)
        k, v = self._kv(steps=4)
        voff.propose(k, v)
        voff.commit(2)
        mx.eval([cache.keys_pool, cache.values_pool])
        assert cache.offset == 2
        assert voff.staged_steps == 0

    def test_commit_zero_is_noop(self, monkeypatch):
        _enable(monkeypatch)
        cache = FusionPagedKVCache(block_size=4, num_blocks=8)
        voff = DraftVirtualAppendOffset(cache)
        k, v = self._kv(steps=4)
        voff.propose(k, v)
        voff.commit(0)
        assert cache.offset == 0

    def test_rollback_discards_staged(self, monkeypatch):
        _enable(monkeypatch)
        cache = FusionPagedKVCache(block_size=4, num_blocks=8)
        voff = DraftVirtualAppendOffset(cache)
        k, v = self._kv(steps=4)
        voff.propose(k, v)
        voff.rollback()
        assert cache.offset == 0
        assert voff.staged_steps == 0
        assert voff.virtual_offset == 0

    def test_rollback_then_real_cache_usable(self, monkeypatch):
        _enable(monkeypatch)
        cache = FusionPagedKVCache(block_size=4, num_blocks=8)
        voff = DraftVirtualAppendOffset(cache)
        k, v = self._kv(steps=4)
        voff.propose(k, v)
        voff.rollback()
        k2, v2 = self._kv(steps=3)
        out = cache.update_and_fetch(k2, v2)
        mx.eval([out[0], out[1]])
        assert cache.offset == 3

    def test_commit_all_then_stats(self, monkeypatch):
        _enable(monkeypatch)
        cache = FusionPagedKVCache(block_size=4, num_blocks=8)
        voff = DraftVirtualAppendOffset(cache)
        k, v = self._kv(steps=4)
        voff.propose(k, v)
        voff.commit(4)
        s = voff.stats()
        assert s["committed_total"] == 4
        assert s["rolled_back_total"] == 0
        assert s["tree_mask_enabled"] is True
        assert s["real_offset"] == 4

    def test_commit_exceeds_staged_clamps(self, monkeypatch):
        _enable(monkeypatch)
        cache = FusionPagedKVCache(block_size=4, num_blocks=8)
        voff = DraftVirtualAppendOffset(cache)
        k, v = self._kv(steps=2)
        voff.propose(k, v)
        voff.commit(10)
        mx.eval([cache.keys_pool, cache.values_pool])
        assert cache.offset == 2
