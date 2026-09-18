# SPDX-License-Identifier: Apache-2.0
"""PR-N tests: MoE route dispatch + gather combine vs numpy reference."""

import numpy as np
import pytest

mlx = pytest.importorskip("mlx.core")
mx = mlx

from fusion_mlx.shim.moe_dispatch import (  # noqa: E402
    DispatchPlan,
    aligned_offsets,
    expert_load_stats,
    gather_combine,
    gather_rows,
    is_moe_dispatch_enabled,
    mul_mat_id,
    route_dispatch,
)

RNG = np.random.default_rng(20260918)


def _plan(inds, num_experts):
    return route_dispatch(mx.array(inds, dtype=mx.int32), num_experts)


def _reference_dispatch(inds, num_experts):
    flat = inds.reshape(-1).astype(np.int64)
    order = np.argsort(flat, kind="stable")
    counts = np.bincount(flat, minlength=num_experts).astype(np.int64)
    offsets = np.concatenate([[0], np.cumsum(counts)[:-1]])
    return flat, order, counts, offsets


class TestRouteDispatch:
    def test_order_is_stable_by_expert(self):
        inds = np.array([[3, 1, 3], [0, 1, 0]], dtype=np.int64)
        plan = _plan(inds, 4)
        flat, order, _, _ = _reference_dispatch(inds, 4)
        np.testing.assert_array_equal(np.array(plan.order), order)
        np.testing.assert_array_equal(np.array(plan.sorted_expert_ids), flat[order])

    def test_counts_and_offsets(self):
        inds = np.array([[2, 0, 2], [2, 1, 0]], dtype=np.int64)
        plan = _plan(inds, 3)
        _, _, counts, offsets = _reference_dispatch(inds, 3)
        np.testing.assert_array_equal(np.array(plan.counts), counts)
        np.testing.assert_array_equal(np.array(plan.offsets), offsets)

    def test_sorted_token_ids_match_order(self):
        inds = RNG.integers(0, 8, size=(16, 4))
        plan = _plan(inds, 8)
        flat, order, _, _ = _reference_dispatch(inds, 8)
        np.testing.assert_array_equal(
            np.array(plan.sorted_token_ids), order // inds.shape[1]
        )
        # expert ids consistent with source rows
        np.testing.assert_array_equal(np.array(plan.sorted_expert_ids), flat[order])

    def test_inverse_restores_original_order(self):
        inds = RNG.integers(0, 5, size=(9, 3))
        plan = _plan(inds, 5)
        order = np.array(plan.order)
        inverse = np.array(plan.inverse)
        np.testing.assert_array_equal(order[inverse], np.arange(order.size))

    def test_counts_sum_to_pairs(self):
        inds = RNG.integers(0, 6, size=(32, 2))
        plan = _plan(inds, 6)
        assert int(plan.counts.sum()) == inds.size

    def test_rejects_bad_shapes(self):
        with pytest.raises(ValueError):
            route_dispatch(mx.array(1.0), 4)
        with pytest.raises(ValueError):
            route_dispatch(mx.zeros((0,), dtype=mx.int32), 4)

    def test_no_expert_gap(self):
        # expert ids beyond num_experts excluded by construction (caller's
        # num_experts), all ids < num_experts land in counts
        inds = np.array([[7, 7, 7]], dtype=np.int64)
        plan = _plan(inds, 8)
        assert int(plan.counts[7]) == 3


class TestGatherRows:
    def test_rows_grouped_by_expert(self):
        x = RNG.normal(size=(2, 8)).astype(np.float32)
        inds = np.array([[2, 0, 1], [0, 2, 1]], dtype=np.int64)
        plan = _plan(inds, 3)
        got = np.array(gather_rows(mx.array(x), plan))
        flat = inds.reshape(-1)
        order = np.argsort(flat, kind="stable")
        np.testing.assert_allclose(got, x[order // 3], rtol=1e-6)

    def test_shape_mismatch_raises(self):
        plan = _plan(np.array([[0, 1]], dtype=np.int64), 2)
        with pytest.raises(ValueError):
            gather_rows(mx.zeros((5, 4)), plan)
        with pytest.raises(ValueError):
            gather_rows(mx.zeros((2, 4, 4)), plan)


class TestMulMatId:
    def test_matches_per_expert_reference(self):
        n_tok, k, d_in, d_out, n_exp = 12, 3, 16, 24, 4
        x = RNG.normal(size=(n_tok, d_in)).astype(np.float32)
        w = RNG.normal(size=(n_exp, d_out, d_in)).astype(np.float32)
        inds = RNG.integers(0, n_exp, size=(n_tok, k))
        plan = _plan(inds, n_exp)
        xs = gather_rows(mx.array(x), plan)
        got = np.array(mul_mat_id(xs, mx.array(w), plan))
        flat = inds.reshape(-1)
        order = np.argsort(flat, kind="stable")
        ref = np.stack([w[flat[p]] @ x[p // k] for p in order])
        np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-4)

    def test_weight_shape_mismatch_raises(self):
        plan = _plan(np.array([[0, 1]], dtype=np.int64), 2)
        with pytest.raises(ValueError):
            mul_mat_id(mx.zeros((2, 4)), mx.zeros((2, 4)), plan)


class TestGatherCombine:
    def test_matches_direct_reference(self):
        n_tok, k, d, n_exp = 10, 2, 8, 3
        scores = RNG.uniform(0.1, 1.0, size=(n_tok, k)).astype(np.float32)
        inds = RNG.integers(0, n_exp, size=(n_tok, k))
        plan = _plan(inds, n_exp)
        # expert outputs depend on expert id so restore order matters
        pair_expert = inds.reshape(-1)
        sorted_y = np.zeros((n_tok * k, d), dtype=np.float32)
        order = np.argsort(pair_expert.reshape(-1), kind="stable")
        base = RNG.normal(size=(n_exp, d)).astype(np.float32)
        sorted_y = base[pair_expert[order]]
        got = np.array(gather_combine(mx.array(sorted_y), mx.array(scores), plan))
        ref = (scores[:, :, None] * base[pair_expert].reshape(n_tok, k, d)).sum(axis=1)
        np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-5)

    def test_full_roundtrip_with_matmul(self):
        n_tok, k, d_in, d_out, n_exp = 16, 2, 12, 12, 4
        x = RNG.normal(size=(n_tok, d_in)).astype(np.float32)
        w = RNG.normal(size=(n_exp, d_out, d_in)).astype(np.float32)
        scores = RNG.uniform(0.1, 1.0, size=(n_tok, k)).astype(np.float32)
        inds = RNG.integers(0, n_exp, size=(n_tok, k))
        plan = _plan(inds, n_exp)
        xs = gather_rows(mx.array(x), plan)
        ys = mul_mat_id(xs, mx.array(w), plan)
        got = np.array(gather_combine(ys, mx.array(scores), plan))
        # reference: direct per-token weighted sum
        ref = np.zeros((n_tok, d_out), dtype=np.float32)
        for t in range(n_tok):
            for j in range(k):
                ref[t] += scores[t, j] * (w[inds[t, j]] @ x[t])
        np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-4)

    def test_row_count_mismatch_raises(self):
        plan = _plan(np.array([[0, 1]], dtype=np.int64), 2)
        with pytest.raises(ValueError):
            gather_combine(mx.zeros((5, 4)), mx.ones((1, 2)), plan)
        with pytest.raises(ValueError):
            gather_combine(mx.zeros((2, 4)), mx.ones((3, 2)), plan)


class TestAlignedOffsets:
    def test_segments_aligned(self):
        inds = np.array([[0, 0, 1, 2, 2, 2, 2]], dtype=np.int64)
        plan = _plan(inds, 3)
        starts, ends = aligned_offsets(plan, 4)
        starts = np.array(starts)
        ends = np.array(ends)
        assert (starts % 4 == 0).all()
        assert (ends % 4 == 0).all()
        counts = np.array(plan.counts)
        np.testing.assert_array_equal(ends - starts, ((counts + 3) // 4) * 4)

    def test_alignment_one_no_padding(self):
        inds = RNG.integers(0, 4, size=(8, 2))
        plan = _plan(inds, 4)
        starts, ends = aligned_offsets(plan, 1)
        np.testing.assert_array_equal(
            np.array(ends) - np.array(starts), np.array(plan.counts)
        )

    def test_rejects_bad_alignment(self):
        plan = _plan(np.array([[0, 1]], dtype=np.int64), 2)
        with pytest.raises(ValueError):
            aligned_offsets(plan, 0)


class TestStatsAndSwitch:
    def test_load_stats(self):
        inds = np.array([[0, 0, 1]], dtype=np.int64)
        plan = _plan(inds, 4)
        stats = expert_load_stats(plan)
        assert stats["pairs"] == 3
        assert stats["experts_active"] == 2
        assert stats["max_pairs_per_expert"] == 2
        assert stats["imbalance"] == pytest.approx(2 * 4 / 3)

    def test_switch_default_off(self, monkeypatch):
        monkeypatch.delenv("FUSION_SHIM_MOE", raising=False)
        assert is_moe_dispatch_enabled() is False

    def test_switch_on(self, monkeypatch):
        monkeypatch.setenv("FUSION_SHIM_MOE", "1")
        assert is_moe_dispatch_enabled() is True

    def test_dispatch_plan_slots(self):
        assert DispatchPlan.__slots__ == (
            "order",
            "sorted_expert_ids",
            "sorted_token_ids",
            "counts",
            "offsets",
            "inverse",
            "num_experts",
            "num_tokens",
            "top_k",
        )
