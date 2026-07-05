import inspect

import mlx.core as mx
import pytest

from fusion_mlx.scheduler.spec_decode import _run_spec_verify

# These tests exercise real mlx array ops (mx.full/argmax/tolist) inside
# _run_spec_verify. The Linux CI conftest stubs mlx as MagicMock, making those
# ops no-ops and breaking n_accepted/bonus computation. KVCache being a real
# class is the reliable "real mlx available" signal (stubbed together on CI).
# Mirrors the 18 existing mlx-only platform-gated tests.
try:
    from mlx_lm.models.cache import KVCache as _KVC

    _HAS_REAL_MLX = inspect.isclass(_KVC)
except Exception:  # noqa: BLE001
    _HAS_REAL_MLX = False

pytestmark = pytest.mark.skipif(
    not _HAS_REAL_MLX,
    reason="needs real mlx array ops (stubbed as MagicMock on CI)",
)


class _MockModel:
    """Returns logits whose argmax per position is a pre-set token id.

    sampled[i] = the model's prediction AFTER processing draft_tokens[i],
    i.e. the prediction for the token at position i+1.
    """

    def __init__(self, sampled_per_position, vocab=128):
        self._sampled = sampled_per_position
        self._vocab = vocab

    def __call__(self, tokens, cache=None):
        K = tokens.shape[-1]
        logits = mx.zeros((1, K, self._vocab))
        for i in range(K):
            logits = mx.array(logits)
            row = mx.zeros(self._vocab)
            row = mx.concatenate(
                [
                    mx.zeros(self._sampled[i]),
                    mx.array([1.0]),
                    mx.zeros(self._vocab - self._sampled[i] - 1),
                ]
            )
            logits[0, i] = row
        return logits


def _make_model(sampled):
    vocab = max(sampled) + 1
    K = len(sampled)

    class M:
        def __call__(self, tokens, cache=None):
            k = tokens.shape[-1]
            out = mx.full((1, k, vocab), -1e9)
            for i in range(k):
                out[0, i, sampled[i]] = 1.0
            return out

    return M()


@pytest.mark.parametrize(
    "drafts,sampled,expected_n_acc,expected_bonus",
    [
        # D2 rejected (n_accepted=1): bonus must be pred AFTER D1 (sampled[0]=99),
        # NOT pred after the rejected D2 (sampled[1]=88).
        pytest.param(
            [10, 20, 30], [99, 88, 77], 1, 99, id="reject_d2_bonus_is_pred_after_d1"
        ),
        # D3 rejected (n_accepted=2): bonus must be pred AFTER D2 (sampled[1]=88),
        # NOT pred after the rejected D3 (sampled[2]=77).
        pytest.param(
            [10, 20, 30], [20, 88, 77], 2, 88, id="reject_d3_bonus_is_pred_after_d2"
        ),
        # Full acceptance (n_accepted=K=3): bonus is sampled[K-1]=44 (unchanged
        # by the fix — both old and new pick sampled[K-1] here).
        pytest.param([10, 20, 30], [20, 30, 44], 3, 44, id="full_accept_bonus_is_last"),
    ],
)
def test_run_spec_verify_resample_idx(drafts, sampled, expected_n_acc, expected_bonus):
    model = _make_model(sampled)
    # D1 (drafts[0]) is "accepted via the regular step" when sampled_from_regular == drafts[0].
    verified, n_accepted, cache_tokens_processed = _run_spec_verify(
        model=model,
        current_token=0,
        draft_tokens=drafts,
        prompt_cache=[],
        sampled_from_regular=drafts[0],
    )
    assert n_accepted == expected_n_acc, (n_accepted, expected_n_acc)
    assert verified[:n_accepted] == drafts[:n_accepted]
    # The bonus token (last element of verified) must be the prediction after
    # the last ACCEPTED draft, i.e. sampled[n_accepted - 1].
    assert verified[-1] == expected_bonus, (
        verified[-1],
        expected_bonus,
        "bonus should be sampled[n_accepted-1]",
    )
    assert verified[-1] == sampled[expected_n_acc - 1]
    assert cache_tokens_processed == len(drafts)


def test_rejection_bonus_is_not_pred_after_rejected_draft():
    """The old buggy code picked sampled[n_accepted] (pred after the first
    REJECTED draft). Verify the fix picks sampled[n_accepted-1] instead."""
    drafts = [10, 20, 30]
    sampled = [99, 88, 77]  # D2 rejected -> n_accepted=1
    model = _make_model(sampled)
    verified, n_accepted, _ = _run_spec_verify(
        model, 0, drafts, [], sampled_from_regular=10
    )
    assert n_accepted == 1
    assert verified == [10, 99]  # [D1, pred_after_D1]
    assert verified != [10, 88]  # NOT [D1, pred_after_rejected_D2]
