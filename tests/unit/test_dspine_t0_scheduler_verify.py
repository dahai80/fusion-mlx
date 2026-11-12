import mlx.core as mx

from fusion_mlx.scheduler.spec_decode import _run_spec_verify_sampled

V = 64


def _probs(rows_seed, T, peak_shift=0):
    out = []
    for i in range(T):
        key = mx.random.key(rows_seed + i)
        logits = mx.random.normal((V,), key=key)
        logits = mx.where(mx.arange(V) == (i + peak_shift) % V, logits + 6.0, logits)
        out.append(mx.softmax(logits.astype(mx.float32)))
    return mx.stack(out)


def _cand_full(T):
    return mx.stack([mx.arange(V) for _ in range(T)])


class _FakeModel:
    """Returns pre-set logits rows for the verify forward, ignoring cache."""

    def __init__(self, logits_rows):
        self.logits_rows = logits_rows
        self.fed = None

    def __call__(self, tokens, cache=None):
        self.fed = [int(x) for x in tokens.reshape(-1).tolist()]
        K = self.fed_len = len(self.fed)
        return mx.stack(self.logits_rows[:K])[None]


class TestRunSpecVerifySampled:
    def test_perfect_draft_accepts_all(self):
        # q rows equal the filtered target rows -> every coin*q < p holds
        # (coin < 1), so all K drafts are accepted and the bonus comes from
        # the last target row (the argmax of row K).
        T = 4
        target = _probs(10, T)
        draft_tokens = [int(mx.argmax(target[i])) for i in range(3)]
        model = _FakeModel([target[i] * 1e6 for i in range(1, T)])
        verified, n_accepted, processed = _run_spec_verify_sampled(
            model,
            draft_tokens,
            target[:3],
            _cand_full(3),
            mx.log(mx.maximum(target[0], 1e-30)),
            prompt_cache=[],
            inv_temp=1.0,
            top_k=V,
            top_p=1.0,
            min_p=0.0,
            accept_uniforms=mx.array([0.999, 0.999, 0.999]),
            final_uniform=0.5,
        )
        assert n_accepted == 3
        assert processed == 3
        assert verified[:3] == draft_tokens
        assert len(verified) == 4

    def test_reject_at_first_position_resamples_from_p0(self):
        # q row puts all mass on a token with p ~ 0 -> coin*q >= p -> reject;
        # emitted must be a single token drawn from the p_0 residual/p mass.
        p0 = _probs(20, 1)[0]
        bad = (int(mx.argmax(p0)) + 1) % V
        q0 = mx.zeros((V,))
        q0 = mx.where(mx.arange(V) == bad, 1.0, q0)
        model = _FakeModel([p0 * 1e6, p0 * 1e6])
        verified, n_accepted, processed = _run_spec_verify_sampled(
            model,
            [bad, bad],
            mx.stack([q0, q0]),
            _cand_full(2),
            mx.log(mx.maximum(p0, 1e-30)),
            prompt_cache=[],
            inv_temp=1.0,
            top_k=V,
            top_p=1.0,
            min_p=0.0,
            accept_uniforms=mx.array([1.0, 1.0]),
            final_uniform=0.5,
        )
        assert n_accepted == 0
        assert processed == 2
        assert len(verified) == 1
        assert verified[0] != bad or float(p0[bad]) > 0.0

    def test_forward_feeds_all_draft_tokens(self):
        target = _probs(30, 3)
        draft_tokens = [1, 2, 3]
        model = _FakeModel([target[0], target[1], target[2]])
        _run_spec_verify_sampled(
            model,
            draft_tokens,
            target[:3],
            _cand_full(3),
            mx.log(mx.maximum(target[0], 1e-30)),
            prompt_cache=[],
            inv_temp=1.0,
            top_k=V,
            top_p=1.0,
            min_p=0.0,
            accept_uniforms=mx.array([0.0, 0.0, 0.0]),
            final_uniform=0.5,
        )
        assert model.fed == [1, 2, 3]

    def test_partial_accept_keeps_prefix(self):
        # q rows equal target rows -> accept all; but force rejection at
        # position 1 by giving q_1 mass on a p~0 token.
        target = _probs(40, 3)
        p0 = target[0]
        p1 = target[1]
        bad = (int(mx.argmax(p1)) + 1) % V
        q0 = p0
        q1 = mx.zeros((V,))
        q1 = mx.where(mx.arange(V) == bad, 1.0, q1)
        model = _FakeModel([target[1], target[2], target[0]])
        verified, n_accepted, processed = _run_spec_verify_sampled(
            model,
            [int(mx.argmax(p0)), bad, int(mx.argmax(p1))],
            mx.stack([q0, q1, q0]),
            _cand_full(3),
            mx.log(mx.maximum(p0, 1e-30)),
            prompt_cache=[],
            inv_temp=1.0,
            top_k=V,
            top_p=1.0,
            min_p=0.0,
            accept_uniforms=mx.array([0.0, 1.0, 0.0]),
            final_uniform=0.5,
        )
        assert n_accepted == 1
        assert processed == 3
        assert verified[0] == int(mx.argmax(p0))
        assert len(verified) == 2

    def test_target_row_count_matches_chain(self):
        # K drafts -> K+1 target rows consumed; the bonus comes from row K.
        target = _probs(50, 4)
        draft_tokens = [int(mx.argmax(target[i])) for i in range(3)]
        model = _FakeModel([target[i] * 1e6 for i in range(1, 4)])
        verified, n_accepted, _ = _run_spec_verify_sampled(
            model,
            draft_tokens,
            target[:3],
            _cand_full(3),
            mx.log(mx.maximum(target[0], 1e-30)),
            prompt_cache=[],
            inv_temp=1.0,
            top_k=V,
            top_p=1.0,
            min_p=0.0,
            accept_uniforms=mx.array([0.999, 0.999, 0.999]),
            final_uniform=0.999,
        )
        assert n_accepted == 3
        # bonus must be the argmax of the last verify row (uniform ~ 1 picks
        # the top-probability token under the cdf walk)
        assert verified[-1] == int(mx.argmax(target[3]))
