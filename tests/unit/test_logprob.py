import os

import mlx.core as mx
import mlx.nn as nn
import pytest

from fusion_mlx.training.logprob import LogprobResult, compute_logprob


class SmoothModel(nn.Module):
    # logits row i = arange(vocab) + i*0.1 so every logit is finite and
    # log_softmax is analytically tractable for verification.
    def __init__(self, vocab=8):
        super().__init__()
        self.vocab = vocab
        self.base = mx.arange(vocab).astype(mx.float32)

    def __call__(self, inputs, cache=None):
        ids = inputs.reshape(-1)
        L = int(ids.shape[0])
        rows = []
        for i in range(L):
            rows.append(self.base + float(i) * 0.1)
        logits = mx.stack(rows)[None, :]
        return logits, cache


def _manual_logprob(sm, prompt, completion, vocab=8):
    full = prompt + completion
    logits, _ = sm(mx.array(full)[None, :])
    lp = nn.log_softmax(logits.astype(mx.float32))
    per = []
    n_prompt = len(prompt)
    for j, tok in enumerate(completion):
        row = lp[0, n_prompt - 1 + j, :]
        per.append(row[tok].item())
    return sum(per), per


def test_compute_logprob_matches_manual():
    sm = SmoothModel(vocab=8)
    prompt = [1, 3, 2]
    completion = [4, 0]
    total, n, per_token = compute_logprob(sm, prompt, completion)
    manual_total, manual_per = _manual_logprob(sm, prompt, completion)
    assert n == 2
    assert per_token == pytest.approx(manual_per, abs=1e-5)
    assert total == pytest.approx(manual_total, abs=1e-5)


def test_compute_logprob_single_completion():
    sm = SmoothModel(vocab=8)
    prompt = [0]
    completion = [5]
    total, n, per_token = compute_logprob(sm, prompt, completion)
    manual_total, _ = _manual_logprob(sm, prompt, completion)
    assert n == 1
    assert total == pytest.approx(manual_total, abs=1e-5)


def test_compute_logprob_empty_completion():
    sm = SmoothModel(vocab=8)
    total, n, per_token = compute_logprob(sm, [0, 1], [])
    assert total == 0.0
    assert n == 0
    assert per_token == []


def test_compute_logprob_deterministic():
    sm = SmoothModel(vocab=8)
    prompt = [2, 1, 3]
    completion = [1, 2, 3]
    t1, _, p1 = compute_logprob(sm, prompt, completion)
    t2, _, p2 = compute_logprob(sm, prompt, completion)
    assert t1 == pytest.approx(t2, abs=1e-6)
    assert p1 == pytest.approx(p2, abs=1e-6)


def test_compute_logprob_accepts_mx_arrays():
    sm = SmoothModel(vocab=8)
    prompt = mx.array([1, 2])
    completion = mx.array([3])
    total, n, _ = compute_logprob(sm, prompt, completion)
    assert n == 1
    assert mx.isfinite(mx.array(total)).item()


def test_logprob_result_to_dict():
    r = LogprobResult(logprob=-1.5, token_count=3, per_token=[-0.5, -0.5, -0.5])
    d = r.to_dict()
    assert d == {"logprob": -1.5, "token_count": 3, "per_token": [-0.5, -0.5, -0.5]}


@pytest.mark.real_model
def test_score_text_real_model():
    if not os.environ.get("FUSION_MLX_REAL_MODEL_TESTS"):
        pytest.skip("set FUSION_MLX_REAL_MODEL_TESTS=1 to run real-model logprob test")
    from fusion_mlx.training.logprob import score_text

    model_path = os.environ.get(
        "FUSION_MLX_TEST_MODEL", "mlx-community/Qwen3-0.6B-4bit"
    )
    result = score_text(model_path, "Hello", " world")
    assert isinstance(result, LogprobResult)
    assert result.token_count > 0
    assert mx.isfinite(mx.array(result.logprob)).item()
    assert len(result.per_token) == result.token_count
    assert sum(result.per_token) == pytest.approx(result.logprob, abs=1e-4)
