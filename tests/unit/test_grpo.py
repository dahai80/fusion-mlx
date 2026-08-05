import os

import mlx.core as mx
import mlx.nn as nn
import pytest

from fusion_mlx.training.grpo import (
    GRPOConfig,
    GRPOTrainer,
    _seq_logprob_differentiable,
)


class GradModel(nn.Module):
    # Tiny differentiable model: embedding + head so trainable params exist
    # and gradients flow. vocab=16, dim=8.
    def __init__(self, vocab=16, dim=8):
        super().__init__()
        self.vocab = vocab
        self.embed = nn.Embedding(vocab, dim)
        self.head = nn.Linear(dim, vocab)

    def __call__(self, inputs, cache=None):
        ids = inputs.reshape(-1)
        x = self.embed(ids)
        logits = self.head(x)[None, :]
        return logits


class FakeTokenizer:
    def __init__(self, vocab=16):
        self.vocab = vocab
        self.eos_token_id = 15

    def encode(self, text):
        return [ord(c) % self.vocab for c in text]

    def decode(self, ids):
        return "x" * len(ids)


def test_advantages_mean_zero_normalized():
    rewards = [1.0, 2.0, 3.0, 4.0]
    adv = GRPOTrainer._advantages(rewards)
    assert sum(adv) == pytest.approx(0.0, abs=1e-6)
    import math

    expected = [(r - 2.5) / (math.sqrt(1.25) + 1e-8) for r in rewards]
    assert adv == pytest.approx(expected, abs=1e-5)


def test_advantages_zero_std_returns_zeros():
    adv = GRPOTrainer._advantages([5.0, 5.0, 5.0])
    assert adv == [0.0, 0.0, 0.0]


def test_advantages_empty():
    assert GRPOTrainer._advantages([]) == []


def test_seq_logprob_differentiable_sum():
    model = GradModel(vocab=16, dim=8)
    prompt = mx.array([1, 2])
    completion = mx.array([3])
    per_token, total = _seq_logprob_differentiable(model, prompt, completion)
    assert int(per_token.shape[0]) == 1
    assert float(total) < 0


def test_seq_logprob_gradient_flows():
    model = GradModel(vocab=16, dim=8)
    prompt = mx.array([1])
    completion = mx.array([2])

    def loss_fn(m, _batch):
        _, s = _seq_logprob_differentiable(m, prompt, completion)
        return s

    lvg = nn.value_and_grad(model, loss_fn)
    val, grad = lvg(model, None)
    from mlx.utils import tree_flatten

    flat = tree_flatten(grad)
    assert len(flat) > 0
    assert any(g is not None for _, g in flat)


def test_compute_rewards_length_fallback():
    trainer = GRPOTrainer(GradModel(), FakeTokenizer(), "fake-path", GRPOConfig())
    rewards = trainer._compute_rewards("hi", ["abc", "xy"])
    assert rewards == [3.0, 2.0]


def test_compute_rewards_http_callback(monkeypatch):
    class _FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return None

        def read(self):
            return b'{"rewards": [0.8, 0.2]}'

    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["data"] = req.data
        captured["timeout"] = timeout
        return _FakeResp()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    cfg = GRPOConfig(reward_endpoint="http://reward.test/score", reward_timeout=5.0)
    trainer = GRPOTrainer(GradModel(), FakeTokenizer(), "fake-path", cfg)
    rewards = trainer._compute_rewards("hello", ["a", "bb"])
    assert rewards == [0.8, 0.2]
    assert captured["url"] == "http://reward.test/score"
    assert captured["timeout"] == 5.0


def test_compute_rewards_mismatched_count_raises(monkeypatch):
    class _FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return None

        def read(self):
            return b'{"rewards": [0.5]}'

    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: _FakeResp())
    cfg = GRPOConfig(reward_endpoint="http://reward.test/score")
    trainer = GRPOTrainer(GradModel(), FakeTokenizer(), "fake-path", cfg)
    with pytest.raises(ValueError, match="reward_endpoint returned"):
        trainer._compute_rewards("hi", ["a", "bb"])


def test_grpo_loss_finite_and_clipped():
    model = GradModel(vocab=16, dim=8)
    cfg = GRPOConfig(clip_ratio=0.2)
    trainer = GRPOTrainer(model, FakeTokenizer(), "fake-path", cfg)
    batch = [
        {
            "prompt_ids": [1, 2],
            "completion_ids": [3],
            "ref_logprob": -1.0,
            "advantage": 1.0,
        },
        {
            "prompt_ids": [1],
            "completion_ids": [4],
            "ref_logprob": -2.0,
            "advantage": -1.0,
        },
    ]
    loss, mean_ratio = trainer._grpo_loss(model, batch)
    assert mx.isfinite(loss).item()
    assert mx.isfinite(mean_ratio).item()
    assert float(mean_ratio) > 0


def test_grpo_config_to_dict_roundtrip():
    cfg = GRPOConfig(group_size=8, iters=10, reward_endpoint="http://x")
    d = cfg.to_dict()
    assert d["group_size"] == 8
    assert d["iters"] == 10
    assert d["reward_endpoint"] == "http://x"
    assert d["clip_ratio"] == 0.2


@pytest.mark.real_model
def test_grpo_real_model_smoke():
    if not os.environ.get("FUSION_MLX_REAL_MODEL_TESTS"):
        pytest.skip("set FUSION_MLX_REAL_MODEL_TESTS=1 to run real-model GRPO test")
    import mlx_lm.utils as mlx_utils
    from mlx_lm.tuner.utils import linear_to_lora_layers

    model_path = os.environ.get(
        "FUSION_MLX_TEST_MODEL", "mlx-community/Qwen3-0.6B-4bit"
    )
    model, tokenizer = mlx_utils.load(model_path)
    model.freeze()
    linear_to_lora_layers(
        model,
        4,
        {"rank": 8, "dropout": 0.0, "scale": 16.0},
        use_dora=False,
    )
    cfg = GRPOConfig(
        group_size=2,
        iters=1,
        batch_size=2,
        max_completion_len=16,
        lora_layers=4,
        temperature=0.0,
    )
    trainer = GRPOTrainer(model, tokenizer, model_path, cfg)
    result = trainer.train_step(["Hello", "The sky is"])
    assert result.n_samples == 4
    assert mx.isfinite(mx.array(result.loss)).item()
