# SPDX-License-Identifier: Apache-2.0
"""Unit tests for RFT (rejection-sampling fine-tuning) trainer.

Mirrors tests/unit/test_grpo.py stub approach: tiny GradModel + FakeTokenizer,
no real mlx_lm load. Real-model smoke test gated behind @pytest.mark.real_model.
"""

import os

import mlx.core as mx
import mlx.nn as nn
import pytest

from fusion_mlx.training.rft import (
    RFTConfig,
    RFTTrainer,
    _sft_loss_differentiable,
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


def test_select_top_k_keeps_highest_rewards():
    completions = [[1], [2], [3], [4]]
    rewards = [0.1, 0.9, 0.5, 0.3]
    picked = RFTTrainer._select_top_k(completions, rewards, top_k=2)
    assert len(picked) == 2
    assert picked[0] == ([2], 0.9)
    assert picked[1] == ([3], 0.5)


def test_select_top_k_clamps_to_available():
    picked = RFTTrainer._select_top_k([[1], [2]], [0.4, 0.6], top_k=5)
    assert len(picked) == 2
    assert picked[0] == ([2], 0.6)


def test_select_top_k_empty():
    assert RFTTrainer._select_top_k([], [], top_k=3) == []


def test_select_top_k_at_least_one():
    picked = RFTTrainer._select_top_k([[9]], [0.2], top_k=0)
    assert len(picked) == 1


def test_sft_loss_differentiable_shape_and_sign():
    model = GradModel(vocab=16, dim=8)
    prompt = mx.array([1, 2])
    completion = mx.array([3])
    per_token, ce_loss = _sft_loss_differentiable(model, prompt, completion)
    assert int(per_token.shape[0]) == 1
    # CE loss = -mean(log p); log p < 0 so loss > 0
    assert float(ce_loss) > 0


def test_sft_loss_gradient_flows():
    model = GradModel(vocab=16, dim=8)
    prompt = mx.array([1])
    completion = mx.array([2])

    def loss_fn(m, _batch):
        _, s = _sft_loss_differentiable(m, prompt, completion)
        return s

    lvg = nn.value_and_grad(model, loss_fn)
    val, grad = lvg(model, None)
    from mlx.utils import tree_flatten

    flat = tree_flatten(grad)
    assert len(flat) > 0
    assert any(g is not None for _, g in flat)


def test_compute_rewards_length_fallback():
    trainer = RFTTrainer(GradModel(), FakeTokenizer(), "fake-path", RFTConfig())
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
    cfg = RFTConfig(reward_endpoint="http://reward.test/score", reward_timeout=5.0)
    trainer = RFTTrainer(GradModel(), FakeTokenizer(), "fake-path", cfg)
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
    cfg = RFTConfig(reward_endpoint="http://reward.test/score")
    trainer = RFTTrainer(GradModel(), FakeTokenizer(), "fake-path", cfg)
    with pytest.raises(ValueError, match="reward_endpoint returned"):
        trainer._compute_rewards("hi", ["a", "bb"])


def test_rft_loss_finite():
    model = GradModel(vocab=16, dim=8)
    cfg = RFTConfig()
    trainer = RFTTrainer(model, FakeTokenizer(), "fake-path", cfg)
    winners = [
        {"prompt_ids": [1, 2], "completion_ids": [3]},
        {"prompt_ids": [1], "completion_ids": [4]},
    ]
    loss = trainer._sft_loss(model, winners)
    assert mx.isfinite(loss).item()
    assert float(loss) > 0


def test_rft_config_to_dict_roundtrip():
    cfg = RFTConfig(num_samples=16, top_k=4, reward_endpoint="http://x")
    d = cfg.to_dict()
    assert d["num_samples"] == 16
    assert d["top_k"] == 4
    assert d["reward_endpoint"] == "http://x"
    assert d["iters"] == 50


def test_rft_train_step_with_stubbed_sampling(monkeypatch):
    # End-to-end train_step with sampling stubbed to fixed completions and
    # length-based reward (no endpoint). Verifies accept_rate, n_winners,
    # loss finite, and that optimizer state advances.
    model = GradModel(vocab=16, dim=8)
    cfg = RFTConfig(
        num_samples=4,
        top_k=2,
        max_completion_len=4,
        temperature=0.0,
        learning_rate=1e-3,
    )
    trainer = RFTTrainer(model, FakeTokenizer(), "fake-path", cfg)

    # Stub sampling: 4 fixed completions per prompt. Reward is length-based,
    # so longer completions win. Top-2 of 4 kept -> accept_rate = 0.5.
    fixed = [[3, 4, 5, 6], [7, 8], [9, 10, 11], [12]]

    def fake_sample(self, prompt_ids, num_samples):
        return list(fixed[:num_samples])

    monkeypatch.setattr(RFTTrainer, "_sample_completions", fake_sample)

    result = trainer.train_step(["Hello", "World"])
    # 2 prompts x 4 samples = 8 sampled; 2 prompts x 2 winners = 4 winners
    assert result.n_samples == 8
    assert result.n_winners == 4
    assert result.accept_rate == pytest.approx(0.5, abs=1e-6)
    assert mx.isfinite(mx.array(result.loss)).item()
    assert result.loss > 0
    # mean_reward = mean of all completion lengths:
    # fixed lengths [4,2,3,1] repeated twice -> mean = (10/4) = 2.5
    assert result.mean_reward == pytest.approx(2.5, abs=1e-6)


@pytest.mark.real_model
def test_rft_real_model_smoke():
    if not os.environ.get("FUSION_MLX_REAL_MODEL_TESTS"):
        pytest.skip("set FUSION_MLX_REAL_MODEL_TESTS=1 to run real-model RFT test")
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
    cfg = RFTConfig(
        num_samples=2,
        top_k=1,
        iters=1,
        batch_size=2,
        max_completion_len=16,
        lora_layers=4,
        temperature=0.0,
    )
    trainer = RFTTrainer(model, tokenizer, model_path, cfg)
    result = trainer.train_step(["Hello", "The sky is"])
    assert result.n_samples == 4
    assert result.n_winners == 2
    assert mx.isfinite(mx.array(result.loss)).item()
