# SPDX-License-Identifier: Apache-2.0
"""Unit tests for reward-adapter loading with value-head tensors (#1742).

Standalone pytest file. Covers _load_reward_model's split-load contract and
GRPO's degraded-batch behavior on reward_endpoint errors.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from fusion_mlx.training import reward_score
from fusion_mlx.training.grpo import GRPOTrainer


class TestLoadRewardModel:
    def test_split_loads_lora_and_head(self, tmp_path):
        adapter = tmp_path / "rm_adapter"
        adapter.mkdir()
        (adapter / "adapter_config.json").write_text(
            json.dumps(
                {
                    "num_layers": 2,
                    "lora_parameters": {},
                    "fine_tune_type": "lora",
                    "reward_model": True,
                }
            )
        )
        lora_tensor = {"lora_a": "LORA", "lora_b": "LORA2"}
        head_tensor = {"value_head.proj.weight": "WH", "value_head.proj.bias": "BH"}

        model = MagicMock()
        param_names = set(lora_tensor) | {"model.embed.weight"}
        model.parameters.return_value = {k: MagicMock() for k in param_names}
        model.value_head = MagicMock()
        fake_tok = MagicMock()
        fake_tok.encode.return_value = [1, 2, 3]

        with (
            patch("mlx_lm.utils.load", return_value=(model, fake_tok)) as load_mock,
            patch(
                "mlx.core.load", return_value={**lora_tensor, **head_tensor}
            ) as mx_load,
            patch("mlx_lm.tuner.utils.linear_to_lora_layers") as lora_mock,
            patch("fusion_mlx.training.reward_score._attach_value_head") as attach_mock,
        ):
            out_model, out_tok = reward_score._load_reward_model(
                "/tmp/model", str(adapter)
            )
        assert out_model is model
        assert out_tok is fake_tok
        load_mock.assert_called_once_with("/tmp/model")
        lora_mock.assert_called_once()
        applied = model.load_weights.call_args[0][0]
        applied_names = [k for k, _ in applied]
        assert applied_names == ["lora_a", "lora_b"]
        head_items = model.value_head.load_weights.call_args[0][0]
        assert head_items == [("proj.weight", "WH"), ("proj.bias", "BH")]
        assert attach_mock.called

    def test_unknown_lora_tensor_raises_loud(self, tmp_path):
        adapter = tmp_path / "rm_adapter"
        adapter.mkdir()
        (adapter / "adapter_config.json").write_text(
            json.dumps(
                {"num_layers": 2, "lora_parameters": {}, "fine_tune_type": "lora"}
            )
        )
        model = MagicMock()
        model.parameters.return_value = {"model.embed.weight": MagicMock()}
        fake_tok = MagicMock()
        with (
            patch("mlx_lm.utils.load", return_value=(model, fake_tok)),
            patch("mlx.core.load", return_value={"bogus.weight": "X"}),
            patch("mlx_lm.tuner.utils.linear_to_lora_layers"),
        ):
            with pytest.raises(ValueError, match="incompatible tensors"):
                reward_score._load_reward_model("/tmp/model", str(adapter))

    def test_no_head_tensors_skips_head_load(self, tmp_path):
        adapter = tmp_path / "rm_adapter"
        adapter.mkdir()
        (adapter / "adapter_config.json").write_text(
            json.dumps(
                {"num_layers": 2, "lora_parameters": {}, "fine_tune_type": "lora"}
            )
        )
        model = MagicMock()
        model.parameters.return_value = {"lora_a": MagicMock()}
        fake_tok = MagicMock()
        with (
            patch("mlx_lm.utils.load", return_value=(model, fake_tok)),
            patch("mlx.core.load", return_value={"lora_a": "L"}),
            patch("mlx_lm.tuner.utils.linear_to_lora_layers"),
        ):
            reward_score._load_reward_model("/tmp/model", str(adapter))
        assert not model.value_head.load_weights.called


class TestGrpoRewardDegradation:
    def _trainer(self):
        from types import SimpleNamespace

        t = GRPOTrainer.__new__(GRPOTrainer)
        t.config = SimpleNamespace(
            reward_endpoint="http://x/reward", reward_timeout=1.0
        )
        return t

    def test_http_400_degrades_to_zero_rewards(self, caplog):
        import io
        import urllib.error

        t = self._trainer()
        err = urllib.error.HTTPError(
            "http://x/reward", 400, "Bad Request", {}, io.BytesIO(b"rejected")
        )
        with patch("urllib.request.urlopen", side_effect=err):
            with caplog.at_level("WARNING", logger="fusion_mlx.training.grpo"):
                rewards = t._compute_rewards("p", ["a", "b"])
        assert rewards == [0.0, 0.0]
        assert "degrading batch" in caplog.text

    def test_unreachable_degrades_to_zero_rewards(self, caplog):
        import urllib.error

        t = self._trainer()
        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("conn refused"),
        ):
            with caplog.at_level("WARNING", logger="fusion_mlx.training.grpo"):
                rewards = t._compute_rewards("p", ["a"])
        assert rewards == [0.0]
        assert "unreachable" in caplog.text

    def test_mismatch_count_still_raises(self):
        t = self._trainer()
        resp = MagicMock()
        resp.read.return_value = json.dumps({"rewards": [1.0]}).encode()
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)
        with patch("urllib.request.urlopen", return_value=resp):
            with pytest.raises(ValueError, match="rewards for"):
                t._compute_rewards("p", ["a", "b"])
