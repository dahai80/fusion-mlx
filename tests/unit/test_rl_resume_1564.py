import json

import pytest

from fusion_mlx.training.dpo import DPOConfig
from fusion_mlx.training.dpo_service import resolve_resume_file
from fusion_mlx.training.grpo import GRPOConfig
from fusion_mlx.training.reward import RewardConfig


class TestResolveResumeFile:
    def test_prefers_final_adapters(self, tmp_path):
        (tmp_path / "adapters.safetensors").write_bytes(b"x")
        (tmp_path / "0000010_adapters.safetensors").write_bytes(b"x")
        assert resolve_resume_file(tmp_path) == tmp_path / "adapters.safetensors"

    def test_falls_back_to_newest_numbered(self, tmp_path):
        (tmp_path / "0000010_adapters.safetensors").write_bytes(b"x")
        (tmp_path / "0000020_adapters.safetensors").write_bytes(b"x")
        (tmp_path / "0000015_adapters.safetensors").write_bytes(b"x")
        assert (
            resolve_resume_file(tmp_path) == tmp_path / "0000020_adapters.safetensors"
        )

    def test_ignores_junk_names(self, tmp_path):
        (tmp_path / "junk_adapters.safetensors").write_bytes(b"x")
        (tmp_path / "final.safetensors").write_bytes(b"x")
        assert resolve_resume_file(tmp_path) is None

    def test_missing_dir_returns_none(self, tmp_path):
        assert resolve_resume_file(tmp_path / "nope") is None

    def test_dpo_chain_uses_numbered_checkpoint(self, tmp_path, monkeypatch):
        # Mid-run resume: an interrupted DPO dir has only numbered
        # checkpoints; _chain_adapter_weights must resolve the newest one
        # instead of demanding adapters.safetensors (#1564).
        from fusion_mlx.training import dpo_service

        ckpt = tmp_path / "0000042_adapters.safetensors"
        ckpt.write_bytes(b"x")
        (tmp_path / "adapter_config.json").write_text(
            json.dumps(
                {
                    "num_layers": 16,
                    "lora_parameters": {"rank": 8, "scale": 16.0},
                }
            )
        )
        cfg = DPOConfig(init_adapter_path=str(tmp_path), lora_layers=16, lora_rank=8)

        loaded = {}

        class FakeModel:
            def load_weights(self, path, strict):
                loaded["path"] = path
                loaded["strict"] = strict

        dpo_service._chain_adapter_weights(FakeModel(), cfg)
        assert loaded["path"] == str(ckpt)
        assert loaded["strict"] is False

    def test_dpo_chain_rank_mismatch_rejected(self, tmp_path):
        from fusion_mlx.training import dpo_service

        (tmp_path / "0000042_adapters.safetensors").write_bytes(b"x")
        (tmp_path / "adapter_config.json").write_text(
            json.dumps({"num_layers": 16, "lora_parameters": {"rank": 32}})
        )
        cfg = DPOConfig(init_adapter_path=str(tmp_path), lora_layers=16, lora_rank=8)

        class FakeModel:
            def load_weights(self, path, strict):
                raise AssertionError("must not load on rank mismatch")

        with pytest.raises(ValueError, match="rank"):
            dpo_service._chain_adapter_weights(FakeModel(), cfg)


class TestRLConfigFields1564:
    def test_grpo_config_accepts_new_fields(self):
        cfg = GRPOConfig(
            memory_limit_gb=32.0, init_adapter_path="/tmp/adapter", save_every=10
        )
        assert cfg.memory_limit_gb == 32.0
        assert cfg.init_adapter_path == "/tmp/adapter"
        assert cfg.save_every == 10
        # to_dict carries them (admin route builds jobs from it)
        d = cfg.to_dict()
        assert d["memory_limit_gb"] == 32.0
        assert d["save_every"] == 10

    def test_grpo_defaults_preserve_old_behavior(self):
        cfg = GRPOConfig()
        assert cfg.memory_limit_gb == 0.0
        assert cfg.init_adapter_path == ""
        assert cfg.save_every == 0

    def test_dpo_config_save_every(self):
        assert DPOConfig().save_every == 0
        assert DPOConfig(save_every=25).save_every == 25

    def test_reward_config_memory_limit(self):
        assert RewardConfig().memory_limit_gb == 0.0
        assert RewardConfig(memory_limit_gb=16.0).memory_limit_gb == 16.0


class TestGRPOPolicyWeightLoad:
    def _service(self):
        from fusion_mlx.training.grpo_service import GRPOService

        return GRPOService.__new__(GRPOService)

    def test_load_uses_newest_checkpoint(self, tmp_path):
        (tmp_path / "0000010_adapters.safetensors").write_bytes(b"x")
        (tmp_path / "0000020_adapters.safetensors").write_bytes(b"x")
        (tmp_path / "adapter_config.json").write_text(
            json.dumps(
                {
                    "num_layers": 16,
                    "lora_parameters": {"rank": 8, "scale": 16.0},
                }
            )
        )
        cfg = GRPOConfig(lora_layers=16, lora_rank=8)
        loaded = {}

        class FakeModel:
            def load_weights(self, path, strict):
                loaded["path"] = path

        self._service()._load_policy_weights(FakeModel(), tmp_path, cfg)
        assert loaded["path"] == str(tmp_path / "0000020_adapters.safetensors")

    def test_missing_dir_raises(self, tmp_path):
        cfg = GRPOConfig(lora_layers=16, lora_rank=8)
        with pytest.raises(ValueError, match="no adapters.safetensors"):
            self._service()._load_policy_weights(object(), tmp_path / "nope", cfg)

    def test_layers_mismatch_rejected(self, tmp_path):
        (tmp_path / "0000001_adapters.safetensors").write_bytes(b"x")
        (tmp_path / "adapter_config.json").write_text(
            json.dumps({"num_layers": 32, "lora_parameters": {"rank": 8}})
        )
        cfg = GRPOConfig(lora_layers=16, lora_rank=8)
        with pytest.raises(ValueError, match="num_layers"):
            self._service()._load_policy_weights(object(), tmp_path, cfg)
