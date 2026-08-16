# SPDX-License-Identifier: Apache-2.0
# P1/P3/P4 reuse-layer checkpoint: scheduler sigmas, model config synthesis,
# VAE single-file split (synthetic keys). No real weights required.
from __future__ import annotations

import pytest

from fusion_mlx.video.ltx2_5 import (
    DISTILLED_STAGE_1_SIGMAS,
    DISTILLED_STAGE_2_SIGMAS,
    LTX2_5Model,
    LTX2_5Variant,
    default_ltx2_5_config,
    load_video_decoder,
    load_video_encoder,
    ltx2_5_scheduler,
    resolve_distilled_sigmas,
)
from fusion_mlx.video.ltx2_5.generate import generate_video
from fusion_mlx.video.ltx2_5.video_vae import _split_vae_weights


class TestSchedulerSigmas:
    def test_stage1_sigmas_nonempty(self):
        assert len(DISTILLED_STAGE_1_SIGMAS) >= 2
        assert DISTILLED_STAGE_1_SIGMAS[0] == 1.0
        assert DISTILLED_STAGE_1_SIGMAS[-1] == 0.0

    def test_stage2_sigmas_nonempty(self):
        assert len(DISTILLED_STAGE_2_SIGMAS) >= 2
        assert DISTILLED_STAGE_2_SIGMAS[-1] == 0.0

    def test_resolve_stage1(self):
        assert resolve_distilled_sigmas(1) is DISTILLED_STAGE_1_SIGMAS

    def test_resolve_stage2(self):
        assert resolve_distilled_sigmas(2) is DISTILLED_STAGE_2_SIGMAS

    def test_resolve_invalid_stage_raises(self):
        with pytest.raises(ValueError, match="unknown distilled stage"):
            resolve_distilled_sigmas(3)
        with pytest.raises(ValueError, match="unknown distilled stage"):
            resolve_distilled_sigmas(0)

    def test_scheduler_returns_array(self):
        import mlx.core as mx

        sigmas = ltx2_5_scheduler(steps=10, num_tokens=4096)
        assert isinstance(sigmas, mx.array)
        assert sigmas.shape[0] == 11  # steps + 1


class TestModelConfigSynthesis:
    def test_default_config_is_25(self):
        cfg = default_ltx2_5_config()
        assert cfg.num_layers == 48
        assert cfg.caption_channels == 3840
        assert cfg.num_attention_heads == 32
        assert cfg.attention_head_dim == 128

    def test_default_config_dev_variant(self):
        cfg = default_ltx2_5_config(LTX2_5Variant.DEV)
        assert cfg.num_layers == 48
        assert cfg.caption_channels == 3840

    def test_model_is_independent_from_ltx2(self):
        # 2.5 transformer is a self-contained nn.Module (user decision:
        # ltx2_5 独立目录全量重写), NOT an LTXModel subclass.
        import mlx.nn as nn

        from fusion_mlx.video.ltx2.ltx2_model import LTXModel

        assert issubclass(LTX2_5Model, nn.Module)
        assert not issubclass(LTX2_5Model, LTXModel)


class TestModelFromPretrainedMissingWeights:
    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not found"):
            LTX2_5Model.from_pretrained(tmp_path / "nonexistent.safetensors")


class TestVAESplit:
    def test_split_separates_prefixes(self, tmp_path):
        import mlx.core as mx

        weights = {
            "encoder.conv.weight": mx.zeros((4, 3, 1, 1, 1)),
            "encoder.conv.bias": mx.zeros((4,)),
            "decoder.conv.weight": mx.zeros((3, 4, 1, 1, 1)),
            "decoder.norm.weight": mx.zeros((4,)),
            "per_channel_statistics.mean-of-means": mx.zeros((4,)),
            "per_channel_statistics.std-of-means": mx.ones((4,)),
        }
        f = tmp_path / "vae.safetensors"
        mx.save_safetensors(str(f), weights)
        enc, dec, stats = _split_vae_weights(f)
        assert "conv.weight" in enc and "conv.bias" in enc
        assert "conv.weight" in dec and "norm.weight" in dec
        assert len(enc) == 2
        assert len(dec) == 2
        assert "per_channel_statistics.mean-of-means" in stats
        assert "per_channel_statistics.std-of-means" in stats
        assert len(stats) == 2

    def test_split_unknown_keys_dropped(self, tmp_path):
        import mlx.core as mx

        weights = {
            "decoder.conv.weight": mx.zeros((3, 4, 1, 1, 1)),
            "audio_decoder.random.weight": mx.zeros((2,)),
        }
        f = tmp_path / "vae.safetensors"
        mx.save_safetensors(str(f), weights)
        enc, dec, stats = _split_vae_weights(f)
        assert len(enc) == 0
        assert len(dec) == 1
        assert len(stats) == 0

    def test_load_encoder_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not found"):
            load_video_encoder(tmp_path / "nope.safetensors")

    def test_load_decoder_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not found"):
            load_video_decoder(tmp_path / "nope.safetensors")


class TestGenerateSkeleton:
    def test_generate_missing_frames_raises(self):
        with pytest.raises(NotImplementedError, match="duration-head"):
            generate_video(
                "Lightricks/LTX-2.5",
                "a cat",
                num_frames=None,
                width=768,
                height=512,
            )

    def test_generate_dim_violation_raises(self):
        with pytest.raises(ValueError, match="divisible by 32"):
            generate_video(
                "Lightricks/LTX-2.5",
                "a cat",
                num_frames=9,
                width=700,
                height=512,
            )

    def test_generate_missing_repo_raises(self):
        # 越过所有输入校验后, 解析不到真实权重目录应 fail visible (FileNotFoundError)。
        with pytest.raises((FileNotFoundError, RuntimeError)):
            generate_video(
                "Lightricks/LTX-2.5-DOES-NOT-EXIST-9f8a2c",
                "a cat",
                num_frames=9,
                width=768,
                height=512,
            )
