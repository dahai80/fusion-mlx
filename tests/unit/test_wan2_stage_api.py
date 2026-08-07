# SPDX-License-Identifier: Apache-2.0
# Tests for issue #410 Wan2Backend stage API (10 methods).
# Monkeypatched / fake-pipeline only - no real MLX model load.

from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from fusion_mlx.engines.video_backends.wan2 import Wan2Backend


def _make_fake_stage(monkeypatch):
    # Fake config object exposing the attributes the stage methods read:
    # text_len, vae_stride, patch_size, vae_z_dim, dual_model, sample_shift,
    # frame_num, in_dim.
    config = SimpleNamespace(
        text_len=512,
        vae_stride=(4, 16, 16),
        patch_size=(1, 2, 2),
        vae_z_dim=16,
        dual_model=False,
        sample_shift=5.0,
        frame_num=81,
        in_dim=16,
        sample_neg_prompt="",
        max_area=0,
    )
    quant = None

    import fusion_mlx.video.wan2.stage as stage_mod
    import fusion_mlx.video.wan2.utils as utils_mod

    monkeypatch.setattr(stage_mod, "load_wan_config", lambda model_dir: (config, quant))
    monkeypatch.setattr(
        stage_mod, "resolve_t5_path", lambda model_dir: Path("/fake/t5")
    )
    monkeypatch.setattr(
        stage_mod, "resolve_vae_path", lambda model_dir: Path("/fake/vae")
    )

    fake_encoder = SimpleNamespace()
    fake_tokenizer = SimpleNamespace()
    context = mx.zeros((1, 512, 4096), dtype=mx.float32)
    monkeypatch.setattr(
        stage_mod,
        "encode_text_stage",
        lambda enc, tok, prompt, text_len: context,
    )

    monkeypatch.setattr(
        utils_mod, "load_t5_encoder", lambda path, cfg, dtype=None: fake_encoder
    )

    fake_dit = SimpleNamespace()
    monkeypatch.setattr(utils_mod, "load_wan_model", lambda path, cfg, q: fake_dit)

    fake_vae = SimpleNamespace()
    monkeypatch.setattr(utils_mod, "load_vae_decoder", lambda path, cfg=None: fake_vae)

    # run_denoise returns a 4D latent (z_dim, t_lat, h_lat, w_lat); the stage
    # denoise() wrapper adds the batch dim -> 5D.
    denoise_result = mx.zeros((16, 5, 32, 32), dtype=mx.float32)
    monkeypatch.setattr(
        stage_mod,
        "run_denoise",
        lambda *a, **k: denoise_result,
    )
    # compute_target_shape returns (target_shape, seq_len, height, width).
    monkeypatch.setattr(
        stage_mod,
        "compute_target_shape",
        lambda cfg, nf, h, w: ((16, 5, 32, 32), 1280, h, w),
    )

    # decode_wan_vae returns uint8 [T, H, W, 3].
    frames_u8 = (np.zeros((5, 512, 512, 3))).astype(np.uint8)
    monkeypatch.setattr(
        stage_mod,
        "decode_wan_vae",
        lambda latent, cfg, vae, tiling_config=None: frames_u8,
    )

    # get_model_path resolves a model name to a directory; fake a real path.
    import fusion_mlx.video.wan2.utils as u2

    monkeypatch.setattr(u2, "get_model_path", lambda name: Path("/fake/wan2"))

    # Override mx.eval to avoid "no Stream(gpu,0)" in test threads.
    _original_eval = mx.eval

    def _safe_eval(*args):
        try:
            _original_eval(*args)
        except RuntimeError:
            pass

    monkeypatch.setattr(mx, "eval", _safe_eval)

    return config, context, denoise_result, frames_u8


class TestWan2StageLoadUnload:
    async def test_load_text_encoder_sets_flag(self, monkeypatch):
        _make_fake_stage(monkeypatch)
        backend = Wan2Backend("wan2.1-t2v-1.3B")
        await backend.start("wan2.1-t2v-1.3B")
        await backend.load_text_encoder()
        assert backend._stage_flags["text_encoder"] is True
        assert backend._t5_encoder is not None

    async def test_load_dit_sets_flag(self, monkeypatch):
        _make_fake_stage(monkeypatch)
        backend = Wan2Backend("wan2.1-t2v-1.3B")
        await backend.start("wan2.1-t2v-1.3B")
        await backend.load_dit()
        assert backend._stage_flags["dit"] is True
        assert backend._stage_dit_models is not None

    async def test_load_vae_sets_flag(self, monkeypatch):
        _make_fake_stage(monkeypatch)
        backend = Wan2Backend("wan2.1-t2v-1.3B")
        await backend.start("wan2.1-t2v-1.3B")
        await backend.load_vae()
        assert backend._stage_flags["vae"] is True
        assert backend._stage_vae is not None

    async def test_unload_text_encoder_clears_flag_and_nones(self, monkeypatch):
        _make_fake_stage(monkeypatch)
        backend = Wan2Backend("wan2.1-t2v-1.3B")
        await backend.start("wan2.1-t2v-1.3B")
        await backend.load_text_encoder()
        await backend.unload_text_encoder()
        assert backend._stage_flags["text_encoder"] is False
        assert backend._t5_encoder is None

    async def test_unload_dit_clears_flag_and_nones(self, monkeypatch):
        _make_fake_stage(monkeypatch)
        backend = Wan2Backend("wan2.1-t2v-1.3B")
        await backend.start("wan2.1-t2v-1.3B")
        await backend.load_dit()
        await backend.unload_dit()
        assert backend._stage_flags["dit"] is False
        assert backend._stage_dit_models is None

    async def test_unload_vae_clears_flag_and_nones(self, monkeypatch):
        _make_fake_stage(monkeypatch)
        backend = Wan2Backend("wan2.1-t2v-1.3B")
        await backend.start("wan2.1-t2v-1.3B")
        await backend.load_vae()
        await backend.unload_vae()
        assert backend._stage_flags["vae"] is False
        assert backend._stage_vae is None

    async def test_stop_resets_all_stage_flags(self, monkeypatch):
        _make_fake_stage(monkeypatch)
        backend = Wan2Backend("wan2.1-t2v-1.3B")
        await backend.start("wan2.1-t2v-1.3B")
        await backend.load_text_encoder()
        await backend.load_dit()
        await backend.load_vae()
        await backend.stop()
        assert backend._stage_flags == {
            "text_encoder": False,
            "dit": False,
            "vae": False,
        }
        assert backend._stage_dit_models is None
        assert backend._stage_vae is None


class TestWan2EncodeText:
    async def test_encode_text_returns_embed(self, monkeypatch):
        _make_fake_stage(monkeypatch)
        backend = Wan2Backend("wan2.1-t2v-1.3B")
        await backend.start("wan2.1-t2v-1.3B")
        await backend.load_text_encoder()
        result = await backend.encode_text("a cat on a mat")
        assert "embed" in result
        assert result["embed"].shape == (1, 512, 4096)

    async def test_encode_text_raises_if_unloaded(self, monkeypatch):
        _make_fake_stage(monkeypatch)
        backend = Wan2Backend("wan2.1-t2v-1.3B")
        await backend.start("wan2.1-t2v-1.3B")
        with pytest.raises(RuntimeError, match="text_encoder is unloaded"):
            await backend.encode_text("hello")


class TestWan2Denoise:
    async def test_denoise_returns_5d_latent(self, monkeypatch):
        _make_fake_stage(monkeypatch)
        backend = Wan2Backend("wan2.1-t2v-1.3B")
        await backend.start("wan2.1-t2v-1.3B")
        await backend.load_dit()

        # 5D empty latent (1, c, t, h, w) from FusionKSampler.
        latent = mx.zeros((1, 16, 5, 32, 32), dtype=mx.float32)
        pos_embed = mx.zeros((1, 512, 4096), dtype=mx.float32)

        result = await backend.denoise(
            latent=latent,
            pos_embed=pos_embed,
            neg_embed=None,
            steps=10,
            cfg=5.0,
            seed=42,
            num_frames=81,
        )
        # run_denoise fake returns (16,5,32,32); wrapper adds batch -> 5D.
        assert result.ndim == 5
        assert result.shape[0] == 1

    async def test_denoise_with_neg_embed(self, monkeypatch):
        _make_fake_stage(monkeypatch)
        backend = Wan2Backend("wan2.1-t2v-1.3B")
        await backend.start("wan2.1-t2v-1.3B")
        await backend.load_dit()

        latent = mx.zeros((1, 16, 5, 32, 32), dtype=mx.float32)
        pos_embed = mx.zeros((1, 512, 4096), dtype=mx.float32)
        neg_embed = mx.zeros((1, 512, 4096), dtype=mx.float32)

        result = await backend.denoise(
            latent=latent,
            pos_embed=pos_embed,
            neg_embed=neg_embed,
            steps=10,
            cfg=5.0,
            seed=42,
            num_frames=81,
        )
        assert result.ndim == 5

    async def test_denoise_raises_if_unloaded(self, monkeypatch):
        _make_fake_stage(monkeypatch)
        backend = Wan2Backend("wan2.1-t2v-1.3B")
        await backend.start("wan2.1-t2v-1.3B")
        latent = mx.zeros((1, 16, 5, 32, 32), dtype=mx.float32)
        pos_embed = mx.zeros((1, 512, 4096), dtype=mx.float32)
        with pytest.raises(RuntimeError, match="dit is unloaded"):
            await backend.denoise(
                latent=latent,
                pos_embed=pos_embed,
                neg_embed=None,
                steps=10,
                cfg=5.0,
                seed=42,
                num_frames=81,
            )


class TestWan2Decode:
    async def test_decode_returns_pixel_tensor(self, monkeypatch):
        _make_fake_stage(monkeypatch)
        backend = Wan2Backend("wan2.1-t2v-1.3B")
        await backend.start("wan2.1-t2v-1.3B")
        await backend.load_vae()

        latent = mx.zeros((1, 16, 5, 32, 32), dtype=mx.float32)
        result = await backend.decode(latent)
        # decode returns float [0,1] (1, T, H, W, 3).
        assert result.ndim == 5
        assert result.shape[0] == 1
        assert result.shape[-1] == 3

    async def test_decode_tiled_returns_pixel_tensor(self, monkeypatch):
        _make_fake_stage(monkeypatch)
        backend = Wan2Backend("wan2.1-t2v-1.3B")
        await backend.start("wan2.1-t2v-1.3B")
        await backend.load_vae()

        latent = mx.zeros((1, 16, 5, 32, 32), dtype=mx.float32)
        result = await backend.decode_tiled(latent, tile_size=256)
        assert result.ndim == 5
        assert result.shape[-1] == 3

    async def test_decode_raises_if_unloaded(self, monkeypatch):
        _make_fake_stage(monkeypatch)
        backend = Wan2Backend("wan2.1-t2v-1.3B")
        await backend.start("wan2.1-t2v-1.3B")
        latent = mx.zeros((1, 16, 5, 32, 32), dtype=mx.float32)
        with pytest.raises(RuntimeError, match="vae is unloaded"):
            await backend.decode(latent)


class TestWan2FullLifecycle:
    async def test_full_pipeline_t2v(self, monkeypatch):
        _make_fake_stage(monkeypatch)
        backend = Wan2Backend("wan2.1-t2v-1.3B")
        await backend.start("wan2.1-t2v-1.3B")

        await backend.load_text_encoder()
        pos = await backend.encode_text("a dog running")
        neg = await backend.encode_text("blurry, low quality")
        await backend.unload_text_encoder()

        await backend.load_dit()
        latent = mx.zeros((1, 16, 5, 32, 32), dtype=mx.float32)
        denoised = await backend.denoise(
            latent=latent,
            pos_embed=pos["embed"],
            neg_embed=neg["embed"],
            steps=20,
            cfg=5.0,
            seed=7,
            num_frames=81,
        )
        await backend.unload_dit()

        await backend.load_vae()
        pixels = await backend.decode(denoised)
        await backend.unload_vae()

        assert pixels.ndim == 5
        assert pixels.shape[-1] == 3
        assert backend._stage_flags == {
            "text_encoder": False,
            "dit": False,
            "vae": False,
        }


class TestWan2SequentialOffload:
    async def test_load_unload_releases_memory(self, monkeypatch):
        _make_fake_stage(monkeypatch)
        backend = Wan2Backend("wan2.1-t2v-1.3B")
        await backend.start("wan2.1-t2v-1.3B")

        await backend.load_dit()
        assert backend._stage_dit_models is not None
        await backend.unload_dit()
        assert backend._stage_dit_models is None

        # VAE load after DiT unload — sequential offload inverts which
        # component holds memory at any time.
        await backend.load_vae()
        assert backend._stage_vae is not None
        await backend.unload_vae()
        assert backend._stage_vae is None


class TestWan2StageDetect:
    def test_detect_wan_path(self):
        assert Wan2Backend.detect("Wan2.1-T2V-14B") is True

    def test_detect_vace_path(self):
        assert Wan2Backend.detect("wan-vace-14B") is True

    def test_detect_non_wan_path(self):
        assert Wan2Backend.detect("llama-3-8b") is False
