# SPDX-License-Identifier: Apache-2.0
# Tests for issue #170/#177 P3 SkyReelsBackend stage API (10 methods).
# Monkeypatched / fake-pipeline only - no real MLX model load.

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import mlx.core as mx
import pytest

from fusion_mlx.engines.video_backends.skyreels import SkyReelsBackend


def _make_fake_pipeline(monkeypatch):
    pipeline = SimpleNamespace(
        text_encoder=object(),
        clip_encoder=object(),
        dit=SimpleNamespace(patch_size=(1, 2, 2)),
        vae=SimpleNamespace(),
        step_strategy=object(),
        config=SimpleNamespace(
            num_inference_steps=30,
            guidance_scale=5.0,
        ),
        _last_spec_stats=None,
    )

    def fake_decode(latent, tiling=False, tile_size=None):
        return mx.zeros(
            (latent.shape[0], 3, latent.shape[3] * 8, latent.shape[4] * 8),
            dtype=mx.float32,
        )

    pipeline.vae.decode = fake_decode

    context_result = mx.zeros((1, 769, 4096), dtype=mx.float32)

    def fake_encode_context(prompt):
        return context_result

    pipeline._encode_context = fake_encode_context

    denoise_result = None

    def fake_denoise_sample(latent, context, seq_lens=None, grid_sizes=None):
        nonlocal denoise_result
        denoise_result = mx.zeros_like(latent)
        return denoise_result

    pipeline._denoise_sample = fake_denoise_sample

    class FakePipelineClass:
        def __init__(self, model_name):
            pass

    import fusion_mlx.video.skyreels_v3.pipelines as pipelines_mod

    monkeypatch.setattr(pipelines_mod, "SkyReelsR2VPipeline", FakePipelineClass)
    monkeypatch.setattr(pipelines_mod, "SkyReelsV2VPipeline", FakePipelineClass)
    monkeypatch.setattr(pipelines_mod, "SkyReelsA2VPipeline", FakePipelineClass)

    async def fake_get_or_create(self, pipeline_class):
        self._pipeline = pipeline
        self._pipeline_class = pipeline_class
        return pipeline

    monkeypatch.setattr(SkyReelsBackend, "_get_or_create_pipeline", fake_get_or_create)

    # Override mx.eval in stage methods to avoid "no Stream(gpu,0)" in test threads.
    # The stage methods run compute via run_in_executor on get_executor("video"),
    # but in test context that thread has no GPU stream. Patch mx.eval to no-op.
    _original_eval = mx.eval

    def _safe_eval(*args):
        try:
            _original_eval(*args)
        except RuntimeError:
            pass

    monkeypatch.setattr(mx, "eval", _safe_eval)

    return pipeline, denoise_result


class TestSkyreelsStageLoadUnload:
    async def test_load_text_encoder_sets_flag(self, monkeypatch):
        pipeline, _ = _make_fake_pipeline(monkeypatch)
        backend = SkyReelsBackend("SkyReels-V3-R2V-14B-MLX")
        await backend.start("SkyReels-V3-R2V-14B-MLX")
        await backend.load_text_encoder()
        assert backend._stage_flags["text_encoder"] is True

    async def test_load_dit_sets_flag(self, monkeypatch):
        pipeline, _ = _make_fake_pipeline(monkeypatch)
        backend = SkyReelsBackend("SkyReels-V3-R2V-14B-MLX")
        await backend.start("SkyReels-V3-R2V-14B-MLX")
        await backend.load_dit()
        assert backend._stage_flags["dit"] is True

    async def test_load_vae_sets_flag(self, monkeypatch):
        pipeline, _ = _make_fake_pipeline(monkeypatch)
        backend = SkyReelsBackend("SkyReels-V3-R2V-14B-MLX")
        await backend.start("SkyReels-V3-R2V-14B-MLX")
        await backend.load_vae()
        assert backend._stage_flags["vae"] is True

    async def test_unload_text_encoder_clears_flag_and_nones(self, monkeypatch):
        pipeline, _ = _make_fake_pipeline(monkeypatch)
        backend = SkyReelsBackend("SkyReels-V3-R2V-14B-MLX")
        await backend.start("SkyReels-V3-R2V-14B-MLX")
        await backend.load_text_encoder()
        await backend.unload_text_encoder()
        assert backend._stage_flags["text_encoder"] is False
        assert pipeline.text_encoder is None
        assert pipeline.clip_encoder is None

    async def test_unload_dit_clears_flag_and_nones(self, monkeypatch):
        pipeline, _ = _make_fake_pipeline(monkeypatch)
        backend = SkyReelsBackend("SkyReels-V3-R2V-14B-MLX")
        await backend.start("SkyReels-V3-R2V-14B-MLX")
        await backend.load_dit()
        await backend.unload_dit()
        assert backend._stage_flags["dit"] is False
        assert pipeline.dit is None
        assert pipeline.step_strategy is None

    async def test_unload_vae_clears_flag_and_nones(self, monkeypatch):
        pipeline, _ = _make_fake_pipeline(monkeypatch)
        backend = SkyReelsBackend("SkyReels-V3-R2V-14B-MLX")
        await backend.start("SkyReels-V3-R2V-14B-MLX")
        await backend.load_vae()
        await backend.unload_vae()
        assert backend._stage_flags["vae"] is False
        assert pipeline.vae is None

    async def test_stop_resets_all_stage_flags(self, monkeypatch):
        pipeline, _ = _make_fake_pipeline(monkeypatch)
        backend = SkyReelsBackend("SkyReels-V3-R2V-14B-MLX")
        await backend.start("SkyReels-V3-R2V-14B-MLX")
        await backend.load_text_encoder()
        await backend.load_dit()
        await backend.load_vae()
        await backend.stop()
        assert backend._stage_flags == {
            "text_encoder": False,
            "dit": False,
            "vae": False,
        }


class TestSkyreelsEncodeText:
    async def test_encode_text_returns_context(self, monkeypatch):
        pipeline, _ = _make_fake_pipeline(monkeypatch)
        backend = SkyReelsBackend("SkyReels-V3-R2V-14B-MLX")
        await backend.start("SkyReels-V3-R2V-14B-MLX")
        result = await backend.encode_text("a cat on a mat")
        assert "context" in result
        assert result["context"].shape == (1, 769, 4096)

    async def test_encode_text_raises_if_unloaded(self, monkeypatch):
        pipeline, _ = _make_fake_pipeline(monkeypatch)
        backend = SkyReelsBackend("SkyReels-V3-R2V-14B-MLX")
        await backend.start("SkyReels-V3-R2V-14B-MLX")
        await backend.load_text_encoder()
        await backend.unload_text_encoder()
        with pytest.raises(RuntimeError, match="text_encoder is unloaded"):
            await backend.encode_text("hello")


class TestSkyreelsDenoise:
    async def test_denoise_returns_latent(self, monkeypatch):
        pipeline, _ = _make_fake_pipeline(monkeypatch)
        backend = SkyReelsBackend("SkyReels-V3-R2V-14B-MLX")
        await backend.start("SkyReels-V3-R2V-14B-MLX")
        await backend.load_dit()

        latent = mx.zeros((1, 16, 9, 64, 64), dtype=mx.float32)
        pos_embed = mx.zeros((1, 769, 4096), dtype=mx.float32)

        result = await backend.denoise(
            latent=latent,
            pos_embed=pos_embed,
            neg_embed=None,
            steps=10,
            cfg=5.0,
            seed=42,
            num_frames=9,
        )
        assert result.shape == latent.shape

    async def test_denoise_with_neg_embed_concatenates(self, monkeypatch):
        pipeline, _ = _make_fake_pipeline(monkeypatch)
        backend = SkyReelsBackend("SkyReels-V3-R2V-14B-MLX")
        await backend.start("SkyReels-V3-R2V-14B-MLX")
        await backend.load_dit()

        latent = mx.zeros((1, 16, 9, 64, 64), dtype=mx.float32)
        pos_embed = mx.zeros((1, 769, 4096), dtype=mx.float32)
        neg_embed = mx.zeros((1, 769, 4096), dtype=mx.float32)

        result = await backend.denoise(
            latent=latent,
            pos_embed=pos_embed,
            neg_embed=neg_embed,
            steps=10,
            cfg=5.0,
            seed=42,
            num_frames=9,
        )
        assert result.shape == latent.shape

    async def test_denoise_restores_config_on_exit(self, monkeypatch):
        pipeline, _ = _make_fake_pipeline(monkeypatch)
        backend = SkyReelsBackend("SkyReels-V3-R2V-14B-MLX")
        await backend.start("SkyReels-V3-R2V-14B-MLX")
        await backend.load_dit()

        latent = mx.zeros((1, 16, 9, 64, 64), dtype=mx.float32)
        pos_embed = mx.zeros((1, 769, 4096), dtype=mx.float32)

        saved_steps = pipeline.config.num_inference_steps
        saved_cfg = pipeline.config.guidance_scale

        await backend.denoise(
            latent=latent,
            pos_embed=pos_embed,
            neg_embed=None,
            steps=5,
            cfg=3.0,
            seed=0,
            num_frames=9,
        )

        assert pipeline.config.num_inference_steps == saved_steps
        assert pipeline.config.guidance_scale == saved_cfg

    async def test_denoise_raises_if_dit_unloaded(self, monkeypatch):
        pipeline, _ = _make_fake_pipeline(monkeypatch)
        backend = SkyReelsBackend("SkyReels-V3-R2V-14B-MLX")
        await backend.start("SkyReels-V3-R2V-14B-MLX")
        await backend.load_dit()
        await backend.unload_dit()
        latent = mx.zeros((1, 16, 9, 64, 64), dtype=mx.float32)
        pos_embed = mx.zeros((1, 769, 4096), dtype=mx.float32)
        with pytest.raises(RuntimeError, match="dit is unloaded"):
            await backend.denoise(latent, pos_embed, None, 5, 3.0, 0, 9)


class TestSkyreelsDecode:
    async def test_decode_returns_pixel_tensor(self, monkeypatch):
        pipeline, _ = _make_fake_pipeline(monkeypatch)
        backend = SkyReelsBackend("SkyReels-V3-R2V-14B-MLX")
        await backend.start("SkyReels-V3-R2V-14B-MLX")
        await backend.load_vae()

        latent = mx.zeros((1, 16, 9, 64, 64), dtype=mx.float32)
        result = await backend.decode(latent)
        assert result.shape[0] == 1
        assert result.shape[1] == 3

    async def test_decode_tiled_passes_tile_size(self, monkeypatch):
        pipeline, _ = _make_fake_pipeline(monkeypatch)
        received_kwargs = {}

        def fake_decode(latent, tiling=False, tile_size=None):
            received_kwargs["tiling"] = tiling
            received_kwargs["tile_size"] = tile_size
            return mx.zeros(
                (latent.shape[0], 3, latent.shape[3] * 8, latent.shape[4] * 8),
                dtype=mx.float32,
            )

        pipeline.vae.decode = fake_decode

        backend = SkyReelsBackend("SkyReels-V3-R2V-14B-MLX")
        await backend.start("SkyReels-V3-R2V-14B-MLX")
        await backend.load_vae()

        latent = mx.zeros((1, 16, 9, 64, 64), dtype=mx.float32)
        await backend.decode_tiled(latent, tile_size=512)
        assert received_kwargs["tiling"] is True
        assert received_kwargs["tile_size"] == (1, 64, 64)

    async def test_decode_raises_if_vae_unloaded(self, monkeypatch):
        pipeline, _ = _make_fake_pipeline(monkeypatch)
        backend = SkyReelsBackend("SkyReels-V3-R2V-14B-MLX")
        await backend.start("SkyReels-V3-R2V-14B-MLX")
        await backend.load_vae()
        await backend.unload_vae()
        latent = mx.zeros((1, 16, 9, 64, 64), dtype=mx.float32)
        with pytest.raises(RuntimeError, match="vae is unloaded"):
            await backend.decode(latent)

    async def test_decode_tiled_raises_if_vae_unloaded(self, monkeypatch):
        pipeline, _ = _make_fake_pipeline(monkeypatch)
        backend = SkyReelsBackend("SkyReels-V3-R2V-14B-MLX")
        await backend.start("SkyReels-V3-R2V-14B-MLX")
        await backend.load_vae()
        await backend.unload_vae()
        latent = mx.zeros((1, 16, 9, 64, 64), dtype=mx.float32)
        with pytest.raises(RuntimeError, match="vae is unloaded"):
            await backend.decode_tiled(latent)


class TestSkyreelsFullStageLifecycle:
    async def test_full_lifecycle_encode_denoise_decode(self, monkeypatch):
        pipeline, _ = _make_fake_pipeline(monkeypatch)
        backend = SkyReelsBackend("SkyReels-V3-R2V-14B-MLX")
        await backend.start("SkyReels-V3-R2V-14B-MLX")

        # Stage 1: text encoding
        await backend.load_text_encoder()
        text_result = await backend.encode_text("a beautiful sunset")
        assert "context" in text_result
        await backend.unload_text_encoder()
        assert pipeline.text_encoder is None

        # Stage 2: denoise (re-load pipeline to get text_encoder back)
        await backend._ensure_pipeline()
        await backend.load_dit()
        latent = mx.zeros((1, 16, 9, 64, 64), dtype=mx.float32)
        pos_embed = text_result["context"]
        result = await backend.denoise(
            latent=latent,
            pos_embed=pos_embed,
            neg_embed=None,
            steps=5,
            cfg=5.0,
            seed=42,
            num_frames=9,
        )
        assert result.shape == latent.shape
        await backend.unload_dit()
        assert pipeline.dit is None

        # Stage 3: decode (re-load pipeline to get dit back)
        await backend._ensure_pipeline()
        await backend.load_vae()
        pixels = await backend.decode(result)
        assert pixels.shape[1] == 3
        await backend.unload_vae()
        assert pipeline.vae is None

    async def test_sequential_offload_memory_pattern(self, monkeypatch):
        pipeline, _ = _make_fake_pipeline(monkeypatch)
        backend = SkyReelsBackend("SkyReels-V3-R2V-14B-MLX")
        await backend.start("SkyReels-V3-R2V-14B-MLX")

        # Simulate ComfyUI sequential offload: only one component loaded at a time
        # Encode text
        await backend.load_text_encoder()
        text_result = await backend.encode_text("test prompt")
        await backend.unload_text_encoder()
        assert backend._stage_flags["text_encoder"] is False

        # Denoise
        await backend._ensure_pipeline()
        await backend.load_dit()
        assert backend._stage_flags["dit"] is True
        # text_encoder was unloaded; dit stage doesn't reload it
        latent = mx.zeros((1, 16, 9, 64, 64), dtype=mx.float32)
        await backend.denoise(
            latent=latent,
            pos_embed=text_result["context"],
            neg_embed=None,
            steps=3,
            cfg=5.0,
            seed=0,
            num_frames=9,
        )
        await backend.unload_dit()

        # Decode
        await backend._ensure_pipeline()
        await backend.load_vae()
        assert backend._stage_flags["vae"] is True
        await backend.decode(latent)
        await backend.unload_vae()


class TestSkyreelsDetectPipelineClass:
    def test_r2v_detection(self, monkeypatch):
        pipeline, _ = _make_fake_pipeline(monkeypatch)
        backend = SkyReelsBackend("SkyReels-V3-R2V-14B-MLX")
        cls = backend._detect_pipeline_class()
        assert cls.__name__ == "FakePipelineClass"

    def test_v2v_detection(self, monkeypatch):
        pipeline, _ = _make_fake_pipeline(monkeypatch)
        backend = SkyReelsBackend("SkyReels-V3-V2V-14B-MLX")
        cls = backend._detect_pipeline_class()
        assert cls.__name__ == "FakePipelineClass"

    def test_a2v_detection(self, monkeypatch):
        pipeline, _ = _make_fake_pipeline(monkeypatch)
        backend = SkyReelsBackend("SkyReels-V3-A2V-19B-MLX")
        cls = backend._detect_pipeline_class()
        assert cls.__name__ == "FakePipelineClass"

    def test_unknown_defaults_to_r2v(self, monkeypatch):
        pipeline, _ = _make_fake_pipeline(monkeypatch)
        backend = SkyReelsBackend("some-unknown-skyreels-model")
        cls = backend._detect_pipeline_class()
        assert cls.__name__ == "FakePipelineClass"
