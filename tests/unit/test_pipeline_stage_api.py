# SPDX-License-Identifier: Apache-2.0
# Tests for issue #170 pipeline stage API on ImageGenEngine / VideoGenEngine.
# Monkeypatched / fake-flux only - no real mflux/MLX model load.

import asyncio
from types import SimpleNamespace

import mlx.core as mx
import pytest

from fusion_mlx.engines.image_gen import ImageGenEngine, get_executor
from fusion_mlx.engines.video import VideoGenEngine
from fusion_mlx.engines.video_backends.base import VideoBackend


def _make_engine_with_fake_flux():
    eng = ImageGenEngine("fake-flux")
    flux = SimpleNamespace(
        text_encoder=object(),
        transformer=object(),
        vae=SimpleNamespace(decode_packed_latents=lambda lat, tiling_config=None: lat),
        tokenizers={"qwen3": object()},
        model_config=object(),
        tiling_config=object(),
        callbacks=None,
    )
    eng._flux = flux
    eng._mflux_missing = False
    return eng, flux


def test_image_stage_requires_started_engine():
    eng = ImageGenEngine("x")
    with pytest.raises(RuntimeError, match="not started"):
        asyncio.run(eng.load_text_encoder())
    with pytest.raises(RuntimeError, match="not started"):
        asyncio.run(eng.decode(mx.zeros((1, 8, 4, 4))))


def test_image_stage_mflux_missing():
    eng = ImageGenEngine("x")
    eng._mflux_missing = True
    with pytest.raises(RuntimeError, match="mflux-fusion not installed"):
        asyncio.run(eng.load_dit())


def test_image_load_unload_lifecycle():
    eng, flux = _make_engine_with_fake_flux()
    asyncio.run(eng.load_text_encoder())
    asyncio.run(eng.load_dit())
    asyncio.run(eng.load_vae())
    asyncio.run(eng.unload_text_encoder())
    assert flux.text_encoder is None
    asyncio.run(eng.unload_dit())
    assert flux.transformer is None
    asyncio.run(eng.unload_vae())
    assert flux.vae is None
    with pytest.raises(RuntimeError, match="text_encoder was unloaded"):
        asyncio.run(eng.load_text_encoder())
    with pytest.raises(RuntimeError, match="transformer .* was unloaded"):
        asyncio.run(eng.load_dit())
    with pytest.raises(RuntimeError, match="vae was unloaded"):
        asyncio.run(eng.load_vae())


def test_image_decode_validates_ndim_and_delegates():
    eng, flux = _make_engine_with_fake_flux()
    with pytest.raises(ValueError, match="batch,c,h,w"):
        asyncio.run(eng.decode(mx.zeros((1, 8, 4))))
    latent = mx.zeros((1, 8, 4, 4))
    out = asyncio.run(eng.decode(latent))
    assert out is latent


def test_image_decode_tiled_passes_tiling_config():
    eng, flux = _make_engine_with_fake_flux()
    received: dict = {}

    def fake_decode(lat, tiling_config=None):
        received["tc"] = tiling_config
        return mx.zeros((1, 3, 4, 4))

    flux.vae.decode_packed_latents = fake_decode
    flux.tiling_config = "TILING_SENTINEL"
    out = asyncio.run(eng.decode_tiled(mx.zeros((1, 8, 4, 4)), tile_size=128))
    assert out.shape == (1, 3, 4, 4)
    assert received["tc"] == "TILING_SENTINEL"


def test_image_denoise_validates_ndim_and_unloaded():
    eng, flux = _make_engine_with_fake_flux()
    with pytest.raises(ValueError, match="batch,c,h,w"):
        asyncio.run(eng.denoise(mx.zeros((1, 8, 4)), None, None, 3, 1.0, 0))
    flux.transformer = None
    with pytest.raises(RuntimeError, match="transformer .* unloaded"):
        asyncio.run(eng.denoise(mx.zeros((1, 8, 4, 4)), None, None, 3, 1.0, 0))


def test_image_denoise_loop_runs_and_returns_4d(monkeypatch):
    eng, flux = _make_engine_with_fake_flux()
    import mflux.models.common.config.config as cfg_mod
    import mflux.models.flux2.latent_creator.flux2_latent_creator as lc_mod
    import mflux.models.flux2.model.flux2_text_encoder.prompt_encoder as pe_mod

    n_steps = 3

    class FakeScheduler:
        def __init__(self, n):
            self.timesteps = mx.array([0.9 - 0.2 * i for i in range(n)])
            self.sigmas = mx.array([1.0] * n)

        def step(self, **kw):
            return kw["latents"]

    class FakeConfig:
        def __init__(self, **kw):
            self.time_steps = list(range(kw.get("num_inference_steps", n_steps)))
            self.scheduler = FakeScheduler(len(self.time_steps))

    class FakeLatentCreator:
        @staticmethod
        def prepare_grid_ids(latent, *, t_coord):
            b, c, h, w = latent.shape
            return mx.zeros((b, h * w, 4), dtype=mx.int32)

        @staticmethod
        def pack_latents(latent):
            b, c, h, w = latent.shape
            return latent.reshape(b, c, h * w).transpose(0, 2, 1)

    class FakePromptEncoder:
        @staticmethod
        def prepare_text_ids(embed):
            return mx.zeros((1, embed.shape[1], 4), dtype=mx.int32)

    monkeypatch.setattr(cfg_mod, "Config", FakeConfig)
    monkeypatch.setattr(lc_mod, "Flux2LatentCreator", FakeLatentCreator)
    monkeypatch.setattr(pe_mod, "Flux2PromptEncoder", FakePromptEncoder)

    predict_calls = {"n": 0}

    def fake_predict(transformer):
        def _predict(**kw):
            predict_calls["n"] += 1
            return mx.zeros_like(kw["latents"])

        return _predict

    flux._predict = fake_predict
    loop = asyncio.new_event_loop()
    ex = get_executor("image")

    def _make_inputs():
        return (
            mx.zeros((1, 8, 4, 4)),
            mx.zeros((1, 5, 8)),
            mx.zeros((1, 5, 8)),
        )

    latent, pos, neg = loop.run_until_complete(loop.run_in_executor(ex, _make_inputs))
    out = loop.run_until_complete(eng.denoise(latent, pos, neg, n_steps, 4.0, 0))
    loop.close()
    assert out.shape == (1, 8, 4, 4)
    assert predict_calls["n"] == n_steps


def test_image_encode_text_with_fake_flux(monkeypatch):
    eng, flux = _make_engine_with_fake_flux()
    import mflux.models.flux2.model.flux2_text_encoder.prompt_encoder as pe_mod

    embed_fake = mx.zeros((1, 5, 8))
    ids_fake = mx.zeros((1, 5, 4), dtype=mx.int32)

    def fake_encode_prompt(**kw):
        assert kw["prompt"] == "hello"
        assert kw["tokenizer"] is flux.tokenizers["qwen3"]
        return embed_fake, ids_fake

    monkeypatch.setattr(pe_mod.Flux2PromptEncoder, "encode_prompt", fake_encode_prompt)
    result = asyncio.run(eng.encode_text("hello"))
    assert result["embed"] is embed_fake
    assert result["text_ids"] is ids_fake
    assert result.get("negative_embed") is None


def test_video_engine_stage_methods_delegate(monkeypatch):
    captured: dict = {}

    class FakeBackend:
        _loaded = True

        async def load_text_encoder(self):
            captured["load_text_encoder"] = True

        async def encode_text(self, prompt):
            captured["encode_text"] = prompt
            return {"embed": "E"}

        async def unload_text_encoder(self):
            captured["unload_text_encoder"] = True

        async def load_dit(self):
            captured["load_dit"] = True

        async def denoise(self, latent, pos, neg, steps, cfg, seed, num_frames):
            captured["denoise"] = (steps, num_frames)
            return latent

        async def unload_dit(self):
            captured["unload_dit"] = True

        async def load_vae(self):
            captured["load_vae"] = True

        async def decode(self, latent):
            return latent

        async def decode_tiled(self, latent, tile_size=256):
            captured["tile_size"] = tile_size
            return latent

        async def unload_vae(self):
            captured["unload_vae"] = True

    monkeypatch.setattr(
        "fusion_mlx.engines.video.resolve_backend", lambda *a, **k: FakeBackend()
    )
    eng = VideoGenEngine("fake-model")
    asyncio.run(eng.load_text_encoder())
    assert asyncio.run(eng.encode_text("p")) == {"embed": "E"}
    asyncio.run(eng.unload_text_encoder())
    asyncio.run(eng.load_dit())
    asyncio.run(eng.denoise("L", "P", None, 3, 4.0, 0, 16))
    asyncio.run(eng.unload_dit())
    asyncio.run(eng.load_vae())
    asyncio.run(eng.decode("L"))
    asyncio.run(eng.decode_tiled("L", tile_size=128))
    asyncio.run(eng.unload_vae())
    assert captured["load_text_encoder"]
    assert captured["encode_text"] == "p"
    assert captured["denoise"] == (3, 16)
    assert captured["tile_size"] == 128


def test_video_backend_default_stage_methods_not_implemented():
    class BareBackend(VideoBackend):
        name = "bare"

        @classmethod
        def detect(cls, model_path):
            return False

        async def start(self, model_path, **kwargs):
            pass

        async def stop(self):
            pass

        async def generate(self, params):
            return []

        def constraints(self):
            from fusion_mlx.engines.video_backends.base import VideoConstraints

            return VideoConstraints()

    b = BareBackend()
    with pytest.raises(NotImplementedError, match="issue #170 phase 2"):
        asyncio.run(b.load_text_encoder())
    with pytest.raises(NotImplementedError, match="issue #170 phase 2"):
        asyncio.run(b.decode(mx.zeros((1, 8, 4, 4))))
    with pytest.raises(NotImplementedError, match="issue #170 phase 2"):
        asyncio.run(b.denoise(mx.zeros((1, 8, 4, 4)), None, None, 3, 1.0, 0, 16))
