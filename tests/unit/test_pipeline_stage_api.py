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
    pytest.importorskip("mflux")
    eng, flux = _make_engine_with_fake_flux()
    with pytest.raises(ValueError, match="batch,c,h,w"):
        asyncio.run(eng.decode(mx.zeros((1, 8, 4))))
    latent = mx.zeros((1, 8, 4, 4))
    out = asyncio.run(eng.decode(latent))
    assert out is latent


def test_image_decode_tiled_passes_tiling_config():
    pytest.importorskip("mflux")
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
    pytest.importorskip("mflux")
    eng, flux = _make_engine_with_fake_flux()
    with pytest.raises(ValueError, match="batch,c,h,w"):
        asyncio.run(eng.denoise(mx.zeros((1, 8, 4)), None, None, 3, 1.0, 0))
    flux.transformer = None
    with pytest.raises(RuntimeError, match="transformer .* unloaded"):
        asyncio.run(eng.denoise(mx.zeros((1, 8, 4, 4)), None, None, 3, 1.0, 0))


def test_image_denoise_loop_runs_and_returns_4d(monkeypatch):
    pytest.importorskip("mflux")
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
    pytest.importorskip("mflux")
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

        async def denoise(
            self, latent, pos, neg, steps, cfg, seed, num_frames, control=None
        ):
            captured["denoise"] = (steps, num_frames, control)
            return latent

        async def unload_dit(self):
            captured["unload_dit"] = True

        async def load_vae_encoder(self):
            captured["load_vae_encoder"] = True

        async def encode_control(self, **kwargs):
            captured["encode_control"] = kwargs
            return "CTRL"

        async def unload_vae_encoder(self):
            captured["unload_vae_encoder"] = True

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
    asyncio.run(eng.denoise("L", "P", None, 3, 4.0, 0, 16, control="C"))
    asyncio.run(eng.unload_dit())
    asyncio.run(eng.load_vae_encoder())
    assert asyncio.run(eng.encode_control(image="i.png")) == "CTRL"
    asyncio.run(eng.unload_vae_encoder())
    asyncio.run(eng.load_vae())
    asyncio.run(eng.decode("L"))
    asyncio.run(eng.decode_tiled("L", tile_size=128))
    asyncio.run(eng.unload_vae())
    assert captured["load_text_encoder"]
    assert captured["encode_text"] == "p"
    assert captured["denoise"] == (3, 16, "C")
    assert captured["load_vae_encoder"]
    assert captured["encode_control"] == {"image": "i.png"}
    assert captured["unload_vae_encoder"]
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
    # #652 conditioning stubs default-raise on backends that don't override them.
    with pytest.raises(NotImplementedError, match="issue #652 conditioning"):
        asyncio.run(b.load_vae_encoder())
    with pytest.raises(NotImplementedError, match="issue #652 conditioning"):
        asyncio.run(b.encode_control(image="x.png"))
    with pytest.raises(NotImplementedError, match="issue #652 conditioning"):
        asyncio.run(b.unload_vae_encoder())


def test_video_encode_not_implemented_base():
    # Backends without an encode override must raise the stage-API default,
    # matching every other unimplemented VideoBackend stage method.
    class _StubBackend(VideoBackend):
        name = "stub"

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

    with pytest.raises(NotImplementedError, match="stage API not implemented"):
        asyncio.run(_StubBackend().encode(mx.zeros((1, 1, 8, 8, 3))))


def test_encode_wan_vae_returns_4d_latent():
    from fusion_mlx.video.wan2.stage import encode_wan_vae

    fake_vae = SimpleNamespace(encode=lambda x: mx.zeros((16, 3, 8, 16)))
    config = {"vae_z_dim": 16}
    out = encode_wan_vae(mx.zeros((1, 3, 7, 512, 512)), config, fake_vae)
    assert out.shape == (16, 3, 8, 16)


def _make_wan2_backend_for_encode():
    from fusion_mlx.engines.video_backends.wan2 import Wan2Backend

    backend = Wan2Backend.__new__(Wan2Backend)
    backend.name = "wan2"
    backend._stage_vae_encoder = None
    backend._stage_flags = {"text_encoder": False, "dit": False, "vae": False}
    backend._stage_config = {"vae_z_dim": 16}
    backend._model_dir = "/fake/wan2"
    backend._stage_quant = None
    return backend


class _InlineExecutor:
    # run_in_executor(ex, fn) runs fn on the calling (main) thread so mx.eval
    # finds the MLX stream the test process owns. The real video worker thread
    # has no stream without a loaded model — the executor path is exercised by
    # the Tier 3 real-model roundtrip, not these unit tests. asyncio's
    # run_in_executor needs a concurrent.futures.Future; submit returns one
    # already-resolved with the inline result.
    def submit(self, fn, *args, **kwargs):
        import concurrent.futures

        fut = concurrent.futures.Future()
        try:
            fut.set_result(fn(*args, **kwargs))
        except Exception as exc:
            fut.set_exception(exc)
        return fut


def _patch_video_executor(monkeypatch):
    import fusion_mlx.engines.video_backends.wan2 as wan2_mod

    monkeypatch.setattr(
        wan2_mod, "get_executor", lambda name: _InlineExecutor(), raising=False
    )


def test_wan2_encode_lazy_loads_encoder(monkeypatch):
    backend = _make_wan2_backend_for_encode()
    fake_vae = SimpleNamespace(encode=lambda x: mx.zeros((16, 3, 8, 16)))
    loaded = {"calls": 0}

    async def fake_load(self):
        loaded["calls"] += 1
        self._stage_vae_encoder = fake_vae
        self._stage_flags["vae_encoder"] = True

    monkeypatch.setattr(
        "fusion_mlx.engines.video_backends.wan2.Wan2Backend._load_vae_encoder_stage",
        fake_load,
    )
    monkeypatch.setattr(
        "fusion_mlx.engines.video_backends.wan2.Wan2Backend._ensure_stage_config",
        lambda self: self._stage_config,
    )
    _patch_video_executor(monkeypatch)
    out = asyncio.run(backend.encode(mx.zeros((1, 7, 512, 512, 3))))
    assert loaded["calls"] == 1
    assert backend._stage_vae_encoder is fake_vae
    assert out.shape == (1, 16, 3, 8, 16)


def test_wan2_encode_shape(monkeypatch):
    backend = _make_wan2_backend_for_encode()
    backend._stage_vae_encoder = SimpleNamespace(
        encode=lambda x: mx.zeros((16, 3, 8, 16))
    )
    monkeypatch.setattr(
        "fusion_mlx.engines.video_backends.wan2.Wan2Backend._ensure_stage_config",
        lambda self: self._stage_config,
    )
    _patch_video_executor(monkeypatch)
    out = asyncio.run(backend.encode(mx.zeros((1, 7, 512, 512, 3))))
    assert out.ndim == 5
    assert out.shape == (1, 16, 3, 8, 16)


def test_wan2_encode_layout(monkeypatch):
    backend = _make_wan2_backend_for_encode()
    captured = {}

    class _FakeVae:
        def encode(self, x):
            captured["x"] = x
            return mx.zeros((16, 3, 8, 16))

    backend._stage_vae_encoder = _FakeVae()
    monkeypatch.setattr(
        "fusion_mlx.engines.video_backends.wan2.Wan2Backend._ensure_stage_config",
        lambda self: self._stage_config,
    )
    _patch_video_executor(monkeypatch)
    asyncio.run(backend.encode(mx.zeros((1, 7, 512, 512, 3))))
    assert captured["x"].shape == (1, 3, 7, 512, 512)


def test_wan2_encode_ndim_guard(monkeypatch):
    backend = _make_wan2_backend_for_encode()
    backend._stage_vae_encoder = SimpleNamespace(encode=lambda x: x)
    monkeypatch.setattr(
        "fusion_mlx.engines.video_backends.wan2.Wan2Backend._ensure_stage_config",
        lambda self: self._stage_config,
    )
    with pytest.raises(ValueError, match="encode expects"):
        asyncio.run(backend.encode(mx.zeros((512, 512, 3))))


def test_wan2_unload_vae_frees_encoder(monkeypatch):
    backend = _make_wan2_backend_for_encode()
    backend._stage_vae_encoder = SimpleNamespace(encode=lambda x: x)
    backend._stage_flags["vae_encoder"] = True

    async def fake_clear():
        pass

    monkeypatch.setattr(
        "fusion_mlx.engines.video_backends.wan2._clear_mlx_cache", fake_clear
    )
    asyncio.run(backend.unload_vae())
    assert backend._stage_vae_encoder is None
    assert "vae_encoder" not in backend._stage_flags


def test_wan2_stop_clears_vae_encoder(monkeypatch):
    backend = _make_wan2_backend_for_encode()
    backend._loaded = True
    backend._stage_vae_encoder = SimpleNamespace(encode=lambda x: x)
    backend._stage_flags["vae_encoder"] = True
    backend._embed_cache_lock = __import__("threading").Lock()
    backend._embed_cache = {}

    import mlx.core as mx

    import fusion_mlx.engines.video_backends.wan2 as wan2_mod

    monkeypatch.setattr(wan2_mod, "get_executor", lambda name: _InlineExecutor())
    monkeypatch.setattr(mx, "synchronize", lambda: None)
    monkeypatch.setattr(mx, "clear_cache", lambda: None)
    asyncio.run(backend.stop())
    assert backend._stage_vae_encoder is None
    assert "vae_encoder" not in backend._stage_flags


def test_video_engine_encode_delegates_to_backend():
    engine = VideoGenEngine.__new__(VideoGenEngine)
    captured = {"pixels": None}

    class _FakeBackend:
        async def encode(self, pixels):
            captured["pixels"] = pixels
            return mx.zeros((1, 16, 3, 8, 16))

    engine._backend = _FakeBackend()
    out = asyncio.run(engine.encode(mx.zeros((1, 7, 512, 512, 3))))
    assert out.shape == (1, 16, 3, 8, 16)
    assert captured["pixels"] is not None
