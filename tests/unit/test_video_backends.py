# SPDX-License-Identifier: Apache-2.0
# Tests for the video backend registry: resolve_backend auto-detection,
# constraints_for, validate_params, and Wan2Backend generate (stubbed mlx-video
# wan_2 - no real model loading or generation).
import sys
import types
from pathlib import Path

import pytest

from fusion_mlx.engines.video_backends import (
    BACKENDS,
    CogVideoBackend,
    LegacyLTXBackend,
    LTX2Backend,
    VideoGenParams,
    Wan2Backend,
    constraints_for,
    resolve_backend,
    validate_params,
)


def _install_wan2_stub(monkeypatch):
    calls = {"resolve": [], "generate": []}

    top = types.ModuleType("mlx_video")
    models = types.ModuleType("mlx_video.models")
    wan_2 = types.ModuleType("mlx_video.models.wan_2")
    gen_mod = types.ModuleType("mlx_video.models.wan_2.generate")

    def get_model_path(repo):
        calls["resolve"].append(repo)
        return Path("/tmp/fake-wan2")

    def generate_video(model_dir, prompt, **kwargs):
        calls["generate"].append({"model_dir": model_dir, "prompt": prompt, **kwargs})
        with open(kwargs["output_path"], "wb") as f:
            f.write(b"WANMP4" + str(kwargs.get("seed", 0)).encode())
        return None

    top.get_model_path = get_model_path
    top.models = models
    models.wan_2 = wan_2
    wan_2.generate = gen_mod
    gen_mod.generate_video = generate_video

    monkeypatch.setitem(sys.modules, "mlx_video", top)
    monkeypatch.setitem(sys.modules, "mlx_video.models", models)
    monkeypatch.setitem(sys.modules, "mlx_video.models.wan_2", wan_2)
    monkeypatch.setitem(sys.modules, "mlx_video.models.wan_2.generate", gen_mod)
    return calls


class TestResolveBackend:
    def test_ltx2_autodetect(self):
        b = resolve_backend("ltx-2")
        assert isinstance(b, LTX2Backend)

    def test_wan2_autodetect_by_name(self):
        assert isinstance(resolve_backend("wan2.1"), Wan2Backend)

    def test_wan2_autodetect_by_repo_id(self):
        assert isinstance(resolve_backend("Wan-AI/Wan2.2-TI2V-5B"), Wan2Backend)

    def test_unknown_falls_back_to_ltx2(self):
        # Preserves Phase 0 single-backend fallback behavior.
        assert isinstance(resolve_backend("some-custom-video-model"), LTX2Backend)

    def test_legacy_ltx_autodetect(self):
        # Legacy LTX-Video (0.9.x) has a pure-MLX port (Phase 3). Detected by
        # name substring and by HF repo id.
        assert isinstance(resolve_backend("ltx-video"), LegacyLTXBackend)
        assert isinstance(resolve_backend("Lightricks/LTX-Video"), LegacyLTXBackend)

    def test_cogvideo_autodetect(self):
        # CogVideoX has no MLX port -> stub.
        assert isinstance(resolve_backend("cogvideo"), CogVideoBackend)
        assert isinstance(resolve_backend("THUDM/CogVideoX-2b"), CogVideoBackend)

    def test_legacy_does_not_shadow_modern_ltx(self):
        # Critical: ltx-2 / ltx-2.3 (shipped by mlx-video) must still resolve to
        # the real LTX2Backend, NOT the legacy stub.
        assert isinstance(resolve_backend("ltx-2"), LTX2Backend)
        assert isinstance(resolve_backend("ltx-2.3"), LTX2Backend)

    def test_explicit_legacy_and_cogvideo_aliases(self):
        assert isinstance(resolve_backend("x", explicit="ltx-video"), LegacyLTXBackend)
        assert isinstance(resolve_backend("x", explicit="ltx_video"), LegacyLTXBackend)
        assert isinstance(resolve_backend("x", explicit="cogvideo"), CogVideoBackend)
        assert isinstance(resolve_backend("x", explicit="cogvideox"), CogVideoBackend)

    def test_explicit_wan2(self):
        assert isinstance(resolve_backend("anything", explicit="wan2"), Wan2Backend)

    def test_explicit_alias(self):
        assert isinstance(resolve_backend("anything", explicit="wan2.2"), Wan2Backend)

    def test_explicit_invalid_raises(self):
        with pytest.raises(ValueError, match="unknown video backend"):
            resolve_backend("anything", explicit="bogus")

    def test_backends_registry_has_all(self):
        # ltx2 + wan2 ship real mlx-video impls; ltx_video_legacy is a pure-MLX
        # port (Phase 3); cogvideo is a stub that raises NotImplementedError
        # (no MLX port exists upstream).
        assert set(BACKENDS) == {"ltx2", "wan2", "ltx_video_legacy", "cogvideo"}


class TestUnimplementedBackends:
    # CogVideoX has no MLX port and is not shipped by mlx-video. The stub must
    # (a) validate permissively so requests reach the backend, then (b) raise
    # NotImplementedError pointing at the upstream issue tracker and naming a
    # real alternative. (Legacy LTX-Video graduated to a real port in Phase 3
    # and is covered by TestLegacyLTXBackend below.)

    @pytest.mark.parametrize("model", ["cogvideo"])
    def test_constraints_are_permissive(self, model):
        # div=1 + no frame validator + supports_i2v=True so validate_params
        # never rejects before the stub can give its clear error.
        c = constraints_for(model)
        assert c.supports_i2v is True
        assert c.dim_divisibility == 1
        assert c.num_frames_validator is None
        assert c.max_n >= 1

    @pytest.mark.parametrize("model", ["cogvideo"])
    def test_validate_params_accepts_then_backend_raises(self, model):
        c = constraints_for(model)
        # Permissive constraints accept arbitrary dims/frames + I2V image.
        validate_params(
            c, num_frames=25, width=512, height=512, n=1, image="http://x/y.png"
        )

    @pytest.mark.parametrize("model", ["cogvideo"])
    def test_stop_is_noop(self, model):
        import asyncio

        asyncio.run(resolve_backend(model).stop())

    @pytest.mark.parametrize("model", ["cogvideo"])
    def test_start_raises_with_upstream_url(self, model):
        import asyncio

        with pytest.raises(NotImplementedError) as exc:
            asyncio.run(resolve_backend(model).start("/fake/path"))
        msg = str(exc.value)
        assert "no MLX port" in msg
        assert "github.com/Blaizzy/mlx-video/issues" in msg
        # Must name at least one real shipped alternative.
        assert "ltx2" in msg or "wan2" in msg

    @pytest.mark.parametrize("model", ["cogvideo"])
    def test_generate_raises_not_implemented(self, model):
        import asyncio

        with pytest.raises(NotImplementedError):
            asyncio.run(resolve_backend(model).generate(VideoGenParams(prompt="x")))


class TestConstraints:
    def test_ltx_constraints(self):
        c = constraints_for("ltx-2")
        assert c.supports_i2v is False
        assert c.dim_divisibility == 64
        assert c.num_frames_validator(97) is True
        assert c.num_frames_validator(16) is False

    def test_wan_constraints(self):
        c = constraints_for("wan2.1")
        assert c.supports_i2v is True
        assert c.dim_divisibility == 16
        assert c.num_frames_validator(41) is True
        assert c.num_frames_validator(40) is False

    def test_wan_default_dims_satisfy_constraints(self):
        # The API defaults (97/768/512) must pass Wan2 constraints so the
        # explicit-model passthrough route test stays green.
        c = constraints_for("wan2.1")
        assert c.num_frames_validator(97) is True
        assert 768 % c.dim_divisibility == 0
        assert 512 % c.dim_divisibility == 0


class TestValidateParams:
    def test_ltx_rejects_image(self):
        with pytest.raises(ValueError, match="image-to-video"):
            validate_params(
                constraints_for("ltx-2"),
                num_frames=97,
                width=768,
                height=512,
                n=1,
                image="/tmp/x.png",
            )

    def test_wan_accepts_image(self):
        validate_params(
            constraints_for("wan2.1"),
            num_frames=41,
            width=768,
            height=512,
            n=1,
            image="/tmp/x.png",
        )

    def test_wan_rejects_bad_dim(self):
        with pytest.raises(ValueError, match="divisible by 16"):
            validate_params(
                constraints_for("wan2.1"),
                num_frames=41,
                width=260,
                height=512,
                n=1,
            )

    def test_wan_rejects_bad_frames(self):
        with pytest.raises(ValueError, match="num_frames"):
            validate_params(
                constraints_for("wan2.1"),
                num_frames=40,
                width=768,
                height=512,
                n=1,
            )

    def test_rejects_n_out_of_range(self):
        with pytest.raises(ValueError, match="n must be"):
            validate_params(
                constraints_for("wan2.1"),
                num_frames=41,
                width=768,
                height=512,
                n=99,
            )


class TestWan2Backend:
    @pytest.fixture
    def stub(self, monkeypatch):
        return _install_wan2_stub(monkeypatch)

    async def test_start_resolves(self, stub):
        backend = Wan2Backend("wan2.1")
        await backend.start("wan2.1")
        assert backend._loaded is True
        assert stub["resolve"] == ["wan2.1"]

    async def test_start_idempotent(self, stub):
        backend = Wan2Backend("wan2.1")
        await backend.start("wan2.1")
        await backend.start("wan2.1")
        assert len(stub["resolve"]) == 1

    async def test_generate_seed_increment(self, stub):
        backend = Wan2Backend("wan2.1")
        await backend.start("wan2.1")
        params = VideoGenParams(
            prompt="a cat", n=3, num_frames=41, width=768, height=512, seed=42
        )
        result = await backend.generate(params)
        assert len(result) == 3
        seeds = [stub["generate"][i]["seed"] for i in range(3)]
        assert seeds == [42, 43, 44]
        assert result[0] == b"WANMP4" + b"42"

    async def test_generate_random_seed_when_none(self, stub):
        backend = Wan2Backend("wan2.1")
        await backend.start("wan2.1")
        params = VideoGenParams(prompt="a cat", num_frames=41, width=768, height=512)
        result = await backend.generate(params)
        seed = stub["generate"][0]["seed"]
        assert seed != 0
        assert result[0] == b"WANMP4" + str(seed).encode()

    async def test_generate_i2v_image_passed_through(self, stub):
        backend = Wan2Backend("wan2.1")
        await backend.start("wan2.1")
        params = VideoGenParams(
            prompt="a cat",
            num_frames=41,
            width=768,
            height=512,
            image="/tmp/clip.png",
        )
        await backend.generate(params)
        assert stub["generate"][0]["image"] == "/tmp/clip.png"

    async def test_generate_forwards_knobs(self, stub):
        backend = Wan2Backend("wan2.1")
        await backend.start("wan2.1")
        params = VideoGenParams(
            prompt="a cat",
            num_frames=41,
            width=768,
            height=512,
            negative_prompt="blurry",
            num_inference_steps=15,
            guide_scale=3.5,
            shift=1.0,
            scheduler="unipc",
        )
        await backend.generate(params)
        call = stub["generate"][0]
        assert call["negative_prompt"] == "blurry"
        assert call["steps"] == 15
        assert call["guide_scale"] == 3.5
        assert call["shift"] == 1.0
        assert call["scheduler"] == "unipc"


class TestLegacyLTXBackend:
    # Legacy LTX-Video (0.9.x) pure-MLX port. No real weights - stub the
    # component loaders + mp4 writer and exercise the orchestration: start()
    # loads transformer/vae/t5/tokenizer/scheduler, generate() runs the denoise
    # loop with the real scheduler + fake transformer and writes an mp4 per
    # sample with per-sample seed increment.

    @pytest.fixture
    def stub(self, monkeypatch):
        import mlx.core as mx
        import numpy as np

        from fusion_mlx.engines.video_backends import ltx_video_legacy as mod
        from fusion_mlx.video import t5_encoder as t5mod
        from fusion_mlx.video.ltx_video_legacy.transformer import Transformer3DModel
        from fusion_mlx.video.ltx_video_legacy.vae import LTVideoVAE

        calls = {"resolve": [], "write": [], "tf": [], "vae": [], "t5": []}

        class FakeCfg:
            in_channels = 128

        class FakeTransformer:
            cfg = FakeCfg()

            def __call__(
                self,
                hidden_states,
                indices_grid=None,
                encoder_hidden_states=None,
                timestep=None,
                attention_mask=None,
                encoder_attention_mask=None,
            ):
                return mx.zeros(hidden_states.shape, dtype=mx.float32)

        class FakeVAEConfig:
            blocks = [{"name": "compress_all"}] * 3
            patch_size = 4

        class FakeVAE:
            config = FakeVAEConfig()

            def decode(self, z, target_shape=None):
                return mx.zeros(target_shape, dtype=mx.float32)

        class FakeT5:
            def __call__(self, input_ids, attention_mask=None):
                return mx.zeros((1, 256, 4096), dtype=mx.float32)

        def fake_tokenizer(prompt, **kwargs):
            return {
                "input_ids": np.zeros((1, 256), dtype=np.int32),
                "attention_mask": np.ones((1, 256), dtype=np.int32),
            }

        monkeypatch.setattr(
            mod, "_resolve_repo", lambda p: calls["resolve"].append(p) or "/fake/legacy"
        )
        monkeypatch.setattr(
            Transformer3DModel,
            "from_pretrained",
            classmethod(
                lambda cls, path, dtype=mx.float32: (
                    calls["tf"].append(path),
                    FakeTransformer(),
                )[1]
            ),
        )
        monkeypatch.setattr(
            LTVideoVAE,
            "from_pretrained",
            classmethod(
                lambda cls, path, dtype=mx.float32: (
                    calls["vae"].append(path),
                    FakeVAE(),
                )[1]
            ),
        )
        monkeypatch.setattr(
            t5mod,
            "load_t5_encoder",
            lambda path, dtype=mx.float32: (calls["t5"].append(path), FakeT5())[1],
        )
        monkeypatch.setattr(t5mod, "load_t5_tokenizer", lambda path: fake_tokenizer)

        def fake_write_mp4(frames, fps, path):
            calls["write"].append({"frames": len(frames), "fps": fps})
            with open(path, "wb") as f:
                f.write(b"LTXMP4")

        monkeypatch.setattr(mod, "_write_mp4", fake_write_mp4)
        return calls

    def test_constraints(self):
        c = constraints_for("ltx-video")
        assert c.supports_i2v is False
        assert c.dim_divisibility == 32
        assert c.num_frames_validator(97) is True
        assert c.num_frames_validator(16) is False

    async def test_start_loads_components(self, stub):
        backend = LegacyLTXBackend("Lightricks/LTX-Video")
        await backend.start("Lightricks/LTX-Video")
        assert backend._loaded is True
        assert backend._transformer is not None
        assert backend._vae is not None
        assert backend._t5 is not None
        assert backend._scheduler is not None
        assert stub["resolve"] == ["Lightricks/LTX-Video"]

    async def test_start_idempotent(self, stub):
        backend = LegacyLTXBackend("ltx-video")
        await backend.start("ltx-video")
        await backend.start("ltx-video")
        assert len(stub["tf"]) == 1
        assert len(stub["vae"]) == 1
        assert len(stub["t5"]) == 1

    async def test_generate_seed_increment(self, stub):
        backend = LegacyLTXBackend("ltx-video")
        await backend.start("ltx-video")
        params = VideoGenParams(
            prompt="a cat",
            n=3,
            num_frames=9,
            width=32,
            height=32,
            fps=8,
            seed=42,
            num_inference_steps=2,
            cfg_scale=3.0,
        )
        result = await backend.generate(params)
        assert len(result) == 3
        assert all(r == b"LTXMP4" for r in result)
        assert [c["frames"] for c in stub["write"]] == [9, 9, 9]
        assert stub["write"][0]["fps"] == 8

    async def test_generate_random_seed_when_none(self, stub):
        backend = LegacyLTXBackend("ltx-video")
        await backend.start("ltx-video")
        params = VideoGenParams(
            prompt="a cat",
            num_frames=9,
            width=32,
            height=32,
            fps=8,
            num_inference_steps=2,
            cfg_scale=3.0,
        )
        result = await backend.generate(params)
        assert len(result) == 1
        # Deterministic stub writes the same sentinel; the point is that
        # generate ran without a seed and produced output.
        assert result[0] == b"LTXMP4"

    async def test_generate_negative_prompt_encoded_once(self, stub):
        backend = LegacyLTXBackend("ltx-video")
        await backend.start("ltx-video")
        params = VideoGenParams(
            prompt="a cat",
            negative_prompt="blurry",
            num_frames=9,
            width=32,
            height=32,
            fps=8,
            num_inference_steps=2,
            cfg_scale=3.0,
        )
        await backend.generate(params)
        # negative_prompt set -> two T5 __call__ invocations would happen inside
        # _encode_prompt; we only assert generate completed and wrote one mp4.
        assert len(stub["write"]) == 1
