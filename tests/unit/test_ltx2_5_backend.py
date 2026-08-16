# SPDX-License-Identifier: Apache-2.0
# P7 checkpoint: LTX-2.5 backend detect / registry / constraints / generate.
from __future__ import annotations

import asyncio

import pytest

from fusion_mlx.engines.video_backends import (
    _ALIASES,
    BACKENDS,
    LTX2_5Backend,
    LTX2Backend,
    constraints_for,
    resolve_backend,
)
from fusion_mlx.engines.video_backends.base import VideoGenParams


class TestDetect:
    @pytest.mark.parametrize(
        "path",
        [
            "Lightricks/LTX-2.5",
            "ltx-2.5-22b-distilled",
            "ltx_2.5",
            "ltx2.5",
            "ltx-2.5-distilled",
            "/models/LTX-2.5-Diffusers",
        ],
    )
    def test_detect_positive(self, path):
        assert LTX2_5Backend.detect(path) is True

    @pytest.mark.parametrize(
        "path",
        [
            "Lightricks/LTX-Video",
            "ltx-2",
            "ltx-2.3",
            "ltx2.3",
            "wan2.1",
            "cosmos-predict2",
            "hunyuanvideo",
        ],
    )
    def test_detect_negative(self, path):
        assert LTX2_5Backend.detect(path) is False

    def test_no_false_hit_on_ltx2(self):
        assert LTX2Backend.detect("Lightricks/LTX-2.5") is False
        assert LTX2Backend.detect("ltx-2.5-22b-distilled") is False

    def test_ltx2_still_matches_2(self):
        assert LTX2Backend.detect("some-ltx-2-model") is True
        assert LTX2Backend.detect("ltx-2.3") is True


class TestRegistry:
    def test_registered(self):
        assert BACKENDS["ltx2_5"] is LTX2_5Backend

    def test_registered_before_ltx2(self):
        keys = list(BACKENDS.keys())
        assert keys.index("ltx2_5") < keys.index("ltx2")

    @pytest.mark.parametrize(
        "alias",
        ["ltx-2.5", "ltx_2.5", "ltx2.5", "ltx-2.5-distilled"],
    )
    def test_aliases_resolve(self, alias):
        assert _ALIASES[alias] == "ltx2_5"

    def test_resolve_via_alias(self):
        b = resolve_backend("Lightricks/LTX-2.5", explicit="ltx-2.5")
        assert isinstance(b, LTX2_5Backend)

    def test_resolve_via_detect_prefers_25(self):
        b = resolve_backend("Lightricks/LTX-2.5")
        assert isinstance(b, LTX2_5Backend)

    def test_resolve_ltx2_unaffected(self):
        b = resolve_backend("Lightricks/LTX-Video")
        assert not isinstance(b, LTX2_5Backend)

    def test_resolve_ltx2_23_unaffected(self):
        b = resolve_backend("ltx-2.3-model")
        assert isinstance(b, LTX2Backend)
        assert not isinstance(b, LTX2_5Backend)

    def test_existing_backends_unaffected(self):
        for path, expected_name in [
            ("cosmos-predict2", "cosmos"),
            ("wan2.1", "wan2"),
            ("hunyuanvideo", "hunyuanvideo"),
        ]:
            b = resolve_backend(path)
            assert b.name == expected_name


class TestConstruction:
    def test_defaults(self):
        b = LTX2_5Backend("Lightricks/LTX-2.5")
        assert b.name == "ltx2_5"
        assert b.supports_i2v is True
        assert b._pipeline == "distilled"
        assert b._two_stage is True

    def test_custom_pipeline(self):
        b = LTX2_5Backend("repo", pipeline="dev")
        assert b._pipeline == "dev"

    def test_two_stage_off(self):
        b = LTX2_5Backend("repo", two_stage=False)
        assert b._two_stage is False


class TestConstraints:
    def test_supports_i2v(self):
        c = LTX2_5Backend("repo").constraints()
        assert c.supports_i2v is True

    def test_dim_divisibility_32(self):
        c = LTX2_5Backend("repo").constraints()
        assert c.dim_divisibility == 32

    def test_max_n(self):
        c = LTX2_5Backend("repo").constraints()
        assert c.max_n == 4

    def test_frames_validator_mod8_plus1(self):
        c = LTX2_5Backend("repo").constraints()
        assert c.num_frames_validator is not None
        for nf in [1, 9, 17, 25, 121]:
            assert c.num_frames_validator(nf) is True
        for nf in [2, 8, 10, 16]:
            assert c.num_frames_validator(nf) is False

    def test_constraints_for(self):
        c = constraints_for("Lightricks/LTX-2.5")
        assert c.dim_divisibility == 32


class TestGenerate:
    def test_generate_passes_params_to_port(self, monkeypatch):
        # Stub _generate_one (no real 22B weights) — verify generate() forwards
        # params + seeds each sample in the n loop.
        calls = {"calls": []}

        def fake_generate_one(model_repo, pipeline, **kwargs):
            calls["calls"].append({"repo": model_repo, "pipeline": pipeline, **kwargs})
            return b"mp4"

        import fusion_mlx.engines.video_backends.ltx2_5 as mod

        monkeypatch.setattr(mod, "_generate_one", fake_generate_one)
        b = LTX2_5Backend("Lightricks/LTX-2.5")
        params = VideoGenParams(
            prompt="a cat",
            num_frames=9,
            width=512,
            height=320,
            fps=24,
            seed=1,
            n=2,
            cfg_scale=4.0,
            tiling="auto",
        )
        out = asyncio.run(b.generate(params))
        assert out == [b"mp4", b"mp4"]
        assert len(calls["calls"]) == 2
        c0 = calls["calls"][0]
        assert c0["repo"] == "Lightricks/LTX-2.5"
        assert c0["pipeline"] == "distilled"
        assert c0["prompt"] == "a cat"
        assert c0["num_frames"] == 9
        assert c0["width"] == 512
        assert c0["height"] == 320
        assert c0["seed"] == 1
        assert c0["seed"] + 1 == calls["calls"][1]["seed"]
        assert c0["two_stage"] is True

    def test_generate_two_stage_off_propagated(self, monkeypatch):
        calls = {"calls": []}

        def fake_generate_one(model_repo, pipeline, **kwargs):
            calls["calls"].append(kwargs)
            return b"mp4"

        import fusion_mlx.engines.video_backends.ltx2_5 as mod

        monkeypatch.setattr(mod, "_generate_one", fake_generate_one)
        b = LTX2_5Backend("repo", two_stage=False)
        params = VideoGenParams(
            prompt="a cat", num_frames=9, width=512, height=320, n=1, seed=0
        )
        asyncio.run(b.generate(params))
        assert calls["calls"][0]["two_stage"] is False

    def test_generate_i2v_passes_image(self, monkeypatch):
        calls = {"calls": []}

        def fake_generate_one(model_repo, pipeline, **kwargs):
            calls["calls"].append(kwargs)
            return b"mp4"

        import fusion_mlx.engines.video_backends.ltx2_5 as mod

        monkeypatch.setattr(mod, "_generate_one", fake_generate_one)
        b = LTX2_5Backend("repo")
        params = VideoGenParams(
            prompt="a cat",
            num_frames=9,
            width=512,
            height=320,
            n=1,
            seed=0,
            image="/tmp/frame.png",
            image_strength=0.8,
        )
        asyncio.run(b.generate(params))
        assert calls["calls"][0]["image"] == "/tmp/frame.png"
        assert calls["calls"][0]["image_strength"] == 0.8
