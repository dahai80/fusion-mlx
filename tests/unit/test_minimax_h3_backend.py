# SPDX-License-Identifier: Apache-2.0
# P5 Backend + registration checkpoint：MiniMaxH3Backend detect/constraints/resolve。
import pytest

from fusion_mlx.engines.video_backends import (
    BACKENDS,
    MiniMaxH3Backend,
    constraints_for,
    resolve_backend,
)
from fusion_mlx.engines.video_backends.base import VideoGenParams, validate_params


class TestDetect:
    @pytest.mark.parametrize(
        "path",
        [
            "/models/MiniMax-H3-FL2VA",
            "/models/minimax_h3_ref2va",
            "/models/h3-fl2va",
            "/models/h3-ref2va",
            "/models/FL2VA",
            "/models/Ref2VA",
        ],
    )
    def test_detect_positive(self, path):
        assert MiniMaxH3Backend.detect(path) is True

    @pytest.mark.parametrize(
        "path",
        [
            "/models/wan2.1",
            "/models/cosmos-predict2",
            "/models/ltx-video",
            "/models/skyreels-v3",
            "/models/cogvideox",
        ],
    )
    def test_detect_no_false_hit(self, path):
        # 不应误命中既有 10 个后端路径。
        assert MiniMaxH3Backend.detect(path) is False


class TestRegistry:
    def test_registered(self):
        assert BACKENDS["minimax_h3"] is MiniMaxH3Backend

    @pytest.mark.parametrize(
        "alias",
        [
            "minimax-h3",
            "minimax_h3",
            "h3",
            "h3-fl2va",
            "h3-ref2va",
            "fl2va",
            "ref2va",
        ],
    )
    def test_aliases_resolve(self, alias):
        b = resolve_backend("/models/x", explicit=alias)
        assert isinstance(b, MiniMaxH3Backend)

    def test_resolve_via_detect(self):
        b = resolve_backend("/models/MiniMax-H3-FL2VA")
        assert isinstance(b, MiniMaxH3Backend)

    def test_existing_backend_unaffected(self):
        # 确保注册 H3 不破坏既有后端。
        b = resolve_backend("/models/cosmos-predict2")
        assert b.name == "cosmos"
        b2 = resolve_backend("/models/wan2.1")
        assert b2.name == "wan2"

    def test_constraints_for(self):
        c = constraints_for("/models/MiniMax-H3-FL2VA")
        assert c.supports_i2v is True
        assert c.max_n == 1
        assert c.dim_divisibility == 16


class TestConstruction:
    def test_defaults(self):
        b = MiniMaxH3Backend("/models/h3-fl2va")
        assert b._partition == "fl2va"
        assert b._resolution == "768p"

    def test_ref2va_path_hint(self):
        b = MiniMaxH3Backend("/models/h3-ref2va")
        assert b._partition == "ref2va"

    def test_explicit_partition(self):
        b = MiniMaxH3Backend("/models/x", partition="ref2va")
        assert b._partition == "ref2va"

    def test_invalid_partition(self):
        with pytest.raises(ValueError):
            MiniMaxH3Backend("/models/x", partition="bad")

    def test_invalid_resolution(self):
        with pytest.raises(ValueError):
            MiniMaxH3Backend("/models/x", resolution="4k")

    def test_resolution_2k(self):
        b = MiniMaxH3Backend("/models/x", resolution="2k")
        assert b._resolution == "2k"


class TestConstraints:
    def test_frames_validator(self):
        b = MiniMaxH3Backend("/models/x")
        c = b.constraints()
        assert c.num_frames_validator(1) is True
        assert c.num_frames_validator(361) is True
        assert c.num_frames_validator(362) is False
        assert c.num_frames_validator(0) is False

    def test_validate_params_ok(self):
        b = MiniMaxH3Backend("/models/x")
        c = b.constraints()
        validate_params(c, num_frames=97, width=768, height=768, n=1, image=None)

    def test_validate_params_n_too_large(self):
        b = MiniMaxH3Backend("/models/x")
        c = b.constraints()
        with pytest.raises(ValueError):
            validate_params(c, num_frames=97, width=768, height=768, n=2, image=None)

    def test_validate_params_dim(self):
        b = MiniMaxH3Backend("/models/x")
        c = b.constraints()
        with pytest.raises(ValueError):
            validate_params(c, num_frames=97, width=770, height=768, n=1, image=None)


class TestNewParams:
    def test_new_fields_default(self):
        p = VideoGenParams(prompt="test")
        assert p.last_frame_image is None
        assert p.reference_audio is None
        assert p.resolution == "768p"
        assert p.use_prompt_skill is None

    def test_new_fields_set(self):
        p = VideoGenParams(
            prompt="test",
            last_frame_image="/img/last.png",
            reference_audio=["/a/1.wav"],
            resolution="2k",
            use_prompt_skill=True,
        )
        assert p.last_frame_image == "/img/last.png"
        assert p.reference_audio == ["/a/1.wav"]
        assert p.resolution == "2k"
        assert p.use_prompt_skill is True


class TestGenerate:
    @pytest.mark.asyncio
    async def test_generate_not_implemented(self):
        b = MiniMaxH3Backend("/models/x")
        p = VideoGenParams(prompt="test", num_frames=97, width=768, height=768, n=1)
        with pytest.raises(NotImplementedError):
            await b.generate(p)
