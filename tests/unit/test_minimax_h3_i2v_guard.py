# SPDX-License-Identifier: Apache-2.0
# Issue #589: MiniMax-H3 only implements t2va. i2va/l2va/fl2va image and
# last-frame conditioning are accepted then silently dropped (silent-wrong-video
# bug). Guard: declare supports_i2v=False so validate_params rejects image= at
# the API (422), and raise in generate() for image/last_frame_image/reference_audio
# (validate_params does not check last_frame_image, so generate() is the backstop).
import pytest

from fusion_mlx.engines.video_backends import MiniMaxH3Backend, constraints_for
from fusion_mlx.engines.video_backends.base import VideoGenParams, validate_params


class TestSupportsI2vFalse:
    def test_class_supports_i2v_false(self):
        assert MiniMaxH3Backend.supports_i2v is False

    def test_constraints_supports_i2v_false(self):
        b = MiniMaxH3Backend("/models/h3-fl2va")
        c = b.constraints()
        assert c.supports_i2v is False

    def test_constraints_for_helper_supports_i2v_false(self):
        c = constraints_for("/models/MiniMax-H3-FL2VA")
        assert c.supports_i2v is False

    def test_validate_params_rejects_image(self):
        # validate_params must reject image= for H3 (backend does not implement i2v).
        c = MiniMaxH3Backend("/models/h3-fl2va").constraints()
        with pytest.raises(ValueError):
            validate_params(
                c, num_frames=97, width=768, height=768, n=1, image="/img/a.png"
            )


class TestGenerateRejectsConditioning:
    def _backend(self):
        b = MiniMaxH3Backend("/nonexistent/h3-model")
        b._loaded = True
        return b

    async def test_generate_rejects_image(self):
        # Even if an image slips past validation (direct call), generate() must
        # fail loudly rather than silently run t2va and discard the image.
        # validate_params (supports_i2v=False) raises first with "I2V"; the
        # generate() guard raises with "i2va". Match case-insensitive "i2v".
        b = self._backend()
        p = VideoGenParams(
            prompt="test", num_frames=97, width=768, height=768, n=1, image="/i.png"
        )
        with pytest.raises(ValueError, match="(?i)i2v"):
            await b.generate(p)

    async def test_generate_rejects_last_frame_image(self):
        # last_frame_image is l2va/fl2va conditioning, not checked by
        # validate_params; generate() is the only backstop, so it must raise.
        b = self._backend()
        p = VideoGenParams(
            prompt="test",
            num_frames=97,
            width=768,
            height=768,
            n=1,
            last_frame_image="/last.png",
        )
        with pytest.raises(ValueError, match="l2va"):
            await b.generate(p)

    async def test_generate_rejects_reference_audio(self):
        b = self._backend()
        p = VideoGenParams(
            prompt="test",
            num_frames=97,
            width=768,
            height=768,
            n=1,
            reference_audio=["/a.wav"],
        )
        with pytest.raises(ValueError, match="reference_audio"):
            await b.generate(p)

    async def test_generate_allows_plain_t2va(self):
        # The guard must NOT reject plain t2va (no conditioning) — that path
        # stays open. Expect a non-guard failure (model load / NotImplementedError),
        # but NOT the i2va/l2va ValueError.
        b = self._backend()
        p = VideoGenParams(prompt="test", num_frames=97, width=768, height=768, n=1)
        raised = None
        try:
            await b.generate(p)
        except ValueError as e:
            raised = e
        except Exception as e:
            raised = e
        # If a ValueError fired, it must be the conditioning guard only — and
        # plain t2va has no conditioning, so no ValueError guard should fire.
        if isinstance(raised, ValueError):
            assert "i2va" not in str(raised)
            assert "l2va" not in str(raised)
            assert "reference_audio" not in str(raised)
