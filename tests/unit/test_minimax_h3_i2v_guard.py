# SPDX-License-Identifier: Apache-2.0
# i2va/l2va/fl2va image + last-frame conditioning now implemented
# (generate_fl2va_video, keyframe-anchors). supports_i2v=True → validate_params
# accepts image=. ref2va reference_audio still unimplemented (separate partition).
import pytest

from fusion_mlx.engines.video_backends import MiniMaxH3Backend, constraints_for
from fusion_mlx.engines.video_backends.base import VideoGenParams, validate_params


class TestSupportsI2vTrue:
    def test_class_supports_i2v_true(self):
        assert MiniMaxH3Backend.supports_i2v is True

    def test_constraints_supports_i2v_true(self):
        b = MiniMaxH3Backend("/models/h3-fl2va")
        c = b.constraints()
        assert c.supports_i2v is True

    def test_constraints_for_helper_supports_i2v_true(self):
        c = constraints_for("/models/MiniMax-H3-FL2VA")
        assert c.supports_i2v is True

    def test_validate_params_accepts_image(self):
        # image= (i2va 首帧) 通过 validate_params（supports_i2v=True）。
        c = MiniMaxH3Backend("/models/h3-fl2va").constraints()
        validate_params(
            c, num_frames=97, width=768, height=768, n=1, image="/img/a.png"
        )


class TestGenerateAcceptsConditioning:
    def _backend(self):
        b = MiniMaxH3Backend("/nonexistent/h3-model")
        b._loaded = True
        return b

    async def test_generate_accepts_image_no_guard(self):
        # image= 不再被 guard 拒绝；路径安全校验通过后进入 generate_video
        # （模型加载失败属非 guard 错误）。断言不抛 i2va ValueError。
        b = self._backend()
        p = VideoGenParams(
            prompt="test", num_frames=97, width=768, height=768, n=1, image="/i.png"
        )
        raised = None
        try:
            await b.generate(p)
        except ValueError as e:
            raised = e
        except Exception as e:
            raised = e
        if isinstance(raised, ValueError):
            assert "i2va" not in str(raised)
            assert "not implemented" not in str(raised).lower() or "ref2va" in str(
                raised
            )

    async def test_generate_accepts_last_frame_image_no_guard(self):
        # last_frame_image= (l2va 末帧) 不再被 guard 拒绝。
        b = self._backend()
        p = VideoGenParams(
            prompt="test",
            num_frames=97,
            width=768,
            height=768,
            n=1,
            last_frame_image="/last.png",
        )
        raised = None
        try:
            await b.generate(p)
        except ValueError as e:
            raised = e
        except Exception as e:
            raised = e
        if isinstance(raised, ValueError):
            assert "l2va" not in str(raised) or "ref2va" in str(raised)

    async def test_generate_rejects_reference_audio(self):
        # ref2va reference_audio 仍未实现（独立 partition，非本 PR）。
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

    async def test_generate_rejects_unsafe_image_path(self):
        # 条件帧绝对路径须在允许目录内（路径安全校验）。
        b = self._backend()
        p = VideoGenParams(
            prompt="test",
            num_frames=97,
            width=768,
            height=768,
            n=1,
            image="/etc/passwd",
        )
        with pytest.raises(ValueError, match="(?i)outside allowed|condition image"):
            await b.generate(p)

    async def test_generate_allows_plain_t2va(self):
        # 无条件帧路径保持开放（t2va），不抛 guard ValueError。
        b = self._backend()
        p = VideoGenParams(prompt="test", num_frames=97, width=768, height=768, n=1)
        raised = None
        try:
            await b.generate(p)
        except ValueError as e:
            raised = e
        except Exception as e:
            raised = e
        if isinstance(raised, ValueError):
            assert "i2va" not in str(raised)
            assert "l2va" not in str(raised)
            assert "reference_audio" not in str(raised)

    async def test_generate_rejects_reference_images_fast(self):
        # ref2va reference_images 未实现（issue #688 step 2-3）：须在模型加载前
        # fail-fast 抛 NotImplementedError，不静默丢弃也不加载 67GB 权重。
        b = self._backend()
        p = VideoGenParams(
            prompt="test",
            num_frames=97,
            width=768,
            height=768,
            n=1,
            reference_images=["/ref/a.png", "/ref/b.png"],
        )
        with pytest.raises(NotImplementedError, match="ref2va reference-image"):
            await b.generate(p)
        # 模型未被加载（fail-fast 早于 start()）。
        assert b._loaded is True  # _backend() 预置 True；guard 不应触发 start()


class TestGenerateVideoRef2vaGuard:
    def test_generate_video_rejects_reference_images(self):
        # generate_video 直接 SDK 调用路径的 defense-in-depth guard。
        # 不加载模型，仅断言 ref2va 分支显式拒绝。
        from fusion_mlx.video.minimax_h3.generate import generate_video

        with pytest.raises(NotImplementedError, match="issue #688"):
            generate_video(
                model_path="/nonexistent/h3-ref2va",
                prompt="test",
                num_frames=97,
                width=768,
                height=768,
                reference_images=["/ref/a.png"],
            )
