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

    async def test_generate_forwards_reference_images_no_rejection(self):
        # ref2va reference_images 已实现（issue #688 step 2-3）：backend 不再
        # 抛 NotImplementedError 拒绝，而是做路径安全校验后转发给 generate_video。
        # 路径在允许目录内（/ref/... 不触发拒绝）→ 进入 model load（nonexistent 失败，
        # 非 guard ValueError/NotImplementedError）。
        b = self._backend()
        p = VideoGenParams(
            prompt="test",
            num_frames=97,
            width=768,
            height=768,
            n=1,
            reference_images=["/ref/a.png", "/ref/b.png"],
        )
        raised = None
        try:
            await b.generate(p)
        except NotImplementedError as e:
            raised = e
        except Exception as e:
            raised = e
        # 不应抛 ref2va NotImplementedError（已实现）。
        if isinstance(raised, NotImplementedError):
            assert "ref2va reference-image" not in str(raised)

    async def test_generate_rejects_unsafe_reference_image_path(self):
        # reference_images 绝对路径须在允许目录内（路径安全校验，#688 step 2-3）。
        b = self._backend()
        p = VideoGenParams(
            prompt="test",
            num_frames=97,
            width=768,
            height=768,
            n=1,
            reference_images=["/etc/passwd"],
        )
        with pytest.raises(ValueError, match="(?i)outside allowed|reference image"):
            await b.generate(p)


class TestGenerateVideoRef2vaBranch:
    def test_generate_video_has_ref2va_branch(self):
        # generate_video 直接 SDK 路径：ref2va 分支已实现（不再 NotImplementedError）。
        # 断言源码含 ref2va 分支标记，且不再含 reference_images 的 NotImplementedError。
        import inspect

        from fusion_mlx.video.minimax_h3.generate import generate_video

        src = inspect.getsource(generate_video)
        assert "is_ref2va" in src
        assert "load_multimodal_text_encoder" in src
        assert "_encode_prompt_ref2va" in src
        # 旧 gate 已移除（issue #688 step 2-3 已实现）。
        assert "ref2va reference-image generation is not implemented" not in src

    def test_generate_video_rejects_empty_reference_images(self):
        # defense-in-depth：空 reference_images 列表显式拒绝（fail-visible）。
        from fusion_mlx.video.minimax_h3.generate import generate_video

        with pytest.raises(ValueError, match="(?i)non-empty"):
            generate_video(
                model_path="/nonexistent/h3-ref2va",
                prompt="test",
                num_frames=97,
                width=768,
                height=768,
                reference_images=[],
            )
