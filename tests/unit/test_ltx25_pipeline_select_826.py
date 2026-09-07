# SPDX-License-Identifier: Apache-2.0
# Tests for issue #826/#827: per-request pipeline (dev vs distilled) selection
# for the LTX-2.5 and legacy ltx2 video backends. No real model weights — these
# validate the forwarding plumbing (route -> VideoGenParams -> backend generate)
# by monkeypatching the executor-bound _generate_one so no MLX weights load.

import asyncio
from unittest import mock

import pytest

from fusion_mlx.api.videos_routes import VideoGenerateRequest
from fusion_mlx.engines.video_backends.base import VideoGenParams
from fusion_mlx.engines.video_backends.ltx2 import LTX2Backend
from fusion_mlx.engines.video_backends.ltx2_5 import LTX2_5Backend


def test_video_generate_request_accepts_pipeline_field():
    # The route model must accept a pipeline field; default None so existing
    # requests are unaffected (fall back to backend construction-time default).
    req = VideoGenerateRequest(prompt="a cat")
    assert req.pipeline is None
    req_dev = VideoGenerateRequest(prompt="a cat", pipeline="dev")
    assert req_dev.pipeline == "dev"


def test_video_gen_params_carries_pipeline():
    # VideoGenParams.pipeline must exist and default to None; the engine layer
    # already populates it from kwargs (engines/video.py:100), so a route that
    # sets gen_kwargs["pipeline"] must see it land on params.pipeline.
    params = VideoGenParams(prompt="x", pipeline="dev")
    assert params.pipeline == "dev"
    params_default = VideoGenParams(prompt="x")
    assert params_default.pipeline is None


def test_ltx2_5_generate_uses_request_pipeline_override():
    # #826/#827: when params.pipeline is set, the backend must pass it to
    # _generate_one instead of self._pipeline (the construction-time default).
    backend = LTX2_5Backend("dgrauet/ltx-2.5-mlx-q8", pipeline="distilled")
    assert backend._pipeline == "distilled"
    captured: dict = {}

    async def fake_run_in_executor(executor, fn, *a):
        fn()
        return [b"x"]

    def fake_generate_one(model_name, pipeline, **kwargs):
        captured["pipeline"] = pipeline
        return b"x"

    params = VideoGenParams(prompt="a cat", num_frames=9, pipeline="dev")
    with (
        mock.patch(
            "fusion_mlx.engines.video_backends.ltx2_5._generate_one",
            side_effect=fake_generate_one,
        ),
        mock.patch(
            "fusion_mlx.engines.video_backends.ltx2_5.get_executor", return_value="io"
        ),
        mock.patch(
            "fusion_mlx.engines.video_backends.ltx2_5.get_video_gen_timeout",
            return_value=60.0,
        ),
        mock.patch(
            "fusion_mlx.engines.video_backends.ltx2_5.asyncio.get_running_loop"
        ) as mloop,
    ):
        mloop.return_value.run_in_executor = fake_run_in_executor
        result = asyncio.run(backend.generate(params))
    assert result == [b"x"]
    assert captured["pipeline"] == "dev", "per-request pipeline override not forwarded"


def test_ltx2_5_generate_falls_back_to_default_when_request_omits_pipeline():
    # Existing requests (pipeline=None) must keep using the construction-time
    # default — no behavior change for callers that don't set the new field.
    backend = LTX2_5Backend("dgrauet/ltx-2.5-mlx-q8", pipeline="distilled")
    captured: dict = {}

    async def fake_run_in_executor(executor, fn, *a):
        fn()
        return [b"x"]

    def fake_generate_one(model_name, pipeline, **kwargs):
        captured["pipeline"] = pipeline
        return b"x"

    params = VideoGenParams(prompt="a cat", num_frames=9)
    with (
        mock.patch(
            "fusion_mlx.engines.video_backends.ltx2_5._generate_one",
            side_effect=fake_generate_one,
        ),
        mock.patch(
            "fusion_mlx.engines.video_backends.ltx2_5.get_executor", return_value="io"
        ),
        mock.patch(
            "fusion_mlx.engines.video_backends.ltx2_5.get_video_gen_timeout",
            return_value=60.0,
        ),
        mock.patch(
            "fusion_mlx.engines.video_backends.ltx2_5.asyncio.get_running_loop"
        ) as mloop,
    ):
        mloop.return_value.run_in_executor = fake_run_in_executor
        asyncio.run(backend.generate(params))
    assert captured["pipeline"] == "distilled", "default pipeline regressed"


def test_ltx2_legacy_generate_uses_request_pipeline_override():
    # Legacy ltx2 backend: same plumbing for PipelineType.DEV /
    # DEV_TWO_STAGE_HQ. The backend forwards params.pipeline or self._pipeline
    # as the third positional arg to _generate_one.
    backend = LTX2Backend("ltx-video-2b", pipeline="distilled")
    captured: dict = {}

    async def fake_run_in_executor(executor, fn, *a):
        fn()
        return [b"x"]

    def fake_generate_one(model_name, text_encoder_repo, pipeline, **kwargs):
        captured["pipeline"] = pipeline
        return b"x"

    params = VideoGenParams(prompt="a cat", num_frames=9, pipeline="dev")
    with (
        mock.patch(
            "fusion_mlx.engines.video_backends.ltx2._generate_one",
            side_effect=fake_generate_one,
        ),
        mock.patch(
            "fusion_mlx.engines.video_backends.ltx2.get_executor", return_value="io"
        ),
        mock.patch(
            "fusion_mlx.engines.video_backends.ltx2.get_video_gen_timeout",
            return_value=60.0,
        ),
        mock.patch(
            "fusion_mlx.engines.video_backends.ltx2.asyncio.get_running_loop"
        ) as mloop,
    ):
        mloop.return_value.run_in_executor = fake_run_in_executor
        asyncio.run(backend.generate(params))
    assert captured["pipeline"] == "dev", "legacy ltx2 pipeline override not forwarded"


def test_ltx2_5_variant_from_str_rejects_unknown_pipeline():
    # An invalid pipeline value must fail visibly (the backend's from_str
    # raises ValueError) rather than silently degrading to distilled.
    from fusion_mlx.video.ltx2_5.config import LTX2_5Variant

    with pytest.raises(ValueError):
        LTX2_5Variant.from_str("turbo")
