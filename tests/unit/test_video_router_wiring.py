# SPDX-License-Identifier: Apache-2.0
"""#gap4: VideoRouter wired into the video API path + /v1/videos/route endpoint.

Verifies the router picks ltx2_5 for general prompts and minimax_h3 for drama
prompts when no explicit model is requested, and that the /route endpoint
reports the decision + loaded status.
"""

from fusion_mlx.api.videos_routes import (
    VideoRouteRequest,
    VideoRouteResponse,
    _resolve_video_model,
)
from fusion_mlx.pipeline.video_router import get_video_router


def test_route_general_prompt_ltx():
    r = get_video_router().route("a UI demo scrolling animation", audio=False)
    assert r == "ltx2_5"


def test_route_drama_prompt_h3():
    r = get_video_router().route("短剧 人物 对话 嘴型", audio=False)
    assert r == "minimax_h3"


def test_route_model_hint_overrides():
    r = get_video_router().route("短剧", model_hint="ltx", audio=False)
    assert r == "ltx2_5"


def test_resolve_with_explicit_model_passthrough():
    # explicit model request bypasses routing
    assert _resolve_video_model("短剧", False, "ltx-2") == "ltx-2"


def test_resolve_drama_no_loaded_falls_back_default():
    # no pool loaded video engines -> returns a default alias (404 path explains)
    mid = _resolve_video_model("短剧 人物", False, None)
    # minimax_h3 routed, not loaded -> default alias for 404
    assert mid == "minimax-h3"


def test_resolve_general_no_loaded_falls_back_ltx():
    mid = _resolve_video_model("UI demo animation", False, None)
    assert mid == "ltx-2"


def test_route_request_model():
    req = VideoRouteRequest(prompt="短剧", audio=True)
    assert req.prompt == "短剧"
    assert req.audio is True


def test_route_response_model():
    resp = VideoRouteResponse(backend="minimax_h3", loaded=False, loaded_model=None)
    assert resp.backend == "minimax_h3"
    assert resp.loaded is False
