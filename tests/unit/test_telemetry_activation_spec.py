# SPDX-License-Identifier: Apache-2.0
from fusion_mlx.telemetry import activation_spec as spec


def test_spec_version_is_3():
    assert spec.ACTIVATION_SPEC_VERSION == 3


def test_nine_milestone_kinds():
    expected = {
        "first_inference",
        "model_pull",
        "agent_setup",
        "first_chat_reply",
        "first_vision_reply",
        "first_dictation",
        "first_image",
        "first_image_generation",
        "first_video_generation",
    }
    assert expected == spec.ACTIVATION_KINDS


def test_multimodal_pairs_on_api_surface():
    assert ("first_image_generation", "api") in spec.ACTIVATION_KIND_SURFACE_PAIRS
    assert ("first_video_generation", "api") in spec.ACTIVATION_KIND_SURFACE_PAIRS


def test_is_allowed_activation_rejects_unknown():
    assert spec.is_allowed_activation("first_inference", "api") is True
    assert spec.is_allowed_activation("bogus", "api") is False
    assert spec.is_allowed_activation("first_inference", "bogus") is False


def test_chat_spawn_env_renamed():
    assert spec.CHAT_SPAWN_ENV == "FUSION_MLX_CHAT_SPAWN"


def test_inference_endpoints_chat_only():
    assert frozenset({"/v1/chat/completions"}) == spec.INFERENCE_ENDPOINTS


def test_is_successful_inference_2xx_nonempty():
    assert spec.is_successful_inference(200, 5) is True
    assert spec.is_successful_inference(200, 0) is False
    assert spec.is_successful_inference(500, 5) is False
    assert spec.is_successful_inference(299, 1) is True
    assert spec.is_successful_inference(300, 5) is False
