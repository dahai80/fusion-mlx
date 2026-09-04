# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

ACTIVATION_SPEC_VERSION = 3

ACTIVATION_FIRST_INFERENCE = "first_inference"
ACTIVATION_MODEL_PULL = "model_pull"
ACTIVATION_AGENT_SETUP = "agent_setup"
ACTIVATION_FIRST_CHAT_REPLY = "first_chat_reply"
ACTIVATION_FIRST_VISION_REPLY = "first_vision_reply"
ACTIVATION_FIRST_DICTATION = "first_dictation"
ACTIVATION_FIRST_IMAGE = "first_image"
ACTIVATION_FIRST_IMAGE_GENERATION = "first_image_generation"
ACTIVATION_FIRST_VIDEO_GENERATION = "first_video_generation"

SURFACE_CLI = "cli"
SURFACE_API = "api"
SURFACE_DESKTOP = "desktop"

ACTIVATION_KIND_SURFACE_PAIRS: frozenset[tuple[str, str]] = frozenset(
    {
        (ACTIVATION_FIRST_INFERENCE, SURFACE_CLI),
        (ACTIVATION_FIRST_INFERENCE, SURFACE_API),
        (ACTIVATION_MODEL_PULL, SURFACE_CLI),
        (ACTIVATION_AGENT_SETUP, SURFACE_CLI),
        (ACTIVATION_FIRST_CHAT_REPLY, SURFACE_DESKTOP),
        (ACTIVATION_FIRST_VISION_REPLY, SURFACE_DESKTOP),
        (ACTIVATION_FIRST_DICTATION, SURFACE_DESKTOP),
        (ACTIVATION_FIRST_IMAGE, SURFACE_DESKTOP),
        (ACTIVATION_FIRST_IMAGE_GENERATION, SURFACE_API),
        (ACTIVATION_FIRST_VIDEO_GENERATION, SURFACE_API),
    }
)
ACTIVATION_KINDS: frozenset[str] = frozenset(
    kind for kind, _ in ACTIVATION_KIND_SURFACE_PAIRS
)
ACTIVATION_SURFACES: frozenset[str] = frozenset(
    surface for _, surface in ACTIVATION_KIND_SURFACE_PAIRS
)
DESKTOP_ACTIVATION_KINDS: frozenset[str] = frozenset(
    kind
    for kind, surface in ACTIVATION_KIND_SURFACE_PAIRS
    if surface == SURFACE_DESKTOP
)


def is_allowed_activation(activation_kind: str, surface: str) -> bool:
    return (activation_kind, surface) in ACTIVATION_KIND_SURFACE_PAIRS


CHAT_SPAWN_ENV = "FUSION_MLX_CHAT_SPAWN"

INFERENCE_ENDPOINTS: frozenset[str] = frozenset({"/v1/chat/completions"})


def is_successful_inference(status: int, completion_tokens: int) -> bool:
    try:
        status_ok = 200 <= int(status) < 300
        nonempty = int(completion_tokens) > 0
    except (TypeError, ValueError):
        logger.warning(
            "activation_spec: bad inference args status=%r tokens=%r",
            status,
            completion_tokens,
        )
        return False
    return status_ok and nonempty
