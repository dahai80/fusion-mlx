# SPDX-License-Identifier: Apache-2.0
"""Audio model registry (stub — no audio models in this build)."""

_AUDIO_ALIASES: dict[str, str] = {}


def resolve_audio_alias(name: str):
    return None


def list_audio_aliases() -> dict[str, str]:
    return _AUDIO_ALIASES
