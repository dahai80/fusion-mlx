# SPDX-License-Identifier: Apache-2.0
"""Audio model alias probe (stub — always returns False in this build)."""

import sys

_AUDIO_ALIAS_TOKENS = frozenset()


def is_audio_model_alias(name: str) -> bool:
    return name.lower() in _AUDIO_ALIAS_TOKENS


def require_audio_or_exit():
    print("Audio models are not available in this build", file=sys.stderr)
    sys.exit(1)
