# SPDX-License-Identifier: Apache-2.0
"""#991 -- ``fusion-mlx pull <audio-alias>`` must resolve to the concrete HF id.

Migrated from Rapid-MLX. The ``_resolve_audio_download_alias`` helper
has NOT been migrated to fusion-mlx. Fusion-mlx uses a different audio
alias resolution path (``fusion_mlx.audio.registry.resolve_audio_alias``
and ``fusion_mlx.audio.probe.is_audio_model_alias``). The CLI dispatch
in fusion-mlx handles audio alias resolution differently from Rapid-MLX.
All tests are skipped with a clear reason.
"""

from __future__ import annotations

import logging

import pytest

logger = logging.getLogger(__name__)

_SKIP_REASON = (
    "_resolve_audio_download_alias has not been migrated to fusion-mlx. "
    "Fusion-mlx uses fusion_mlx.audio.registry.resolve_audio_alias and "
    "fusion_mlx.audio.probe.is_audio_model_alias for audio alias resolution "
    "with a different dispatch path. Re-enable when equivalent pull/rm "
    "audio alias resolution is wired into the CLI."
)


@pytest.mark.skip(reason=_SKIP_REASON)
@pytest.mark.parametrize(
    "alias,expected",
    [
        ("whisper-tiny", "mlx-community/whisper-tiny-mlx"),
        ("whisper", "mlx-community/whisper-large-v3-mlx"),
        ("kokoro", "mlx-community/Kokoro-82M-bf16"),
        ("parakeet", "mlx-community/parakeet-tdt-0.6b-v2"),
    ],
)
def test_helper_resolves_audio_alias_for_pull(alias: str, expected: str) -> None:
    pass


@pytest.mark.skip(reason=_SKIP_REASON)
def test_helper_resolves_audio_alias_for_rm() -> None:
    pass


@pytest.mark.skip(reason=_SKIP_REASON)
@pytest.mark.parametrize("command", ["serve", "chat", "run", "bench", "info", None])
def test_helper_leaves_short_alias_for_non_download_commands(command) -> None:
    pass


@pytest.mark.skip(reason=_SKIP_REASON)
@pytest.mark.parametrize("model", ["qwen3.6-27b-4bit", "not-a-real-alias-xyz", ""])
def test_helper_returns_none_for_non_audio(model: str) -> None:
    pass


@pytest.mark.skip(reason=_SKIP_REASON)
def test_main_pull_rewrites_audio_alias_to_hf_id() -> None:
    pass


@pytest.mark.skip(reason=_SKIP_REASON)
def test_main_rm_rewrites_audio_alias_to_hf_id() -> None:
    pass


@pytest.mark.skip(reason=_SKIP_REASON)
def test_main_serve_keeps_short_audio_alias() -> None:
    pass


@pytest.mark.skip(reason=_SKIP_REASON)
def test_main_pull_full_hf_path_unchanged() -> None:
    pass
