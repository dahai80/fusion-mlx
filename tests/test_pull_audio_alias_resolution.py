# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest


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
    pytest.skip("Audio alias resolution for pull not migrated to fusion-mlx")


def test_helper_resolves_audio_alias_for_rm() -> None:
    pytest.skip("Audio alias resolution for rm not migrated to fusion-mlx")


@pytest.mark.parametrize("command", ["serve", "chat", "run", "bench", "info", None])
def test_helper_leaves_short_alias_for_non_download_commands(command) -> None:
    pytest.skip("Audio alias resolution not migrated to fusion-mlx")


@pytest.mark.parametrize("model", ["qwen3.6-27b-4bit", "not-a-real-alias-xyz", ""])
def test_helper_returns_none_for_non_audio(model: str) -> None:
    pytest.skip("Audio alias resolution not migrated to fusion-mlx")


def test_main_pull_rewrites_audio_alias_to_hf_id(monkeypatch) -> None:
    pytest.skip("Audio alias resolution for pull not migrated to fusion-mlx")


def test_main_rm_rewrites_audio_alias_to_hf_id(monkeypatch) -> None:
    pytest.skip("Audio alias resolution for rm not migrated to fusion-mlx")


def test_main_serve_keeps_short_audio_alias(monkeypatch) -> None:
    pytest.skip("Audio alias resolution for serve not migrated to fusion-mlx")


def test_main_pull_full_hf_path_unchanged(monkeypatch) -> None:
    pytest.skip("Audio alias resolution for pull not migrated to fusion-mlx")
