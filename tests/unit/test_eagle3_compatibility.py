# SPDX-License-Identifier: Apache-2.0
# Phase-2 item 2: Eagle3 is_compatible family guard.
#
# Eagle3 draft weights are trained against a specific target family
# (llama3 / qwen3). Running a llama3-trained Eagle3 against a Qwen
# target (or vice versa) produces garbage drafts silently. The guard
# in Eagle3Speculator.is_compatible() must return False on a family
# mismatch and True on a match, so engine_core._init_draft() can
# disable spec decode instead of emitting garbage.

from __future__ import annotations

from fusion_mlx.speculative.eagle3.speculator import (
    Eagle3DraftConfig,
    Eagle3Speculator,
)


def _spec(key: str) -> Eagle3Speculator:
    return Eagle3Speculator(Eagle3DraftConfig(draft_model_key=key))


def test_llama3_draft_matches_llama_target():
    spec = _spec("llama3.1-8b")
    assert spec.is_compatible("mlx-community/Meta-Llama-3.1-8B-Instruct-4bit") is True


def test_llama3_draft_rejects_qwen_target():
    spec = _spec("llama3.1-8b")
    assert spec.is_compatible("mlx-community/Qwen3-8B-4bit") is False


def test_qwen3_draft_matches_qwen_target():
    spec = _spec("qwen3-8b")
    assert spec.is_compatible("mlx-community/Qwen3-8B-4bit") is True


def test_qwen3_draft_rejects_llama_target():
    spec = _spec("qwen3-8b")
    assert spec.is_compatible("mlx-community/Meta-Llama-3.1-8B-Instruct-4bit") is False


def test_local_path_basename_matched():
    # A local model dir ~/.fusion-mlx/models/.../Llama-3.1-... must match
    # the llama3 family even though "llama" only appears in the basename.
    spec = _spec("llama3.1-8b")
    assert (
        spec.is_compatible(
            "/Users/dahai/.fusion-mlx/models/mlx-community/Llama-3.1-8B-Instruct-4bit"
        )
        is True
    )


def test_empty_name_allowed_when_no_matcher():
    # Unknown family -> no matcher -> allow (best-effort), but should not crash.
    spec = _spec("llama3.1-8b")
    assert spec.is_compatible("") is False  # empty never matches llama3


def test_case_insensitive_match():
    spec = _spec("llama3.1-8b")
    assert spec.is_compatible("LLAMA-3.1-something") is True
