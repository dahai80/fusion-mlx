# SPDX-License-Identifier: Apache-2.0
"""Tests for central patch registry."""

from __future__ import annotations

import pytest


def test_register_and_apply(reset_registry):
    from fusion_mlx.patches import apply_all_for_model, register

    called = []
    register(
        "test_patch_1",
        target="qwen3",
        reason="test reason",
        apply_fn=lambda model, config: called.append("patch_1"),
    )
    apply_all_for_model("qwen3")
    assert called == ["patch_1"]


def test_idempotent_double_apply(reset_registry):
    from fusion_mlx.patches import apply_all_for_model, register

    called = []
    register(
        "test_patch_2",
        target="llama",
        reason="idempotency test",
        apply_fn=lambda model, config: called.append("fired"),
    )
    apply_all_for_model("llama")
    apply_all_for_model("llama")  # second call should be no-op
    assert called == ["fired"]


def test_target_mismatch_skips(reset_registry):
    from fusion_mlx.patches import apply_all_for_model, register

    called = []
    register(
        "qwen_only_patch",
        target="qwen3",
        reason="targeted",
        apply_fn=lambda model, config: called.append("hit"),
    )
    apply_all_for_model("llama")  # wrong target
    assert called == []


def test_global_patch_applies_to_all(reset_registry):
    from fusion_mlx.patches import apply_all_for_model, register

    called = []
    register(
        "global_patch",
        target="global",
        reason="unconditional",
        apply_fn=lambda model, config: called.append("global"),
        is_global=True,
    )
    apply_all_for_model("llama")
    apply_all_for_model("qwen3")
    assert called == ["global"]  # idempotent: only once


def test_list_patches(reset_registry):
    from fusion_mlx.patches import list_patches, register

    register("p_a", target="qwen3", reason="reason A", apply_fn=lambda m, c: None)
    register(
        "p_b",
        target="global",
        reason="reason B",
        apply_fn=lambda m, c: None,
        is_global=True,
    )
    patches = list_patches()
    assert len(patches) == 2
    ids = [p["patch_id"] for p in patches]
    assert "p_a" in ids and "p_b" in ids


def test_duplicate_register_ignored(reset_registry):
    from fusion_mlx.patches import list_patches, register

    register("dup", target="x", reason="first", apply_fn=lambda m, c: None)
    register("dup", target="x", reason="second", apply_fn=lambda m, c: None)
    patches = list_patches()
    assert len(patches) == 1
    assert patches[0]["reason"] == "first"


def test_patch_failure_does_not_crash(reset_registry):
    from fusion_mlx.patches import apply_all_for_model, register

    def bad_fn(model, config):
        raise RuntimeError("boom")

    register("bad_patch", target="x", reason="fails", apply_fn=bad_fn)
    # Should not raise
    apply_all_for_model("x")


@pytest.fixture
def reset_registry():
    from fusion_mlx.patches import reset_for_tests

    reset_for_tests()
    yield
    reset_for_tests()
