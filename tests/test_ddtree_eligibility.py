# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest


def test_check_passes_for_good_profile() -> None:
    pytest.skip("DDTree not migrated to fusion-mlx")


def test_check_rejects_alias_without_supports_ddtree() -> None:
    pytest.skip("DDTree not migrated to fusion-mlx")


def test_check_rejects_moe_alias() -> None:
    pytest.skip("DDTree not migrated to fusion-mlx")


def test_check_rejects_4bit_main_model() -> None:
    pytest.skip("DDTree not migrated to fusion-mlx")


def test_report_collects_all_failures() -> None:
    pytest.skip("DDTree not migrated to fusion-mlx")


def test_qwen3_5_9b_8bit_alias_passes_check() -> None:
    pytest.skip("DDTree not migrated to fusion-mlx")


def test_qwen3_5_9b_4bit_alias_fails_with_4bit_reason() -> None:
    pytest.skip("DDTree not migrated to fusion-mlx")


def test_runtime_patches_rope_parameters_without_copying_weights(
    tmp_path, monkeypatch
) -> None:
    pytest.skip("DDTree not migrated to fusion-mlx")


def test_runtime_replaces_stale_ddtree_patch_dir(tmp_path, monkeypatch) -> None:
    pytest.skip("DDTree not migrated to fusion-mlx")


def test_runtime_cleans_temp_patch_dir_on_write_failure(tmp_path, monkeypatch) -> None:
    pytest.skip("DDTree not migrated to fusion-mlx")


def test_eligible_aliases_surfaces_alias_registry_errors(monkeypatch) -> None:
    pytest.skip("DDTree not migrated to fusion-mlx")


def test_runtime_patches_qwen35_split_prefill() -> None:
    pytest.skip("DDTree not migrated to fusion-mlx")
