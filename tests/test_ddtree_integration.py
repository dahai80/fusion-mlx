# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest


def test_serve_parser_exposes_enable_ddtree() -> None:
    pytest.skip("DDTree not migrated to fusion-mlx")


def test_speculative_config_ddtree_preflight_uses_config_overrides(
    monkeypatch,
) -> None:
    pytest.skip("DDTree not migrated to fusion-mlx")


def test_speculative_config_ddtree_preflight_falls_back_to_alias_defaults(
    monkeypatch,
) -> None:
    pytest.skip("DDTree not migrated to fusion-mlx")


def test_enable_ddtree_legacy_flag_is_speculative_config_shorthand(
    monkeypatch,
) -> None:
    pytest.skip("DDTree not migrated to fusion-mlx")


def test_enable_ddtree_conflicts_with_explicit_speculative_config(capsys) -> None:
    pytest.skip("DDTree not migrated to fusion-mlx")


def test_info_renders_ddtree_block_for_eligible_alias(capsys, monkeypatch) -> None:
    pytest.skip("DDTree not migrated to fusion-mlx")


def test_info_ddtree_marks_4bit_alias_ineligible(capsys) -> None:
    pytest.skip("DDTree not migrated to fusion-mlx")


def test_models_listing_renders_ddtree_column(capsys) -> None:
    pytest.skip("DDTree not migrated to fusion-mlx")


def test_build_app_healthz_models_and_completion() -> None:
    pytest.skip("DDTree not migrated to fusion-mlx")


def test_build_app_healthz_works_while_runtime_loads() -> None:
    pytest.skip("DDTree not migrated to fusion-mlx")


def test_build_app_honors_api_key_and_model_name() -> None:
    pytest.skip("DDTree not migrated to fusion-mlx")


def test_build_app_runtime_load_failure_is_sanitized() -> None:
    pytest.skip("DDTree not migrated to fusion-mlx")


def test_chat_completions_rejects_unsupported_ddtree_params() -> None:
    pytest.skip("DDTree not migrated to fusion-mlx")


def test_run_ddtree_server_loads_runtime_on_separate_executor(monkeypatch) -> None:
    pytest.skip("DDTree not migrated to fusion-mlx")
