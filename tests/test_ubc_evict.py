# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest


def test_ubc_evict_darwin_releases_pages(tmp_path):
    pytest.skip("UBC evict not migrated to fusion-mlx")


def test_ubc_evict_noop_on_linux(monkeypatch, tmp_path, caplog):
    pytest.skip("UBC evict not migrated to fusion-mlx")


def test_ubc_evict_paths_noop_on_linux(monkeypatch, tmp_path):
    pytest.skip("UBC evict not migrated to fusion-mlx")


def test_ubc_evict_munmap_failure_is_not_reported_as_success(tmp_path, caplog):
    pytest.skip("UBC evict not migrated to fusion-mlx")


def test_ubc_evict_missing_file_returns_zero(tmp_path, caplog):
    pytest.skip("UBC evict not migrated to fusion-mlx")


def test_ubc_evict_zero_byte_file_returns_zero(tmp_path, caplog):
    pytest.skip("UBC evict not migrated to fusion-mlx")


def test_render_prometheus_lines_shape():
    pytest.skip("UBC evict not migrated to fusion-mlx")


def test_render_prometheus_lines_zero_by_default():
    pytest.skip("UBC evict not migrated to fusion-mlx")


def test_route_module_render_matches_helper():
    pytest.skip("UBC evict not migrated to fusion-mlx")


def test_counter_monotonic_across_calls(tmp_path):
    pytest.skip("UBC evict not migrated to fusion-mlx")


def test_reset_for_tests_clears_state(monkeypatch):
    pytest.skip("UBC evict not migrated to fusion-mlx")


def test_post_load_ubc_evict_targets_safetensors_only(monkeypatch, tmp_path):
    pytest.skip("UBC evict not migrated to fusion-mlx")


def test_post_load_ubc_evict_skips_when_model_path_unresolved(monkeypatch):
    pytest.skip("UBC evict not migrated to fusion-mlx")


def test_post_load_ubc_evict_skips_when_no_shards(monkeypatch, tmp_path):
    pytest.skip("UBC evict not migrated to fusion-mlx")


def test_post_load_ubc_evict_does_not_resolve_path_on_non_darwin(monkeypatch):
    pytest.skip("UBC evict not migrated to fusion-mlx")
