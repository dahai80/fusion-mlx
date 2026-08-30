# SPDX-License-Identifier: Apache-2.0
"""Tests for `fusion-mlx models` --cached / `fusion-mlx ls` cache view + argparse wiring.

Rescue 2026-08-30: the original file pinned the Rapid-MLX per-alias
capability table (Tools/Reasoning/Spec-Decode/Suffix Tier columns, hybrid
marker, "rapid-mlx" footer tip, "Available models" header, qwen3.5-35b-4bit
alias). That table was intentionally reverted to the released server-query
contract (see cli_commands.models_command docstring) and is already covered
by test_models_command_layout.py — those 7 tests were dead contract
(Rule 9: false coverage worse than no test) and were removed.

The --cached / ls cache-view helpers (_scan_hf_cache_models, _format_bytes,
_print_cached_models) are a KEPT live feature in cli_commands.py but were
not re-exported into cli.py, so the cache tests raised AttributeError on
``cli.<helper>``. Rewritten to import/patch from fusion_mlx.cli_commands
(test-only seam; no prod change). The vllm_mlx package was renamed to
fusion_mlx, so the version-check patch target moved to
fusion_mlx._version_check.
"""

from __future__ import annotations

import sys
from unittest.mock import patch

from fusion_mlx import cli, cli_commands

# ----------------------------------------------------------------------
# Argparse wiring (live, unchanged)
# ----------------------------------------------------------------------


def test_models_command_subparser_smoke():
    import pytest

    with (
        patch.object(sys, "argv", ["fusion-mlx", "models", "--help"]),
        pytest.raises(SystemExit) as exc,
    ):
        cli.main()
    assert exc.value.code == 0


def test_ls_subcommand_registered():
    import pytest

    with (
        patch.object(sys, "argv", ["fusion-mlx", "ls", "--help"]),
        pytest.raises(SystemExit) as exc,
    ):
        cli.main()
    assert exc.value.code == 0


def test_ls_routes_to_models_with_cached():
    captured: list = []
    with (
        patch.object(sys, "argv", ["fusion-mlx", "ls"]),
        patch.object(cli, "models_command", side_effect=captured.append),
    ):
        cli.main()
    assert len(captured) == 1
    assert captured[0].cached is True


# ----------------------------------------------------------------------
# --cached / ls view (helpers live in cli_commands, not re-exported to cli)
# ----------------------------------------------------------------------


def test_models_cached_flag_routes_to_cached_view(monkeypatch, capsys):
    monkeypatch.setenv("HF_HOME", "/nonexistent_path_for_this_test_xyz")
    monkeypatch.setattr(cli_commands, "_scan_hf_cache_models", lambda: [])

    with (
        patch.object(sys, "argv", ["fusion-mlx", "models", "--cached"]),
        patch("fusion_mlx._version_check.print_staleness_warning_if_any"),
    ):
        cli.main()
    out = capsys.readouterr().out
    assert "No models cached yet" in out
    assert "Available models" not in out


def test_cached_view_renders_alias_for_known_repo(monkeypatch, capsys):
    from fusion_mlx.model_aliases import list_profiles

    profiles = list_profiles()
    alias = next(iter(profiles))
    hf_path = profiles[alias].hf_path

    monkeypatch.setattr(
        cli_commands,
        "_scan_hf_cache_models",
        lambda: [(hf_path, 1024 * 1024 * 100, 0.0)],
    )
    cli_commands._print_cached_models()
    out = capsys.readouterr().out
    assert alias in out, f"expected alias {alias!r} in cached view"
    assert hf_path[:40] in out, "expected HF path in cached view"


def test_cached_view_renders_unmapped_for_unknown_repo(monkeypatch, capsys):
    monkeypatch.setattr(
        cli_commands,
        "_scan_hf_cache_models",
        lambda: [("some/totally-unmapped-repo", 1024, 0.0)],
    )
    cli_commands._print_cached_models()
    out = capsys.readouterr().out
    assert "(unmapped)" in out
    assert "totally-unmapped-repo" in out


def test_format_bytes_unit_selection():
    assert cli_commands._format_bytes(0) == "0 B"
    assert cli_commands._format_bytes(512) == "512 B"
    assert cli_commands._format_bytes(2048) == "2.0 KiB"
    assert cli_commands._format_bytes(5 * 1024 * 1024) == "5.0 MiB"
    assert cli_commands._format_bytes(int(2.5 * 1024**3)) == "2.5 GiB"


def test_scan_hf_cache_models_filters_to_models_only(tmp_path, monkeypatch):
    cache_root = tmp_path / "hub"
    cache_root.mkdir()
    (cache_root / "models--mlx-community--FakeModel").mkdir()
    (cache_root / "models--mlx-community--FakeModel" / "blob1").write_bytes(b"x" * 128)
    (cache_root / "datasets--squad").mkdir()
    (cache_root / "datasets--squad" / "data").write_bytes(b"y" * 999)
    (cache_root / "spaces--gradio--demo").mkdir()

    # _scan_hf_cache_models reads the cache root via
    # ``from huggingface_hub.constants import HF_HUB_CACHE`` (call-time).
    # huggingface_hub is a lazy module; under the conftest vllm_mlx alias
    # meta-path finder a plain ``setattr`` on the constants attribute is
    # not observed by the submodule ``from``-import (stale real path wins).
    # Swap a stub module into sys.modules so the ``from``-import binds the
    # tmp cache root deterministically; monkeypatch.setitem restores it.
    import sys
    import types

    stub = types.ModuleType("huggingface_hub.constants")
    stub.HF_HUB_CACHE = str(cache_root)
    monkeypatch.setitem(sys.modules, "huggingface_hub.constants", stub)
    rows = cli_commands._scan_hf_cache_models()
    repos = [r[0] for r in rows]
    assert "mlx-community/FakeModel" in repos
    assert all("squad" not in r for r in repos)
    assert all("demo" not in r for r in repos)


if __name__ == "__main__":  # pragma: no cover — convenience only
    import pytest

    pytest.main([__file__, "-v"])
