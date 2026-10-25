# SPDX-License-Identifier: Apache-2.0
# #1386: the draft model default path named the flat ``<root>/org/name`` form,
# but downloads live in the HF cache layout (``models--org--name/snapshots``),
# so mlx_lm.load raised "Repo id must be in the form 'repo_name' ..." and
# speculative draft decode was silently off on every stock install.

import os

from fusion_mlx.speculative.draft_model import _resolve_draft_path


def _make_cache(root, org="mlx-community", name="Qwen3-0.6B-4bit", sha="abc123"):
    snapshot = root / f"models--{org}--{name}" / "snapshots" / sha
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")
    return snapshot


def test_flat_path_resolves_to_snapshot(tmp_path):
    snapshot = _make_cache(tmp_path)
    resolved = _resolve_draft_path(str(tmp_path / "mlx-community" / "Qwen3-0.6B-4bit"))
    assert resolved == str(snapshot)


def test_existing_directory_is_untouched(tmp_path):
    local = tmp_path / "local-draft"
    local.mkdir()
    assert _resolve_draft_path(str(local)) == str(local)


def test_newest_snapshot_wins(tmp_path):
    old = _make_cache(tmp_path, sha="old")
    new = _make_cache(tmp_path, sha="new")
    os.utime(old, (1_000_000, 1_000_000))
    resolved = _resolve_draft_path(str(tmp_path / "mlx-community" / "Qwen3-0.6B-4bit"))
    assert resolved == str(new)


def test_missing_cache_falls_back_to_input(tmp_path):
    raw = str(tmp_path / "mlx-community" / "Qwen3-0.6B-4bit")
    assert _resolve_draft_path(raw) == raw


def test_empty_path_is_passthrough():
    assert _resolve_draft_path("") == ""


def test_repo_id_without_root_is_passthrough():
    assert _resolve_draft_path("mlx-community/Qwen3-0.6B-4bit") == (
        "mlx-community/Qwen3-0.6B-4bit"
    )
