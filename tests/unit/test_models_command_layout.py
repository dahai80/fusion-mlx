# SPDX-License-Identifier: Apache-2.0
"""Tests for the `rapid-mlx models` table column-alignment contract.

Dogfood-driven: 0.9.5 had a hardcoded 24-char alias column. The actual
registry has names up to 31 chars (``deepseek-coder-v2-lite-16b-4bit``),
which overflowed and shifted the rest of that row's columns. 0.9.6 sizes
the column from the data with a 24-char floor.
"""

from __future__ import annotations

from types import SimpleNamespace

from fusion_mlx.cli import models_command
from fusion_mlx.model_aliases import list_profiles


def _capture(capsys, **arg_overrides):
    args = SimpleNamespace(cached=False, **arg_overrides)
    models_command(args)
    return capsys.readouterr().out


def test_every_row_aligns_with_the_header_separator(capsys):
    """Each data row must have the same number of visible columns and
    the same column positions as the header. With the old fixed 24-char
    alias column, the 31-char ``deepseek-coder-v2-lite-16b-4bit`` row
    pushed Tools / Reasoning / Spec-Decode out one full column position.
    """
    out = _capture(capsys)
    lines = [ln for ln in out.splitlines() if ln.startswith("  ")]
    # Find the header line ("  Alias ... DFlash") and the data rows
    # immediately following (between two separator lines).
    header_idx = next(
        i
        for i, ln in enumerate(lines)
        if ln.lstrip().startswith("Alias") and "Capabilities" in ln
    )
    header = lines[header_idx]
    data_rows: list[str] = []
    for ln in lines[header_idx + 2 :]:
        if set(ln.strip()) == {"─"}:
            break
        data_rows.append(ln)
    assert len(data_rows) >= 10, f"expected at least 10 aliases, got {len(data_rows)}"

    caps_col = header.index("Capabilities")
    for row in data_rows:
        stripped = row[2:]
        first_gap = stripped.find(" ")
        second_col_abs = (
            2
            + len(stripped[:first_gap])
            + (len(stripped[first_gap:]) - len(stripped[first_gap:].lstrip()))
        )
        assert second_col_abs == caps_col, (
            f"Row mis-aligned: caps col at {second_col_abs}, header at "
            f"{caps_col}. Row: {row!r}"
        )


def test_alias_column_width_floor_is_24(capsys, monkeypatch):
    """If the registry only has short names, the alias column must
    still be 24 wide so short tables don't feel cramped."""
    from fusion_mlx import model_aliases
    from fusion_mlx.model_aliases import AliasProfile

    short_profile = AliasProfile(hf_path="x/y")
    monkeypatch.setattr(model_aliases, "list_profiles", lambda: {"qwen": short_profile})
    out = _capture(capsys)
    # Header has "Alias" followed by at least 19 spaces before "Capabilities"
    # → column starts at position 2 + 24 + 1 = 27.
    header_line = next(
        ln for ln in out.splitlines() if "Alias" in ln and "Capabilities" in ln
    )
    assert header_line.index("Capabilities") - header_line.index("Alias") == 25, (
        "Alias-column floor regression: short registry should still pad "
        "to 24 chars (Capabilities header at offset 25 from Alias)."
    )


def test_longest_real_alias_does_not_overflow(capsys):
    """End-to-end: with the real registry, the longest alias still gets
    its column with at least 1 space before the next column."""
    out = _capture(capsys)
    longest_alias = max(list_profiles().keys(), key=len)
    data_line = next(
        ln for ln in out.splitlines() if ln.lstrip().startswith(longest_alias)
    )
    after_alias = data_line[2 + len(longest_alias) :]
    assert after_alias.startswith(" "), (
        f"No padding between alias and Capabilities column for {longest_alias!r}"
    )
