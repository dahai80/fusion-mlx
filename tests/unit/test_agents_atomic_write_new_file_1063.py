# SPDX-License-Identifier: Apache-2.0
"""Regression tests for issue #1063.

#1063: _atomic_write used a plain write_text for non-existent targets — a
crash mid-write left a corrupted half-written agent config. Fix: all writes
go through tmp + os.replace().
"""

from __future__ import annotations

import inspect

import pytest

from fusion_mlx.agents.adapter import _atomic_write


class TestAtomicWriteNewFile1063:
    """_atomic_write is atomic for both new and existing files."""

    def test_source_no_plain_write_early_return(self):
        from fusion_mlx.agents import adapter

        source = inspect.getsource(adapter)
        # pre-fix had an early `if not resolved.exists(): ... write_text`
        # branch; post-fix routes all writes through tmp + os.replace.
        assert "plain write is safe" not in source
        assert "write_text(content" not in source

    def test_new_file_written_atomically(self, tmp_path):
        target = tmp_path / "subdir" / "agent_config.json"
        content = '{"name": "test-agent", "version": 1}'
        _atomic_write(target, content)
        assert target.exists()
        assert target.read_text(encoding="utf-8") == content
        # no tmp files left behind
        leftovers = list((tmp_path / "subdir").glob(".fusion-mlx-*.tmp"))
        assert leftovers == []

    def test_existing_file_replaced_atomically(self, tmp_path):
        target = tmp_path / "config.json"
        target.write_text('{"old": true}', encoding="utf-8")
        _atomic_write(target, '{"new": true}')
        assert target.read_text(encoding="utf-8") == '{"new": true}'

    def test_new_file_gets_default_mode(self, tmp_path):
        target = tmp_path / "new_config.json"
        _atomic_write(target, "{}")
        import stat

        mode = stat.S_IMODE(target.stat().st_mode)
        assert mode == 0o644

    def test_existing_file_preserves_mode(self, tmp_path):
        target = tmp_path / "config.json"
        target.write_text("old", encoding="utf-8")
        import os

        os.chmod(target, 0o600)
        _atomic_write(target, "new")
        import stat

        mode = stat.S_IMODE(target.stat().st_mode)
        assert mode == 0o600


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
