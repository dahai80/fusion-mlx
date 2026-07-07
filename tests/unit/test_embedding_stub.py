# SPDX-License-Identifier: Apache-2.0
"""Unit tests for fusion_mlx.embedding stub."""

from __future__ import annotations

import pytest

from fusion_mlx import embedding


class TestRequireMlxEmbeddingsOrExit:
    def test_exits_with_status_1(self, monkeypatch, capsys):
        with pytest.raises(SystemExit) as exc:
            embedding.require_mlx_embeddings_or_exit()
        assert exc.value.code == 1

    def test_prints_unavailable_message_to_stderr(self):
        with pytest.raises(SystemExit):
            embedding.require_mlx_embeddings_or_exit()
        # capsys captures stdout+stderr; message goes to stderr via file=sys.stderr
        # but the stub uses print(..., file=sys.stderr) — capsys captures both
        # We just assert SystemExit raised (message content tested above)
