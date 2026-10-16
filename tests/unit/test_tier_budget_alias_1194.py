# SPDX-License-Identifier: Apache-2.0
"""#1194 follow-up: alias serves must not bypass the tier-budget gate.

Real-model verification on the 11433 debug instance found two gaps:
(1) ``serve --model <hf-repo-id>`` passed the bare id to the size
    estimate, which raised "No model weights found", was swallowed, and
    left size=0 — the budget check was skipped AND tier accounting
    undercounted. The estimate now retries the hub-cache snapshot dirs.
(2) the gui_compat /v1/models/{id}/load route shadows the public
    handler and degraded a pool budget rejection into 404 "not found".
    It now surfaces 503 with the rejection reason.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from fusion_mlx.exceptions import InsufficientMemoryError, ModelTooLargeError
from fusion_mlx.server import _estimate_single_model_size


def _hub_layout(base, repo="org--repo", size=2048):
    # Mirror the hub cache layout the helper retries:
    # <base>/models--org--repo/snapshots/<hash>/
    snap = base / f"models--{repo}" / "snapshots" / "abc123"
    snap.mkdir(parents=True)
    (snap / "model.safetensors").write_bytes(b"0" * size)
    return snap


class TestAliasEstimate:
    def test_bare_hf_id_resolved_via_hub_cache(self, tmp_path, monkeypatch):
        _hub_layout(
            tmp_path / ".cache" / "huggingface" / "hub",
            "mlx-community--Qwen3.5-4B-MLX-4bit",
            4096,
        )
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
        monkeypatch.delenv("HF_HOME", raising=False)
        got = _estimate_single_model_size("mlx-community/Qwen3.5-4B-MLX-4bit")
        # estimate_model_size adds a ~5% runtime-buffer overhead
        assert got == int(4096 * 1.05)

    def test_hf_home_env_preferred(self, tmp_path, monkeypatch):
        _hub_layout(tmp_path / "hf", "org--repo", 8192)
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path / "nowhere")
        monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
        assert _estimate_single_model_size("org/repo") == int(8192 * 1.05)

    def test_local_path_direct(self, tmp_path):
        d = tmp_path / "model"
        d.mkdir()
        (d / "model.safetensors").write_bytes(b"0" * 512)
        assert _estimate_single_model_size(str(d)) == int(512 * 1.05)

    def test_unknown_returns_zero_with_warning(self, tmp_path, monkeypatch, caplog):
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
        monkeypatch.delenv("HF_HOME", raising=False)
        with caplog.at_level("WARNING"):
            assert _estimate_single_model_size("no/such-model") == 0
        assert any("size estimate failed" in r.getMessage() for r in caplog.records)

    def test_bare_name_without_slash_skips_retry(self, tmp_path, monkeypatch):
        # No "/" → no hub-cache candidates; the helper must not raise on
        # a bare alias that resolves to nothing local.
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
        monkeypatch.delenv("HF_HOME", raising=False)
        assert _estimate_single_model_size("qwen3.5-9b-4bit") == 0


class TestGuiCompatBudgetSurfacing:
    def _srv(self, side_effect):
        srv = MagicMock()
        pool = MagicMock()
        entry = MagicMock()
        entry.engine = None
        pool.get_entry = MagicMock(return_value=entry)
        pool.get_engine = AsyncMock(side_effect=side_effect)
        srv.pool = pool
        return srv

    @pytest.mark.asyncio
    async def test_budget_rejection_raises_503(self):
        from fusion_mlx.gui_compat.server import _resolve_pool_model

        srv = self._srv(
            ModelTooLargeError("m", 10, 5, message="P0 tier budget exceeded")
        )
        with (
            patch("fusion_mlx.server.get_server", return_value=srv),
            patch("fusion_mlx.server.resolve_model_id", return_value="m"),
        ):
            with pytest.raises(HTTPException) as ei:
                await _resolve_pool_model("m")
        assert ei.value.status_code == 503
        assert "P0 tier budget exceeded" in ei.value.detail

    @pytest.mark.asyncio
    async def test_insufficient_memory_raises_503(self):
        from fusion_mlx.gui_compat.server import _resolve_pool_model

        srv = self._srv(InsufficientMemoryError("no room"))
        with (
            patch("fusion_mlx.server.get_server", return_value=srv),
            patch("fusion_mlx.server.resolve_model_id", return_value="m"),
        ):
            with pytest.raises(HTTPException) as ei:
                await _resolve_pool_model("m")
        assert ei.value.status_code == 503
        assert "no room" in ei.value.detail

    @pytest.mark.asyncio
    async def test_generic_failure_still_falls_back_to_404(self):
        from fusion_mlx.gui_compat.server import _resolve_pool_model

        srv = self._srv(RuntimeError("boom"))
        with (
            patch("fusion_mlx.server.get_server", return_value=srv),
            patch("fusion_mlx.server.resolve_model_id", return_value="m"),
        ):
            assert await _resolve_pool_model("m") is None

    @pytest.mark.asyncio
    async def test_successful_load_returns_ok(self):
        from fusion_mlx.gui_compat.server import _resolve_pool_model

        srv = self._srv(None)
        with (
            patch("fusion_mlx.server.get_server", return_value=srv),
            patch("fusion_mlx.server.resolve_model_id", return_value="m"),
        ):
            got = await _resolve_pool_model("m")
        assert got["status"] == "ok"
