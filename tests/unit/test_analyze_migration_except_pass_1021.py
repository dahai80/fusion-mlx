# SPDX-License-Identifier: Apache-2.0
"""Regression tests for issue #1021.

#1021: analyze_routes + migration_routes had `except Exception: pass` that
swallowed failures and returned incomplete results masquerading as success.

Fix (#1021):
- analyze_routes: safetensors shape parse + special-ops detection failures
  now log WARNING and append to a `warnings` list surfaced in AnalyzeResponse.
- migration_routes: resolve_model probe failure now logs WARNING and threads
  the warning into MigrationLevelResponse.warnings instead of silent pass.
"""

from __future__ import annotations

import logging
from unittest.mock import patch

import pytest

from fusion_mlx.api.analyze_routes import AnalyzeRequest, AnalyzeResponse, analyze_model
from fusion_mlx.api.migration_routes import MigrationLevel, _assess_level


class TestAnalyzePartialWarnings:
    """analyze_routes: failures surface in warnings, not silent pass."""

    @pytest.mark.asyncio
    async def test_analyze_response_has_warnings_field(self):
        resp = AnalyzeResponse(
            model_id="x",
            architecture="test",
            params_total=0,
            params_by_layer={},
            layer_types=[],
            num_layers=0,
            hidden_size=0,
            num_attention_heads=0,
            special_ops=[],
            safetensors_files=[],
            config_json={},
        )
        assert hasattr(resp, "warnings")
        assert resp.warnings == []

    @pytest.mark.asyncio
    async def test_safetensors_parse_failure_logged_and_warned(self, tmp_path, caplog):
        cfg_file = tmp_path / "config.json"
        cfg_file.write_text('{"model_type":"qwen2","num_hidden_layers":2}')
        sf = tmp_path / "model.safetensors"
        sf.write_bytes(b"garbage")

        req = AnalyzeRequest(model_path=str(tmp_path))
        with caplog.at_level(logging.WARNING, logger="fusion_mlx.api.analyze_routes"):
            result = await analyze_model(req)

        assert isinstance(result, AnalyzeResponse)
        assert any(
            "shape parse failed" in w for w in result.warnings
        ), f"expected shape-parse warning, got {result.warnings}"
        assert any("failed to parse shapes" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_special_ops_detection_failure_warned(self, tmp_path, caplog):
        cfg_file = tmp_path / "config.json"
        cfg_file.write_text('{"model_type":"qwen2","num_hidden_layers":2}')

        req = AnalyzeRequest(model_path=str(tmp_path))
        with patch(
            "fusion_mlx.api.analyze_routes.detect_model_config",
            side_effect=RuntimeError("boom"),
        ):
            with caplog.at_level(
                logging.WARNING, logger="fusion_mlx.api.analyze_routes"
            ):
                result = await analyze_model(req)

        assert any("special-ops detection failed" in w for w in result.warnings)
        assert any("special-ops detection failed" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_clean_analysis_no_warnings(self, tmp_path):
        cfg_file = tmp_path / "config.json"
        cfg_file.write_text('{"model_type":"qwen2","num_hidden_layers":2}')

        req = AnalyzeRequest(model_path=str(tmp_path))
        with patch(
            "fusion_mlx.api.analyze_routes.detect_model_config",
            return_value=None,
        ):
            result = await analyze_model(req)

        assert result.warnings == [], f"expected no warnings, got {result.warnings}"


class TestMigrationProbeWarning:
    """migration_routes: resolve_model failure surfaces warning, not silent pass."""

    def test_resolve_failure_adds_warning(self, caplog):
        with patch(
            "fusion_mlx.api.migration_routes.resolve_model",
            side_effect=RuntimeError("network down"),
        ):
            with patch(
                "fusion_mlx.api.migration_routes.detect_model_config",
                side_effect=RuntimeError("also down"),
            ):
                with caplog.at_level(
                    logging.WARNING, logger="fusion_mlx.api.migration_routes"
                ):
                    level, matched, missing, warnings = _assess_level(
                        "some-unknown-model", None
                    )

        assert level == MigrationLevel.L4
        assert any(
            "model probe failed" in w for w in warnings
        ), f"expected probe-failure warning, got {warnings}"
        assert any(
            "model probe (resolve_model) failed" in r.message for r in caplog.records
        )

    def test_resolve_success_no_probe_warning(self):
        with patch("fusion_mlx.api.migration_routes.resolve_model", return_value=None):
            with patch(
                "fusion_mlx.api.migration_routes.detect_model_config",
                side_effect=RuntimeError("down"),
            ):
                level, matched, missing, warnings = _assess_level(
                    "some-unknown-model", None
                )

        assert not any(
            "model probe failed" in w for w in warnings
        ), f"unexpected probe warning, got {warnings}"

    def test_alias_fast_path_no_warnings(self):
        aliases_mod = "fusion_mlx.api.migration_routes.list_aliases"
        with patch(aliases_mod, return_value={"qwen3.5-9b-4bit"}):
            level, matched, missing, warnings = _assess_level("qwen3.5-9b-4bit", None)

        assert level == MigrationLevel.L0
        assert warnings == []


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
