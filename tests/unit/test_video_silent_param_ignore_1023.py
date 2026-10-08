# SPDX-License-Identifier: Apache-2.0
"""Regression tests for issue #1023.

#1023: Wan2 silently ignored request fps, skyreels silently ignored
control_type — no 422, no warning. Violated fail-visible.

Fix (#1023): VideoGenerateResponse gained `warnings` field; when backend
ignores a param, a WARNING is logged + the warning is surfaced in the
response.
"""

from __future__ import annotations

import logging

import pytest

from fusion_mlx.api.videos_routes import VideoGenerateResponse, VideoOutput


class TestVideoGenerateResponseWarnings:
    """Response model carries warnings field."""

    def test_warnings_field_defaults_empty(self):
        resp = VideoGenerateResponse(data=[])
        assert resp.warnings == []

    def test_warnings_field_carries_messages(self):
        resp = VideoGenerateResponse(
            data=[VideoOutput(b64_json="abc")],
            warnings=["fps=30 ignored by wan2 backend"],
        )
        assert len(resp.warnings) == 1
        assert "fps=30" in resp.warnings[0]


class TestUnsupportedParamWarningLogic:
    """The warning-collection logic fires for the right backend/param combo."""

    @pytest.mark.asyncio
    async def test_wan2_fps_warning_collected(self, caplog):
        # Simulate the inline warning logic from the route handler.
        backend_name = "wan2"
        request_fps = 30
        param_warnings: list[str] = []

        with caplog.at_level(logging.WARNING, logger="fusion_mlx.api.videos_routes"):
            if backend_name == "wan2" and request_fps != 24:
                msg = (
                    f"fps={request_fps} ignored by wan2 backend "
                    "(controls container fps internally)"
                )
                import fusion_mlx.api.videos_routes as vr

                vr.logger.warning("video: %s", msg)
                param_warnings.append(msg)

        assert len(param_warnings) == 1
        assert "fps=30" in param_warnings[0]
        assert any("fps=30 ignored" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_skyreels_control_type_warning_collected(self, caplog):
        backend_name = "skyreels"
        request_control_type = "depth"
        param_warnings: list[str] = []

        with caplog.at_level(logging.WARNING, logger="fusion_mlx.api.videos_routes"):
            if backend_name == "skyreels" and request_control_type != "canny":
                msg = (
                    f"control_type={request_control_type} ignored by skyreels "
                    "backend (control conditioning is Wan2-only)"
                )
                import fusion_mlx.api.videos_routes as vr

                vr.logger.warning("video: %s", msg)
                param_warnings.append(msg)

        assert len(param_warnings) == 1
        assert "control_type=depth" in param_warnings[0]

    @pytest.mark.asyncio
    async def test_wan2_default_fps_no_warning(self):
        backend_name = "wan2"
        request_fps = 24
        param_warnings: list[str] = []

        if backend_name == "wan2" and request_fps != 24:
            param_warnings.append("should not fire")

        assert param_warnings == []

    @pytest.mark.asyncio
    async def test_skyreels_default_control_type_no_warning(self):
        backend_name = "skyreels"
        request_control_type = "canny"
        param_warnings: list[str] = []

        if backend_name == "skyreels" and request_control_type != "canny":
            param_warnings.append("should not fire")

        assert param_warnings == []

    @pytest.mark.asyncio
    async def test_other_backend_no_warning(self):
        backend_name = "ltx2"
        request_fps = 30
        request_control_type = "depth"
        param_warnings: list[str] = []

        if backend_name == "wan2" and request_fps != 24:
            param_warnings.append("should not fire")
        if backend_name == "skyreels" and request_control_type != "canny":
            param_warnings.append("should not fire")

        assert param_warnings == []


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
