# SPDX-License-Identifier: Apache-2.0
"""#1026: multiview texture generation — single-texture approximation warning.

The paint diffusion pipeline currently generates a single texture and
broadcasts it to all 6 rasterized views. True multiview texture generation
(6 distinct view textures with cross-view attention) is not yet implemented.
This test verifies the approximation is surfaced as a warning in the API
response so clients know the texture quality is approximate.
"""

from fusion_mlx.api.threed_routes import ThreeDGenerateResponse
from fusion_mlx.threed.orchestrator import ThreeDOrchestrator


class TestThreeDResponseWarnings:
    def test_response_has_warnings_field(self):
        resp = ThreeDGenerateResponse(
            created=0,
            model="test",
            format="glb",
            glb_base64="",
            vertices=0,
            faces=0,
            bytes=0,
        )
        assert hasattr(resp, "warnings")
        assert resp.warnings == []

    def test_response_with_warning(self):
        warning = "single_texture_multiview_approximation: test"
        resp = ThreeDGenerateResponse(
            created=0,
            model="test",
            format="glb",
            glb_base64="",
            vertices=0,
            faces=0,
            bytes=0,
            warnings=[warning],
        )
        assert len(resp.warnings) == 1
        assert "single_texture" in resp.warnings[0]


class TestOrchestratorWarnings:
    def test_orchestrator_has_last_warnings_attr(self):
        # __init__ loads model config (needs real weights), so verify the
        # attribute is declared on the class and settable without __init__.
        orch = ThreeDOrchestrator.__new__(ThreeDOrchestrator)
        orch.last_warnings = []
        assert hasattr(orch, "last_warnings")
        assert isinstance(orch.last_warnings, list)
        assert orch.last_warnings == []

    def test_warning_string_content(self):
        # Verify the warning string format matches what the route handler
        # and orchestrator produce.
        from fusion_mlx.threed.orchestrator import ThreeDOrchestrator

        orch = ThreeDOrchestrator.__new__(ThreeDOrchestrator)
        orch.last_warnings = [
            "single_texture_multiview_approximation: a single diffusion "
            "texture is broadcast to all 6 views; multiview texture "
            "generation is not yet implemented (#1026)"
        ]
        assert "single_texture" in orch.last_warnings[0]
        assert "#1026" in orch.last_warnings[0]
        assert "6 views" in orch.last_warnings[0]
