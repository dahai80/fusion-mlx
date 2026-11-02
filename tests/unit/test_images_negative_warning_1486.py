# SPDX-License-Identifier: Apache-2.0
# #1486 item 3: negative_prompt honesty on /v1/images/generate. Qwen-Image-2.1
# and other CFG-capable variants honor it silently; FLUX variants drop it —
# the response must carry a warning instead of only a server log line.

from unittest.mock import AsyncMock

from fusion_mlx.api.images import (
    _NEG_PROMPT_HONORED_VARIANTS,
    ImageGenerateRequest,
    ImageGenerateResponse,
    generate_image,
)
from fusion_mlx.engines import ImageGenEngine


def _engine(variant: str | None) -> ImageGenEngine:
    eng = ImageGenEngine(model_name="flux-2", variant=variant)
    eng.generate = AsyncMock(return_value=[b"fake-png-bytes"])
    return eng


def _pool_with(eng) -> AsyncMock:
    pool = AsyncMock()
    pool.get_engine = AsyncMock(return_value=eng)
    return pool


class TestNegativePromptWarning:
    async def test_flux_variant_gets_warning(self, monkeypatch):
        eng = _engine("flux2_dev")
        monkeypatch.setattr(
            "fusion_mlx.api.images._pool", _pool_with(eng), raising=False
        )
        req = ImageGenerateRequest(prompt="a cat", negative_prompt="blurry")
        resp = await generate_image(req)
        assert any("negative_prompt" in w for w in resp.warnings)
        # The engine still receives the param (upstream logs + drops it) —
        # the warning documents the drop; behavior unchanged.
        assert eng.generate.call_args.kwargs.get("negative_prompt") == "blurry"

    async def test_honoring_variant_no_warning(self, monkeypatch):
        eng = _engine("qwen_image_21")
        monkeypatch.setattr(
            "fusion_mlx.api.images._pool", _pool_with(eng), raising=False
        )
        req = ImageGenerateRequest(prompt="a cat", negative_prompt="blurry")
        resp = await generate_image(req)
        assert resp.warnings == []

    async def test_no_negative_prompt_no_warning(self, monkeypatch):
        eng = _engine("flux2_dev")
        monkeypatch.setattr(
            "fusion_mlx.api.images._pool", _pool_with(eng), raising=False
        )
        req = ImageGenerateRequest(prompt="a cat")
        resp = await generate_image(req)
        assert resp.warnings == []

    def test_response_schema_has_warnings_default_empty(self):
        resp = ImageGenerateResponse(data=[])
        assert resp.warnings == []

    def test_honored_set_matches_engine(self):
        # Keep the route-level set in sync with the engine's honored list.
        from fusion_mlx.engines.image_gen import ImageGenEngine as _E

        src = open(_E.__module__.replace(".", "/") + ".py").read()
        for variant in _NEG_PROMPT_HONORED_VARIANTS:
            assert f'"{variant}"' in src, f"{variant} missing from engine list"
