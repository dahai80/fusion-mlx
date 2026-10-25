# SPDX-License-Identifier: Apache-2.0
"""Issue #1374: reference images are only consumed by the redux variant.

fusion-drama attaches character/scene references to /v1/images/edits. On any
other variant the extra images were dropped without a word, so the generated
shot drifted and the caller had no way to tell why.
"""

import logging
from unittest.mock import AsyncMock, patch

import pytest

from fusion_mlx.engines.image_gen import ImageGenEngine


def _engine(variant: str) -> ImageGenEngine:
    engine = ImageGenEngine(model_name="black-forest-labs/FLUX.1-dev")
    engine._variant = variant
    engine._admit_generation = AsyncMock()
    engine._generate_subprocess = AsyncMock(return_value=[b"png"])
    return engine


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "variant", ["flux1_dev", "flux2_dev", "qwen_image_21", "fill", "kontext"]
)
async def test_reference_images_warned_and_dropped(variant, caplog):
    engine = _engine(variant)
    with (
        patch("fusion_mlx.engines.image_gen._subprocess_enabled", return_value=True),
        caplog.at_level(logging.WARNING, logger="fusion_mlx.engines.image_gen"),
    ):
        out = await engine.generate(
            prompt="a man in a forest",
            edit_image="target.png",
            reference_images=["char.png", "scene.png"],
        )
    assert out == [b"png"]
    assert f"image variant '{variant}' ignores 2 reference image(s)" in caplog.text
    # The first image still reaches the engine as the edit base.
    kwargs = engine._generate_subprocess.await_args.kwargs
    assert kwargs["edit_image"] == "target.png"
    assert kwargs["reference_images"] == ["char.png", "scene.png"]


@pytest.mark.asyncio
async def test_redux_consumes_references_without_warning(caplog):
    engine = _engine("redux")
    with (
        patch("fusion_mlx.engines.image_gen._subprocess_enabled", return_value=True),
        caplog.at_level(logging.WARNING, logger="fusion_mlx.engines.image_gen"),
    ):
        await engine.generate(
            prompt="a man in a forest",
            reference_images=["char.png", "scene.png"],
        )
    assert "ignores" not in caplog.text


@pytest.mark.asyncio
async def test_no_reference_no_warning(caplog):
    engine = _engine("flux1_dev")
    with (
        patch("fusion_mlx.engines.image_gen._subprocess_enabled", return_value=True),
        caplog.at_level(logging.WARNING, logger="fusion_mlx.engines.image_gen"),
    ):
        await engine.generate(prompt="a man", edit_image="target.png")
    assert "reference image" not in caplog.text
