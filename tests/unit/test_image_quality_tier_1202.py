# SPDX-License-Identifier: Apache-2.0
"""#1202: image quality-tier routing (quality=fast|high).

Covers: distilled-sibling derivation, request-model routing with fallback,
variant/image-input exemption, catalog alias for the distilled tier, and
the /metrics series. Real ImageGenEngine instances with a mocked generate
so isinstance checks pass; the pool is a dict-backed AsyncMock.
"""

from unittest.mock import AsyncMock, patch

import pytest
from pydantic import ValidationError

from fusion_mlx.api.images import ImageGenerateRequest, generate_image
from fusion_mlx.engines import ImageGenEngine
from fusion_mlx.exceptions import ModelNotFoundError

DEV = "black-forest-labs/FLUX.1-dev"
SCHNELL = "black-forest-labs/FLUX.1-schnell"
KLEIN = "black-forest-labs/FLUX.2-klein"
POOL_KEY_DEV = "black-forest-labs--FLUX.1-dev"
POOL_KEY_SCHNELL = "black-forest-labs--FLUX.1-schnell"


def _engine(model_name: str, variant: str | None = None) -> ImageGenEngine:
    # Real ImageGenEngine instance (the route isinstance-checks it);
    # generate is mocked so nothing actually loads.
    eng = ImageGenEngine(model_name=model_name, variant=variant)
    eng.generate = AsyncMock(return_value=[b"fake-png-bytes"])
    return eng


def _pool_with(engines: dict) -> AsyncMock:
    # Dict-backed pool: get_engine(name) returns engines[name] or raises
    # the REAL ModelNotFoundError so the route's except path is exercised.
    # Mirrors the server's alias resolution loosely: a HF repo id falls
    # back to its pool-key form ("a/b" -> "a--b").
    pool = AsyncMock()

    async def get_engine(name):
        key = name if name in engines else name.replace("/", "--")
        if key in engines:
            return engines[key]
        raise ModelNotFoundError(
            f"model '{name}' not found",
            available_models=list(engines),
        )

    pool.get_engine = AsyncMock(side_effect=get_engine)
    return pool


class TestFastSiblingModel:
    def test_flux1_dev_variants(self):
        from fusion_mlx.engines.image_gen import _fast_sibling_model

        # HF repo id: derived sibling == canonical, deduped to one.
        assert _fast_sibling_model(DEV) == [SCHNELL]
        # Pool key (double hyphen): derived sibling is a pool key too.
        assert _fast_sibling_model(POOL_KEY_DEV) == [
            "black-forest-labs--FLUX.1-schnell",
            SCHNELL,
        ]
        # Symlink key: derived sibling is NOT a pool key; canonical repo
        # id is the fallback candidate.
        assert _fast_sibling_model("FLUX.1-dev") == ["FLUX.1-schnell", SCHNELL]

    def test_flux2_dev_to_klein(self):
        from fusion_mlx.engines.image_gen import _fast_sibling_model

        assert _fast_sibling_model("black-forest-labs/FLUX.2-dev") == [KLEIN]
        assert _fast_sibling_model("black-forest-labs--FLUX.2-dev") == [
            "black-forest-labs--FLUX.2-klein",
            KLEIN,
        ]

    def test_distilled_nonflux_empty(self):
        from fusion_mlx.engines.image_gen import _fast_sibling_model

        # Distilled models have no faster sibling; non-flux and empty
        # paths too.
        assert _fast_sibling_model(SCHNELL) == []
        assert _fast_sibling_model(KLEIN) == []
        assert _fast_sibling_model("Qwen-Image-2512") == []
        assert _fast_sibling_model("") == []


class TestQualityField:
    def test_default_auto(self):
        # #1251: rerouting is opt-in. The default must NOT silently swap
        # the caller's requested model for a distilled sibling.
        req = ImageGenerateRequest(prompt="x")
        assert req.quality == "auto"

    def test_fast_accepted(self):
        req = ImageGenerateRequest(prompt="x", quality="fast")
        assert req.quality == "fast"

    def test_high_accepted(self):
        req = ImageGenerateRequest(prompt="x", quality="high")
        assert req.quality == "high"

    def test_invalid_rejected(self):
        with pytest.raises(ValidationError):
            ImageGenerateRequest(prompt="x", quality="premium")


class TestRouteRouting:
    async def test_fast_routes_to_distilled(self):
        eng = _engine(POOL_KEY_DEV, variant="flux1_dev")
        fast_eng = _engine(POOL_KEY_SCHNELL, variant="flux1_schnell")
        pool = _pool_with({POOL_KEY_DEV: eng, POOL_KEY_SCHNELL: fast_eng})
        req = ImageGenerateRequest(
            model=DEV, prompt="x", quality="fast", response_format="b64_json"
        )
        with (
            patch("fusion_mlx.server.resolve_model_id", side_effect=lambda m: m),
            patch("fusion_mlx.api.images._pool", pool),
        ):
            resp = await generate_image(req)
        fast_eng.generate.assert_awaited_once()
        eng.generate.assert_not_awaited()
        assert resp.data[0].b64_json
        # #1251: the substitution is echoed so the caller never mistakes a
        # distilled render for the full model it asked for.
        assert resp.model == SCHNELL
        assert resp.requested_model == DEV

    async def test_auto_keeps_requested_model(self):
        # #1251: default quality=auto must use exactly the requested model
        # even when a distilled sibling is loaded.
        eng = _engine(POOL_KEY_DEV, variant="flux1_dev")
        fast_eng = _engine(POOL_KEY_SCHNELL, variant="flux1_schnell")
        pool = _pool_with({POOL_KEY_DEV: eng, POOL_KEY_SCHNELL: fast_eng})
        req = ImageGenerateRequest(model=DEV, prompt="x")
        with (
            patch("fusion_mlx.server.resolve_model_id", side_effect=lambda m: m),
            patch("fusion_mlx.api.images._pool", pool),
        ):
            resp = await generate_image(req)
        eng.generate.assert_awaited_once()
        fast_eng.generate.assert_not_awaited()
        assert resp.model == DEV
        assert resp.requested_model == DEV

    async def test_fallback_when_sibling_missing(self):
        eng = _engine(POOL_KEY_DEV, variant="flux1_dev")
        pool = _pool_with({POOL_KEY_DEV: eng})
        req = ImageGenerateRequest(model=DEV, prompt="x", quality="fast")
        with (
            patch("fusion_mlx.server.resolve_model_id", side_effect=lambda m: m),
            patch("fusion_mlx.api.images._pool", pool),
        ):
            resp = await generate_image(req)
        eng.generate.assert_awaited_once()
        assert resp.model == DEV
        assert resp.requested_model == DEV

    async def test_high_keeps_model(self):
        eng = _engine(POOL_KEY_DEV, variant="flux1_dev")
        pool = _pool_with({POOL_KEY_DEV: eng})
        req = ImageGenerateRequest(model=DEV, prompt="x", quality="high")
        with (
            patch("fusion_mlx.server.resolve_model_id", side_effect=lambda m: m),
            patch("fusion_mlx.api.images._pool", pool),
        ):
            await generate_image(req)
        eng.generate.assert_awaited_once()

    async def test_variant_keeps_model(self):
        eng = _engine(POOL_KEY_DEV, variant="flux1_dev")
        fast_eng = _engine(POOL_KEY_SCHNELL, variant="flux1_schnell")
        pool = _pool_with({POOL_KEY_DEV: eng, POOL_KEY_SCHNELL: fast_eng})
        req = ImageGenerateRequest(
            model=DEV, prompt="x", quality="fast", variant="kontext"
        )
        with (
            patch("fusion_mlx.server.resolve_model_id", side_effect=lambda m: m),
            patch("fusion_mlx.api.images._pool", pool),
        ):
            await generate_image(req)
        eng.generate.assert_awaited_once()
        fast_eng.generate.assert_not_awaited()

    async def test_image_input_keeps_model(self):
        eng = _engine(POOL_KEY_DEV, variant="flux1_dev")
        fast_eng = _engine(POOL_KEY_SCHNELL, variant="flux1_schnell")
        pool = _pool_with({POOL_KEY_DEV: eng, POOL_KEY_SCHNELL: fast_eng})
        req = ImageGenerateRequest(
            model=DEV, prompt="x", quality="fast", control_image="/tmp/c.png"
        )
        with (
            patch("fusion_mlx.server.resolve_model_id", side_effect=lambda m: m),
            patch("fusion_mlx.api.images._pool", pool),
        ):
            await generate_image(req)
        eng.generate.assert_awaited_once()
        fast_eng.generate.assert_not_awaited()

    async def test_canonical_fallback(self):
        # Pool has the canonical schnell repo id but not the derived
        # pool-key sibling — the second candidate must resolve it.
        eng = _engine(POOL_KEY_DEV, variant="flux1_dev")
        fast_eng = _engine(SCHNELL, variant="flux1_schnell")
        pool = _pool_with({POOL_KEY_DEV: eng, SCHNELL: fast_eng})
        req = ImageGenerateRequest(model=POOL_KEY_DEV, prompt="x", quality="fast")
        with (
            patch("fusion_mlx.server.resolve_model_id", side_effect=lambda m: m),
            patch("fusion_mlx.api.images._pool", pool),
        ):
            await generate_image(req)
        fast_eng.generate.assert_awaited_once()
        eng.generate.assert_not_awaited()

    async def test_not_image_engine_skipped(self):
        # Sibling resolves to a non-ImageGenEngine object: skipped with a
        # warning, request falls back to the requested model.
        eng = _engine(POOL_KEY_DEV, variant="flux1_dev")
        pool = _pool_with({POOL_KEY_DEV: eng, POOL_KEY_SCHNELL: object()})
        req = ImageGenerateRequest(model=DEV, prompt="x", quality="fast")
        with (
            patch("fusion_mlx.server.resolve_model_id", side_effect=lambda m: m),
            patch("fusion_mlx.api.images._pool", pool),
        ):
            await generate_image(req)
        eng.generate.assert_awaited_once()


class TestCatalogAlias:
    def test_flux1_alias(self):
        from fusion_mlx.config import DEFAULT_ALIASES

        assert DEFAULT_ALIASES.get("flux-1") == SCHNELL


class TestTierMetric:
    def test_counts_increment(self):
        from fusion_mlx.server_metrics import ServerMetrics

        sm = ServerMetrics()
        sm.record_image_quality_tier("fast")
        sm.record_image_quality_tier("fast")
        sm.record_image_quality_tier("high")
        assert sm.image_quality_tier_total == 3
        assert sm.image_quality_tier_by_tier == {"fast": 2, "high": 1}

    def test_render_series(self):
        from unittest.mock import patch as _patch

        from fusion_mlx.routes_internal.metrics import (
            _render_image_quality_tier_metrics,
        )
        from fusion_mlx.server_metrics import ServerMetrics

        sm = ServerMetrics()
        sm.record_image_quality_tier("fast")
        sm.record_image_quality_tier("high")
        with _patch(
            "fusion_mlx.routes_internal.metrics.get_server_metrics",
            return_value=sm,
        ):
            lines = _render_image_quality_tier_metrics()
        text = "\n".join(lines)
        assert "fusion_mlx_image_quality_tier_total 2" in text
        assert 'tier="fast"' in text
        assert 'tier="high"' in text

    def test_render_empty_until_first_request(self):
        from unittest.mock import patch as _patch

        from fusion_mlx.routes_internal.metrics import (
            _render_image_quality_tier_metrics,
        )
        from fusion_mlx.server_metrics import ServerMetrics

        sm = ServerMetrics()
        with _patch(
            "fusion_mlx.routes_internal.metrics.get_server_metrics",
            return_value=sm,
        ):
            lines = _render_image_quality_tier_metrics()
        assert lines == []
