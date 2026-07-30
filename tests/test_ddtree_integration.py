# SPDX-License-Identifier: Apache-2.0
"""Integration tests for DFlash/DDTree speculative decoding.

Tests the CLI surface, eligibility gates, and DFlash server endpoints
without requiring a GPU or mlx-vlm runtime.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock, patch

import pytest

from fusion_mlx.speculative.dflash.eligibility import (
    DFlashUnavailable,
    EligibilityReport,
    _looks_like_4bit,
    check,
    eligible_aliases,
    have_runtime,
    report,
)
from fusion_mlx.speculative.registry import SpecDecoderPlugin, get_spec_decoder


class TestDDTreeRegistry:
    def test_ddtree_registered(self):
        p = get_spec_decoder("ddtree")
        assert p is not None
        assert p.method == "ddtree"
        assert "dflash" in p.aliases

    def test_dflash_alias_resolves(self):
        p = get_spec_decoder("dflash")
        assert p is not None
        assert p.method == "ddtree"

    def test_block_diffusion_alias(self):
        p = get_spec_decoder("block-diffusion")
        assert p is not None
        assert p.method == "ddtree"


class TestEligibility4bit:
    def test_4bit_suffix_detected(self):
        assert _looks_like_4bit("Qwen3.5-27B-4bit") is True

    def test_mxfp4_detected(self):
        assert _looks_like_4bit("Qwen3.5-27B-mxfp4") is True

    def test_nvfp4_detected(self):
        assert _looks_like_4bit("Qwen3.5-27B-nvfp4") is True

    def test_8bit_not_flagged(self):
        assert _looks_like_4bit("Qwen3.5-27B-8bit") is False

    def test_bf16_not_flagged(self):
        assert _looks_like_4bit("Qwen3.5-27B-bf16") is False


class TestEligibilityReport:
    def _make_profile(self, **overrides):
        defaults = dict(
            hf_path="Qwen3.5-27B-8bit",
            supports_dflash=True,
            is_moe=False,
            dflash_draft_model="z-lab/Qwen3.5-27B-DFlash",
        )
        defaults.update(overrides)
        return MagicMock(**defaults)

    def test_eligible_alias_passes(self):
        p = self._make_profile()
        r = report(p, alias="test-alias")
        assert len(r.reasons) == 0
        assert r.supports_dflash is True
        assert r.is_moe is False
        assert r.is_4bit is False
        assert r.has_drafter is True

    def test_moe_alias_rejected(self):
        p = self._make_profile(is_moe=True)
        r = report(p, alias="moe-alias")
        assert any("MoE" in reason for reason in r.reasons)
        assert r.is_moe is True

    def test_4bit_alias_rejected(self):
        p = self._make_profile(hf_path="Qwen3.5-27B-4bit")
        r = report(p, alias="4bit-alias")
        assert any("4-bit" in reason for reason in r.reasons)
        assert r.is_4bit is True

    def test_no_drafter_rejected(self):
        p = self._make_profile(dflash_draft_model=None, drafter_hf_path=None)
        r = report(p, alias="no-drafter")
        assert any("dflash_draft_model" in reason for reason in r.reasons)
        assert r.has_drafter is False

    def test_unsupported_dflash_rejected(self):
        p = self._make_profile(supports_dflash=False)
        r = report(p, alias="no-dflash")
        assert any("not DFlash-enabled" in reason for reason in r.reasons)


class TestEligibilityCheck:
    def _make_profile(self, **overrides):
        defaults = dict(
            hf_path="Qwen3.5-27B-8bit",
            supports_dflash=True,
            is_moe=False,
            dflash_draft_model="z-lab/Qwen3.5-27B-DFlash",
        )
        defaults.update(overrides)
        return MagicMock(**defaults)

    def test_check_eligible_passes(self):
        p = self._make_profile()
        check(p, alias="ok-alias")

    def test_check_4bit_raises(self):
        p = self._make_profile(hf_path="Qwen3.5-27B-4bit")
        with pytest.raises(DFlashUnavailable, match="4-bit"):
            check(p, alias="4bit-alias")

    def test_check_moe_raises(self):
        p = self._make_profile(is_moe=True)
        with pytest.raises(DFlashUnavailable, match="MoE"):
            check(p, alias="moe-alias")


class TestHaveRuntime:
    def test_returns_bool(self):
        result = have_runtime()
        assert isinstance(result, bool)


class TestEligibleAliases:
    def test_returns_list(self):
        result = eligible_aliases()
        assert isinstance(result, list)


class TestDFlashServerBuild:
    def _make_runtime(self):
        rt = MagicMock()
        rt.drafter_repo = "z-lab/Qwen3.5-27B-DFlash"
        rt.kind = "dflash"
        rt.reset_accept_lens = MagicMock()
        rt.accept_lens_snapshot.return_value = []
        return rt

    def test_build_app_creates_healthz(self):
        from fusion_mlx.speculative.dflash.server import _build_app

        rt = self._make_runtime()
        app = _build_app(
            model=MagicMock(),
            processor=MagicMock(),
            runtime=rt,
            served_model_name="test-model",
            default_max_tokens=4096,
            cors_origins=["*"],
        )
        routes = [r.path for r in app.routes]
        assert "/healthz" in routes

    def test_build_app_creates_models_endpoint(self):
        from fusion_mlx.speculative.dflash.server import _build_app

        rt = self._make_runtime()
        app = _build_app(
            model=MagicMock(),
            processor=MagicMock(),
            runtime=rt,
            served_model_name="test-model",
            default_max_tokens=4096,
            cors_origins=["*"],
        )
        routes = [r.path for r in app.routes]
        assert "/v1/models" in routes

    def test_build_app_creates_completions_endpoint(self):
        from fusion_mlx.speculative.dflash.server import _build_app

        rt = self._make_runtime()
        app = _build_app(
            model=MagicMock(),
            processor=MagicMock(),
            runtime=rt,
            served_model_name="test-model",
            default_max_tokens=4096,
            cors_origins=["*"],
        )
        routes = [r.path for r in app.routes]
        assert "/v1/chat/completions" in routes


class TestDFlashCompletionRejectsUnsupported:
    @pytest.fixture
    def dflash_client(self):
        from httpx import ASGITransport, AsyncClient

        from fusion_mlx.speculative.dflash.server import _build_app

        rt = MagicMock()
        rt.drafter_repo = "z-lab/Qwen3.5-27B-DFlash"
        rt.kind = "dflash"
        rt.reset_accept_lens = MagicMock()
        rt.accept_lens_snapshot.return_value = []
        app = _build_app(
            model=MagicMock(),
            processor=MagicMock(),
            runtime=rt,
            served_model_name="test-model",
            default_max_tokens=4096,
            cors_origins=["*"],
        )
        transport = ASGITransport(app=app)
        return AsyncClient(transport=transport, base_url="http://test")

    @pytest.mark.asyncio
    async def test_rejects_tools(self, dflash_client):
        async with dflash_client as c:
            resp = await c.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "hi"}],
                    "tools": [{"type": "function", "function": {"name": "f"}}],
                },
            )
            assert resp.status_code == 400
            assert "Tool calling" in resp.json()["error"]["message"]

    @pytest.mark.asyncio
    async def test_rejects_logprobs(self, dflash_client):
        async with dflash_client as c:
            resp = await c.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "hi"}],
                    "logprobs": True,
                },
            )
            assert resp.status_code == 400
            assert "logprobs" in resp.json()["error"]["message"]

    @pytest.mark.asyncio
    async def test_rejects_response_format(self, dflash_client):
        async with dflash_client as c:
            resp = await c.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "hi"}],
                    "response_format": {"type": "json_object"},
                },
            )
            assert resp.status_code == 400
            assert "response_format" in resp.json()["error"]["message"]


class TestDFlashExecutor:
    def test_executor_is_single_thread(self):
        from fusion_mlx.speculative.dflash.server import _dflash_executor

        assert _dflash_executor._max_workers == 1

    def test_executor_thread_prefix(self):
        from fusion_mlx.speculative.dflash.server import _dflash_executor

        assert _dflash_executor._thread_name_prefix == "dflash-worker"
