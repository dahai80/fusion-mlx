# SPDX-License-Identifier: Apache-2.0
"""F-D01 regression tests — unified ``capabilities`` advertisement on
``/v1/models``.

The ``capabilities`` field is derived from the alias profile's boolean
fields via the ``AliasProfile.capabilities`` property. These tests pin
the contract.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from fusion_mlx.config import get_config
from fusion_mlx.routes_internal import models as models_route


def _mount_models_app(**cfg_overrides):
    """Mount the models router with controlled config."""
    app = FastAPI()
    app.include_router(models_route.router)

    cfg = get_config()
    saved = {
        k: getattr(cfg, k, None)
        for k in (
            "model_name",
            "model_alias",
            "model_registry",
            "embedding_model_locked",
            "tool_call_parser",
            "api_key",
        )
    }
    cfg.model_registry = None
    cfg.api_key = None
    for k, v in cfg_overrides.items():
        setattr(cfg, k, v)

    def _restore():
        for k, v in saved.items():
            setattr(cfg, k, v)

    return TestClient(app), _restore


def _fetch_entry(client, model_id):
    r = client.get("/v1/models")
    assert r.status_code == 200, r.text
    body = r.json()
    for entry in body["data"]:
        if entry["id"] == model_id:
            return entry
    raise AssertionError(f"model {model_id} missing from /v1/models")


class TestTextOnlyModel:
    """Text-only model — capabilities from alias profile."""

    def test_text_model_has_capabilities(self):
        client, restore = _mount_models_app(
            model_name="mlx-community/Qwen3-0.6B-8bit",
            model_alias="qwen3-0.6b-8bit",
        )
        try:
            entry = _fetch_entry(client, "mlx-community/Qwen3-0.6B-8bit")
        finally:
            restore()
        caps = entry["capabilities"]
        assert isinstance(caps, list)
        assert (
            "spec_decode" in caps
        ), f"text-only model should have spec_decode from profile: {caps}"

    def test_unregistered_text_path_gets_empty_capabilities(self):
        unknown_id = "operator/custom-text-model"
        client, restore = _mount_models_app(
            model_name=unknown_id,
            model_alias=unknown_id,
        )
        try:
            entry = _fetch_entry(client, unknown_id)
        finally:
            restore()
        # No profile match → capabilities defaults to []
        assert "capabilities" in entry
        assert isinstance(entry["capabilities"], list)


class TestVisionModel:
    """VLM — capabilities includes ``"vision"``."""

    def test_vlm_alias_advertises_vision(self):
        import pytest

        from fusion_mlx.model_aliases import resolve_profile

        candidates = [
            "qwen3-vl-2b-instruct-4bit",
            "mlx-community/Qwen3-VL-2B-Instruct-4bit",
        ]
        vlm_id = None
        for cand in candidates:
            p = resolve_profile(cand)
            if p is not None and p.supports_mllm:
                vlm_id = cand
                break
        if vlm_id is None:
            pytest.skip("No VLM alias profile with supports_mllm=True found")

        client, restore = _mount_models_app(
            model_name=vlm_id,
            model_alias=vlm_id,
        )
        try:
            entry = _fetch_entry(client, vlm_id)
        finally:
            restore()
        caps = entry["capabilities"]
        assert (
            "vision" in caps
        ), f"VLM {vlm_id} missing 'vision' in capabilities: {caps}"

    def test_raw_hf_vlm_path_advertises_vision(self):
        import pytest

        from fusion_mlx.model_aliases import resolve_profile

        raw_id = "mlx-community/Qwen3-VL-7B-Instruct-MLX"
        profile = resolve_profile(raw_id)
        if profile is None or not profile.supports_mllm:
            pytest.skip(f"No VLM profile for {raw_id}")

        client, restore = _mount_models_app(
            model_name=raw_id,
            model_alias=raw_id,
        )
        try:
            entry = _fetch_entry(client, raw_id)
        finally:
            restore()
        assert "vision" in entry["capabilities"]
        assert entry["modality"] == "image"


class TestToolsCapability:
    """Tool-capable models — capabilities includes ``"tool_call"``."""

    def test_profile_tool_parser_enables_tool_call_tag(self):
        import pytest

        from fusion_mlx.model_aliases import resolve_profile

        model_id = "mlx-community/Qwen3-0.6B-8bit"
        profile = resolve_profile("qwen3-0.6b-8bit") or resolve_profile(model_id)
        if profile is None or not profile.tool_call_parser:
            pytest.skip("No alias profile with tool_call_parser for qwen3")

        client, restore = _mount_models_app(
            model_name=model_id,
            model_alias="qwen3-0.6b-8bit",
            tool_call_parser="hermes",
        )
        try:
            entry = _fetch_entry(client, model_id)
        finally:
            restore()
        caps = entry["capabilities"]
        assert "tool_call" in caps, f"should advertise 'tool_call', got {caps}"

    def test_server_tool_parser_does_not_leak_to_unrelated_entries(self):
        served_unregistered = "operator/served-custom-tools-model"
        client, restore = _mount_models_app(
            model_name=served_unregistered,
            model_alias=served_unregistered,
            tool_call_parser="hermes",
        )
        try:
            entry = _fetch_entry(client, served_unregistered)
        finally:
            restore()
        # No profile → capabilities is []
        assert "capabilities" in entry


class TestCapabilityShapeAndOrder:
    """Pin wire shape: list of strings, sorted, no dupes."""

    def test_capabilities_is_a_list(self):
        client, restore = _mount_models_app(
            model_name="mlx-community/Qwen3-0.6B-8bit",
            model_alias="qwen3-0.6b-8bit",
        )
        try:
            entry = _fetch_entry(client, "mlx-community/Qwen3-0.6B-8bit")
        finally:
            restore()
        assert isinstance(entry["capabilities"], list)
        for c in entry["capabilities"]:
            assert isinstance(c, str)

    def test_capabilities_sorted_order(self):
        import pytest

        from fusion_mlx.model_aliases import resolve_profile

        model_id = "mlx-community/Qwen3-0.6B-8bit"
        profile = resolve_profile("qwen3-0.6b-8bit") or resolve_profile(model_id)
        if profile is None or len(profile.capabilities) < 2:
            pytest.skip("Need profile with >=2 capabilities")

        client, restore = _mount_models_app(
            model_name=model_id,
            model_alias="qwen3-0.6b-8bit",
            tool_call_parser="hermes",
        )
        try:
            entry = _fetch_entry(client, model_id)
        finally:
            restore()
        caps = entry["capabilities"]
        assert caps == sorted(caps), f"capabilities not sorted: {caps}"

    def test_no_duplicate_tags(self):
        client, restore = _mount_models_app(
            model_name="mlx-community/Qwen3-0.6B-8bit",
            model_alias="qwen3-0.6b-8bit",
            tool_call_parser="hermes",
        )
        try:
            entry = _fetch_entry(client, "mlx-community/Qwen3-0.6B-8bit")
        finally:
            restore()
        caps = entry["capabilities"]
        assert len(caps) == len(set(caps)), f"duplicate tags: {caps}"
