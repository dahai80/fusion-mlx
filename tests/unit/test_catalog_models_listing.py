import json
import logging

import pytest

from fusion_mlx.api.openai import _common
from fusion_mlx.api.openai_models import ModelInfo

CATALOG_ENTRY = {
    "schemaVersion": 1,
    "name": "Qwen--Qwen3-8B",
    "hubId": "",
    "revision": "",
    "contextWindowTokens": 32768,
    "maxOutputTokens": 8192,
    "modality": {"input": ["text"], "output": ["text"]},
    "capabilities": {
        "functionCall": True,
        "reasoning": True,
        "vision": False,
        "embedding": False,
    },
    "quantizations": [],
}


@pytest.fixture()
def catalog_dir(tmp_path, monkeypatch):
    d = tmp_path / "catalog" / "models"
    d.mkdir(parents=True)
    monkeypatch.setenv("FUSION_MLX_MODEL_CATALOG_DIR", str(d))
    _common._catalog_cache.clear()
    _common._catalog_warned.clear()
    return d


def _write(d, name, data):
    (d / f"{name}.json").write_text(json.dumps(data), encoding="utf-8")


def test_enrichment_fields_parsed(catalog_dir):
    _write(catalog_dir, "Qwen--Qwen3-8B", CATALOG_ENTRY)
    extra = _common._catalog_enrichment("Qwen--Qwen3-8B")
    assert extra["context_length"] == 32768
    assert extra["max_output_tokens"] == 8192
    assert extra["capabilities"]["tool_calling"] is True
    assert extra["capabilities"]["reasoning"] is True
    assert extra["capabilities"]["vision"] is False


def test_enrichment_accepts_alias_forms(catalog_dir):
    _write(catalog_dir, "Qwen--Qwen3-8B", CATALOG_ENTRY)
    for mid in ("Qwen/Qwen3-8B", "QWEN--qwen3-8b"):
        extra = _common._catalog_enrichment(mid)
        assert extra.get("context_length") == 32768, mid


def test_missing_entry_degrades_with_one_warning(catalog_dir, caplog):
    with caplog.at_level(logging.WARNING, logger="fusion_mlx.api.openai._common"):
        first = _common._catalog_enrichment("Unknown--Model-9")
        second = _common._catalog_enrichment("Unknown--Model-9")
    assert first == {}
    assert second == {}
    warns = [r for r in caplog.records if "no entry" in r.getMessage()]
    assert len(warns) == 1


def test_malformed_json_degrades(catalog_dir):
    (catalog_dir / "Broken--Model.json").write_text("{not json", encoding="utf-8")
    assert _common._catalog_enrichment("Broken--Model") == {}


def test_no_env_and_no_sibling_repo(monkeypatch):
    monkeypatch.delenv("FUSION_MLX_MODEL_CATALOG_DIR", raising=False)
    _common._catalog_cache.clear()
    monkeypatch.setattr(_common, "_model_catalog_dir", lambda: None, raising=False)
    assert _common._catalog_enrichment("Whatever--Model") == {}


def test_mlx_community_prefix_stripped(catalog_dir):
    _write(catalog_dir, "Qwen3-8B", CATALOG_ENTRY)
    extra = _common._catalog_enrichment("mlx-community--Qwen3-8B")
    assert extra.get("context_length") == 32768


def test_apply_fills_only_missing_fields(catalog_dir):
    _write(catalog_dir, "Qwen--Qwen3-8B", CATALOG_ENTRY)
    info = ModelInfo(
        id="Qwen--Qwen3-8B",
        capabilities={"tool_calling": True, "vision": False},
        context_length=99999,
    )
    _common._apply_catalog_enrichment(info)
    assert info.context_length == 99999
    assert info.max_output_tokens == 8192
    assert info.capabilities["tool_calling"] is True
    assert info.capabilities["reasoning"] is True
    assert info.capabilities["vision"] is False


def test_apply_fills_context_when_runtime_none(catalog_dir):
    _write(catalog_dir, "Qwen--Qwen3-8B", CATALOG_ENTRY)
    info = ModelInfo(id="Qwen--Qwen3-8B")
    _common._apply_catalog_enrichment(info)
    assert info.context_length == 32768


def test_apply_image_modality_when_resolver_default(catalog_dir):
    entry = dict(CATALOG_ENTRY)
    entry["modality"] = {"input": ["image", "text"], "output": ["text"]}
    _write(catalog_dir, "Qwen--Qwen3-8B", entry)
    info = ModelInfo(id="Qwen--Qwen3-8B")
    _common._apply_catalog_enrichment(info)
    assert info.modality == "image"


def test_apply_keeps_resolver_modality(catalog_dir):
    entry = dict(CATALOG_ENTRY)
    entry["modality"] = {"input": ["image", "text"], "output": ["text"]}
    _write(catalog_dir, "Qwen--Qwen3-8B", entry)
    info = ModelInfo(id="Qwen--Qwen3-8B", modality="audio")
    _common._apply_catalog_enrichment(info)
    assert info.modality == "audio"


def test_non_bool_capability_flags_ignored(catalog_dir):
    entry = dict(CATALOG_ENTRY)
    entry["capabilities"] = {"functionCall": "yes", "vision": True}
    _write(catalog_dir, "Qwen--Qwen3-8B", entry)
    extra = _common._catalog_enrichment("Qwen--Qwen3-8B")
    assert "tool_calling" not in extra["capabilities"]
    assert extra["capabilities"]["vision"] is True


def test_registry_knows_catalog_env_var():
    from tests.unit.test_no_out_of_band_routing import (
        ALLOWED_FUSION_MLX_ENV_VARS,
    )

    assert "FUSION_MLX_MODEL_CATALOG_DIR" in ALLOWED_FUSION_MLX_ENV_VARS
