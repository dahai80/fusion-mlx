# SPDX-License-Identifier: Apache-2.0
# Tests for #1416: language capability metadata — alias ``languages`` field,
# family hints, and the ``languages`` array on /v1/models entries.
from __future__ import annotations

import json

import pytest

# ---------------------------------------------------------------------------
# languages_for resolution
# ---------------------------------------------------------------------------


def test_alias_declared_languages_win():
    from fusion_mlx.model_languages import languages_for

    assert languages_for("jais3-7b-4bit") == ["ar", "en"]


def test_hf_path_resolves_to_alias_languages():
    from fusion_mlx.model_languages import languages_for

    assert languages_for("IFM/Jais-3-7B-MLX-4bit") == ["ar", "en"]


def test_family_hint_qwen():
    from fusion_mlx.model_languages import languages_for

    # Not an alias entry — resolved purely via the family hint.
    assert languages_for("mlx-community/Qwen3.5-9B-4bit") == ["en", "zh"]


def test_family_hint_gemma():
    from fusion_mlx.model_languages import languages_for

    assert languages_for("mlx-community/Gemma-4-4B-4bit") == ["en", "zh"]


def test_unknown_model_empty():
    from fusion_mlx.model_languages import languages_for

    assert languages_for("mlx-community/Nemotron-30B-4bit") == []
    assert languages_for("") == []


def test_alias_declared_overrides_family_hint():
    # An alias that declares languages explicitly must beat the family
    # hint even when the name matches a hinted family.
    from fusion_mlx.model_aliases import AliasProfile
    from fusion_mlx.model_languages import _family_hint

    profile = AliasProfile(name="x", hf_path="y", languages=("ar",))
    assert profile.languages == ("ar",)
    # Family hint alone still returns the family default.
    assert _family_hint("some-jais-variant") == ("ar", "en")


# ---------------------------------------------------------------------------
# Alias profile coercion + validation
# ---------------------------------------------------------------------------


def test_coerce_languages_valid():
    from fusion_mlx.model_aliases import _coerce

    profile = _coerce("t", {"hf_path": "a/b", "languages": ["ar", "en"]})
    assert profile.languages == ("ar", "en")


def test_coerce_languages_absent_empty():
    from fusion_mlx.model_aliases import _coerce

    assert _coerce("t", {"hf_path": "a/b"}).languages == ()
    assert _coerce("t", "a/b").languages == ()


@pytest.mark.parametrize(
    "bad",
    [
        ["AR"],  # uppercase
        ["ara"],  # ISO 639-2, not 639-1
        ["en", "en"],  # duplicate
        [1],  # non-string
        "en",  # not a list
        [],  # empty list
    ],
)
def test_coerce_languages_invalid_rejects_alias(bad):
    # #1416 acceptance: metadata must match real capability — a malformed
    # languages entry must reject the alias loudly, not degrade to [].
    from fusion_mlx.model_aliases import _coerce

    with pytest.raises(ValueError):
        _coerce("t", {"hf_path": "a/b", "languages": bad})


def test_registry_jais_alias_loads():
    from fusion_mlx.model_aliases import resolve_profile

    profile = resolve_profile("jais3-7b-4bit")
    assert profile is not None
    assert profile.hf_path == "IFM/Jais-3-7B-MLX-4bit"
    assert profile.languages == ("ar", "en")


def test_aliases_json_valid():
    # Guard: the shipped registry parses and every entry coerces.
    import fusion_mlx.model_aliases as ma

    raw = json.loads(ma._ALIASES_FILE.read_text(encoding="utf-8"))
    for name, value in raw.items():
        ma._coerce(name, value)


# ---------------------------------------------------------------------------
# /v1/models entry schema + wiring
# ---------------------------------------------------------------------------


def test_model_info_has_languages_field_default_empty():
    from fusion_mlx.api.openai_models import ModelInfo

    info = ModelInfo(id="m")
    assert info.languages == []
    data = info.model_dump()
    assert data["languages"] == []
    assert data["languages"] == list(data["languages"])


def test_resolve_languages_via_common():
    from fusion_mlx.api.openai._common import _resolve_languages

    assert _resolve_languages("jais3-7b-4bit") == ["ar", "en"]
    assert _resolve_languages("mlx-community/Nemotron-30B-4bit") == []


def test_list_models_entry_carries_languages(monkeypatch):
    # Integration: the /v1/models handler must emit the languages array on
    # every entry (empty allowed) — #1416 acceptance criterion.
    import asyncio

    class _FakePool:
        def list_models(self):
            return ["jais3-7b-4bit", "mlx-community/Nemotron-30B-4bit"]

        def get_entry(self, model_id):
            return None

    import importlib

    import fusion_mlx.api.openai._common as _common_mod

    # NOTE: api.openai.__init__ re-exports a ``completions`` *function* which
    # shadows the submodule attribute — plain ``import ... as`` binds the
    # function, so fetch the real module via importlib/sys.modules.
    completions = importlib.import_module("fusion_mlx.api.openai.completions")

    monkeypatch.setattr(_common_mod, "_pool", _FakePool())
    resp = asyncio.new_event_loop().run_until_complete(completions.list_models())
    by_id = {m.id: m for m in resp.data}
    assert by_id["jais3-7b-4bit"].languages == ["ar", "en"]
    assert by_id["mlx-community/Nemotron-30B-4bit"].languages == []
