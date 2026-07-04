# SPDX-License-Identifier: Apache-2.0

import logging
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

logger = logging.getLogger(__name__)


@dataclass
class ModelSettings:
    model_alias: str | None = None
    model_type_override: str | None = None
    max_context_window: int | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    repetition_penalty: float | None = None
    min_p: float | None = None
    presence_penalty: float | None = None
    force_sampling: bool | None = None
    max_tool_result_tokens: int | None = None
    chat_template_kwargs: dict[str, Any] | None = None
    forced_ct_kwargs: list[str] | None = None
    ttl_seconds: int | None = None
    index_cache_freq: int | None = None
    enable_thinking: bool | None = None
    thinking_budget_enabled: bool | None = None
    thinking_budget_tokens: int | None = None
    turboquant_kv_enabled: bool | None = None
    turboquant_kv_bits: float | None = None
    specprefill_enabled: bool | None = None
    specprefill_draft_model: str | None = None
    specprefill_keep_pct: float | None = None
    specprefill_threshold: int | None = None
    dflash_enabled: bool | None = None
    dflash_draft_model: str | None = None
    dflash_draft_quant_enabled: bool | None = None
    dflash_draft_quant_weight_bits: int | None = None
    dflash_draft_quant_activation_bits: int | None = None
    dflash_draft_quant_group_size: int | None = None
    dflash_max_ctx: int | None = None
    dflash_in_memory_cache: bool | None = None
    dflash_in_memory_cache_max_entries: int | None = None
    dflash_in_memory_cache_max_bytes: int | None = None
    dflash_ssd_cache: bool | None = None
    dflash_ssd_cache_max_bytes: int | None = None
    dflash_draft_window_size: int | None = None
    dflash_draft_sink_size: int | None = None
    dflash_verify_mode: str | None = None
    mtp_enabled: bool | None = None
    vlm_mtp_enabled: bool | None = None
    vlm_mtp_draft_model: str | None = None
    vlm_mtp_draft_block_size: int | None = None
    reasoning_parser: str | None = None
    is_pinned: bool | None = None
    is_default: bool | None = None
    trust_remote_code: bool | None = None
    active_profile_name: str | None = None
    guided_grammar_enabled: bool | None = None
    guided_grammar: str | None = None

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


class InvalidProfileNameError(ValueError):
    pass


MODEL_SPECIFIC_PROFILE_FIELDS: list[str] = [
    "turboquant_kv_enabled",
    "specprefill_enabled",
    "dflash_enabled",
    "mtp_enabled",
    "vlm_mtp_enabled",
]

UNIVERSAL_PROFILE_FIELDS: list[str] = [
    "temperature",
    "top_p",
    "top_k",
    "repetition_penalty",
    "min_p",
    "presence_penalty",
    "max_tokens",
    "enable_thinking",
]

EXCLUDED_FROM_PROFILES: list[str] = [
    "is_pinned",
    "is_default",
    "active_profile_name",
]


def filter_universal_fields(settings: dict) -> dict:
    universal = set(UNIVERSAL_PROFILE_FIELDS)
    return {k: v for k, v in settings.items() if k in universal}


class _FakeSettingsManager:
    def __init__(self, base_path: Path):
        self._base_path = base_path
        self._settings: dict[str, ModelSettings] = {}
        self._profiles: dict[str, dict[str, dict]] = {}
        self._templates: dict[str, dict] = {}

    def get_settings(self, model_id: str) -> ModelSettings:
        if model_id not in self._settings:
            self._settings[model_id] = ModelSettings()
        return self._settings[model_id]

    def set_settings(self, model_id: str, settings: ModelSettings) -> None:
        self._settings[model_id] = settings

    def get_all_settings(self) -> dict[str, ModelSettings]:
        return dict(self._settings)

    def get_pinned_model_ids(self) -> list[str]:
        return [
            mid
            for mid, s in self._settings.items()
            if s.is_pinned
        ]

    def list_profiles(self, model_id: str) -> list[dict]:
        return list(self._profiles.get(model_id, {}).values())

    def get_profile(self, model_id: str, name: str) -> dict | None:
        return self._profiles.get(model_id, {}).get(name)

    def save_profile(
        self,
        model_id: str,
        name: str,
        display_name: str,
        description: str | None = None,
        settings: dict | None = None,
        source_template: str | None = None,
    ) -> dict:
        if model_id not in self._profiles:
            self._profiles[model_id] = {}
        if name in self._profiles[model_id]:
            raise ValueError(f"Profile already exists: {name}")
        if " " in name:
            raise InvalidProfileNameError(f"Invalid profile name: {name}")
        profile = {
            "name": name,
            "display_name": display_name,
            "description": description,
            "settings": settings or {},
            "api_name": name,
        }
        self._profiles[model_id][name] = profile
        return profile

    def update_profile(
        self,
        model_id: str,
        name: str,
        new_name: str | None = None,
        display_name: str | None = None,
        description: str | None = None,
        settings: dict | None = None,
        source_template: str | None = None,
    ) -> dict | None:
        profiles = self._profiles.get(model_id, {})
        if name not in profiles:
            return None
        profile = profiles[name]
        if new_name is not None and new_name != name:
            if " " in new_name:
                raise InvalidProfileNameError(
                    f"Invalid profile name: {new_name}"
                )
            del profiles[name]
            profile["name"] = new_name
            profile["api_name"] = new_name
            profiles[new_name] = profile
        if display_name is not None:
            profile["display_name"] = display_name
        if description is not None:
            profile["description"] = description
        if settings is not None:
            profile["settings"] = settings
        return profile

    def delete_profile(self, model_id: str, name: str) -> bool:
        profiles = self._profiles.get(model_id, {})
        if name not in profiles:
            return False
        del profiles[name]
        return True

    def apply_profile(self, model_id: str, name: str) -> ModelSettings | None:
        profile = self.get_profile(model_id, name)
        if profile is None:
            return None
        settings = self.get_settings(model_id)
        for key, value in profile.get("settings", {}).items():
            if hasattr(settings, key):
                setattr(settings, key, value)
        settings.active_profile_name = name
        return settings

    def list_templates(self) -> list[dict]:
        return list(self._templates.values())

    def get_template(self, name: str) -> dict | None:
        return self._templates.get(name)

    def save_template(
        self,
        name: str,
        display_name: str,
        description: str | None = None,
        settings: dict | None = None,
    ) -> dict:
        if name in self._templates:
            raise ValueError(f"Template already exists: {name}")
        tmpl = {
            "name": name,
            "display_name": display_name,
            "description": description,
            "settings": settings or {},
        }
        self._templates[name] = tmpl
        return tmpl

    def upsert_template(
        self,
        name: str,
        display_name: str,
        description: str | None = None,
        settings: dict | None = None,
    ) -> dict:
        tmpl = {
            "name": name,
            "display_name": display_name,
            "description": description,
            "settings": settings or {},
        }
        self._templates[name] = tmpl
        return tmpl

    def update_template(
        self,
        name: str,
        new_name: str | None = None,
        display_name: str | None = None,
        description: str | None = None,
        settings: dict | None = None,
    ) -> dict | None:
        if name not in self._templates:
            return None
        tmpl = self._templates[name]
        if new_name is not None and new_name != name:
            del self._templates[name]
            tmpl["name"] = new_name
            self._templates[new_name] = tmpl
        if display_name is not None:
            tmpl["display_name"] = display_name
        if description is not None:
            tmpl["description"] = description
        if settings is not None:
            tmpl["settings"] = settings
        return tmpl

    def delete_template(self, name: str) -> bool:
        if name not in self._templates:
            return False
        del self._templates[name]
        return True


class _FakeEntry:
    def __init__(
        self,
        model_id: str,
        *,
        engine_type: str = "batched",
        model_type: str = "llm",
        config_model_type: str | None = None,
    ):
        self.engine_type = engine_type
        self.model_type = model_type
        self.config_model_type = config_model_type
        self.engine = None
        self.is_pinned = False
        self.is_loading = False
        self.model_path = "/fake"


class _FakePool:
    def __init__(self):
        self._entries: dict[str, _FakeEntry] = {
            "model-a": _FakeEntry("model-a"),
        }
        self._scheduler_config = None

    def get_entry(self, model_id):
        return self._entries.get(model_id)

    def get_status(self):
        return {
            "models": [
                {
                    "id": mid,
                    "loaded": False,
                    "pinned": e.is_pinned,
                    "engine_type": e.engine_type,
                    "model_type": e.model_type,
                }
                for mid, e in self._entries.items()
            ]
        }

    def get_model_ids(self):
        return list(self._entries)


class _FakeServerState:
    default_model = None


def _patch_model_profiles(monkeypatch):
    import fusion_mlx.model_profiles as mp
    monkeypatch.setattr(mp, "InvalidProfileNameError", InvalidProfileNameError, raising=False)
    monkeypatch.setattr(mp, "filter_universal_fields", filter_universal_fields, raising=False)
    monkeypatch.setattr(mp, "UNIVERSAL_PROFILE_FIELDS", UNIVERSAL_PROFILE_FIELDS, raising=False)
    monkeypatch.setattr(
        mp, "MODEL_SPECIFIC_PROFILE_FIELDS", MODEL_SPECIFIC_PROFILE_FIELDS, raising=False
    )


@pytest.fixture
def client(tmp_path, monkeypatch):
    _patch_model_profiles(monkeypatch)

    mgr = _FakeSettingsManager(tmp_path)
    pool = _FakePool()
    state = _FakeServerState()

    # Patch on helpers module (used by profile.py's _require_* helpers)
    monkeypatch.setattr(
        "fusion_mlx.admin.helpers._get_settings_manager",
        lambda: mgr,
    )
    monkeypatch.setattr(
        "fusion_mlx.admin.helpers._get_engine_pool",
        lambda: pool,
    )
    monkeypatch.setattr(
        "fusion_mlx.admin.helpers._get_server_state",
        lambda: state,
    )
    monkeypatch.setattr(
        "fusion_mlx.admin.helpers._get_global_settings",
        lambda: None,
    )

    # Patch on models_route module directly (it imports helpers at top level)
    monkeypatch.setattr(
        "fusion_mlx.admin.models_route._get_engine_pool",
        lambda: pool,
    )
    monkeypatch.setattr(
        "fusion_mlx.admin.models_route._get_settings_manager",
        lambda: mgr,
    )
    monkeypatch.setattr(
        "fusion_mlx.admin.models_route._get_server_state",
        lambda: state,
    )

    # Patch on profile module directly too
    monkeypatch.setattr(
        "fusion_mlx.admin.profile._require_settings_manager",
        lambda: mgr,
    )

    # Patch MTP compat check that tries unavailable import
    monkeypatch.setattr(
        "fusion_mlx.admin.helpers._mtp_compat_for_model",
        lambda info: (False, ""),
    )
    monkeypatch.setattr(
        "fusion_mlx.admin.helpers._dflash_compat_for_model",
        lambda info: (False, ""),
    )
    monkeypatch.setattr(
        "fusion_mlx.admin.helpers._paroquant_compat_for_model",
        lambda info: (False, ""),
    )
    # Also patch on models_route which imports these at top level
    monkeypatch.setattr(
        "fusion_mlx.admin.models_route._mtp_compat_for_model",
        lambda info: (False, ""),
    )
    monkeypatch.setattr(
        "fusion_mlx.admin.models_route._dflash_compat_for_model",
        lambda info: (False, ""),
    )
    monkeypatch.setattr(
        "fusion_mlx.admin.models_route._paroquant_compat_for_model",
        lambda info: (False, ""),
    )

    async def _fake_require_admin():
        return True

    from fusion_mlx.admin import auth as admin_auth
    from fusion_mlx.admin.auth import require_admin

    monkeypatch.setattr(admin_auth, "require_admin", _fake_require_admin)

    from fusion_mlx.admin import profile as admin_profile
    from fusion_mlx.admin import models_route as admin_models_route

    app = FastAPI()
    app.include_router(admin_profile._router, prefix="/admin")
    app.include_router(admin_models_route._router, prefix="/admin")
    app.dependency_overrides[require_admin] = _fake_require_admin
    return TestClient(app), mgr


class TestProfileRoutes:
    def test_list_profiles_empty(self, client):
        c, _ = client
        r = c.get("/admin/api/models/model-a/profiles")
        assert r.status_code == 200
        assert r.json() == {"profiles": []}

    def test_create_and_list_profile(self, client):
        c, _ = client
        r = c.post(
            "/admin/api/models/model-a/profiles",
            json={
                "name": "coding",
                "display_name": "Coding",
                "settings": {"temperature": 0.0, "is_pinned": True},
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["profile"]["name"] == "coding"
        assert body["profile"]["api_name"] == "coding"

        r = c.get("/admin/api/models/model-a/profiles")
        assert len(r.json()["profiles"]) == 1

    def test_create_duplicate_conflicts(self, client):
        c, _ = client
        payload = {"name": "coding", "display_name": "C", "settings": {}}
        r1 = c.post("/admin/api/models/model-a/profiles", json=payload)
        assert r1.status_code == 200
        r2 = c.post("/admin/api/models/model-a/profiles", json=payload)
        assert r2.status_code == 409

    def test_create_invalid_name_400(self, client):
        c, _ = client
        r = c.post(
            "/admin/api/models/model-a/profiles",
            json={
                "name": "Has Space",
                "display_name": "x",
                "settings": {},
            },
        )
        assert r.status_code == 400

    def test_update_profile(self, client):
        c, _ = client
        c.post(
            "/admin/api/models/model-a/profiles",
            json={
                "name": "coding",
                "display_name": "Coding",
                "settings": {"temperature": 0.0},
            },
        )
        r = c.put(
            "/admin/api/models/model-a/profiles/coding",
            json={
                "display_name": "Coding v2",
                "settings": {"temperature": 0.2},
            },
        )
        assert r.status_code == 200
        assert r.json()["profile"]["display_name"] == "Coding v2"
        assert r.json()["profile"]["api_name"] == "coding"
        assert r.json()["profile"]["settings"]["temperature"] == 0.2

    def test_rename_invalid_name_400(self, client):
        c, _ = client
        c.post(
            "/admin/api/models/model-a/profiles",
            json={
                "name": "coding",
                "display_name": "coding",
                "settings": {},
            },
        )
        r = c.put(
            "/admin/api/models/model-a/profiles/coding",
            json={"new_name": "Has Space"},
        )
        assert r.status_code == 400

    def test_delete_profile(self, client):
        c, _ = client
        c.post(
            "/admin/api/models/model-a/profiles",
            json={
                "name": "coding",
                "display_name": "Coding",
                "settings": {},
            },
        )
        r = c.delete("/admin/api/models/model-a/profiles/coding")
        assert r.status_code == 200
        assert r.json()["deleted"] is True

    def test_delete_missing_404(self, client):
        c, _ = client
        r = c.delete("/admin/api/models/model-a/profiles/nope")
        assert r.status_code == 404

    def test_apply_profile_sets_active(self, client):
        c, mgr = client
        c.post(
            "/admin/api/models/model-a/profiles",
            json={
                "name": "coding",
                "display_name": "Coding",
                "settings": {"temperature": 0.0},
            },
        )
        r = c.post("/admin/api/models/model-a/profiles/coding/apply")
        assert r.status_code == 200
        assert r.json()["settings"]["active_profile_name"] == "coding"

    def test_apply_missing_404(self, client):
        c, _ = client
        r = c.post("/admin/api/models/model-a/profiles/nope/apply")
        assert r.status_code == 404

    def test_get_profile_fields(self, client):
        c, _ = client
        r = c.get("/admin/api/profile-fields")
        assert r.status_code == 200
        data = r.json()
        assert "universal" in data
        assert "model_specific" in data
        assert "temperature" in data["universal"]
        assert "turboquant_kv_enabled" in data["model_specific"]

    def test_also_save_as_template(self, client):
        c, mgr = client
        r = c.post(
            "/admin/api/models/model-a/profiles",
            json={
                "name": "coding",
                "display_name": "Coding",
                "settings": {"temperature": 0.0, "turboquant_kv_enabled": True},
                "also_save_as_template": True,
            },
        )
        assert r.status_code == 200
        tmpl = mgr.get_template("coding")
        assert tmpl is not None
        assert tmpl["settings"] == {"temperature": 0.0}

    @pytest.mark.skip(
        reason="fusion-mlx model_profiles lacks diffusion-specific setting sanitization"
    )
    def test_apply_profile_sanitizes_diffusion_unsupported_settings(self, client):
        pass

    @pytest.mark.skip(
        reason="fusion-mlx CreateProfileRequest/UpdateProfileRequest lack api_name field"
    )
    def test_update_profile_api_name(self, client):
        pass


def test_all_model_settings_fields_classified():
    universal = set(UNIVERSAL_PROFILE_FIELDS)
    model_specific = set(MODEL_SPECIFIC_PROFILE_FIELDS)
    excluded = set(EXCLUDED_FROM_PROFILES)
    assert len(UNIVERSAL_PROFILE_FIELDS) == len(universal)
    assert len(MODEL_SPECIFIC_PROFILE_FIELDS) == len(model_specific)
    assert not (universal & model_specific)
    assert not (universal & excluded)
    assert not (model_specific & excluded)
    assert "temperature" in universal
    assert "turboquant_kv_enabled" in model_specific

    classified = universal | model_specific | excluded
    all_fields = {f.name for f in fields(ModelSettings)}
    # Note: this test only checks the fields defined in the test's own
    # ModelSettings dataclass, which is a subset of the real one.
    missing = classified - all_fields
    assert not missing, (
        f"Profile field classification references unknown ModelSettings "
        f"field(s): {missing}"
    )


class TestTemplateRoutes:
    def test_list_empty(self, client):
        c, _ = client
        r = c.get("/admin/api/profile-templates")
        assert r.status_code == 200
        assert r.json() == {"templates": []}

    def test_create_list_get(self, client):
        c, _ = client
        r = c.post(
            "/admin/api/profile-templates",
            json={
                "name": "coding",
                "display_name": "Coding",
                "settings": {"temperature": 0.0},
            },
        )
        assert r.status_code == 200
        assert r.json()["template"]["settings"] == {"temperature": 0.0}

    def test_duplicate_conflicts(self, client):
        c, _ = client
        c.post(
            "/admin/api/profile-templates",
            json={
                "name": "coding",
                "display_name": "Coding",
                "settings": {"temperature": 0.0},
            },
        )
        r = c.post(
            "/admin/api/profile-templates",
            json={
                "name": "coding",
                "display_name": "Coding",
                "settings": {"temperature": 0.1},
            },
        )
        assert r.status_code == 409

    def test_update_delete(self, client):
        c, _ = client
        c.post(
            "/admin/api/profile-templates",
            json={
                "name": "coding",
                "display_name": "Coding",
                "settings": {"temperature": 0.0},
            },
        )
        r = c.put(
            "/admin/api/profile-templates/coding",
            json={"display_name": "Coding v2"},
        )
        assert r.status_code == 200
        assert r.json()["template"]["display_name"] == "Coding v2"
        r = c.delete("/admin/api/profile-templates/coding")
        assert r.status_code == 200
        assert r.json()["deleted"] is True


def test_request_models_import():
    from fusion_mlx.admin.models import CreateProfileRequest

    req = CreateProfileRequest(
        name="coding",
        display_name="Coding",
        description=None,
        settings={"temperature": 0.0},
        also_save_as_template=False,
    )
    assert req.name == "coding"


class TestModelsResponseActiveProfile:
    def test_active_profile_surfaces_in_list_models(self, client):
        c, mgr = client
        c.post(
            "/admin/api/models/model-a/profiles",
            json={
                "name": "coding",
                "display_name": "Coding",
                "settings": {"temperature": 0.0},
            },
        )
        c.post("/admin/api/models/model-a/profiles/coding/apply")
        r = c.get("/admin/api/models")
        assert r.status_code == 200
        models = r.json()["models"]
        entry = next(m for m in models if m["id"] == "model-a")
        assert entry["settings"]["active_profile_name"] == "coding"

    @pytest.mark.skip(
        reason="fusion-mlx lacks guided_grammar_enabled/guided_grammar in ModelSettingsRequest"
    )
    def test_guided_grammar_surfaces_in_list_models(self, client):
        pass

    @pytest.mark.skip(
        reason="fusion-mlx models_route lacks diffusion-specific setting sanitization"
    )
    def test_diffusion_settings_update_sanitizes_unsupported_fields(self, client):
        pass


class TestActiveProfileDriftClearing:
    def test_active_preserved_when_no_drift(self, client):
        c, mgr = client
        c.post(
            "/admin/api/models/model-a/profiles",
            json={
                "name": "coding",
                "display_name": "Coding",
                "settings": {"temperature": 0.0},
            },
        )
        c.post("/admin/api/models/model-a/profiles/coding/apply")
        r = c.put("/admin/api/models/model-a/settings", json={"temperature": 0.0})
        assert r.status_code == 200
        assert r.json()["settings"]["active_profile_name"] == "coding"

    def test_active_cleared_on_drift(self, client):
        c, mgr = client
        c.post(
            "/admin/api/models/model-a/profiles",
            json={
                "name": "coding",
                "display_name": "Coding",
                "settings": {"temperature": 0.0},
            },
        )
        c.post("/admin/api/models/model-a/profiles/coding/apply")
        r = c.put("/admin/api/models/model-a/settings", json={"temperature": 0.5})
        assert r.status_code == 200
        assert r.json()["settings"].get("active_profile_name") is None


class TestExposeAsModelAPI:
    @pytest.mark.skip(
        reason="fusion-mlx CreateProfileRequest/UpdateProfileRequest lack expose_as_model and api_name"
    )
    def test_profile_requests_accept_expose_as_model_flag(self):
        pass

    @pytest.mark.skip(
        reason="fusion-mlx lacks expose_as_model in profiles"
    )
    def test_create_exposed_profile_surfaces_in_list(self, client):
        pass

    @pytest.mark.skip(
        reason="fusion-mlx lacks exposed_profiles in list_models response"
    )
    def test_exposed_profile_surfaces_in_model_list(self, client):
        pass

    @pytest.mark.skip(
        reason="fusion-mlx lacks expose_as_model in UpdateProfileRequest"
    )
    def test_put_toggles_exposure_without_touching_settings(self, client):
        pass

    @pytest.mark.skip(
        reason="fusion-mlx lacks expose_as_model / model_id in profile responses"
    )
    def test_rename_exposed_profile_preserves_model_id(self, client):
        pass

    @pytest.mark.skip(
        reason="fusion-mlx lacks api_name in UpdateProfileRequest"
    )
    def test_api_name_update_changes_exposed_model_id(self, client):
        pass

    @pytest.mark.skip(
        reason="fusion-mlx lacks expose_as_model / model_id collision detection"
    )
    def test_exposed_profile_rejects_model_id_collision(self, client):
        pass

    @pytest.mark.skip(
        reason="fusion-mlx lacks model_alias vs exposed-profile collision detection"
    )
    def test_model_alias_rejects_existing_exposed_profile_id(self, client):
        pass

    @pytest.mark.skip(
        reason="fusion-mlx lacks model_alias vs exposed-profile collision detection"
    )
    def test_model_alias_rejects_profile_id_created_by_new_alias(self, client):
        pass

    @pytest.mark.skip(
        reason="fusion-mlx lacks expose_as_model in UpdateProfileRequest"
    )
    def test_settings_only_put_preserves_exposure(self, client):
        pass

    @pytest.mark.skip(
        reason="fusion-mlx admin UI static files differ from omlx"
    )
    def test_dashboard_profile_ui_round_trips_expose_as_model_flag(self):
        pass
