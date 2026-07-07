"""Per-model settings management for fusion-mlx."""

from __future__ import annotations

import copy
import json
import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Optional

from ..model_profiles import (
    MODEL_SPECIFIC_PROFILE_FIELDS,
    filter_profile_fields,
    filter_universal_fields,
    slugify_profile_api_name,
    utcnow,
    validate_profile_name,
)

logger = logging.getLogger(__name__)

SETTINGS_VERSION = 1
PROFILES_VERSION = 1
TEMPLATES_VERSION = 1


@dataclass
class ModelSettings:
    max_context_window: int | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    repetition_penalty: float | None = None
    min_p: float | None = None
    presence_penalty: float | None = None
    force_sampling: bool = False
    max_tool_result_tokens: int | None = None
    chat_template_kwargs: dict[str, Any] | None = None
    forced_ct_kwargs: list[str] | None = None
    ttl_seconds: int | None = None
    model_type_override: str | None = None
    model_alias: str | None = None
    index_cache_freq: int | None = None
    enable_thinking: bool | None = None
    preserve_thinking: bool | None = None
    thinking_budget_enabled: bool = False
    thinking_budget_tokens: int | None = None
    reasoning_parser: str | None = None
    guided_grammar_enabled: bool = False
    guided_grammar: str | None = None

    turboquant_kv_enabled: bool = False
    turboquant_kv_bits: float = 4
    turboquant_skip_last: bool = True

    specprefill_enabled: bool = False
    specprefill_draft_model: str | None = None
    specprefill_keep_pct: float | None = None
    specprefill_threshold: int | None = None

    dflash_enabled: bool = False
    dflash_draft_model: str | None = None
    dflash_draft_quant_enabled: bool | None = None
    dflash_draft_quant_weight_bits: int | None = None
    dflash_draft_quant_activation_bits: int | None = None
    dflash_draft_quant_group_size: int | None = None
    dflash_max_ctx: int | None = None
    dflash_in_memory_cache: bool = True
    dflash_in_memory_cache_max_entries: int = 4
    dflash_in_memory_cache_max_bytes: int = 8 * 1024 * 1024 * 1024
    dflash_ssd_cache: bool = False
    dflash_ssd_cache_max_bytes: int = 20 * 1024 * 1024 * 1024
    dflash_draft_window_size: int | None = None
    dflash_draft_sink_size: int | None = None
    dflash_verify_mode: str | None = None

    mtp_enabled: bool = False

    vlm_mtp_enabled: bool = False
    vlm_mtp_draft_model: str | None = None
    vlm_mtp_draft_block_size: int | None = None

    ngram_spec_enabled: bool = False
    ngram_spec_order: int | None = None
    ngram_spec_num_draft: int | None = None
    ngram_spec_break_even: float | None = None

    is_pinned: bool = False
    is_default: bool = False

    trust_remote_code: bool = False

    display_name: str | None = None
    description: str | None = None
    active_profile_name: str | None = None

    def __post_init__(self) -> None:
        if self.mtp_enabled and self.dflash_enabled:
            raise ValueError(
                "mtp_enabled and dflash_enabled cannot both be True; "
                "choose one speculative-decoding path per model"
            )
        if self.mtp_enabled and self.turboquant_kv_enabled:
            raise ValueError(
                "mtp_enabled and turboquant_kv_enabled cannot both be True; "
                "TurboQuant patches the attention path that MTP relies on"
            )
        if self.vlm_mtp_enabled:
            conflicts = [
                ("dflash_enabled", self.dflash_enabled),
                ("specprefill_enabled", self.specprefill_enabled),
                ("mtp_enabled", self.mtp_enabled),
                ("turboquant_kv_enabled", self.turboquant_kv_enabled),
            ]
            for name, value in conflicts:
                if value:
                    raise ValueError(
                        f"vlm_mtp_enabled and {name} cannot both be True; "
                        "choose one speculative path per model"
                    )

    def to_dict(self) -> dict:
        result = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if value is not None:
                result[f.name] = value
        return result

    @classmethod
    def from_dict(cls, data: dict) -> ModelSettings:
        valid_fields = {f.name for f in fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)


class ModelSettingsManager:
    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
        self.settings_file = self.base_path / "model_settings.json"
        self.profiles_file = self.base_path / "model_profiles.json"
        self.templates_file = self.base_path / "global_templates.json"
        self._lock = threading.Lock()
        self._settings: dict[str, ModelSettings] = {}
        self._profiles: dict[str, dict[str, dict[str, Any]]] = {}
        self._templates: dict[str, dict[str, Any]] = {}

        self.base_path.mkdir(parents=True, exist_ok=True)

        self._load()
        self._load_profiles()
        self._load_templates()

    def _load(self) -> None:
        if not self.settings_file.exists():
            logger.debug("Settings file not found: %s", self.settings_file)
            self._settings = {}
            return
        try:
            with open(self.settings_file, encoding="utf-8") as f:
                data = json.load(f)
            version = data.get("version", 1)
            if version != SETTINGS_VERSION:
                logger.warning(
                    "Settings file version %d differs from current %d",
                    version,
                    SETTINGS_VERSION,
                )
            models_data = data.get("models", {})
            self._settings = {}
            for model_id, model_data in models_data.items():
                try:
                    self._settings[model_id] = ModelSettings.from_dict(model_data)
                except Exception as e:
                    logger.warning(
                        "Failed to load settings for model '%s': %s", model_id, e
                    )
            logger.info("Loaded settings for %d models", len(self._settings))
        except json.JSONDecodeError as e:
            logger.error("Invalid JSON in settings file: %s", e)
            self._settings = {}
        except Exception as e:
            logger.error("Failed to load settings file: %s", e)
            self._settings = {}

    def _save(self) -> None:
        data = {
            "version": SETTINGS_VERSION,
            "models": {
                model_id: settings.to_dict()
                for model_id, settings in self._settings.items()
            },
        }
        try:
            temp_file = self.settings_file.with_suffix(".tmp")
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            temp_file.replace(self.settings_file)
            logger.debug("Saved settings for %d models", len(self._settings))
        except Exception as e:
            logger.error("Failed to save settings file: %s", e)
            raise

    def get_settings(self, model_id: str) -> ModelSettings:
        with self._lock:
            if model_id in self._settings:
                settings = self._settings[model_id]
                return ModelSettings.from_dict(settings.to_dict())
            return ModelSettings()

    def get_settings_for_request(
        self,
        model_id: str,
        resolved_model_id: str | None = None,
    ) -> ModelSettings:
        with self._lock:
            candidates = [model_id]
            if "/" in model_id:
                candidates.append(model_id.split("/", 1)[1])
            for candidate in candidates:
                profile_match = self._find_exposed_profile_locked(candidate)
                if profile_match is not None:
                    base_model_id, profile = profile_match
                    return self._settings_with_profile_locked(base_model_id, profile)
        return self.get_settings(resolved_model_id or model_id)

    def set_settings(self, model_id: str, settings: ModelSettings) -> None:
        with self._lock:
            if settings.is_default:
                for mid, s in self._settings.items():
                    if mid != model_id and s.is_default:
                        s.is_default = False
                        logger.info(
                            "Cleared is_default from model '%s' (new default: '%s')",
                            mid,
                            model_id,
                        )
            self._settings[model_id] = ModelSettings.from_dict(settings.to_dict())
            logger.info("Updated settings for model '%s'", model_id)
            self._save()

    def delete_settings(self, model_id: str) -> bool:
        with self._lock:
            removed = False
            if model_id in self._settings:
                del self._settings[model_id]
                self._save()
                removed = True
            if model_id in self._profiles:
                del self._profiles[model_id]
                self._save_profiles()
                removed = True
            if removed:
                logger.info("Deleted settings for model '%s'", model_id)
            return removed

    def get_default_model_id(self) -> str | None:
        with self._lock:
            for model_id, settings in self._settings.items():
                if settings.is_default:
                    return model_id
            return None

    def get_pinned_model_ids(self) -> list[str]:
        with self._lock:
            return [
                model_id
                for model_id, settings in self._settings.items()
                if settings.is_pinned
            ]

    def get_all_settings(self) -> dict[str, ModelSettings]:
        with self._lock:
            return {
                model_id: ModelSettings.from_dict(settings.to_dict())
                for model_id, settings in self._settings.items()
            }

    # ==================== Profiles ====================

    def _load_profiles(self) -> None:
        if not self.profiles_file.exists():
            self._profiles = {}
            return
        try:
            with open(self.profiles_file, encoding="utf-8") as f:
                data = json.load(f)
            version = data.get("version", 1)
            if version != PROFILES_VERSION:
                logger.warning(
                    "Profiles file version %d differs from current %d",
                    version,
                    PROFILES_VERSION,
                )
            self._profiles = data.get("profiles", {}) or {}
            changed = False
            for model_id, profiles in self._profiles.items():
                used_api_names: set[str] = set()
                for name, profile in profiles.items():
                    current_api_name = profile.get("api_name")
                    if current_api_name:
                        try:
                            validate_profile_name(current_api_name)
                            base_api_name = current_api_name
                        except Exception:
                            base_api_name = slugify_profile_api_name(
                                profile.get("display_name") or name,
                                fallback="profile",
                            )
                    else:
                        base_api_name = slugify_profile_api_name(
                            profile.get("display_name") or name,
                            fallback="profile",
                        )
                    api_name = self._dedupe_profile_api_name(
                        base_api_name, used_api_names
                    )
                    if current_api_name != api_name:
                        profile["api_name"] = api_name
                        changed = True
            if changed:
                self._save_profiles()
        except Exception as e:
            logger.error("Failed to load profiles file: %s", e)
            self._profiles = {}

    def _save_profiles(self) -> None:
        data = {"version": PROFILES_VERSION, "profiles": self._profiles}
        temp_file = self.profiles_file.with_suffix(".tmp")
        try:
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            temp_file.replace(self.profiles_file)
        except Exception as e:
            logger.error("Failed to save profiles file: %s", e)
            if temp_file.exists():
                temp_file.unlink(missing_ok=True)
            raise

    @staticmethod
    def _dedupe_profile_api_name(base: str, used: set[str]) -> str:
        validate_profile_name(base)
        candidate = base
        index = 2
        while candidate in used:
            suffix = f"-{index}"
            root = base[: 32 - len(suffix)].rstrip("-_") or "profile"
            candidate = f"{root}{suffix}"
            index += 1
        used.add(candidate)
        return candidate

    @staticmethod
    def _profile_api_name(profile: dict[str, Any]) -> str:
        api_name = profile.get("api_name") or profile["name"]
        validate_profile_name(api_name)
        return api_name

    def _allocate_profile_api_name_locked(
        self,
        profiles: dict[str, dict[str, Any]],
        value: str | None,
        *,
        display_name: str | None,
        internal_name: str,
        exclude_name: str | None = None,
    ) -> str:
        if value:
            validate_profile_name(value)
            base = value
        else:
            base = slugify_profile_api_name(
                display_name or internal_name,
                fallback="profile",
            )
        used = {
            self._profile_api_name(profile)
            for name, profile in profiles.items()
            if name != exclude_name
        }
        return self._dedupe_profile_api_name(base, used)

    def _profile_model_id(self, model_id: str, api_name: str) -> str:
        return f"{model_id}:{api_name}"

    def _display_profile_model_id_locked(
        self, model_id: str, profile: dict[str, Any]
    ) -> str:
        base = self._settings.get(model_id)
        display_base = base.model_alias if base and base.model_alias else model_id
        return self._profile_model_id(display_base, self._profile_api_name(profile))

    def _find_exposed_profile_locked(
        self, model_id: str
    ) -> tuple[str, dict[str, Any]] | None:
        for base_model_id, profiles in self._profiles.items():
            base = self._settings.get(base_model_id)
            alias = base.model_alias if base else None
            for profile in profiles.values():
                if not profile.get("expose_as_model"):
                    continue
                api_name = self._profile_api_name(profile)
                if model_id == self._profile_model_id(base_model_id, api_name):
                    return base_model_id, profile
                if alias and model_id == self._profile_model_id(alias, api_name):
                    return base_model_id, profile
        return None

    def _settings_with_profile_locked(
        self, model_id: str, profile: dict[str, Any]
    ) -> ModelSettings:
        base = self._settings.get(model_id)
        merged = base.to_dict() if base is not None else {}
        merged.update(filter_universal_fields(profile.get("settings", {}) or {}))
        return ModelSettings.from_dict(merged)

    def _runtime_settings_with_profile_locked(
        self, model_id: str, profile: dict[str, Any]
    ) -> ModelSettings:
        base = self._settings.get(model_id)
        merged = base.to_dict() if base is not None else {}
        merged.update(filter_profile_fields(profile.get("settings", {}) or {}))
        return ModelSettings.from_dict(merged)

    def get_exposed_profile_source_model_id(self, model_id: str) -> str | None:
        with self._lock:
            candidates = [model_id]
            if "/" in model_id:
                candidates.append(model_id.split("/", 1)[1])
            for candidate in candidates:
                match = self._find_exposed_profile_locked(candidate)
                if match is not None:
                    return match[0]
            return None

    def get_exposed_profile_runtime_settings_for_request(
        self,
        model_id: str,
    ) -> tuple[str, ModelSettings] | None:
        with self._lock:
            candidates = [model_id]
            if "/" in model_id:
                candidates.append(model_id.split("/", 1)[1])
            for candidate in candidates:
                match = self._find_exposed_profile_locked(candidate)
                if match is not None:
                    base_model_id, profile = match
                    return (
                        base_model_id,
                        self._runtime_settings_with_profile_locked(
                            base_model_id, profile
                        ),
                    )
            return None

    def get_exposed_profile_model_ids(
        self,
        *,
        exclude_model_id: str | None = None,
        exclude_profile_name: str | None = None,
    ) -> set[str]:
        with self._lock:
            model_ids: set[str] = set()
            for base_model_id, profiles in self._profiles.items():
                base = self._settings.get(base_model_id)
                alias = base.model_alias if base else None
                for profile in profiles.values():
                    if not profile.get("expose_as_model"):
                        continue
                    if (
                        exclude_model_id == base_model_id
                        and exclude_profile_name == profile["name"]
                    ):
                        continue
                    api_name = self._profile_api_name(profile)
                    model_ids.add(self._profile_model_id(base_model_id, api_name))
                    if alias:
                        model_ids.add(self._profile_model_id(alias, api_name))
            return model_ids

    def _profile_request_ids_locked(
        self,
        model_id: str,
        profile: dict[str, Any],
    ) -> set[str]:
        base = self._settings.get(model_id)
        base_ids = {model_id}
        if base and base.model_alias:
            base_ids.add(base.model_alias)
        api_name = self._profile_api_name(profile)
        return {self._profile_model_id(base_id, api_name) for base_id in base_ids}

    def _validate_exposed_profile_ids_available_locked(
        self,
        model_id: str,
        profile: dict[str, Any],
        *,
        exclude_profile_name: str | None = None,
        reserved_model_ids: set[str] | None = None,
    ) -> None:
        if not profile.get("expose_as_model"):
            return
        candidate_ids = self._profile_request_ids_locked(model_id, profile)
        if reserved_model_ids:
            for candidate_id in candidate_ids:
                if candidate_id != model_id and candidate_id in reserved_model_ids:
                    raise ValueError(
                        f"Exposed profile model ID '{candidate_id}' conflicts "
                        "with a model directory name"
                    )
        for mid, settings in self._settings.items():
            if settings.model_alias and settings.model_alias in candidate_ids:
                raise ValueError(
                    f"Exposed profile model ID '{settings.model_alias}' "
                    f"conflicts with model alias for '{mid}'"
                )
        existing_ids: set[str] = set()
        for base_model_id, profiles in self._profiles.items():
            for other in profiles.values():
                if not other.get("expose_as_model"):
                    continue
                if base_model_id == model_id and other["name"] == exclude_profile_name:
                    continue
                existing_ids.update(
                    self._profile_request_ids_locked(base_model_id, other)
                )
        conflict = candidate_ids & existing_ids
        if conflict:
            conflict_id = sorted(conflict)[0]
            raise ValueError(f"Exposed profile model ID '{conflict_id}' already exists")

    @staticmethod
    def _has_engine_fields(profile: dict[str, Any]) -> bool:
        settings = profile.get("settings", {}) or {}
        return any(
            k in MODEL_SPECIFIC_PROFILE_FIELDS and v is not None
            for k, v in settings.items()
        )

    def list_exposed_profile_models(self) -> list[dict]:
        with self._lock:
            exposed = []
            for base_model_id, profiles in self._profiles.items():
                for profile in profiles.values():
                    if not profile.get("expose_as_model"):
                        continue
                    item = dict(profile)
                    item["model_id"] = self._display_profile_model_id_locked(
                        base_model_id, profile
                    )
                    item["source_model_id"] = base_model_id
                    item["settings"] = self._settings_with_profile_locked(
                        base_model_id, item
                    ).to_dict()
                    exposed.append(item)
            return exposed

    def list_profiles(self, model_id: str) -> list[dict]:
        with self._lock:
            per_model = self._profiles.get(model_id, {})
            return [
                {
                    **p,
                    "model_id": self._display_profile_model_id_locked(model_id, p),
                    "has_engine_fields": self._has_engine_fields(p),
                }
                for p in per_model.values()
            ]

    def get_profile(self, model_id: str, name: str) -> dict | None:
        with self._lock:
            return dict(self._profiles.get(model_id, {}).get(name, {})) or None

    def save_profile(
        self,
        model_id: str,
        name: str,
        display_name: str,
        description: str | None,
        settings: dict[str, Any],
        source_template: str | None = None,
        expose_as_model: bool = False,
        api_name: str | None = None,
        reserved_model_ids: set[str] | None = None,
    ) -> dict:
        validate_profile_name(name)
        filtered = filter_profile_fields(settings or {})
        with self._lock:
            per_model = self._profiles.setdefault(model_id, {})
            if name in per_model:
                raise ValueError(
                    f"Profile '{name}' already exists for model '{model_id}'"
                )
            now = utcnow().isoformat()
            profile_api_name = self._allocate_profile_api_name_locked(
                per_model,
                api_name,
                display_name=display_name,
                internal_name=name,
            )
            profile_record = {
                "name": name,
                "display_name": display_name or name,
                "api_name": profile_api_name,
                "description": description,
                "created_at": now,
                "updated_at": now,
                "settings": filtered,
                "source_template": source_template,
                "expose_as_model": bool(expose_as_model),
            }
            self._validate_exposed_profile_ids_available_locked(
                model_id,
                profile_record,
                reserved_model_ids=reserved_model_ids,
            )
            per_model[name] = profile_record
            self._save_profiles()
            return dict(per_model[name])

    def update_profile(
        self,
        model_id: str,
        name: str,
        *,
        new_name: str | None = None,
        display_name: str | None = None,
        description: str | None = None,
        settings: dict[str, Any] | None = None,
        source_template: str | None = None,
        expose_as_model: bool | None = None,
        api_name: str | None = None,
        reserved_model_ids: set[str] | None = None,
    ) -> dict | None:
        with self._lock:
            per_model = self._profiles.get(model_id, {})
            if name not in per_model:
                return None
            profile = dict(per_model[name])
            target_name = name
            rename_mode = False
            if new_name is not None and new_name != name:
                validate_profile_name(new_name)
                if new_name in per_model:
                    raise ValueError(
                        f"Profile '{new_name}' already exists for model '{model_id}'"
                    )
                target_name = new_name
                profile["name"] = new_name
                rename_mode = True
            if display_name is not None:
                profile["display_name"] = display_name
            if api_name is not None:
                profile["api_name"] = self._allocate_profile_api_name_locked(
                    per_model,
                    api_name,
                    display_name=profile.get("display_name"),
                    internal_name=target_name,
                    exclude_name=name,
                )
            if description is not None:
                profile["description"] = description
            if settings is not None:
                profile["settings"] = filter_profile_fields(settings)
            if source_template is not None:
                profile["source_template"] = source_template or None
            if expose_as_model is not None:
                profile["expose_as_model"] = bool(expose_as_model)
            profile["updated_at"] = utcnow().isoformat()
            self._validate_exposed_profile_ids_available_locked(
                model_id,
                profile,
                exclude_profile_name=name,
                reserved_model_ids=reserved_model_ids,
            )
            profiles_snapshot = copy.deepcopy(self._profiles)
            settings_snapshot = copy.deepcopy(self._settings)
            old_active = None
            if rename_mode:
                old_active = self._settings.get(model_id)
                if old_active is not None and old_active.active_profile_name == name:
                    old_active.active_profile_name = target_name
                del per_model[name]
            per_model[target_name] = profile
            try:
                self._save_profiles()
                if rename_mode and old_active is not None:
                    self._save()
            except Exception:
                self._profiles = profiles_snapshot
                self._settings = settings_snapshot
                raise
            return dict(profile)

    def delete_profile(self, model_id: str, name: str) -> bool:
        with self._lock:
            per_model = self._profiles.get(model_id, {})
            if name not in per_model:
                return False
            profiles_snapshot = copy.deepcopy(self._profiles)
            settings_snapshot = copy.deepcopy(self._settings)
            del per_model[name]
            if not per_model and model_id in self._profiles:
                del self._profiles[model_id]
            old_active = self._settings.get(model_id)
            if old_active is not None and old_active.active_profile_name == name:
                old_active.active_profile_name = None
            try:
                self._save_profiles()
                if old_active is not None and old_active.active_profile_name is None:
                    self._save()
            except Exception:
                self._profiles = profiles_snapshot
                self._settings = settings_snapshot
                raise
            return True

    def apply_profile(
        self,
        model_id: str,
        name: str,
        settings_sanitizer: Callable[[dict[str, Any]], None] | None = None,
    ) -> ModelSettings | None:
        with self._lock:
            per_model = self._profiles.get(model_id, {})
            if name not in per_model:
                return None
            profile_settings = per_model[name].get("settings", {}) or {}
            settings_snapshot = copy.deepcopy(self._settings)
            current = self._settings.get(model_id)
            if current is None:
                current = ModelSettings()
            merged = current.to_dict()
            for k, v in profile_settings.items():
                merged[k] = v
            merged["active_profile_name"] = name
            if settings_sanitizer is not None:
                settings_sanitizer(merged)
            new_settings = ModelSettings.from_dict(merged)
            self._settings[model_id] = new_settings
            try:
                self._save()
            except Exception:
                self._settings = settings_snapshot
                raise
            return ModelSettings.from_dict(new_settings.to_dict())

    # ==================== Templates ====================

    def _load_templates(self) -> None:
        if not self.templates_file.exists():
            self._templates = {}
            return
        try:
            with open(self.templates_file, encoding="utf-8") as f:
                data = json.load(f)
            version = data.get("version", 1)
            if version != TEMPLATES_VERSION:
                logger.warning(
                    "Templates file version %d differs from current %d",
                    version,
                    TEMPLATES_VERSION,
                )
            self._templates = data.get("templates", {}) or {}
        except Exception as e:
            logger.error("Failed to load templates file: %s", e)
            self._templates = {}

    def _save_templates(self) -> None:
        data = {"version": TEMPLATES_VERSION, "templates": self._templates}
        try:
            temp_file = self.templates_file.with_suffix(".tmp")
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            temp_file.replace(self.templates_file)
        except Exception as e:
            logger.error("Failed to save templates file: %s", e)
            raise

    def list_templates(self) -> list[dict]:
        with self._lock:
            return [dict(t) for t in self._templates.values()]

    def get_template(self, name: str) -> dict | None:
        with self._lock:
            u = self._templates.get(name)
            return dict(u) if u is not None else None

    def save_template(
        self,
        name: str,
        display_name: str,
        description: str | None,
        settings: dict[str, Any],
    ) -> dict:
        validate_profile_name(name)
        filtered = filter_universal_fields(settings or {})
        with self._lock:
            if name in self._templates:
                raise ValueError(f"Template '{name}' already exists")
            now = utcnow().isoformat()
            self._templates[name] = {
                "name": name,
                "display_name": display_name or name,
                "description": description,
                "created_at": now,
                "updated_at": now,
                "settings": filtered,
            }
            self._save_templates()
            return dict(self._templates[name])

    def upsert_template(
        self,
        name: str,
        display_name: str,
        description: str | None,
        settings: dict[str, Any],
    ) -> dict:
        validate_profile_name(name)
        filtered = filter_universal_fields(settings or {})
        with self._lock:
            now = utcnow().isoformat()
            existing = self._templates.get(name)
            created_at = existing["created_at"] if existing else now
            self._templates[name] = {
                "name": name,
                "display_name": display_name or name,
                "description": description,
                "created_at": created_at,
                "updated_at": now,
                "settings": filtered,
            }
            self._save_templates()
            return dict(self._templates[name])

    def update_template(
        self,
        name: str,
        *,
        new_name: str | None = None,
        display_name: str | None = None,
        description: str | None = None,
        settings: dict[str, Any] | None = None,
    ) -> dict | None:
        with self._lock:
            if name not in self._templates:
                return None
            template = dict(self._templates[name])
            target = name
            if new_name is not None and new_name != name:
                validate_profile_name(new_name)
                if new_name in self._templates:
                    raise ValueError(f"Template '{new_name}' already exists")
                target = new_name
                template["name"] = new_name
            if display_name is not None:
                template["display_name"] = display_name
            if description is not None:
                template["description"] = description
            if settings is not None:
                template["settings"] = filter_universal_fields(settings)
            template["updated_at"] = utcnow().isoformat()
            if target != name:
                del self._templates[name]
            self._templates[target] = template
            self._save_templates()
            return dict(template)

    def delete_template(self, name: str) -> bool:
        with self._lock:
            if name not in self._templates:
                return False
            del self._templates[name]
            self._save_templates()
            return True
