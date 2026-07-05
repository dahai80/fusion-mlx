"""Settings management for fusion-mlx."""

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SubKeyEntry:
    """Represents a sub-API key entry."""

    name: str
    key_hash: str
    created_at: str
    expires_at: str | None = None
    usage_count: int = 0
    is_active: bool = True


class _SettingsAuthView:
    # Live view exposing the released flat Settings auth fields under the
    # nested .auth shape that merged admin routes (auth_routes/subkey/stats/
    # settings/helpers) expect. Backed by the parent Settings so mutations
    # (api_key assign, sub_keys append/pop) propagate and persist on save.
    __slots__ = ("_settings",)

    def __init__(self, settings: "Settings") -> None:
        self._settings = settings

    @property
    def api_key(self) -> str | None:
        return self._settings.api_key

    @api_key.setter
    def api_key(self, value: str | None) -> None:
        self._settings.api_key = value

    @property
    def sub_keys(self) -> list[SubKeyEntry]:
        return self._settings.sub_keys

    @property
    def skip_api_key_verification(self) -> bool:
        return bool(
            self._settings.global_settings.get("skip_api_key_verification", False)
        )


@dataclass
class Settings:
    """Persistent server settings."""

    api_key: str | None = None
    sub_keys: list[SubKeyEntry] = field(default_factory=list)
    model_settings: dict[str, Any] = field(default_factory=dict)
    global_settings: dict[str, Any] = field(default_factory=dict)

    @property
    def auth(self) -> _SettingsAuthView:
        # Compatibility shim: admin routes access global_settings.auth.api_key
        # and .auth.sub_keys; the released Settings stores these flat.
        return _SettingsAuthView(self)

    def _save_sync(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "api_key": self.api_key,
            "sub_keys": [
                {
                    "name": k.name,
                    "key_hash": k.key_hash,
                    "created_at": k.created_at,
                    "expires_at": k.expires_at,
                    "usage_count": k.usage_count,
                    "is_active": k.is_active,
                }
                for k in self.sub_keys
            ],
            "model_settings": self.model_settings,
            "global_settings": self.global_settings,
        }
        path.write_text(json.dumps(data, indent=2))

    def save(self, path: Path) -> None:
        """Serialize settings to JSON (sync, for CLI/init context)."""
        self._save_sync(path)

    async def save_async(self, path: Path) -> None:
        """Non-blocking async save for FastAPI routes."""
        await asyncio.to_thread(self._save_sync, path)

    @classmethod
    def _load_sync(cls, path: Path) -> "Settings":
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text())
            # Support both flat api_key and nested auth.api_key formats
            api_key = data.get("api_key")
            if not api_key:
                auth_data = data.get("auth", {})
                api_key = (
                    auth_data.get("api_key") if isinstance(auth_data, dict) else None
                )
            sub_keys = [SubKeyEntry(**sk) for sk in data.get("sub_keys", [])]
            return cls(
                api_key=api_key,
                sub_keys=sub_keys,
                model_settings=data.get("model_settings", {}),
                global_settings=data.get("global_settings", {}),
            )
        except (json.JSONDecodeError, KeyError):
            return cls()

    @classmethod
    def load(cls, path: Path) -> "Settings":
        """Load settings from JSON file (sync, for CLI/init context)."""
        return cls._load_sync(path)

    @classmethod
    async def load_async(cls, path: Path) -> "Settings":
        """Non-blocking async load for FastAPI routes."""
        return await asyncio.to_thread(cls._load_sync, path)
