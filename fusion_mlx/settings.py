"""Settings management for fusion-mlx."""

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class SubKeyEntry:
    """Represents a sub-API key entry."""

    name: str
    key_hash: str
    created_at: str
    expires_at: Optional[str] = None
    usage_count: int = 0
    is_active: bool = True


@dataclass
class Settings:
    """Persistent server settings."""

    api_key: Optional[str] = None
    sub_keys: List[SubKeyEntry] = field(default_factory=list)
    model_settings: Dict[str, Any] = field(default_factory=dict)
    global_settings: Dict[str, Any] = field(default_factory=dict)

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
            sub_keys = [
                SubKeyEntry(**sk) for sk in data.get("sub_keys", [])
            ]
            return cls(
                api_key=data.get("api_key"),
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
