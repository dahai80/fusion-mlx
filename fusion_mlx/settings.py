"""Settings management for fusion-mlx."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class IntegrationSettings:
    markitdown_enabled: bool = True
    markitdown_expose_model: bool = False
    markitdown_max_file_size_mb: int = 25
    markitdown_max_files_per_request: int = 5
    markitdown_pdf_processing_engine: str = "markitdown"

    def to_dict(self) -> dict[str, Any]:
        return {
            "markitdown_enabled": self.markitdown_enabled,
            "markitdown_expose_model": self.markitdown_expose_model,
            "markitdown_max_file_size_mb": self.markitdown_max_file_size_mb,
            "markitdown_max_files_per_request": self.markitdown_max_files_per_request,
            "markitdown_pdf_processing_engine": self.markitdown_pdf_processing_engine,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IntegrationSettings:
        return cls(
            markitdown_enabled=data.get("markitdown_enabled", True),
            markitdown_expose_model=data.get("markitdown_expose_model", False),
            markitdown_max_file_size_mb=data.get("markitdown_max_file_size_mb", 25),
            markitdown_max_files_per_request=data.get(
                "markitdown_max_files_per_request", 5
            ),
            markitdown_pdf_processing_engine=data.get(
                "markitdown_pdf_processing_engine", "markitdown"
            ),
        )


@dataclass
class SubKeyEntry:
    """Represents a sub-API key entry."""

    name: str
    key_hash: str
    created_at: str
    expires_at: str | None = None
    usage_count: int = 0
    is_active: bool = True


def _clear_plaintext_api_key(path: Path) -> None:
    """#770: remove the api_key field from settings.json on disk so the
    secret lives only in the Keychain. Rewrites the file at 0o600."""
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return
    changed = False
    if "api_key" in data:
        data["api_key"] = None
        changed = True
    auth_data = data.get("auth")
    if isinstance(auth_data, dict) and "api_key" in auth_data:
        auth_data["api_key"] = None
        data["auth"] = auth_data
        changed = True
    if not changed:
        return
    raw = json.dumps(data, indent=2).encode("utf-8")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, raw)
    finally:
        os.close(fd)
    os.chmod(path, 0o600)


class _SettingsAuthView:
    # Live view exposing the released flat Settings auth fields under the
    # nested .auth shape that merged admin routes (auth_routes/subkey/stats/
    # settings/helpers) expect. Backed by the parent Settings so mutations
    # (api_key assign, sub_keys append/pop) propagate and persist on save.
    __slots__ = ("_settings",)

    def __init__(self, settings: Settings) -> None:
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
    integrations: IntegrationSettings = field(default_factory=IntegrationSettings)

    @property
    def auth(self) -> _SettingsAuthView:
        # Compatibility shim: admin routes access global_settings.auth.api_key
        # and .auth.sub_keys; the released Settings stores these flat.
        return _SettingsAuthView(self)

    def _save_sync(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # #770: when Keychain is enabled, the API key lives in the Keychain,
        # not on disk. Persist None to settings.json so no plaintext lands.
        save_api_key = self.api_key
        try:
            from .admin.keychain import is_available, is_enabled, set_key

            if is_enabled() and is_available():
                if self.api_key:
                    set_key(self.api_key)
                save_api_key = None
        except Exception:
            logger.warning("keychain: save-path check failed, keeping plaintext")
        data = {
            "api_key": save_api_key,
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
            "integrations": self.integrations.to_dict(),
        }
        # 审计0830 P0-6: settings.json 含明文 api_key, 原 write_text 无权限约束,
        # 默认 umask 022 -> 0644 任意本地进程可读。os.open 0o600 原子建新文件
        # (无 write_text+chmod TOCTOU 窗口); 已存在文件 mode 不变, 补 chmod 0o600 收敛。
        raw = json.dumps(data, indent=2).encode("utf-8")
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.write(fd, raw)
        finally:
            os.close(fd)
        os.chmod(path, 0o600)

    def save(self, path: Path) -> None:
        """Serialize settings to JSON (sync, for CLI/init context)."""
        self._save_sync(path)

    async def save_async(self, path: Path) -> None:
        """Non-blocking async save for FastAPI routes."""
        await asyncio.to_thread(self._save_sync, path)

    @classmethod
    def _load_sync(cls, path: Path) -> Settings:
        # #770: Keychain takes precedence over plaintext when FUSION_KEYCHAIN=on.
        # On load, if Keychain is empty but settings.json carries a plaintext
        # api_key, migrate it into the Keychain and clear the field so the
        # secret leaves disk. Fail visibly and fall back to plaintext if the
        # Keychain is unavailable.
        kc_key = None
        kc_enabled = False
        try:
            from .admin.keychain import get_key, is_available, is_enabled, set_key

            kc_enabled = is_enabled() and is_available()
            if kc_enabled:
                kc_key = get_key()
        except Exception:
            logger.warning("keychain: import/check failed, falling back to plaintext")
            kc_enabled = False

        if not path.exists():
            return cls(api_key=kc_key) if kc_enabled else cls()
        # 审计0830 P0-6: 读时收敛既有 0644 文件权限 (历史 write_text 遗留),
        # 防此前已落盘的明文 api_key 在下次 save 前仍可被本地越权读。
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
        try:
            data = json.loads(path.read_text())
            # Support both flat api_key and nested auth.api_key formats
            api_key = data.get("api_key")
            if not api_key:
                auth_data = data.get("auth", {})
                api_key = (
                    auth_data.get("api_key") if isinstance(auth_data, dict) else None
                )

            if kc_enabled:
                if kc_key is not None:
                    api_key = kc_key
                    if data.get("api_key") or (
                        isinstance(data.get("auth"), dict)
                        and data["auth"].get("api_key")
                    ):
                        logger.info(
                            "keychain: plaintext api_key present in settings.json "
                            "but Keychain holds the key; clearing plaintext"
                        )
                        _clear_plaintext_api_key(path)
                elif api_key:
                    if set_key(api_key):
                        logger.info(
                            "keychain: migrated plaintext api_key from settings.json "
                            "into Keychain; clearing plaintext"
                        )
                        _clear_plaintext_api_key(path)
                    else:
                        logger.warning(
                            "keychain: migration failed, keeping plaintext api_key"
                        )

            sub_keys = [SubKeyEntry(**sk) for sk in data.get("sub_keys", [])]
            integrations_data = data.get("integrations", {})
            integrations = (
                IntegrationSettings.from_dict(integrations_data)
                if isinstance(integrations_data, dict)
                else IntegrationSettings()
            )
            return cls(
                api_key=api_key,
                sub_keys=sub_keys,
                model_settings=data.get("model_settings", {}),
                global_settings=data.get("global_settings", {}),
                integrations=integrations,
            )
        except (json.JSONDecodeError, KeyError):
            return cls()

    @classmethod
    def load(cls, path: Path) -> Settings:
        """Load settings from JSON file (sync, for CLI/init context)."""
        return cls._load_sync(path)

    @classmethod
    async def load_async(cls, path: Path) -> Settings:
        """Non-blocking async load for FastAPI routes."""
        return await asyncio.to_thread(cls._load_sync, path)
