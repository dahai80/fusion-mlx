# SPDX-License-Identifier: Apache-2.0
"""OPS-P4-7 (#0907 audit): Pydantic schema for settings.json.

settings.json was parsed with bare ``json.loads`` + ``.get()``, so a typo
(wrong-case enum, string-for-int) silently fell back to the default with no
signal. This module provides a Pydantic ``SettingsSchema`` that validates the
known typed structure on load and on hot-reload.

Scope: the *safe reloadable subset* (log level, idle timeout, memory guard
tier/ceiling, prefill guard, chunked prefill, route-guard toggles). Fields
that need a restart (host, port, loaded models, max_concurrent_requests,
cache dirs) are NOT validated-for-reload here — they are persisted and take
effect on next boot; the reload path logs them as ``not_applied``.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

logger = logging.getLogger(__name__)

VALID_MEMORY_GUARD_TIERS = {"safe", "balanced", "aggressive", "custom"}
VALID_LOG_LEVELS = {"debug", "info", "warning", "error", "trace"}


class ServerSection(BaseModel):
    model_config = ConfigDict(extra="allow")

    host: str | None = None
    port: int | None = None
    log_level: str | None = None
    sse_keepalive_mode: str | None = None

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, v: str | None) -> str | None:
        if v is None:
            return None
        lvl = v.strip().lower()
        if lvl not in VALID_LOG_LEVELS:
            raise ValueError(f"log_level={v!r} not in {sorted(VALID_LOG_LEVELS)}")
        return lvl

    @field_validator("sse_keepalive_mode")
    @classmethod
    def _validate_sse(cls, v: str | None) -> str | None:
        if v is None:
            return None
        if v not in {"chunk", "comment", "off"}:
            raise ValueError(f"sse_keepalive_mode={v!r} not chunk/comment/off")
        return v

    @field_validator("port")
    @classmethod
    def _validate_port(cls, v: int | None) -> int | None:
        if v is None:
            return None
        if not isinstance(v, int) or not (1 <= v <= 65535):
            raise ValueError(f"port={v!r} must be int in 1..65535")
        return v


class MemorySection(BaseModel):
    model_config = ConfigDict(extra="allow")

    memory_guard_tier: str | None = None
    memory_guard_custom_ceiling_gb: float | None = None
    prefill_memory_guard: bool | None = None

    @field_validator("memory_guard_tier")
    @classmethod
    def _validate_tier(cls, v: str | None) -> str | None:
        if v is None:
            return None
        tier = v.strip().lower()
        if tier not in VALID_MEMORY_GUARD_TIERS:
            raise ValueError(
                f"memory_guard_tier={v!r} not in {sorted(VALID_MEMORY_GUARD_TIERS)}"
            )
        return tier

    @field_validator("memory_guard_custom_ceiling_gb")
    @classmethod
    def _validate_ceiling(cls, v: Any) -> Any:
        if v is None:
            return None
        try:
            f = float(v)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"memory_guard_custom_ceiling_gb={v!r} must be a number"
            ) from exc
        if f < 0:
            raise ValueError("memory_guard_custom_ceiling_gb must be >= 0")
        return f


class SchedulerSection(BaseModel):
    model_config = ConfigDict(extra="allow")

    max_concurrent_requests: int | None = None
    chunked_prefill: bool | None = None
    embedding_batch_size: int | None = None

    @field_validator("max_concurrent_requests")
    @classmethod
    def _validate_mcr(cls, v: int | None) -> int | None:
        if v is None:
            return None
        if not isinstance(v, int) or v < 1:
            raise ValueError("max_concurrent_requests must be int >= 1")
        return v

    @field_validator("embedding_batch_size")
    @classmethod
    def _validate_ebs(cls, v: int | None) -> int | None:
        if v is None:
            return None
        if not isinstance(v, int) or v < 1:
            raise ValueError("embedding_batch_size must be int >= 1")
        return v


class IdleTimeoutSection(BaseModel):
    model_config = ConfigDict(extra="allow")

    idle_timeout_seconds: int | None = None

    @field_validator("idle_timeout_seconds")
    @classmethod
    def _validate_it(cls, v: Any) -> Any:
        if v is None:
            return None
        try:
            i = int(v)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"idle_timeout_seconds={v!r} must be int") from exc
        if i < 0:
            raise ValueError("idle_timeout_seconds must be >= 0")
        return i


class RouteGuardSection(BaseModel):
    # Route-guard enforcement is env-var driven (FUSION_ROUTE_WARN_ONLY /
    # FUSION_ROUTE_ENFORCE / FUSION_ROUTE_TOKEN); the middleware reads env
    # per-request, so mutating os.environ on reload flips it with no restart.
    model_config = ConfigDict(extra="allow")

    warn_only: bool | None = None
    enforce: bool | None = None
    token: str | None = None


class SettingsSchema(BaseModel):
    """Top-level settings.json schema (safe reloadable subset)."""

    model_config = ConfigDict(extra="allow")

    server: ServerSection = Field(default_factory=ServerSection)
    memory: MemorySection = Field(default_factory=MemorySection)
    scheduler: SchedulerSection = Field(default_factory=SchedulerSection)
    idle_timeout: IdleTimeoutSection = Field(default_factory=IdleTimeoutSection)
    route_guard: RouteGuardSection = Field(default_factory=RouteGuardSection)

    @classmethod
    def validate_dict(cls, data: dict) -> SettingsSchema:
        """Validate a raw settings.json dict, returning the parsed schema.

        Raises pydantic.ValidationError on bad types/enums; callers log the
        detail and fail visibly rather than silently falling back.
        """
        return cls.model_validate(data)


RELOAD_SAFE_FIELDS = (
    "server.log_level",
    "memory.memory_guard_tier",
    "memory.memory_guard_custom_ceiling_gb",
    "memory.prefill_memory_guard",
    "scheduler.chunked_prefill",
    "idle_timeout.idle_timeout_seconds",
    "route_guard.warn_only",
    "route_guard.enforce",
    "route_guard.token",
)

RESTART_NEEDED_FIELDS = (
    "server.host",
    "server.port",
    "scheduler.max_concurrent_requests",
    "scheduler.embedding_batch_size",
    "model.model_dirs",
    "cache.ssd_cache_dir",
    "mcp.config_path",
)
