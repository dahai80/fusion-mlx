# SPDX-License-Identifier: Apache-2.0
"""OPS-P4-6 (#0907 audit): SIGHUP hot-reload + POST /v1/config/reload.

Single-maintainer Beta had no hot-reload: changing log level / idle timeout /
memory tier / route-guard toggle required a full restart, dropping loaded
models. This module re-reads settings.json, validates it against
``config_schema.SettingsSchema`` (OPS-P4-7), and applies the *safe reloadable
subset* at runtime via the existing ``_apply_*_runtime`` helpers — no restart,
no model reload.

Restart-needed fields (host, port, max_concurrent_requests, model_dirs,
cache dir, mcp config) are logged as ``not_applied`` and left on disk for
the next boot.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends
from pydantic import ValidationError

from ..middleware.auth import verify_api_key_or_x_api_key

logger = logging.getLogger(__name__)

router = APIRouter(dependencies=[Depends(verify_api_key_or_x_api_key)])

# A-P0-1 (#0908 audit): serialize concurrent reloads. A rapid double-SIGHUP
# (or SIGHUP during POST /v1/config/reload) ran two reload_config() bodies
# concurrently — both mutating enforcer.prefill_memory_guard,
# scheduler.config.chunked_prefill, os.environ[...] interleaved, with a
# TOCTOU between the schema read and the _apply_*_runtime await that could
# apply a stale tier on top of a newer one. One module-level Lock guards
# the whole body; a reentrant reload awaits the in-flight one and returns
# its result instead of racing.
_reload_lock = asyncio.Lock()
_in_flight: asyncio.Future | None = None


def _settings_json_path() -> Path:
    return Path.home() / ".fusion-mlx" / "settings.json"


def _read_raw_settings() -> dict[str, Any]:
    path = _settings_json_path()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.warning(
            "settings.json malformed on reload — keeping live config",
            exc_info=True,
        )
        return {}
    return data if isinstance(data, dict) else {}


async def reload_config(source: str = "manual") -> dict[str, Any]:
    """Re-read settings.json, validate, apply safe reloadable subset.

    Returns a dict with ``applied``, ``not_applied``, ``errors`` lists so the
    HTTP route and the SIGHUP handler both report the same shape.

    A-P0-1: serialized by _reload_lock — concurrent callers await the
    in-flight reload and receive its result (no double-apply race).
    """
    global _in_flight
    # If a reload is already running, piggyback on its result rather than
    # queueing a second one that would mutate the same singletons.
    if _in_flight is not None and not _in_flight.done():
        logger.info("config reload (%s): awaiting in-flight reload", source)
        try:
            return await asyncio.shield(_in_flight)
        except Exception:
            pass  # fall through and run a fresh reload
    async with _reload_lock:
        loop = asyncio.get_running_loop()
        _in_flight = loop.create_future()
        try:
            result = await _reload_config_impl(source)
            _in_flight.set_result(result)
            return result
        except Exception as e:
            _in_flight.set_exception(e)
            raise
        finally:
            _in_flight = None


async def _reload_config_impl(source: str) -> dict[str, Any]:
    from ..config_schema import SettingsSchema

    raw = _read_raw_settings()
    applied: list[str] = []
    not_applied: list[str] = []
    errors: list[str] = []

    try:
        schema = SettingsSchema.validate_dict(raw)
    except ValidationError as e:
        for err in e.errors():
            loc = ".".join(str(p) for p in err.get("loc", []))
            errors.append(f"{loc}: {err.get('msg', 'invalid')}")
        logger.warning("config reload (%s) rejected by schema: %s", source, errors)
        return {"source": source, "applied": [], "not_applied": [], "errors": errors}

    try:
        from ..server import _server_state

        enforcer = _server_state.get("process_memory_enforcer")
        pool = _server_state.get("engine_pool")
    except Exception:
        enforcer = None
        pool = None

    # 1) log level — live via _apply_log_level_runtime
    log_level = schema.server.log_level
    if log_level is not None:
        try:
            from ..admin.helpers import _apply_log_level_runtime

            _apply_log_level_runtime(log_level)
            applied.append("server.log_level")
            logger.info("config reload (%s): log_level=%s applied", source, log_level)
        except Exception:
            errors.append("server.log_level: apply failed")
            logger.warning("config reload: log_level apply failed", exc_info=True)

    # 2) memory guard tier + custom ceiling — live via enforcer
    tier = schema.memory.memory_guard_tier
    ceiling = schema.memory.memory_guard_custom_ceiling_gb
    if tier is not None or ceiling is not None:
        if enforcer is not None:
            try:
                from ..admin.helpers import _apply_memory_guard_tier_runtime

                ok, msg = await _apply_memory_guard_tier_runtime(
                    tier=tier, custom_ceiling_gb=ceiling
                )
                if ok:
                    applied.append("memory.memory_guard_tier")
                    logger.info("config reload (%s): %s", source, msg)
                else:
                    errors.append(f"memory.memory_guard_tier: {msg}")
            except Exception:
                errors.append("memory.memory_guard_tier: apply failed")
                logger.warning("config reload: tier apply failed", exc_info=True)
        else:
            not_applied.append("memory.memory_guard_tier (enforcer not up)")

    # 3) prefill_memory_guard — live on enforcer
    if schema.memory.prefill_memory_guard is not None and enforcer is not None:
        enforcer.prefill_memory_guard = schema.memory.prefill_memory_guard
        applied.append("memory.prefill_memory_guard")
        logger.info(
            "config reload (%s): prefill_memory_guard=%s",
            source,
            schema.memory.prefill_memory_guard,
        )

    # 4) chunked_prefill — live on loaded schedulers
    if schema.scheduler.chunked_prefill is not None and pool is not None:
        try:
            for mid, entry in pool._entries.items():
                if entry is None or getattr(entry, "engine", None) is None:
                    continue
                async_core = getattr(entry.engine, "_engine", None)
                core = (
                    getattr(async_core, "engine", None)
                    if async_core is not None
                    else None
                )
                scheduler = getattr(core, "scheduler", None) if core else None
                if scheduler is not None and hasattr(scheduler, "config"):
                    scheduler.config.chunked_prefill = schema.scheduler.chunked_prefill
            applied.append("scheduler.chunked_prefill")
            logger.info(
                "config reload (%s): chunked_prefill=%s",
                source,
                schema.scheduler.chunked_prefill,
            )
        except Exception:
            errors.append("scheduler.chunked_prefill: apply failed")
            logger.warning("config reload: chunked_prefill apply failed", exc_info=True)

    # 5) idle_timeout_seconds — live via enforcer hot-reload field
    idle = schema.idle_timeout.idle_timeout_seconds
    if idle is not None:
        if enforcer is not None:
            enforcer.set_reloaded_idle_timeout(idle if idle > 0 else None)
            applied.append("idle_timeout.idle_timeout_seconds")
        else:
            not_applied.append("idle_timeout.idle_timeout_seconds (enforcer not up)")

    # 6) route_guard toggles — env-var driven, mutate os.environ
    rg = schema.route_guard
    if rg.warn_only is not None:
        os.environ["FUSION_ROUTE_WARN_ONLY"] = "true" if rg.warn_only else "false"
        applied.append("route_guard.warn_only")
    if rg.enforce is not None:
        os.environ["FUSION_ROUTE_ENFORCE"] = "true" if rg.enforce else "false"
        applied.append("route_guard.enforce")
    if rg.token is not None:
        os.environ["FUSION_ROUTE_TOKEN"] = rg.token
        applied.append("route_guard.token")

    # Restart-needed fields: report not_applied so the operator knows.
    if schema.server.host is not None:
        not_applied.append("server.host (restart needed)")
    if schema.server.port is not None:
        not_applied.append("server.port (restart needed)")
    if schema.scheduler.max_concurrent_requests is not None:
        not_applied.append("scheduler.max_concurrent_requests (restart needed)")
    if schema.scheduler.embedding_batch_size is not None:
        not_applied.append("scheduler.embedding_batch_size (restart needed)")

    logger.info(
        "config reload (%s) done: applied=%s not_applied=%s errors=%s",
        source,
        applied,
        not_applied,
        errors,
    )
    return {
        "source": source,
        "applied": applied,
        "not_applied": not_applied,
        "errors": errors,
    }


@router.post("/v1/config/reload")
async def config_reload_route():
    """Hot-reload the safe reloadable subset of settings.json without restart.

    Gated by the same dual Bearer/x-api-key auth as other admin routes.
    """
    result = await reload_config(source="http")
    return result
