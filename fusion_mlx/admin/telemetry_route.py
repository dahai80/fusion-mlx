# SPDX-License-Identifier: Apache-2.0
"""Admin routes for the telemetry module (#5).

Exposes the internal telemetry subsystem to the admin panel so an
operator can see — without reading the on-disk state files — whether
telemetry is enabled, what the queue is doing, which activation
milestones have fired, and which endpoint is configured.

All routes are admin-gated (``require_admin``). The route never returns
user content or the client_id; it returns only operational counters and
the consent/enabled booleans needed for a dashboard.
"""

from __future__ import annotations

import logging
import os

from fastapi import APIRouter, Depends

from .auth import require_admin

logger = logging.getLogger(__name__)

router = APIRouter()


def _queue_snapshot() -> dict:
    from ..telemetry import emit

    q = emit._queue
    if q is None:
        return {
            "active": False,
            "pending": 0,
            "enqueued_total": 0,
            "dropped_total": 0,
            "flushes_ok": 0,
            "flushes_failed": 0,
        }
    snap = q.snapshot()
    snap["active"] = True
    return snap


def _activation_state() -> dict:
    from ..telemetry.activation_spec import ACTIVATION_KINDS
    from ..telemetry.state import activation_marker_path

    out: dict[str, bool] = {}
    for kind in sorted(ACTIVATION_KINDS):
        try:
            out[kind] = activation_marker_path(kind).exists()
        except (OSError, ValueError):
            out[kind] = False
    return out


@router.get("/api/telemetry/status")
async def telemetry_status(is_admin: bool = Depends(require_admin)):
    from ..telemetry import emit
    from ..telemetry.state import (
        CURRENT_CONSENT_SCHEMA_VERSION,
        consent_source,
        get_consent_state,
        is_enabled,
    )
    from ..telemetry.transport import endpoint

    consent = get_consent_state()
    enabled = is_enabled()
    logger.info(
        "telemetry status: enabled=%s source=%s queue_active=%s",
        enabled,
        consent_source(),
        emit._queue is not None,
    )
    return {
        "enabled": enabled,
        "consent_source": consent_source(),
        "consent": {
            "given": consent.consent if consent else None,
            "prompted_at": consent.prompted_at if consent else None,
            "prompted_version": consent.prompted_version if consent else None,
            "schema_version": consent.schema_version if consent else None,
            "current_schema_version": CURRENT_CONSENT_SCHEMA_VERSION,
        },
        "endpoint": endpoint(),
        "session_id": emit.session_id() if emit._queue is not None else None,
        "queue": _queue_snapshot(),
        "activations": _activation_state(),
        "env": {
            "FUSION_MLX_TELEMETRY": os.environ.get("FUSION_MLX_TELEMETRY", ""),
            "FUSION_MLX_TELEMETRY_ENDPOINT": os.environ.get(
                "FUSION_MLX_TELEMETRY_ENDPOINT", ""
            ),
            "FUSION_MLX_TELEMETRY_REQUEST_SAMPLE": os.environ.get(
                "FUSION_MLX_TELEMETRY_REQUEST_SAMPLE", ""
            ),
        },
    }


@router.get("/api/telemetry/queue")
async def telemetry_queue(is_admin: bool = Depends(require_admin)):
    logger.debug("telemetry queue snapshot requested")
    return _queue_snapshot()


@router.get("/api/telemetry/activations")
async def telemetry_activations(is_admin: bool = Depends(require_admin)):
    return _activation_state()


@router.get("/api/telemetry/alerts")
async def telemetry_alerts(is_admin: bool = Depends(require_admin)):
    # Derive simple alert signals from the queue counters so the dashboard
    # can surface problems without polling each counter separately. The
    # thresholds are conservative — a single failed flush is worth surfacing;
    # drops only matter once they accumulate past a small floor.
    snap = _queue_snapshot()
    alerts: list[dict] = []

    if snap.get("flushes_failed", 0) > 0:
        alerts.append(
            {
                "level": "warning",
                "code": "flush_failures",
                "message": (
                    f"Telemetry upload has failed {snap['flushes_failed']} "
                    f"time(s). Events may back up or be dropped."
                ),
                "count": snap["flushes_failed"],
            }
        )

    dropped = snap.get("dropped_total", 0)
    if dropped > 0:
        alerts.append(
            {
                "level": "warning",
                "code": "events_dropped",
                "message": (
                    f"{dropped} telemetry event(s) dropped — the queue is "
                    f"overflowing or the endpoint is unreachable."
                ),
                "count": dropped,
            }
        )

    pending = snap.get("pending", 0)
    if pending > 1000:
        alerts.append(
            {
                "level": "info",
                "code": "queue_backlog",
                "message": (
                    f"Telemetry queue backlog is {pending} events — flush "
                    f"latency may be elevated."
                ),
                "count": pending,
            }
        )

    if not alerts:
        alerts.append(
            {"level": "ok", "code": "no_alerts", "message": "No telemetry alerts."}
        )

    return {"alerts": alerts}
