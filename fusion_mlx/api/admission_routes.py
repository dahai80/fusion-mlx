# SPDX-License-Identifier: Apache-2.0
"""#1187: heavy-workload admission control routes.

- POST /v1/admission/pause  -> hold training jobs while a serving window runs
- POST /v1/admission/resume -> end the window; deferred jobs recheck
- GET  /v1/admission/status -> gate snapshot + counters

Auth is middleware-level (no Depends), matching videos_routes. Byte fields
are rendered in GB (rounded to 2 decimals) for operator readability.
"""

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..admission import get_heavy_admission_gate
from ..server_metrics import get_server_metrics

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/admission", tags=["admission"])


class AdmissionPauseRequest(BaseModel):
    scope: str = "training"


def _snapshot_payload() -> dict:
    snap = get_heavy_admission_gate().snapshot()
    for key in (
        "physical_bytes",
        "safety_line_bytes",
        "usage_bytes",
        "training_budget_default_bytes",
    ):
        snap[key.removesuffix("_bytes") + "_gb"] = round(snap.get(key, 0) / 2**30, 2)
        snap.pop(key, None)
    sm = get_server_metrics()
    snap["admitted_total"] = sm.admission_admitted_total
    snap["rejected_total"] = sm.admission_rejected_total
    snap["deferred_total"] = sm.admission_deferred_total
    return snap


@router.post("/pause")
async def pause_admission(body: AdmissionPauseRequest):
    if body.scope != "training":
        raise HTTPException(
            status_code=400,
            detail=(
                f"unsupported pause scope: {body.scope!r}; "
                "only 'training' is supported"
            ),
        )
    logger.info("admission: training paused by /v1/admission/pause")
    get_heavy_admission_gate().pause(body.scope)
    return _snapshot_payload()


@router.post("/resume")
async def resume_admission():
    logger.info("admission: serving window ended by /v1/admission/resume")
    get_heavy_admission_gate().resume()
    return _snapshot_payload()


@router.get("/status")
async def admission_status():
    return _snapshot_payload()
