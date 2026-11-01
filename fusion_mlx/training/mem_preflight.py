# SPDX-License-Identifier: Apache-2.0
"""#1473 item (1): training memory budget preflight.

Importers/callers: training services (_process_queue) — dpo_service,
grpo_service, reward_service, rft_service, service (SFT), vlm_service.

In-process training shares the serving process. A job whose projected
peak (policy weights + reference weights + activation/optimizer working
set) exceeds the memory ceiling drives the whole box into a swap storm
and host watchdog panics — observed 72.8GB RSS + 5 kernel panics in 12h
(#1473). The heavy admission gate charges every job a blind 60%-of-RAM
default budget when the config carries no memory_limit_gb, so a 27B DPO
job and a 4B SFT job look identical to it. This module estimates the
REAL per-job peak from the model on disk and fails the job closed BEFORE
any weights load, naming the job in the log line.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Operator ceiling for training jobs (#1473). Caps resident usage + job
# peak; 0/unset = fall back to the memory_enforcer final ceiling.
ENV_TRAIN_MEM_BUDGET_GB = "FUSION_MLX_TRAIN_MEM_BUDGET_GB"

# Activation + optimizer working-set headroom above resident weights, as
# a fraction of model bytes. Measured (#1469/#1472, 27B-8bit DPO with
# compiled step + gradient checkpointing + resident reference): steady
# peak ~= policy 27GB + ref 27GB + ~12GB streaming activations ~= 0.22x
# weights. 0.35 leaves margin for the eager fallback path (compiled step
# failure) and larger batch shapes.
ACT_HEADROOM_FRACTION = 0.35
# Small models: absolute floor so a tiny model still reserves a
# realistic working set (4B-4bit step peak measured ~6.7GB total on a
# ~2.3GB weight set — headroom alone was ~4.4GB).
ACT_HEADROOM_FLOOR_BYTES = 4 * 1024**3


def estimate_training_peak_bytes(model_path, needs_reference: bool) -> int:
    # Projected steady-state job footprint: policy weights + (optional)
    # reference weights + activation/optimizer headroom. File-size based
    # (MLX keeps quantized weights compressed in memory), same estimator
    # the engine pool uses for pre-load admission.
    from ..pool.model_discovery import estimate_model_size

    weights = int(estimate_model_size(Path(model_path)))
    ref = weights if needs_reference else 0
    headroom = max(ACT_HEADROOM_FLOOR_BYTES, int(weights * ACT_HEADROOM_FRACTION))
    return weights + ref + headroom


def train_mem_budget_ceiling_bytes() -> int:
    # Explicit operator ceiling for training jobs. 0 = unset.
    raw = os.environ.get(ENV_TRAIN_MEM_BUDGET_GB, "").strip()
    if not raw:
        return 0
    try:
        gb = float(raw)
    except ValueError:
        logger.warning(
            "invalid %s=%r; ignoring training memory budget",
            ENV_TRAIN_MEM_BUDGET_GB,
            raw,
        )
        return 0
    if gb <= 0:
        return 0
    return int(gb * 1024**3)


def _enforcer_ceiling_bytes() -> int:
    try:
        from ..server import _server_state

        enforcer = _server_state.get("process_memory_enforcer")
        if enforcer is not None:
            return int(enforcer.get_final_ceiling())
    except Exception:
        pass
    return 0


def resolve_training_budget_bytes(
    job_id: str,
    memory_limit_gb: float,
    model_path,
    needs_reference: bool,
) -> tuple[int, str | None]:
    # Returns (budget_bytes, reject_reason). reject_reason non-empty =
    # fail the job closed before weights load: projected resident usage +
    # job peak over the effective ceiling means the enforcer would
    # fatal-exit mid-job and take the serving plane down (#1473).
    # budget_bytes feeds the heavy admission gate so its safety-line
    # check sees the real job size instead of the blind 60% default.
    from ..admission import get_heavy_admission_gate

    gate = get_heavy_admission_gate()
    if not model_path:
        # Unresolvable model id: the executor already fails such jobs
        # with its own clear error; keep legacy budget here.
        logger.warning(
            "#1473 preflight: no model path for job %s; using default budget",
            job_id,
        )
        return gate.training_budget_bytes(memory_limit_gb), None
    try:
        peak = estimate_training_peak_bytes(model_path, needs_reference)
    except Exception as exc:
        # Estimation failure is not a job error — fall back to the legacy
        # budget, loudly. Fail-closed applies to a KNOWN oversize, not to
        # a broken estimator.
        logger.warning(
            "#1473 preflight: peak estimate failed for job %s (%s); "
            "using default budget",
            job_id,
            exc,
        )
        return gate.training_budget_bytes(memory_limit_gb), None

    ceiling = train_mem_budget_ceiling_bytes()
    ceiling_source = ENV_TRAIN_MEM_BUDGET_GB
    if ceiling <= 0:
        ceiling = _enforcer_ceiling_bytes()
        ceiling_source = "memory_enforcer final ceiling"
    if ceiling > 0:
        usage = gate.current_usage_bytes()
        projected = usage + peak
        if projected > ceiling:
            reason = (
                f"training job '{job_id}' rejected by memory preflight "
                f"(#1473): projected {projected / 2**30:.1f}GB exceeds "
                f"ceiling {ceiling / 2**30:.1f}GB (resident "
                f"{usage / 2**30:.1f}GB + job peak {peak / 2**30:.1f}GB "
                f"= weights"
                f"{'+ reference' if needs_reference else ''}"
                f" + activation headroom; ceiling from {ceiling_source})"
            )
            logger.error("%s", reason)
            return 0, reason
        logger.info(
            "#1473 preflight: job %s pass — resident %.1fGB + peak %.1fGB "
            "<= ceiling %.1fGB (%s)",
            job_id,
            usage / 2**30,
            peak / 2**30,
            ceiling / 2**30,
            ceiling_source,
        )

    if memory_limit_gb and memory_limit_gb > 0:
        # Operator set an explicit per-job budget: honor it for the gate.
        return int(memory_limit_gb * 1024**3), None
    return peak, None
