# SPDX-License-Identifier: Apache-2.0
"""Ablations isolation zone (L16 / PR-D11.2).

Experimental / half-validated code lives here, env-gated, so the main
trunk stays clean of experiment corpses. An ablation module MUST:

1. Be importable only via ``load_ablation(name)`` — never imported
   directly by trunk code.
2. Gate its effects behind an env var (``FUSION_ABLATION_<NAME>=1``)
   defaulting OFF. With the env var unset, ``load_ablation`` returns
   ``None`` and the trunk behaves exactly as before.
3. Log its activation loudly (fail-visible) so an operator reading
   logs knows non-trunk code is running.

Move candidate code here when:
- A feature is falsified (e.g. speculative denoise: 0% acceptance) but
  kept for reproducibility / future re-evaluation.
- A spike is merged but not yet promoted to trunk (gated behind env
  until the soak / bench passes).
- An experiment is in progress and must not leak into production paths.

Do NOT move stable, promoted, or production code here — this is a
quarantine, not a dumping ground. Promote out (delete the env gate +
move to the real module) when an ablation graduates.
"""

from __future__ import annotations

import importlib
import logging
import os

logger = logging.getLogger(__name__)


def load_ablation(name: str) -> object | None:
    # Import ``fusion_mlx.ablations.<name>`` only if
    # ``FUSION_ABLATION_<NAME_UPPER>`` is set to a truthy value.
    # Returns the module or ``None`` (when gated off / not found).
    # Never raises on a gated-off ablation — trunk callers get ``None``
    # and proceed with default behavior.
    env_var = f"FUSION_ABLATION_{name.upper()}"
    enabled = os.environ.get(env_var, "").strip().lower() in ("1", "true", "on")
    if not enabled:
        logger.debug("ablation %s: gated off (set %s=1 to enable)", name, env_var)
        return None
    try:
        mod = importlib.import_module(f"fusion_mlx.ablations.{name}")
    except ModuleNotFoundError:
        logger.warning(
            "ablation %s: env enabled (%s=1) but module not found", name, env_var
        )
        return None
    logger.warning(
        "ABLATION ACTIVE: %s — non-trunk experimental code running (env %s=1)",
        name,
        env_var,
    )
    return mod


__all__ = ["load_ablation"]
