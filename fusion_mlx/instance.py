# SPDX-License-Identifier: Apache-2.0
"""Stable instance identity for server-side HA (#754).

A fusion-mlx process exposes a stable ``instance_id`` so a gateway or CLI
doing health-driven failover can distinguish replicas and route around a
draining/down instance. The id is sourced from, in priority order:

1. ``FUSION_INSTANCE_ID`` env var (operator-set, e.g. ``mlx-node-1``).
2. A derived ``<hostname>:<pid>`` fallback (unique per host+process,
   stable for the process lifetime).

The derived fallback is computed once and cached; an operator-provided id
is read live so a config reload is not required to pick one up.
"""

from __future__ import annotations

import logging
import os
import socket

logger = logging.getLogger(__name__)

_ENV_VAR = "FUSION_INSTANCE_ID"
_derived: str | None = None


def _derived_instance_id() -> str:
    global _derived
    if _derived is None:
        host = socket.gethostname() or "unknown-host"
        _derived = f"{host}:{os.getpid()}"
        logger.info(
            "instance_id derived fallback: %s (env %s unset)", _derived, _ENV_VAR
        )
    return _derived


def get_instance_id() -> str:
    explicit = os.environ.get(_ENV_VAR, "").strip()
    if explicit:
        return explicit
    return _derived_instance_id()
