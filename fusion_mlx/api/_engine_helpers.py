# SPDX-License-Identifier: Apache-2.0
"""Shared engine resolve/release helpers for API route modules.

R-P1-1 (#0908 audit): previously duplicated identically in
openai_routes.py and anthropic_routes.py.
"""

import logging

log = logging.getLogger(__name__)


async def resolve_engine(model_name: str, pool, adapter_path=None):
    if pool is not None:
        return await pool.get_engine(model_name, _lease=True, adapter_path=adapter_path)
    from ..service.helpers import get_engine

    log.debug("pool None, falling back to cfg.engine for %s", model_name)
    return get_engine(model_name)


async def release_engine(model_name: str, pool, adapter_path=None):
    if pool is not None:
        await pool.release_engine(model_name, adapter_path=adapter_path)
