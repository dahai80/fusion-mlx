# SPDX-License-Identifier: Apache-2.0
"""Shared client-disconnect guard for streaming route generators.

R-P1-3 (#0908 audit): openai_routes had CancelledError handling that
aborts the engine request; anthropic_routes and ollama_routes did not,
so a client disconnect left the engine generating tokens into the void.
"""

import asyncio
import logging

logger = logging.getLogger(__name__)

_pending_abort_tasks: set[asyncio.Task] = set()


def handle_disconnect(request_id: str, engine) -> None:
    try:
        task = asyncio.create_task(engine.abort_request(request_id))
        _pending_abort_tasks.add(task)
        task.add_done_callback(_pending_abort_tasks.discard)
        task.add_done_callback(
            lambda t: (
                logger.warning(
                    "abort_request failed for %s: %s",
                    request_id,
                    t.exception(),
                )
                if not t.cancelled() and t.exception()
                else None
            )
        )
    except Exception:
        logger.debug("disconnect guard: abort_request scheduling failed", exc_info=True)
