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
    # #903: probe the engine abort surface instead of assuming
    # abort_request exists (VLMBatchedEngine only had abort_all_requests —
    # the AttributeError left disconnected requests generating into the
    # void). Prefer per-request abort; fall back to abort-all only when the
    # engine offers nothing narrower.
    try:
        abort = getattr(engine, "abort_request", None)
        if callable(abort):
            coro = abort(request_id)
        else:
            abort_all = getattr(engine, "abort_all_requests", None)
            if not callable(abort_all):
                logger.debug(
                    "disconnect guard: engine %s has no abort surface; "
                    "request %s left running",
                    type(engine).__name__,
                    request_id,
                )
                return
            logger.warning(
                "disconnect guard: %s lacks abort_request; falling back to "
                "abort_all_requests for %s",
                type(engine).__name__,
                request_id,
            )
            coro = abort_all()
        task = asyncio.create_task(coro)
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
