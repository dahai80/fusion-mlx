# SPDX-License-Identifier: Apache-2.0
# Streaming progress bridge for engine generate() callbacks.
# mflux/MLX denoise loops run in a thread executor (run_in_executor). The
# public on_step callback is async (Callable[[int, int], Awaitable[None]]),
# so a sync in-thread hook schedules it back onto the running event loop via
# asyncio.run_coroutine_threadsafe. Fire-and-forget: the denoise thread never
# blocks on the WebSocket send. Errors in the callback are logged, never
# raised into the denoise loop (generation must not abort on a UI hiccup).

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)

StepCallback = Callable[[int, int], Awaitable[None]]


def make_sync_step_callback(
    on_step: StepCallback | None,
    loop: asyncio.AbstractEventLoop | None,
) -> Callable[[int, int], None] | None:
    if on_step is None or loop is None:
        return None

    def _sync(step: int, total: int) -> None:
        try:
            fut = asyncio.run_coroutine_threadsafe(on_step(step, total), loop)

            def _log_err(f: asyncio.Future) -> None:
                if f.cancelled():
                    return
                exc = f.exception()
                if exc is not None:
                    logger.warning("on_step callback raised: %r", exc)

            fut.add_done_callback(_log_err)
        except RuntimeError:
            logger.debug("on_step schedule skipped (event loop closed)")
        except Exception:
            logger.debug("on_step scheduling failed", exc_info=True)

    return _sync
