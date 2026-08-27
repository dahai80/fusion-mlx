# SPDX-License-Identifier: Apache-2.0
"""Non-stream request disconnect detection with cancel-metric tick."""

from __future__ import annotations

import asyncio
import logging

from fastapi import HTTPException
from starlette.requests import Request

logger = logging.getLogger(__name__)


async def _wait_with_disconnect(
    coro,
    raw_request: Request,
    timeout: float,
    poll_interval: float = 0.5,
):
    import time as _time

    from ..scheduler import BackpressureError

    _t0 = _time.monotonic()

    task = asyncio.ensure_future(coro)

    async def _wait_disconnect():
        poll_count = 0
        while True:
            await asyncio.sleep(poll_interval)
            poll_count += 1
            is_disc = await raw_request.is_disconnected()
            if poll_count % 10 == 0 or is_disc:
                logger.info(
                    f"[disconnect_guard] poll #{poll_count} "
                    f"disconnected={is_disc} elapsed={_time.monotonic() - _t0:.1f}s"
                )
            if is_disc:
                return

    disconnect_task = asyncio.create_task(_wait_disconnect())

    try:
        done, _ = await asyncio.wait(
            [task, disconnect_task],
            timeout=timeout,
            return_when=asyncio.FIRST_COMPLETED,
        )

        if not done:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
            raise HTTPException(
                status_code=504,
                detail=f"Request timed out after {timeout:.1f} seconds",
            )

        if disconnect_task in done:
            logger.info(
                f"[disconnect_guard] CLIENT DISCONNECTED (non-stream) "
                f"elapsed={_time.monotonic() - _t0:.1f}s"
            )
            try:
                from ..server_metrics import record_llm_disconnect_cancel

                record_llm_disconnect_cancel()
            except Exception:
                logger.debug("disconnect cancel metric tick failed", exc_info=True)
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
            return None

        try:
            return task.result()
        except BackpressureError as exc:
            raise HTTPException(
                status_code=503,
                detail={
                    "error": {
                        "message": "Server is temporarily overloaded. Please retry.",
                        "type": "server_busy",
                        "code": "backpressure",
                    }
                },
                headers={"Retry-After": "5"},
            ) from exc

    finally:
        if not disconnect_task.done():
            disconnect_task.cancel()
        if not task.done():
            task.cancel()
