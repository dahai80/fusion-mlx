# SPDX-License-Identifier: Apache-2.0
"""Disconnect detection and force-abort for streaming routes."""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import AsyncIterator

from fastapi import HTTPException
from starlette.requests import Request

logger = logging.getLogger(__name__)

_disconnect_abort_recorder = None
_disconnect_abort_lock = threading.Lock()

# Strong refs for fire-and-forget force-abort tasks so they are not GC'd
# before completion; entries self-remove via the done-callback.
_pending_force_abort_tasks: set[asyncio.Task] = set()


def _resolve_sync_scheduler_for_abort(engine):
    if engine is None:
        return None
    scheduler = getattr(engine, "scheduler", None)
    if scheduler is not None and hasattr(scheduler, "abort_request"):
        return scheduler
    return None


def _resolve_disconnect_abort_recorder():
    global _disconnect_abort_recorder
    if _disconnect_abort_recorder is not None:
        return _disconnect_abort_recorder
    try:
        from ..telemetry import get_disconnect_abort_recorder

        _disconnect_abort_recorder = get_disconnect_abort_recorder()
    except Exception:
        pass
    return _disconnect_abort_recorder


def _unresolved_engine_dedupe_key(engine, request_id) -> str | None:
    if engine is None:
        return None
    model_name = getattr(engine, "model_name", None) or getattr(engine, "name", None)
    if model_name is None:
        return None
    return f"{model_name}:{request_id}"


def _record_disconnect_abort_on_scheduler(engine, request_id) -> None:
    recorder = _resolve_disconnect_abort_recorder()
    if recorder is None:
        return
    try:
        key = _unresolved_engine_dedupe_key(engine, request_id)
        recorder.record_abort(key)
    except Exception:
        logger.debug("_record_disconnect_abort_on_scheduler failed", exc_info=True)


def _force_abort_request(engine, request_id_holder) -> bool:
    if (
        request_id_holder is None
        or not isinstance(request_id_holder, list)
        or not request_id_holder
    ):
        return False
    request_id = request_id_holder[0]
    if request_id is None:
        return False
    scheduler = _resolve_sync_scheduler_for_abort(engine)
    if scheduler is not None:
        try:
            result = scheduler.abort_request(request_id)
            if result:
                _record_disconnect_abort_on_scheduler(engine, request_id)
            logger.info(
                f"[disconnect_guard] force-abort scheduler.abort_request("
                f"{str(request_id)[:12]}) -> {result}"
            )
            return True
        except Exception:
            logger.warning(
                "[disconnect_guard] scheduler.abort_request raised; "
                "falling back to engine.abort_request",
                exc_info=True,
            )
    if engine is not None and hasattr(engine, "abort_request"):
        try:
            result = engine.abort_request(request_id)
            if asyncio.iscoroutine(result):

                async def _await_and_record():
                    try:
                        r = await result
                        if r:
                            _record_disconnect_abort_on_scheduler(engine, request_id)
                        logger.info(
                            f"[disconnect_guard] async force-abort "
                            f"engine.abort_request({str(request_id)[:12]}) -> {r}"
                        )
                    except Exception:
                        logger.warning(
                            "[disconnect_guard] async force-abort failed",
                            exc_info=True,
                        )

                _t = asyncio.ensure_future(_await_and_record())
                _pending_force_abort_tasks.add(_t)
                _t.add_done_callback(_pending_force_abort_tasks.discard)
                logger.warning(
                    f"[disconnect_guard] force-abort fell back to async "
                    f"engine.abort_request({str(request_id)[:12]}); the "
                    f"scheduler abort is NOT guaranteed in flight by the "
                    f"time disconnect handling returns."
                )
                return False
            logger.info(
                f"[disconnect_guard] force-abort engine.abort_request("
                f"{str(request_id)[:12]}) -> {result}"
            )
            if result:
                _record_disconnect_abort_on_scheduler(engine, request_id)
            return True
        except Exception:
            logger.warning(
                "[disconnect_guard] force-abort raised; falling back to "
                "generator-close cascade",
                exc_info=True,
            )
    return False


async def _disconnect_guard(
    generator: AsyncIterator[str],
    raw_request: Request,
    poll_interval: float = 0.5,
    engine=None,
    keepalive_seconds: float | None = None,
    request_id_holder: list | None = None,
) -> AsyncIterator[str]:
    import time as _time

    _t0 = _time.monotonic()

    def _elapsed():
        return f"{_time.monotonic() - _t0:.1f}s"

    if keepalive_seconds is None:
        try:
            from ..config import get_config

            keepalive_seconds = float(get_config().sse_keepalive_seconds)
        except Exception:
            keepalive_seconds = 20.0
    keepalive_enabled = keepalive_seconds and keepalive_seconds > 0

    logger.info(
        f"[disconnect_guard] START poll_interval={poll_interval}s "
        f"keepalive_seconds={keepalive_seconds}"
    )

    async def _wait_disconnect():
        poll_count = 0
        while True:
            await asyncio.sleep(poll_interval)
            poll_count += 1
            is_disc = await raw_request.is_disconnected()
            if poll_count % 10 == 0 or is_disc:
                logger.info(
                    f"[disconnect_guard] poll #{poll_count} "
                    f"disconnected={is_disc} elapsed={_elapsed()}"
                )
            if is_disc:
                return

    chunk_count = 0
    keepalive_count = 0
    disconnect_task: asyncio.Task | None = None
    anext_task: asyncio.Task | None = None
    finished_normally = False
    try:
        aiter = generator.__aiter__()
        disconnect_task = asyncio.create_task(_wait_disconnect())
        while True:
            if anext_task is None or anext_task.done():
                anext_task = asyncio.ensure_future(aiter.__anext__())
            wait_kwargs: dict = {"return_when": asyncio.FIRST_COMPLETED}
            if keepalive_enabled:
                wait_kwargs["timeout"] = keepalive_seconds
            done, _pending = await asyncio.wait(
                [anext_task, disconnect_task],
                **wait_kwargs,
            )
            if disconnect_task in done:
                logger.info(
                    f"[disconnect_guard] CLIENT DISCONNECTED after "
                    f"{chunk_count} chunks ({keepalive_count} keepalives), "
                    f"elapsed={_elapsed()}"
                )
                _force_abort_request(engine, request_id_holder)
                anext_task.cancel()
                try:
                    await anext_task
                except (asyncio.CancelledError, StopAsyncIteration):
                    pass
                break
            if anext_task not in done:
                keepalive_count += 1
                if keepalive_count == 1 or keepalive_count % 5 == 0:
                    logger.info(
                        f"[disconnect_guard] emitting keepalive "
                        f"#{keepalive_count}, elapsed={_elapsed()}"
                    )
                yield ": keepalive\n\n"
                continue
            try:
                chunk = anext_task.result()
            except StopAsyncIteration:
                logger.info(
                    f"[disconnect_guard] generator exhausted normally, "
                    f"{chunk_count} chunks ({keepalive_count} keepalives), "
                    f"elapsed={_elapsed()}"
                )
                finished_normally = True
                break
            except Exception as exc:
                logger.error(
                    f"[disconnect_guard] generator raised {type(exc).__name__}: "
                    f"{exc}, {chunk_count} chunks, elapsed={_elapsed()}",
                    exc_info=True,
                )
                import json as _json

                error_data = _json.dumps(
                    {
                        "error": {
                            "message": "Internal error during streaming",
                            "type": "internal_error",
                        }
                    }
                )
                yield f"data: {error_data}\n\n"
                yield "data: [DONE]\n\n"
                break
            chunk_count += 1
            if chunk_count == 1:
                logger.info(
                    f"[disconnect_guard] first chunk arrived, elapsed={_elapsed()}"
                )
            yield chunk
            if chunk_count == 1 and keepalive_enabled:
                keepalive_count += 1
                logger.info(
                    f"[disconnect_guard] post-first-chunk keepalive, "
                    f"elapsed={_elapsed()}"
                )
                yield ": keepalive\n\n"
    except GeneratorExit:
        logger.info(
            f"[disconnect_guard] GeneratorExit after {chunk_count} chunks, elapsed={_elapsed()}"
        )
        if anext_task is not None and not anext_task.done():
            anext_task.cancel()
        _force_abort_request(engine, request_id_holder)
    finally:
        if disconnect_task and not disconnect_task.done():
            disconnect_task.cancel()
        if anext_task and not anext_task.done():
            anext_task.cancel()
        try:
            await generator.aclose()
        except Exception:
            pass
        if not finished_normally:
            _force_abort_request(engine, request_id_holder)
        if engine is not None:
            release = getattr(engine, "release_admission_reservation", None)
            if release is not None:
                try:
                    release()
                except Exception:
                    logger.warning(
                        "[disconnect_guard] release_admission_reservation raised",
                        exc_info=True,
                    )
        logger.info(
            f"[disconnect_guard] CLEANUP done, {chunk_count} chunks total, elapsed={_elapsed()}"
        )


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
