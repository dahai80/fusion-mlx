# SPDX-License-Identifier: Apache-2.0
"""Request-level concurrency backpressure semaphore.

§2.4 (#0911 audit): the API layer had no concurrency guard — only the
scheduler capped inflight generation (max_num_seqs*4). Before a request
reached the scheduler it traversed parse / auth / resolve / tokenize with
no bound, so a request flood could exhaust memory/event-loop capacity
before admission control kicked in.

This module owns a process-wide ``asyncio.Semaphore`` shared by every
route surface (openai / anthropic / ollama) so a single capacity budget
applies across all of them. Cache-HIT short-circuits bypass the semaphore
(handled at call sites) since they never touch the engine.

The semaphore is created lazily in ``init_request_semaphore`` (called from
``set_openai_context`` at server startup) so it binds to the running loop
rather than import time. Defaults to ``max(max_num_seqs*4, 32)``; override
via ``FUSION_MAX_CONCURRENT_REQUESTS``.
"""

import asyncio
import logging
import os

logger = logging.getLogger(__name__)

_request_sem: asyncio.Semaphore | None = None
_max_concurrent: int = 0


def _resolve_default_max(max_num_seqs: int) -> int:
    env_val = os.environ.get("FUSION_MAX_CONCURRENT_REQUESTS", "").strip()
    if env_val:
        try:
            n = int(env_val)
            if n > 0:
                return n
        except ValueError:
            logger.warning(
                "FUSION_MAX_CONCURRENT_REQUESTS=%r not a positive int, ignoring",
                env_val,
            )
    base = max(max_num_seqs * 4, 32)
    return base


def init_request_semaphore(max_num_seqs: int = 8) -> None:
    global _request_sem, _max_concurrent
    if _request_sem is not None:
        return
    _max_concurrent = _resolve_default_max(max_num_seqs)
    _request_sem = asyncio.Semaphore(_max_concurrent)
    logger.info(
        "request concurrency semaphore initialized: max=%d (env=%s)",
        _max_concurrent,
        bool(os.environ.get("FUSION_MAX_CONCURRENT_REQUESTS")),
    )


def get_request_semaphore() -> asyncio.Semaphore | None:
    return _request_sem


async def acquire_request_slot() -> None:
    if _request_sem is not None:
        await _request_sem.acquire()


def release_request_slot() -> None:
    if _request_sem is not None:
        try:
            _request_sem.release()
        except ValueError:
            logger.warning("release_request_slot called with no held slot")


async def concurrency_guarded(gen):
    # Wrap an async generator so the semaphore is held across its full
    # iteration. Acquire happens before the first yield (so a queued
    # request waits before receiving any SSE bytes); release in finally
    # covers normal completion, client disconnect (GeneratorExit), and
    # exceptions. Used by streaming routes where the StreamingResponse is
    # returned before generation begins.
    await acquire_request_slot()
    try:
        async for chunk in gen:
            yield chunk
    finally:
        release_request_slot()
