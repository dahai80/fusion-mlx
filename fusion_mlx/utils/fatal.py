# SPDX-License-Identifier: Apache-2.0
"""Fatal process-exit helpers."""

from __future__ import annotations

import faulthandler
import logging
import os
from collections.abc import Callable
from typing import NoReturn

logger = logging.getLogger(__name__)

FATAL_EXIT_CODE = 70
FATAL_TEARDOWN_TIMEOUT_S = 60.0

# #1205: best-effort pre-exit hooks. Callers register callbacks that must
# run before os._exit (e.g. mark an in-flight fine-tune job interrupted so
# its adapter checkpoint is discoverable on restart). Hooks are bounded by
# _FATAL_HOOK_TIMEOUT_S total and exceptions never block the exit.
_fatal_hooks: list[Callable[[], None]] = []
_FATAL_HOOK_TIMEOUT_S = 10.0


def register_fatal_hook(hook: Callable[[], None]) -> None:
    _fatal_hooks.append(hook)


def _run_fatal_hooks() -> None:
    import threading

    if not _fatal_hooks:
        return
    done = threading.Event()

    def _run():
        for hook in _fatal_hooks:
            try:
                hook()
            except Exception:
                logger.exception("Fatal-exit hook failed (continuing)")

    t = threading.Thread(target=lambda: (_run(), done.set()), daemon=True)
    t.start()
    if not done.wait(timeout=_FATAL_HOOK_TIMEOUT_S):
        logger.warning(
            "Fatal-exit hooks exceeded %.0fs; proceeding to exit",
            _FATAL_HOOK_TIMEOUT_S,
        )


def fatal_exit(reason: str, exit_code: int = FATAL_EXIT_CODE) -> NoReturn:
    """Terminate immediately so an external supervisor can restart cleanly."""
    logger.critical(
        "%s; exiting process so the supervisor can restart with a clean state",
        reason,
    )
    _run_fatal_hooks()
    try:
        faulthandler.dump_traceback(all_threads=True)
    except Exception:
        logger.exception("Failed to dump thread tracebacks before fatal exit")
    os._exit(exit_code)
