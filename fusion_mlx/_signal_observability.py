# SPDX-License-Identifier: Apache-2.0
"""Process-death observability for fusion-mlx servers.

Installs a signal handler chain + faulthandler so the operator can tell
the difference between SIGKILL (no handler), SIGTERM/SIGHUP (logged +
stack dump before shutdown), and C-level segfault (faulthandler traceback).
"""

from __future__ import annotations

import faulthandler
import logging
import signal
import sys
import threading
from collections.abc import Callable

logger = logging.getLogger(__name__)

_OBSERVED_SIGNALS: tuple[int, ...] = tuple(
    sig
    for sig in (
        getattr(signal, "SIGTERM", None),
        getattr(signal, "SIGHUP", None),
    )
    if sig is not None
)

_install_lock = threading.Lock()
_prior_handlers: dict[int, signal.Handlers | Callable[..., object] | int | None] = {}


def _signal_name(signum: int) -> str:
    try:
        return signal.Signals(signum).name
    except (ValueError, AttributeError):
        return f"signal {signum}"


def _on_signal(signum: int, frame) -> None:
    name = _signal_name(signum)
    try:
        logger.warning(
            "fusion-mlx received signal %s; thread stacks follow (faulthandler)",
            name,
        )
    except Exception:
        pass

    try:
        faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
    except Exception:
        pass

    prior = _prior_handlers.get(signum)
    is_sighup = getattr(signal, "SIGHUP", None) is not None and signum == signal.SIGHUP
    if callable(prior):
        try:
            prior(signum, frame)
        except Exception:
            logger.debug(
                "prior signal handler for %s raised during chain", name, exc_info=True
            )
            raise
    elif prior == signal.SIG_DFL and is_sighup:
        return
    elif prior == signal.SIG_DFL:
        terminate_failed = False
        try:
            signal.signal(signum, signal.SIG_DFL)
            signal.raise_signal(signum)
        except Exception:
            terminate_failed = True
            logger.error(
                "could not chain SIGTERM-class signal %s to SIG_DFL"
                " for termination; forcing os._exit(128+%d)",
                name,
                signum,
                exc_info=True,
            )
        if terminate_failed:
            import os

            os._exit(128 + signum)


def install_signal_observability(
    *,
    observed_signals: tuple[int, ...] | None = None,
) -> bool:
    with _install_lock:
        if threading.current_thread() is not threading.main_thread():
            logger.debug(
                "signal observability skipped: not on main thread"
                " (current=%s); faulthandler/signal install requires"
                " the main thread on POSIX",
                threading.current_thread().name,
            )
            return False

        try:
            faulthandler.enable(file=sys.stderr, all_threads=True)
        except (ValueError, RuntimeError) as exc:
            logger.debug("faulthandler.enable failed: %r", exc)

        signals_to_install = (
            observed_signals if observed_signals is not None else _OBSERVED_SIGNALS
        )

        installed_any = False
        for sig in signals_to_install:
            if sig in _prior_handlers:
                installed_any = True
                continue
            try:
                prior = signal.signal(sig, _on_signal)
            except (OSError, ValueError) as exc:
                logger.debug(
                    "could not install fusion-mlx handler for %s: %r",
                    _signal_name(sig),
                    exc,
                )
                continue
            _prior_handlers[sig] = prior
            installed_any = True
            logger.debug(
                "fusion-mlx signal handler installed for %s (prior=%r)",
                _signal_name(sig),
                prior,
            )

        return installed_any


def _reset_for_tests() -> None:
    with _install_lock:
        for sig, prior in list(_prior_handlers.items()):
            try:
                signal.signal(sig, prior if prior is not None else signal.SIG_DFL)
            except (OSError, ValueError):
                pass
        _prior_handlers.clear()


def _get_installed_handlers() -> dict[int, object]:
    return dict(_prior_handlers)
