# SPDX-License-Identifier: Apache-2.0
"""Parent-process watchdog with crash auto-restart.

When fusion-mlx is launched by a supervisor (launchd, brew services, or
start.sh), the child process installs a watchdog thread that:

1. Monitors the parent process via heartbeat pipe/socket.
2. Detects parent death (ppid=1 reparenting) and exits cleanly.
3. On SIGTERM/SIGINT: triggers graceful shutdown, releases GPU, waits
   for inflight requests, then exits.
4. On crash (SIGKILL/unhandled): the supervisor restarts the process.

The supervisor side (launchd plist or start.sh --watchdog) is responsible
for auto-restart with exponential backoff.
"""

import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

ENV_VAR = "FUSION_MLX_WATCHDOG_PPID"

_SHUTDOWN_EVENT = threading.Event()
# Set by the server lifespan teardown AFTER prefix-cache save + pool
# shutdown complete — distinguishes "shutdown finished cleanly" from
# _SHUTDOWN_EVENT ("shutdown initiated"). The orphan self-kill path polls
# this so a SIGKILL does not truncate cache serialization (fix-0911 §3).
_SHUTDOWN_COMPLETE = threading.Event()
_MAX_SHUTDOWN_WAIT = 20.0
_STATUS_DIR = Path.home() / ".fusion-mlx" / "runtime"
_STATUS_FILE = _STATUS_DIR / "server.status"
# ENG-04 (#0909 audit): port-suffixed PID file so multi-instance
# deployments don't overwrite each other's PID. The legacy fixed
# "server.pid" caused the second instance to clobber the first; the
# supervisor then SIGTERMs the wrong PID, killing a running instance.
# _active_pid_file tracks which file this instance wrote so
# remove_pid_file() cleans up the correct one.
_PID_FILE = _STATUS_DIR / "server.pid"
_active_pid_file: Path | None = None
_CRASH_COUNTER_FILE = _STATUS_DIR / "crash.counter"
_MAX_CRASH_COUNT = 5
_CRASH_WINDOW = 300  # 5 minutes

# ENG-05: track the watchdog thread so stop_watchdog() can signal it.
_active_watchdog_thread: threading.Thread | None = None
_active_watchdog_stop_event: threading.Event | None = None


def _resolve_pid_file(port: int | None = None) -> Path:
    if port and port > 0:
        return _STATUS_DIR / f"server.{port}.pid"
    return _PID_FILE


def ensure_status_dir() -> None:
    try:
        _STATUS_DIR.mkdir(parents=True, exist_ok=True)
    except OSError:
        logger.debug("status dir creation failed (non-fatal)")


def write_pid_file(port: int | None = None) -> None:
    global _active_pid_file
    ensure_status_dir()
    pid_file = _resolve_pid_file(port)
    try:
        # ENG-04: atomic write via temp + rename. Path.write_text()
        # truncates then writes — a concurrent reader (supervisor
        # checking PID) can see a truncated/empty file. os.rename is
        # atomic on POSIX.
        tmp = pid_file.with_suffix(".pid.tmp")
        tmp.write_text(str(os.getpid()))
        os.rename(tmp, pid_file)
        _active_pid_file = pid_file
    except OSError as exc:
        logger.debug("pid file write failed: %s", exc)


def remove_pid_file() -> None:
    global _active_pid_file
    target = _active_pid_file or _PID_FILE
    try:
        target.unlink(missing_ok=True)
    except OSError:
        pass
    _active_pid_file = None


def write_status(status: str) -> None:
    ensure_status_dir()
    try:
        _STATUS_FILE.write_text(f"{status}\n{time.time():.0f}")
    except OSError as exc:
        logger.debug("status file write failed: %s", exc)


def read_status() -> tuple[str, float]:
    try:
        parts = _STATUS_FILE.read_text().strip().split("\n")
        status = parts[0] if parts else "unknown"
        ts = float(parts[1]) if len(parts) > 1 else 0.0
        return status, ts
    except (OSError, ValueError):
        return "unknown", 0.0


def record_crash() -> int:
    now = time.time()
    timestamps: list[float] = []
    try:
        if _CRASH_COUNTER_FILE.exists():
            for line in _CRASH_COUNTER_FILE.read_text().strip().splitlines():
                ts = float(line.strip())
                if now - ts < _CRASH_WINDOW:
                    timestamps.append(ts)
    except (OSError, ValueError):
        timestamps = []
    timestamps.append(now)
    ensure_status_dir()
    try:
        _CRASH_COUNTER_FILE.write_text("\n".join(f"{t:.0f}" for t in timestamps))
    except OSError:
        pass
    return len(timestamps)


def clear_crash_counter() -> None:
    try:
        _CRASH_COUNTER_FILE.unlink(missing_ok=True)
    except OSError:
        pass


def get_crash_count() -> int:
    now = time.time()
    timestamps: list[float] = []
    try:
        if _CRASH_COUNTER_FILE.exists():
            for line in _CRASH_COUNTER_FILE.read_text().strip().splitlines():
                ts = float(line.strip())
                if now - ts < _CRASH_WINDOW:
                    timestamps.append(ts)
    except (OSError, ValueError):
        pass
    return len(timestamps)


def _default_on_orphan(expected_ppid: int, observed_ppid: int) -> None:
    logger.critical(
        "[rapid-mlx] parent watchdog: expected PPID %d, "
        "observed PPID %d — parent died, self-terminating",
        expected_ppid,
        observed_ppid,
    )
    print(
        f"[rapid-mlx] parent watchdog: expected PPID {expected_ppid}, "
        f"observed PPID {observed_ppid} — self-terminating",
        file=sys.stderr,
    )
    try:
        os.kill(os.getpid(), signal.SIGTERM)
    except OSError:
        pass
    # fix-0911 §3: poll for graceful-shutdown completion (prefix-cache
    # save + pool teardown) instead of a fixed 5s sleep. Cache
    # serialization on large models can exceed 5s; a fixed window risks
    # SIGKILL truncating the write. Wait up to 15s (matches uvicorn
    # timeout_graceful_shutdown); if the server signals completion the
    # main thread will exit the process cleanly — only SIGKILL if it
    # hangs past the deadline.
    if not wait_for_shutdown_complete(timeout=15.0):
        try:
            os.kill(os.getpid(), signal.SIGKILL)
        except OSError:
            pass
    os._exit(1)


def install_parent_watchdog(
    ppid: int | None,
    *,
    interval: float = 2.0,
    on_orphan=None,
) -> threading.Thread | None:
    if ppid is None or ppid <= 1:
        logger.debug("parent watchdog skipped, ppid=%s", ppid)
        return None

    callback = on_orphan or _default_on_orphan

    # Install-time short-circuit: if the supervisor already died between
    # spawn and install, fire the callback synchronously — no thread.
    current_ppid = os.getppid()
    if current_ppid != ppid:
        logger.warning(
            "parent already gone at install time (expected=%d, actual=%d)",
            ppid,
            current_ppid,
        )
        callback(ppid, current_ppid)
        return None

    stop_event = threading.Event()

    def _watch():
        while not stop_event.is_set():
            live_ppid = os.getppid()
            if live_ppid != ppid:
                callback(ppid, live_ppid)
                return
            stop_event.wait(interval)

    t = threading.Thread(target=_watch, name="parent-watchdog", daemon=True)
    t._rapid_mlx_stop_event = stop_event  # type: ignore[attr-defined]
    # ENG-05 (#0909 audit): track the thread + stop_event at module
    # level so stop_watchdog() can signal it without callers needing
    # to hold a reference to the thread.
    global _active_watchdog_thread, _active_watchdog_stop_event
    _active_watchdog_thread = t
    _active_watchdog_stop_event = stop_event
    t.start()
    logger.info("parent watchdog installed, monitoring ppid=%d", ppid)
    return t


def stop_watchdog() -> None:
    """Signal the parent watchdog thread to stop.

    ENG-05 (#0909 audit): the watchdog had no public stop API — the
    stop_event was only on a non-standard thread attribute. During
    graceful shutdown, if stop_event was not set, the watchdog could
    detect PPID change (reparenting during shutdown) and fire SIGKILL,
    interrupting in-flight request draining. This must be called early
    in the shutdown sequence.
    """
    global _active_watchdog_thread, _active_watchdog_stop_event
    if _active_watchdog_stop_event is not None:
        _active_watchdog_stop_event.set()
    _active_watchdog_thread = None
    _active_watchdog_stop_event = None
    logger.debug("parent watchdog stop signaled")


def install_signal_handlers() -> None:
    # Capture uvicorn's existing signal handlers (installed by Server.run()
    # before the ASGI lifespan startup) so we can chain into them. uvicorn's
    # handle_exit sets Server.should_exit, which is what actually drives the
    # ASGI shutdown phase (resuming the lifespan past `yield`). Without
    # chaining, replacing the handler leaves should_exit unset, the lifespan
    # never resumes, and _shutdown() never runs — every stop becomes a
    # SIGKILL from start.sh (#807 P0-4).
    _prev_handlers: dict[int, object] = {
        signal.SIGTERM: signal.getsignal(signal.SIGTERM),
        signal.SIGINT: signal.getsignal(signal.SIGINT),
    }

    def _handler(signum, frame):
        sig_name = signal.Signals(signum).name
        logger.info("received %s, initiating graceful shutdown", sig_name)
        _trigger_shutdown(0)
        prev = _prev_handlers.get(signum)
        if prev is not None and callable(prev):
            try:
                prev(signum, frame)
            except Exception:
                logger.debug("previous %s handler raised", sig_name, exc_info=True)

    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)
    logger.debug("signal handlers installed (SIGTERM, SIGINT), chained to previous")


def _trigger_shutdown(exit_code: int = 0) -> None:
    _SHUTDOWN_EVENT.set()
    write_status("shutting_down")


def signal_shutdown_complete() -> None:
    # Called by server lifespan teardown after cache save + pool shutdown.
    # Lets the orphan self-kill path distinguish "graceful shutdown
    # finished" from "shutdown initiated" and skip the SIGKILL fallback.
    _SHUTDOWN_COMPLETE.set()


def wait_for_shutdown_complete(timeout: float = 15.0) -> bool:
    return _SHUTDOWN_COMPLETE.wait(timeout=timeout)


def wait_for_shutdown(timeout: float = _MAX_SHUTDOWN_WAIT) -> bool:
    return _SHUTDOWN_EVENT.wait(timeout=timeout)


def is_shutting_down() -> bool:
    return _SHUTDOWN_EVENT.is_set()


def resolve_expected_ppid(ppid: int | None) -> int | None:
    if ppid is not None:
        if ppid <= 1:
            return None
        return ppid
    env_val = os.environ.get(ENV_VAR, "").strip()
    if env_val:
        try:
            env_ppid = int(env_val)
            if env_ppid > 1:
                return env_ppid
        except (ValueError, TypeError):
            pass
    return None


def should_auto_restart() -> bool:
    count = get_crash_count()
    if count >= _MAX_CRASH_COUNT:
        logger.error(
            "too many crashes (%d in %ds), NOT auto-restarting",
            count,
            _CRASH_WINDOW,
        )
        return False
    return True


def write_exit_status(status: str) -> None:
    ensure_status_dir()
    try:
        (_STATUS_DIR / "exit.status").write_text(
            f"{status}\n{time.time():.0f}\n{os.getpid()}"
        )
    except OSError:
        pass
