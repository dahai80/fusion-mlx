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
_MAX_SHUTDOWN_WAIT = 20.0
_STATUS_DIR = Path.home() / ".fusion-mlx" / "runtime"
_STATUS_FILE = _STATUS_DIR / "server.status"
_PID_FILE = _STATUS_DIR / "server.pid"
_CRASH_COUNTER_FILE = _STATUS_DIR / "crash.counter"
_MAX_CRASH_COUNT = 5
_CRASH_WINDOW = 300  # 5 minutes


def ensure_status_dir() -> None:
    try:
        _STATUS_DIR.mkdir(parents=True, exist_ok=True)
    except OSError:
        logger.debug("status dir creation failed (non-fatal)")


def write_pid_file() -> None:
    ensure_status_dir()
    try:
        _PID_FILE.write_text(str(os.getpid()))
    except OSError as exc:
        logger.debug("pid file write failed: %s", exc)


def remove_pid_file() -> None:
    try:
        _PID_FILE.unlink(missing_ok=True)
    except OSError:
        pass


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
    time.sleep(5.0)
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
    t.start()
    logger.info("parent watchdog installed, monitoring ppid=%d", ppid)
    return t


def install_signal_handlers() -> None:
    def _handler(signum, frame):
        sig_name = signal.Signals(signum).name
        logger.info("received %s, initiating graceful shutdown", sig_name)
        _trigger_shutdown(0)

    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)
    logger.debug("signal handlers installed (SIGTERM, SIGINT)")


def _trigger_shutdown(exit_code: int = 0) -> None:
    _SHUTDOWN_EVENT.set()
    write_status("shutting_down")


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
