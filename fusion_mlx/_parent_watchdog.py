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

# #1456 guard mode: defer SIGTERM shutdown while work is in flight. The
# 2026-10-30 22:53 supervisor stop-all wave SIGTERMed a busy server and the
# in-flight jobs were lost. Probes are registered by the server lifespan
# (EnginePool.any_active_requests, video scheduler); a probe returning True
# (or raising — conservative) means "jobs in flight, don't die yet".
ENV_GUARD_MODE = "FUSION_MLX_GUARD_MODE"
ENV_GUARD_MAX_DEFER_SEC = "FUSION_MLX_GUARD_MAX_DEFER_SEC"
_GUARD_DEFAULT_MAX_DEFER_SEC = 900.0
_GUARD_POLL_INTERVAL = 2.0
_busy_probes: dict[str, object] = {}
_busy_probe_lock = threading.Lock()
_GUARD_DEFERRING = threading.Event()
_GUARD_FORCE = False


def _guard_mode_enabled() -> bool:
    return os.environ.get(ENV_GUARD_MODE, "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _guard_max_defer_sec() -> float:
    raw = os.environ.get(ENV_GUARD_MAX_DEFER_SEC, "").strip()
    if not raw:
        return _GUARD_DEFAULT_MAX_DEFER_SEC
    try:
        return max(0.0, float(raw))
    except ValueError:
        logger.warning(
            "invalid %s=%r, using default %.0fs",
            ENV_GUARD_MAX_DEFER_SEC,
            raw,
            _GUARD_DEFAULT_MAX_DEFER_SEC,
        )
        return _GUARD_DEFAULT_MAX_DEFER_SEC


def register_busy_probe(name: str, probe) -> None:
    if not callable(probe):
        raise TypeError("busy probe must be callable")
    with _busy_probe_lock:
        _busy_probes[name] = probe
    logger.debug("guard busy probe registered: %s", name)


def unregister_busy_probe(name: str) -> None:
    with _busy_probe_lock:
        _busy_probes.pop(name, None)
    logger.debug("guard busy probe unregistered: %s", name)


def _jobs_in_flight() -> bool:
    with _busy_probe_lock:
        probes = list(_busy_probes.values())
    for probe in probes:
        try:
            if probe() is True:
                return True
        except Exception:
            # Conservative: an erroring probe counts as busy so a broken
            # probe never green-lights killing live work.
            return True
    return False


def _run_guard_deferral(sig_name: str, max_defer: float) -> None:
    global _GUARD_FORCE
    deadline = time.monotonic() + max_defer
    while True:
        if not _jobs_in_flight():
            logger.warning(
                "guard: in-flight work drained, proceeding with %s shutdown",
                sig_name,
            )
            break
        if time.monotonic() >= deadline:
            logger.error(
                "guard: max defer window (%.0fs) expired with work still in "
                "flight, forcing shutdown",
                max_defer,
            )
            break
        time.sleep(_GUARD_POLL_INTERVAL)
    _GUARD_DEFERRING.clear()
    _GUARD_FORCE = True
    # Re-deliver the signal so the normal shutdown path (including the
    # chained uvicorn handler) runs in the main thread.
    os.kill(os.getpid(), signal.SIGTERM)


def _start_guard_deferral(sig_name: str) -> None:
    max_defer = _guard_max_defer_sec()
    _GUARD_DEFERRING.set()
    write_status("guard_deferring")
    logger.warning(
        "guard mode: %s received with in-flight work, deferring shutdown up "
        "to %.0fs (poll %.1fs). Send %s again to force shutdown now. Note: "
        "an external stopper (start.sh stop) still SIGKILLs after its own "
        "grace window.",
        sig_name,
        max_defer,
        _GUARD_POLL_INTERVAL,
        sig_name,
    )
    threading.Thread(
        target=_run_guard_deferral,
        args=(sig_name, max_defer),
        name="fusion-mlx-guard-defer",
        daemon=True,
    ).start()


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
    # #951-downstream: macOS jetsam kills the supervisor shell (parent) first
    # under memory pressure — the serve child then detects the PPID change and
    # self-SIGTERMs here, BEFORE the video step-boundary check_step_pressure
    # can fire, so a 49-frame 1344x768 run dies at step ~26 with no 507. If a
    # video generation is in-flight, arm the video abort event and give the
    # video thread a short grace window to raise VideoMemoryPressureError
    # (clean 507, server survives the request) before we self-terminate. The
    # self-SIGTERM is deferred, not skipped — once the generation ends (or the
    # grace expires) we proceed with the normal shutdown path below.
    try:
        from fusion_mlx.scheduler.video_unified_scheduler import (
            is_video_generating,
            signal_video_abort,
            wait_video_generation_done,
        )

        if is_video_generating():
            signal_video_abort(
                f"parent process died (ppid {expected_ppid} -> "
                f"{observed_ppid}), likely jetsam; aborting in-flight video "
                f"to 507 before self-terminate"
            )
            logger.warning(
                "[rapid-mlx] parent watchdog: video generation in-flight on "
                "orphan — armed abort, waiting up to 12s for 507 before exit"
            )
            wait_video_generation_done(timeout=12.0)
    except Exception as exc:
        logger.debug("parent watchdog video-abort hook failed: %s", exc)
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
        # #1256: attribute the sender. A graceful exit with no attributable
        # sender is indistinguishable from an operator kill, which made the
        # "server silently exited" class of incident uninvestigable. si_pid /
        # si_uid are only populated on siginfo-delivering handlers
        # (signal.signal does NOT provide them on macOS — log what we can:
        # the frame's caller chain and whether the handler was chained).
        sender = "unknown"
        try:
            # /proc-style attribution is not available on macOS. ppid is NOT
            # the sender: a reparented process (original parent gone) reports
            # ppid=1 no matter who signalled it — the 2026-10-30 stop-all
            # incident read as "SIGTERM from ppid=1" while the real sender
            # was fusion-supervisor's start.sh stop. Report it as context
            # only (#1456).
            ppid = os.getppid()
            sender = f"unknown (ppid={ppid} is context, not the sender)"
        except Exception:
            pass
        in_flight = ""
        try:
            from .scheduler.video_unified_scheduler import is_video_generating

            if is_video_generating():
                in_flight = " (video generation IN FLIGHT — job output lost)"
        except Exception:
            pass
        # #1456 guard mode: a SIGTERM with work in flight defers shutdown
        # until drained (bounded), instead of killing live jobs. A second
        # SIGTERM during deferral is the operator override — fall through
        # and shut down now. SIGINT (Ctrl-C at a terminal) never defers.
        if (
            signum == signal.SIGTERM
            and not _GUARD_FORCE
            and not _SHUTDOWN_EVENT.is_set()
            and _guard_mode_enabled()
            and _jobs_in_flight()
        ):
            if _GUARD_DEFERRING.is_set():
                logger.warning(
                    "second SIGTERM during guard deferral — operator "
                    "override, shutting down now"
                )
            else:
                logger.warning(
                    "received %s from %s%s (#1256 attribution); " "chained_previous=%s",
                    sig_name,
                    sender,
                    in_flight,
                    _prev_handlers.get(signum) is not None,
                )
                _start_guard_deferral(sig_name)
                return
        logger.warning(
            "received %s from %s%s, initiating graceful shutdown (#1256 "
            "attribution); chained_previous=%s",
            sig_name,
            sender,
            in_flight,
            _prev_handlers.get(signum) is not None,
        )
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
