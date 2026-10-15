# SPDX-License-Identifier: Apache-2.0
"""#1186: run fine-tune train() in a separate OS process.

The serving process hosts resident inference weights on the same Metal GPU;
an in-process train() saturates the shared command stream and allocator for
hours (laya P50 5.7-10ms -> 100-900ms, #1184). This module spawns the
training job in a `spawn`-context child process with its own Metal memory
limit (`mx.metal.set_memory_limit` is per-process), streams progress events
over a duplex Pipe, and surfaces three outcomes:

- success: returns the child's result dict
- `TrainingProcessFailed`: the child reported an error (bad config, empty
  dataset, load failure) — same semantics as the in-process path
- `TrainingProcessKilled`: the child died without reporting (kill -9, OOM
  killer, crash) — the caller applies the configured kill policy

Importers/callers: fusion_mlx.training.service._run_job (process mode).
"""

from __future__ import annotations

import logging
import multiprocessing
import os

logger = logging.getLogger(__name__)

# Default child budget when memory_limit_gb==0: fraction of physical RAM.
# Calibration (#1186/#1184): observed training peak 64.9GB on a 128GB
# machine while serving weights were resident — 60% (76.8GB) leaves the
# resident serving set + OS headroom intact. Overridable per job.
DEFAULT_MEMORY_LIMIT_FRACTION = 0.60


class TrainingProcessFailed(RuntimeError):
    """Child process reported an error before exiting."""


class TrainingProcessKilled(RuntimeError):
    """Child process died without reporting (signal/OOM/crash)."""


def physical_memory_gb() -> float:
    try:
        import psutil

        return psutil.virtual_memory().total / (2**30)
    except Exception:
        page = os.sysconf("SC_PAGE_SIZE")
        count = os.sysconf("SC_PHYS_PAGES")
        return (page * count) / (2**30)


def resolve_memory_limit_gb(memory_limit_gb: float) -> float:
    # 0 (or negative) = auto: default fraction of physical memory.
    if memory_limit_gb and memory_limit_gb > 0:
        return float(memory_limit_gb)
    auto = physical_memory_gb() * DEFAULT_MEMORY_LIMIT_FRACTION
    logger.info(
        "training memory_limit_gb not set; auto = %.1f GB "
        "(%.0f%% of %.1f GB physical, calibration in #1186)",
        auto,
        DEFAULT_MEMORY_LIMIT_FRACTION * 100,
        physical_memory_gb(),
    )
    return auto


def _child_main(conn, spec: dict, limit_gb: float):
    # Child entry (spawn context). Must stay module-level for pickling.
    import signal
    import sys

    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        if limit_gb > 0:
            import mlx.core as mx

            # mx.metal.set_memory_limit is deprecated (removed in newer MLX);
            # prefer the top-level mx.set_memory_limit when available.
            set_limit = getattr(mx, "set_memory_limit", None) or (
                mx.metal.set_memory_limit
            )
            set_limit(int(limit_gb * (2**30)))
            logger.info("training child: memory limit set to %.1f GB", limit_gb)
        entrypoint = spec.get("entrypoint", "")
        if not entrypoint:
            raise ValueError("spec.entrypoint is required for process mode")
        import importlib

        mod_name, _, fn_name = entrypoint.partition(":")
        execute = getattr(importlib.import_module(mod_name), fn_name)
        result = execute(
            spec,
            report=lambda event: conn.send({"type": "progress", "event": event}),
        )
        conn.send({"type": "done", "result": result})
    except BaseException as exc:
        logger.exception("training child failed")
        try:
            conn.send({"type": "error", "error": f"{type(exc).__name__}: {exc}"})
        except Exception:
            pass
        sys.exit(1)
    finally:
        try:
            conn.close()
        except Exception:
            pass


def run_training_process(
    spec: dict,
    on_event=None,
    memory_limit_gb: float = 0.0,
    on_started=None,
) -> dict:
    # Blocking (call from a worker thread). on_event(event: dict) receives
    # progress events as they stream; on_started(proc) hands the child
    # handle to the caller as soon as it exists (for cancel/kill).
    ctx = multiprocessing.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=True)
    limit_gb = resolve_memory_limit_gb(memory_limit_gb)
    proc = ctx.Process(
        target=_child_main, args=(child_conn, spec, limit_gb), daemon=True
    )
    proc.start()
    child_conn.close()
    logger.info(
        "training child started pid=%d limit_gb=%.1f entrypoint=%s",
        proc.pid,
        limit_gb,
        spec.get("entrypoint", "?"),
    )
    if on_started is not None:
        try:
            on_started(proc)
        except Exception:
            logger.exception("on_started callback failed")

    outcome = {"result": None, "error": None}
    conn = parent_conn

    def _handle(msg: dict) -> bool:
        mtype = msg.get("type")
        if mtype == "progress":
            if on_event is not None:
                try:
                    on_event(msg.get("event", {}))
                except Exception:
                    logger.exception("on_event callback failed")
            return False
        if mtype == "done":
            outcome["result"] = msg.get("result") or {}
            return True
        if mtype == "error":
            outcome["error"] = msg.get("error") or "unknown training error"
            return True
        return False

    def _drain():
        try:
            while True:
                if conn.poll(0.5):
                    try:
                        msg = conn.recv()
                    except EOFError:
                        break
                    if _handle(msg):
                        break
                elif not proc.is_alive():
                    # Child exited; drain any final message already in the
                    # pipe buffer before declaring a kill.
                    while conn.poll(0.2):
                        try:
                            msg = conn.recv()
                        except EOFError:
                            break
                        if _handle(msg):
                            break
                    break
        except Exception:
            logger.exception("training pipe drain failed")
        finally:
            try:
                conn.close()
            except Exception:
                pass

    # Caller already runs on a worker thread (asyncio.to_thread); a plain
    # blocking drain keeps ordering simple and avoids races.
    _drain()
    proc.join(5)
    if proc.is_alive():
        logger.warning("training child %d still alive after final message", proc.pid)
    if outcome["error"] is not None:
        raise TrainingProcessFailed(outcome["error"])
    if outcome["result"] is not None:
        return outcome["result"]
    raise TrainingProcessKilled(
        f"training process died without reporting (pid={proc.pid}, "
        f"exitcode={proc.exitcode})"
    )


def kill_training_process(proc, grace_seconds: float = 3.0) -> bool:
    # SIGTERM then SIGKILL escalation. Returns True if the process is gone.
    if proc is None:
        return False
    try:
        if not proc.is_alive():
            return True
        logger.warning("terminating training child pid=%d", proc.pid)
        proc.terminate()
        proc.join(grace_seconds)
        if proc.is_alive():
            logger.warning("training child pid=%d ignored SIGTERM, SIGKILL", proc.pid)
            proc.kill()
            proc.join(2)
        return not proc.is_alive()
    except Exception:
        logger.exception("failed to kill training child")
        return False
