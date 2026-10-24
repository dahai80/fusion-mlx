# SPDX-License-Identifier: Apache-2.0
# MediaJobManager (S3, audit 0910 §6.2): spawns image_worker subprocess,
# reads JSONL progress events, enforces timeout via process kill (not
# thread cancel), returns output file paths.
#
# Process boundary = memory boundary: image gen Metal allocations cannot
# crash the LLM in the main process. Timeout = kill subprocess (real
# cancellation, unlike uncancellable threads).
from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# #1332: first-worker-event window. The worker emits nothing on stdout until
# its first denoise progress event, so "no event" covers model load plus the
# first Metal op. A silent worker past this window is a load hang (cold
# cache / missing shard / stuck Metal init), not a generation hang — fail
# fast instead of burning the full hang deadline. Env override:
# FUSION_IMAGE_LOAD_TIMEOUT (seconds); <= 0 disables the window.
_IMAGE_LOAD_TIMEOUT_DEFAULT_S = 120.0


def get_image_load_timeout() -> float:
    raw = os.environ.get("FUSION_IMAGE_LOAD_TIMEOUT")
    if raw is None or raw == "":
        return _IMAGE_LOAD_TIMEOUT_DEFAULT_S
    try:
        val = float(raw)
    except (TypeError, ValueError):
        logger.warning(
            "FUSION_IMAGE_LOAD_TIMEOUT=%r is not a number, using default %.0fs",
            raw,
            _IMAGE_LOAD_TIMEOUT_DEFAULT_S,
        )
        return _IMAGE_LOAD_TIMEOUT_DEFAULT_S
    return val


class MediaJobLoadHangError(RuntimeError):
    # #1332: distinct from the overall TimeoutError so the caller can tell a
    # load hang (no first event) from a generation hang (deadline exceeded).
    pass


@dataclass
class MediaJobResult:
    outputs: list[str] = field(default_factory=list)
    error: str | None = None
    elapsed: float = 0.0
    exit_code: int = 0


class MediaJobManager:
    # Manages subprocess image generation jobs. One job at a time (image
    # executor max_workers=1). Each job spawns a fresh worker process that
    # loads the model, generates, exits — memory fully released per job.

    def __init__(self):
        self._python_exe = sys.executable
        self._worker_module = "fusion_mlx.media.image_worker"

    def _build_spec(
        self,
        variant: str,
        model_path: str,
        quantize,
        config_label,
        output_format: str,
        n_images: int,
        gen_params: dict[str, Any],
    ) -> tuple[str, dict]:
        job_id = f"img_{int(time.monotonic() * 1000)}_{os.getpid()}"
        output_dir = tempfile.mkdtemp(prefix=f"fusion_{job_id}_")
        lease_bytes = self._compute_lease_bytes()
        spec = {
            "variant": variant,
            "model_path": model_path,
            "quantize": quantize,
            "config_label": config_label,
            "output_dir": output_dir,
            "output_format": output_format,
            "n_images": n_images,
            "gen_params": gen_params,
            "lease_bytes": lease_bytes,
        }
        logger.info("image job %s lease_bytes=%d", job_id, lease_bytes)
        return output_dir, spec

    def _compute_lease_bytes(self) -> int:
        """Allocate a memory budget for the subprocess based on current
        system pressure. 60% of available memory, floored at 4 GB,
        capped at 32 GB.
        """
        try:
            import psutil

            avail = psutil.virtual_memory().available
            lease = int(avail * 0.6)
            lease = max(4 * 1024**3, min(lease, 32 * 1024**3))
            return lease
        except Exception:
            return 8 * 1024**3

    async def run_image_job(
        self,
        variant: str,
        model_path: str,
        quantize,
        config_label,
        output_format: str,
        n_images: int,
        gen_params: dict[str, Any],
        timeout: float,
        on_step: Callable[[int, int, int], None] | None = None,
        load_timeout: float | None = None,
    ) -> MediaJobResult:
        # Returns MediaJobResult with output file paths. Raises TimeoutError
        # if the worker exceeds timeout (subprocess killed). Raises
        # MediaJobLoadHangError when no event arrives within load_timeout
        # (#1332). Raises RuntimeError on worker crash/error.
        if load_timeout is None:
            load_timeout = get_image_load_timeout()
        output_dir, spec = self._build_spec(
            variant,
            model_path,
            quantize,
            config_label,
            output_format,
            n_images,
            gen_params,
        )
        spec_path = os.path.join(output_dir, "spec.json")
        with open(spec_path, "w") as f:
            json.dump(spec, f)

        t0 = time.monotonic()
        env = dict(os.environ)
        proc = await asyncio.create_subprocess_exec(
            self._python_exe,
            "-m",
            self._worker_module,
            spec_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        logger.info(
            "MediaJobManager: spawned worker pid=%d variant=%s model=%s",
            proc.pid,
            variant,
            model_path,
        )

        result = MediaJobResult()
        # #1332: track our own kills so an external-signal death (rc < 0) is
        # logged as external instead of being blamed on the timeout/teardown
        # kill paths (issue #1332's undiagnosed SIGTERM at ~485s).
        killed_by_us = False

        def _kill_worker() -> None:
            nonlocal killed_by_us
            killed_by_us = True
            self._kill(proc)

        try:
            await self._read_stdout(proc, timeout, on_step, result, load_timeout, t0)
        except MediaJobLoadHangError as exc:
            _kill_worker()
            result.error = str(exc)
            result.exit_code = -1
            raise
        except TimeoutError:
            _kill_worker()
            result.error = f"worker exceeded {timeout:.0f}s timeout (killed)"
            result.exit_code = -1
            raise TimeoutError(result.error)
        finally:
            if proc.returncode is None:
                # If we already received a terminal event (result or error),
                # the worker is shutting down cleanly — give it a short grace
                # window to exit with its real exit code instead of racing a
                # SIGTERM that rewrites returncode to -15 and masks the cause.
                if result.outputs or result.error:
                    try:
                        await asyncio.wait_for(proc.wait(), timeout=3.0)
                    except TimeoutError:
                        _kill_worker()
                else:
                    _kill_worker()
            await proc.wait()
            result.elapsed = time.monotonic() - t0

        if proc.returncode not in (0, None):
            rc = proc.returncode
            if result.outputs:
                # #1330: the worker delivered a terminal result event, so the
                # outputs are complete on disk. A non-zero exit code here is
                # teardown noise — most commonly our own 3s grace SIGTERM
                # while the interpreter releases Metal buffers. Do not
                # discard a successful job over it.
                logger.warning(
                    "worker exited %d after delivering outputs (teardown kill); treating as success",
                    rc,
                )
                result.exit_code = rc
                return result
            stderr_data = await proc.stderr.read() if proc.stderr else b""
            stderr_text = stderr_data.decode(errors="replace")[-2000:]
            elapsed = result.elapsed
            if rc < 0 and not killed_by_us:
                sig = -rc
                try:
                    sig_name = signal.Signals(sig).name
                except ValueError:
                    sig_name = f"signal {sig}"
                # #1332: an external kill (observed: SIGTERM at ~485s with no
                # enforcer cancel and no timeout kill logged) is undiagnosable
                # from "worker exited -15" alone. Log pid + signal + elapsed
                # so the sender can be traced (jetsam / watchdog / operator /
                # a peer process). No sender pid is available on macOS.
                logger.error(
                    "#1332 image worker pid=%d died from EXTERNAL signal %d "
                    "(%s) after %.1fs — no MediaJobManager kill path fired "
                    "(not timeout, not load-hang, not teardown). Investigate "
                    "the sender. stderr tail: %s",
                    proc.pid,
                    sig,
                    sig_name,
                    elapsed,
                    stderr_text[-500:] or "(empty)",
                )
                result.error = (
                    f"worker exited {rc} ({sig_name}) after {elapsed:.1f}s: "
                    "killed by an external signal, not by fusion-mlx. This is "
                    "not a persistent fault; the next request will spawn a "
                    f"fresh worker: {stderr_text}"
                )
            elif rc < 0:
                try:
                    sig_name = signal.Signals(-rc).name
                except ValueError:
                    sig_name = f"signal {-rc}"
                result.error = (
                    f"worker exited {rc} ({sig_name}) after {elapsed:.1f}s: "
                    f"stopped by MediaJobManager's own kill path: {stderr_text}"
                )
            elif rc == 137:
                result.error = f"worker killed by OOM/jetsam (exit {rc})"
            elif rc == 139:
                result.error = f"worker segfaulted/Metal crash (exit {rc})"
            else:
                result.error = f"worker exited {rc}: {stderr_text}"
            result.exit_code = rc
            raise RuntimeError(result.error)

        if not result.outputs:
            result.error = "worker produced no outputs"
            raise RuntimeError(result.error)

        return result

    @staticmethod
    def _load_hang_message(load_timeout: float) -> str:
        # #1332: actionable text — says WHAT hung (load, not generate) and
        # that the subsystem is not poisoned, so the next request is safe.
        return (
            f"image worker produced no events within {load_timeout:g}s of "
            "spawn: model load produced no events (load hang, not a "
            "generation hang). The worker was killed; the image subsystem "
            "is not poisoned and the next request will spawn a fresh worker. "
            "Raise FUSION_IMAGE_LOAD_TIMEOUT if cold model loads legitimately "
            "need longer."
        )

    async def _read_stdout(
        self,
        proc: asyncio.subprocess.Process,
        timeout: float,
        on_step: Callable[[int, int, int], None] | None,
        result: MediaJobResult,
        load_timeout: float = 0.0,
        spawned_at: float | None = None,
    ) -> None:
        # Read JSONL progress/result events from worker stdout. Raises
        # TimeoutError if no event arrives within timeout. #1332: when
        # load_timeout > 0, a first event that is still missing at that
        # window raises MediaJobLoadHangError instead (fast-fail load hang).
        assert proc.stdout is not None
        if spawned_at is None:
            spawned_at = time.monotonic()
        deadline = spawned_at + timeout
        load_window = load_timeout if load_timeout and load_timeout > 0 else 0.0
        saw_event = False
        while True:
            now = time.monotonic()
            remaining = deadline - now
            if remaining <= 0:
                raise TimeoutError()
            wait_s = remaining
            if load_window and not saw_event:
                load_remaining = load_window - (now - spawned_at)
                if load_remaining <= 0:
                    raise MediaJobLoadHangError(self._load_hang_message(load_window))
                wait_s = min(wait_s, load_remaining)
            try:
                line = await asyncio.wait_for(proc.stdout.readline(), timeout=wait_s)
            except TimeoutError:
                if (
                    load_window
                    and not saw_event
                    and time.monotonic() - spawned_at >= load_window
                ):
                    raise MediaJobLoadHangError(
                        self._load_hang_message(load_window)
                    ) from None
                raise
            if not line:
                if proc.returncode is not None:
                    break
                if (
                    load_window
                    and not saw_event
                    and time.monotonic() - spawned_at >= load_window
                ):
                    raise MediaJobLoadHangError(self._load_hang_message(load_window))
                continue
            line_str = line.decode(errors="replace").strip()
            if not line_str:
                continue
            saw_event = True
            try:
                event = json.loads(line_str)
            except json.JSONDecodeError:
                logger.debug("worker stdout non-JSON: %s", line_str[:200])
                continue
            etype = event.get("type")
            if etype == "progress" and on_step is not None:
                on_step(
                    event.get("image", 0), event.get("step", 0), event.get("total", 0)
                )
            elif etype == "result":
                result.outputs = event.get("outputs", [])
                return
            elif etype == "error":
                result.error = event.get("message", "unknown worker error")
                return

    @staticmethod
    def _kill(proc: asyncio.subprocess.Process) -> None:
        try:
            proc.send_signal(signal.SIGTERM)
        except (ProcessLookupError, OSError):
            return
        try:
            import threading

            timer = threading.Timer(5.0, lambda: _sigkill(proc))
            timer.start()
        except Exception:
            pass


def _sigkill(proc: asyncio.subprocess.Process) -> None:
    try:
        proc.send_signal(signal.SIGKILL)
    except (ProcessLookupError, OSError):
        pass
