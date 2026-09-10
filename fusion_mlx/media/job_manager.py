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
        spec = {
            "variant": variant,
            "model_path": model_path,
            "quantize": quantize,
            "config_label": config_label,
            "output_dir": output_dir,
            "output_format": output_format,
            "n_images": n_images,
            "gen_params": gen_params,
        }
        return output_dir, spec

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
    ) -> MediaJobResult:
        # Returns MediaJobResult with output file paths. Raises TimeoutError
        # if the worker exceeds timeout (subprocess killed). Raises
        # RuntimeError on worker crash/error.
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
        try:
            await self._read_stdout(proc, timeout, on_step, result)
        except TimeoutError:
            self._kill(proc)
            result.error = f"worker exceeded {timeout:.0f}s timeout (killed)"
            result.exit_code = -1
            raise TimeoutError(result.error)
        finally:
            if proc.returncode is None:
                self._kill(proc)
            await proc.wait()
            result.elapsed = time.monotonic() - t0

        if proc.returncode not in (0, None):
            rc = proc.returncode
            stderr_data = await proc.stderr.read() if proc.stderr else b""
            stderr_text = stderr_data.decode(errors="replace")[-2000:]
            if rc in (137, -9):
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

    async def _read_stdout(
        self,
        proc: asyncio.subprocess.Process,
        timeout: float,
        on_step: Callable[[int, int, int], None] | None,
        result: MediaJobResult,
    ) -> None:
        # Read JSONL progress/result events from worker stdout. Raises
        # TimeoutError if no event arrives within timeout.
        assert proc.stdout is not None
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError()
            try:
                line = await asyncio.wait_for(proc.stdout.readline(), timeout=remaining)
            except TimeoutError:
                raise
            if not line:
                if proc.returncode is not None:
                    break
                continue
            line_str = line.decode(errors="replace").strip()
            if not line_str:
                continue
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
