# SPDX-License-Identifier: Apache-2.0
"""Vision-language fine-tuning service wrapping mlx_vlm.trainer (#797).

Importers/callers:
  - fusion_mlx.admin.fine_tune_route imports VLMFineTuneService for route handlers
  - fusion_mlx.server imports VLMFineTuneService for startup wiring

Mirrors fusion_mlx.training.service.FineTuneService but drives mlx-vlm SFT/LoRA
instead of mlx-lm, so a visual-language model can be fine-tuned over HTTP (blocks
fusion-trainer #55). Image + text input, structured text target.

Data schemas:
  VLMFineTuneConfig  — training hyperparameters (lora rank/layers, lr, iters, ...)
  VLMFineTuneProgress — per-step metrics
  VLMFineTuneJob     — job record (job_id, model_id, dataset, status, events[])
  VLMFineTuneService — singleton (queue, CRUD, _execute_training, SSE progress)
"""

from __future__ import annotations

import asyncio
import contextlib
import gc
import io
import json
import logging
import os
import sys
import threading
import time
import types
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path

import mlx.core as mx

from fusion_mlx.training.proc import TrainingProcessKilled
from fusion_mlx.training.service import format_job_error

logger = logging.getLogger(__name__)

ADAPTER_BASE_DIR = Path(os.path.expanduser("~/.fusion-mlx/adapters"))


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class VLMFineTuneConfig:
    # LoRA knobs. full_finetune=True unfreezes the language model instead.
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    lora_layers: int = 16
    full_finetune: bool = False
    train_vision: bool = False
    optimizer: str = "adamw"
    learning_rate: float = 1e-5
    batch_size: int = 4
    iters: int = 100
    val_batches: int = 25
    steps_per_report: int = 10
    steps_per_eval: int = 200
    steps_per_save: int = 100
    max_seq_length: int = 2048
    grad_checkpoint: bool = False
    grad_accumulation_steps: int = 1
    seed: int = 0
    # Dataset column mapping. mlx-vlm expects a HF dataset whose rows carry
    # an image field and a conversations/messages field. image_column /
    # text_column let the caller point at non-default field names; None means
    # use the dataset's native columns (mlx-vlm handles common schemas).
    image_column: str | None = None
    text_column: str | None = None
    # Number of data-loader workers (passed to VisionDataset via config).
    num_processes: int = 1
    # #1186: process isolation (same knobs as text SFT FineTuneConfig).
    execution_mode: str = "process"
    memory_limit_gb: float = 0.0
    on_process_kill: str = "mark_failed"
    max_restarts: int = 1

    def validate(self):
        if self.full_finetune and self.train_vision:
            # Allowed but memory-heavy; warn, do not fail.
            logger.warning(
                "full_finetune + train_vision both true: unfreezing the full "
                "language + vision stack is memory-heavy on Apple Silicon."
            )
        if self.lora_layers <= 0 and not self.full_finetune:
            raise ValueError("lora_layers must be > 0 for LoRA fine-tuning")
        if self.iters <= 0:
            raise ValueError("iters must be > 0")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        # #1186: isolation knobs.
        if self.execution_mode not in ("process", "thread"):
            raise ValueError(
                f"execution_mode must be 'process' or 'thread', "
                f"got {self.execution_mode!r}"
            )
        if self.on_process_kill not in ("mark_failed", "auto_restart"):
            raise ValueError(
                f"on_process_kill must be 'mark_failed' or 'auto_restart', "
                f"got {self.on_process_kill!r}"
            )
        if self.memory_limit_gb < 0:
            raise ValueError(
                f"memory_limit_gb must be >= 0 (0 = auto), got {self.memory_limit_gb}"
            )


@dataclass
class VLMFineTuneProgress:
    step: int = 0
    total_steps: int = 0
    train_loss: float = 0.0
    val_loss: float | None = None
    learning_rate: float = 0.0
    elapsed_seconds: float = 0.0
    eta_seconds: float = 0.0


@dataclass
class VLMFineTuneJob:
    job_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    model_id: str = ""
    dataset: str = ""
    config: VLMFineTuneConfig = field(default_factory=VLMFineTuneConfig)
    status: JobStatus = JobStatus.QUEUED
    progress: VLMFineTuneProgress = field(default_factory=VLMFineTuneProgress)
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    adapter_path: str = ""
    adapter_name: str = ""
    error: str = ""
    events: list[dict] = field(default_factory=list)
    terminal: bool = False
    # #1186: auto_restart attempts consumed by the kill policy.
    attempts: int = 0
    cond: asyncio.Condition = field(default_factory=asyncio.Condition)

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "model_id": self.model_id,
            "dataset": self.dataset,
            "config": asdict(self.config),
            "status": self.status.value,
            "progress": asdict(self.progress),
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "adapter_path": self.adapter_path,
            "adapter_name": self.adapter_name,
            "error": self.error,
            "attempts": self.attempts,
        }


class _StdoutProgressReader:
    """Parse mlx_vlm.trainer.train() stdout progress lines into job events.

    mlx_vlm's train() reports via print() (no TrainingCallback like mlx-lm),
    so we redirect stdout to a pipe and parse the "Iter N: Train loss X"
    lines. Runs in its own thread; pushes events into the job via the loop.
    """

    _PREFIX = "Iter "

    def __init__(
        self,
        job: VLMFineTuneJob,
        loop: asyncio.AbstractEventLoop | None,
        emit=None,
    ):
        # #1186: emit=None keeps the in-process behavior (job mutation +
        # cond notify); emit=setter routes events to the process pipe.
        self._job = job
        self._loop = loop
        self._emit = emit
        self._start = time.monotonic()
        self._buf = ""
        self._stop = False

    def feed(self, chunk: str):
        self._buf += chunk
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._parse_line(line.strip())

    def _parse_line(self, line: str):
        if not line.startswith(self._PREFIX):
            return
        # Lines look like: "Iter 10: Train loss 1.23, Learning Rate ..., It/sec ..."
        try:
            rest = line[len(self._PREFIX) :]
            step_str = rest.split(":", 1)[0].strip()
            step = int(step_str)
        except (ValueError, IndexError):
            return
        train_loss = 0.0
        learning_rate = 0.0
        it_sec = 0.0
        if "Train loss" in line:
            try:
                train_loss = float(line.split("Train loss", 1)[1].split(",")[0].strip())
            except (ValueError, IndexError):
                pass
        if "Learning Rate" in line:
            try:
                learning_rate = float(
                    line.split("Learning Rate", 1)[1].split(",")[0].strip()
                )
            except (ValueError, IndexError):
                pass
        if "It/sec" in line:
            try:
                it_sec = float(line.split("It/sec", 1)[1].split(",")[0].strip())
            except (ValueError, IndexError):
                pass
        total = self._job.config.iters
        elapsed = time.monotonic() - self._start
        remaining = (total - step) / it_sec if it_sec > 0 else 0.0
        self._job.progress = VLMFineTuneProgress(
            step=step,
            total_steps=total,
            train_loss=train_loss,
            val_loss=self._job.progress.val_loss,
            learning_rate=learning_rate,
            elapsed_seconds=round(elapsed, 1),
            eta_seconds=round(remaining, 1),
        )
        self._push(
            {
                "type": "train_loss",
                "step": step,
                "total_steps": total,
                "train_loss": train_loss,
                "learning_rate": learning_rate,
                "it_sec": it_sec,
                "elapsed_seconds": self._job.progress.elapsed_seconds,
                "eta_seconds": self._job.progress.eta_seconds,
            }
        )
        if "Val loss" in line:
            try:
                val_loss = float(line.split("Val loss", 1)[1].split(",")[0].strip())
                self._job.progress.val_loss = val_loss
                self._push({"type": "val_loss", "step": step, "val_loss": val_loss})
            except (ValueError, IndexError):
                pass

    def _push(self, event: dict):
        if self._emit is not None:
            self._emit(event)
            return
        self._job.events.append(event)

        async def _notify():
            async with self._job.cond:
                self._job.cond.notify_all()

        try:
            self._loop.call_soon_threadsafe(lambda: asyncio.ensure_future(_notify()))
        except RuntimeError:
            pass


class VLMFineTuneService:
    """Singleton managing VL fine-tune job queue and execution (#797).

    Apple Silicon constraint: only 1 concurrent training job (training needs
    full GPU memory, inference model is evicted).
    """

    def __init__(self):
        self._jobs: dict[str, VLMFineTuneJob] = {}
        self._queue: list[str] = []
        self._current_job_id: str | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._engine_pool = None
        self._running = False
        self._pending_tasks: set = set()
        # #1186: spawned training child handle (process mode).
        self._proc = None

    def _kill_training_process(self) -> bool:
        # #1186: SIGTERM then SIGKILL the training child, if any.
        from fusion_mlx.training.proc import kill_training_process

        proc = self._proc
        self._proc = None
        killed = kill_training_process(proc)
        if killed:
            logger.info("VL training child process terminated")
        return killed

    def set_engine_pool(self, pool):
        self._engine_pool = pool

    def set_loop(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop

    async def shutdown(self):
        # ENG-02 (#0909 audit): cancel pending tasks on server shutdown.
        # #1186: the training child dies with the server.
        self._kill_training_process()
        cancelled = 0
        for task in list(self._pending_tasks):
            if not task.done():
                task.cancel()
                cancelled += 1
        if self._pending_tasks:
            await asyncio.gather(*self._pending_tasks, return_exceptions=True)
        self._pending_tasks.clear()
        self._running = False
        if cancelled:
            logger.info(
                "VLMFineTuneService shutdown: cancelled %d pending task(s)", cancelled
            )

    # =========================================================================
    # Job CRUD
    # =========================================================================

    def create_job(
        self,
        model_id: str,
        dataset: str,
        config: VLMFineTuneConfig | None = None,
        adapter_name: str = "",
    ) -> VLMFineTuneJob:
        cfg = config or VLMFineTuneConfig()
        adapter_name = adapter_name or f"vlm-lora-{uuid.uuid4().hex[:6]}"
        adapter_path = str(ADAPTER_BASE_DIR / model_id / adapter_name)
        job = VLMFineTuneJob(
            model_id=model_id,
            dataset=dataset,
            config=cfg,
            adapter_path=adapter_path,
            adapter_name=adapter_name,
        )
        self._jobs[job.job_id] = job
        self._queue.append(job.job_id)
        logger.info(
            f"VL fine-tune job created: {job.job_id} model={model_id} "
            f"adapter={adapter_name} queued={len(self._queue)}"
        )
        return job

    def get_job(self, job_id: str) -> VLMFineTuneJob | None:
        return self._jobs.get(job_id)

    def list_jobs(self) -> list[VLMFineTuneJob]:
        return list(self._jobs.values())

    def cancel_job(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if job is None:
            return False
        if job.status == JobStatus.QUEUED:
            job.status = JobStatus.CANCELLED
            job.terminal = True
            if job_id in self._queue:
                self._queue.remove(job_id)
            self._notify_job(job)
            logger.info(f"VL fine-tune job cancelled (queued): {job_id}")
            return True
        if job.status == JobStatus.RUNNING:
            # #1186: the training child must die with the job.
            self._kill_training_process()
            job.status = JobStatus.CANCELLED
            job.terminal = True
            job.finished_at = time.time()
            self._current_job_id = None
            self._notify_job(job)
            self._process_queue()
            logger.info(f"VL fine-tune job cancelled (running): {job_id}")
            return True
        return False

    def delete_job(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if job is None:
            return False
        if job.status == JobStatus.RUNNING:
            return False
        self._jobs.pop(job_id, None)
        if job_id in self._queue:
            self._queue.remove(job_id)
        logger.info(f"VL fine-tune job deleted: {job_id}")
        return True

    # =========================================================================
    # Queue processing
    # =========================================================================

    def start_processing(self):
        self._running = True
        if self._loop is None:
            try:
                self._loop = asyncio.get_running_loop()
            except RuntimeError:
                self._loop = asyncio.new_event_loop()
        self._process_queue()
        logger.info("VL fine-tune service processing queue (jobs=%d)", len(self._queue))

    def _process_queue(self):
        if not self._running:
            return
        if self._current_job_id is not None:
            return
        if not self._queue:
            return
        job_id = self._queue.pop(0)
        job = self._jobs.get(job_id)
        if job is None or job.status == JobStatus.CANCELLED:
            self._process_queue()
            return
        # #1187: unified heavy-admission gate before the job claims the
        # execution slot. Deferred while paused (recheck scheduled); FAILED
        # when the safety line is exceeded — a clear error instead of
        # saturating the GPU for hours (#1184/#1186).
        from ..admission import (
            ADMISSION_RECHECK_SECONDS,
            get_heavy_admission_gate,
            schedule_recheck,
        )

        _gate = get_heavy_admission_gate()
        _budget = _gate.training_budget_bytes(
            getattr(job.config, "memory_limit_gb", 0.0)
        )
        _decision = _gate.admit_training(_budget)
        if not _decision.admitted:
            if _decision.deferred:
                self._queue.insert(0, job_id)
                schedule_recheck(self)
                logger.info(
                    "VL fine-tune job %s deferred by admission gate; recheck in %.0fs",
                    job_id,
                    ADMISSION_RECHECK_SECONDS,
                )
                return
            job.status = JobStatus.FAILED
            job.error = _decision.reason
            job.terminal = True
            job.finished_at = time.time()
            self._notify_job(job)
            logger.warning("VL fine-tune job %s rejected by admission gate", job_id)
            self._process_queue()
            return

        self._current_job_id = job_id
        job.status = JobStatus.RUNNING
        job.started_at = time.time()
        self._notify_job(job)
        if self._loop is None:
            logger.error("No event loop for VL fine-tune execution")
            job.status = JobStatus.FAILED
            job.error = "No event loop available"
            job.terminal = True
            self._current_job_id = None
            return
        _t = asyncio.ensure_future(self._run_job(job), loop=self._loop)
        self._pending_tasks.add(_t)
        _t.add_done_callback(self._pending_tasks.discard)

    async def _run_job(self, job: VLMFineTuneJob):
        try:
            if job.config.execution_mode == "process":
                await self._run_job_in_process(job)
            else:
                await asyncio.to_thread(self._execute_training, job)
            self._mark_job_completed(job)
        except TrainingProcessKilled as exc:
            # #1186: child died without reporting — apply the kill policy.
            self._handle_training_killed(job, exc)
        except Exception as exc:
            logger.exception(f"VL fine-tune job {job.job_id} failed: {exc}")
            job.status = JobStatus.FAILED
            job.error = format_job_error(exc)
            job.terminal = True
            job.finished_at = time.time()
            self._notify_job(job)
        finally:
            self._proc = None
            self._current_job_id = None
            self._process_queue()

    async def _run_job_in_process(self, job: VLMFineTuneJob):
        # #1186: spawn train() into a child process with its own Metal
        # memory budget; stdout progress lines stream over the pipe.
        from fusion_mlx.training.proc import run_training_process

        cfg = job.config
        cfg.validate()
        model_path = self._resolve_model_path(job.model_id)
        if model_path is None:
            raise ValueError(f"Model not found: {job.model_id}")

        # Evict inference model to free GPU memory (same constraint as text SFT).
        self._evict_model(job.model_id)

        spec = {
            "config": cfg,
            "model_path": model_path,
            "dataset": job.dataset,
            "adapter_path": job.adapter_path,
            "entrypoint": "fusion_mlx.training.vlm_service:execute_vlm_training_spec",
        }

        def _on_event(event: dict):
            _apply_vlm_progress_event(job, event)
            self._notify_job(job)

        return await asyncio.to_thread(
            run_training_process,
            spec,
            _on_event,
            cfg.memory_limit_gb,
            lambda proc: setattr(self, "_proc", proc),
        )

    def _mark_job_completed(self, job: VLMFineTuneJob):
        job.status = JobStatus.COMPLETED
        job.terminal = True
        job.finished_at = time.time()
        job.events.append(
            {
                "type": "completed",
                "adapter_path": job.adapter_path,
                "adapter_file": str(Path(job.adapter_path) / "adapters.safetensors"),
            }
        )
        self._notify_job(job)
        logger.info(f"VL fine-tune job completed: {job.job_id}")

    def _handle_training_killed(self, job: VLMFineTuneJob, exc: Exception):
        # #1186 kill policy (mirror of text SFT).
        if job.status == JobStatus.CANCELLED:
            logger.info(
                "VL training child death after cancel (job %s) — policy skipped",
                job.job_id,
            )
            return
        if (
            job.config.on_process_kill == "auto_restart"
            and job.attempts < job.config.max_restarts
        ):
            job.attempts += 1
            job.status = JobStatus.QUEUED
            job.terminal = False
            job.error = ""
            logger.warning(
                "VL training child died (%s); auto_restart attempt %d/%d",
                exc,
                job.attempts,
                job.config.max_restarts,
            )
            self._queue.append(job.job_id)
            return
        logger.error("VL training child died (%s); marking job FAILED", exc)
        job.status = JobStatus.FAILED
        job.error = format_job_error(exc)
        job.terminal = True
        job.finished_at = time.time()
        self._notify_job(job)

    def _execute_training(self, job: VLMFineTuneJob):
        """Legacy in-process path (execution_mode="thread")."""

        cfg = job.config
        model_path = self._resolve_model_path(job.model_id)
        if model_path is None:
            raise ValueError(f"Model not found: {job.model_id}")

        # Evict inference model to free GPU memory (same constraint as text SFT).
        self._evict_model(job.model_id)

        loop = self._loop or asyncio.get_event_loop()
        reader = _StdoutProgressReader(job, loop)

        return execute_vlm_training_core(
            cfg=cfg,
            model_path=model_path,
            dataset_path=job.dataset,
            adapter_path=job.adapter_path,
            emit=None,
            reader=reader,
        )

    @staticmethod
    def _drain_pipe(read_fd: int, reader: _StdoutProgressReader):
        """Read captured stdout from the pipe and feed the progress reader."""
        try:
            with os.fdopen(read_fd, "r", buffering=1) as f:
                for chunk in iter(lambda: f.read(256), ""):
                    reader.feed(chunk)
        except (OSError, ValueError):
            pass

    # =========================================================================
    # Helpers
    # =========================================================================

    def _resolve_model_path(self, model_id: str) -> str | None:
        if self._engine_pool is None:
            return model_id
        entry = self._engine_pool.get_entry(model_id)
        if entry is not None and hasattr(entry, "model_path"):
            return entry.model_path
        models_dir = Path(os.path.expanduser("~/.fusion-mlx/models"))
        candidate = models_dir / model_id
        if candidate.exists():
            return str(candidate)
        return model_id

    def _evict_model(self, model_id: str):
        if self._engine_pool is None:
            return
        try:
            result = self._engine_pool.unload_if_idle_unpinned(model_id)
            if result:
                logger.info(f"Evicted inference model for VL training: {model_id}")
            else:
                logger.warning(
                    f"Could not evict model {model_id} (pinned or in-use). "
                    f"VL training will proceed but may OOM."
                )
        except Exception as exc:
            logger.warning(f"Failed to evict model {model_id}: {exc}")

    def _notify_job(self, job: VLMFineTuneJob):
        async def _do():
            async with job.cond:
                job.cond.notify_all()

        if self._loop and self._loop.is_running():
            asyncio.ensure_future(_do(), loop=self._loop)


def execute_vlm_training_core(
    cfg: VLMFineTuneConfig,
    model_path: str,
    dataset_path: str,
    adapter_path: str,
    emit=None,
    reader=None,
) -> dict:
    # #1186: VL training body shared by the in-process thread path and the
    # spawned child process. emit=None uses the given reader (job-backed);
    # emit=setter routes parsed stdout events to the caller (pipe).
    import mlx.optimizers as optim
    from mlx_vlm.lora import VisionDataset, load_dataset
    from mlx_vlm.trainer.sft_trainer import TrainingArgs, train
    from mlx_vlm.trainer.utils import (
        save_adapter,
        setup_model_for_training,
    )
    from mlx_vlm.utils import load as vlm_load

    cfg.validate()

    adapter_dir = Path(adapter_path)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    adapter_file = adapter_dir / "adapters.safetensors"

    # Load VLM + processor (mlx_vlm returns (model, processor), no tokenizer).
    logger.info(f"Loading VLM for training: {model_path}")
    model, processor = vlm_load(model_path)
    processor_config = getattr(processor, "config", None) or {}

    # Load dataset via HuggingFace datasets (Hub id or local dir with
    # images + jsonl). mlx_vlm.lora.load_dataset = datasets.load_dataset.
    logger.info(f"Loading VL dataset: {dataset_path}")
    hf_dataset = load_dataset(dataset_path)
    if hasattr(hf_dataset, "keys"):
        # DatasetDict — take the first split as train.
        split = next(iter(hf_dataset.keys()))
        hf_dataset = hf_dataset[split]

    train_ds = VisionDataset(
        hf_dataset,
        processor_config,
        processor,
        image_resize_shape=None,
    )
    val_ds = None

    mx.random.seed(cfg.seed)

    # Build a peft-style args namespace for setup_model_for_training, which
    # expects lora_rank/lora_alpha/lora_dropout/full_finetune/train_vision.
    setup_args = types.SimpleNamespace(
        full_finetune=cfg.full_finetune,
        train_vision=cfg.train_vision,
        lora_rank=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
    )
    model = setup_model_for_training(model, setup_args, adapter_path=None)

    # Save adapter config for reproducibility (mirrors text SFT).
    config_record = {
        "model": model_path,
        "fine_tune_type": "full" if cfg.full_finetune else "lora",
        "num_layers": cfg.lora_layers,
        "lora_parameters": {
            "rank": cfg.lora_rank,
            "alpha": cfg.lora_alpha,
            "dropout": cfg.lora_dropout,
        },
        "train_vision": cfg.train_vision,
        "iters": cfg.iters,
        "batch_size": cfg.batch_size,
        "learning_rate": cfg.learning_rate,
    }
    with open(adapter_dir / "adapter_config.json", "w") as f:
        json.dump(config_record, f, indent=2)

    # Build training args. mlx_vlm TrainingArgs carries learning_rate
    # directly (no external schedule builder), and adapter_file.
    training_args = TrainingArgs(
        batch_size=cfg.batch_size,
        iters=cfg.iters,
        val_batches=cfg.val_batches,
        steps_per_report=cfg.steps_per_report,
        steps_per_eval=cfg.steps_per_eval,
        steps_per_save=cfg.steps_per_save,
        max_seq_length=cfg.max_seq_length,
        adapter_file=str(adapter_file),
        grad_checkpoint=cfg.grad_checkpoint,
        learning_rate=cfg.learning_rate,
        full_finetune=cfg.full_finetune,
        gradient_accumulation_steps=cfg.grad_accumulation_steps,
    )

    # Build optimizer.
    opt_map = {
        "adam": optim.Adam,
        "adamw": optim.AdamW,
        "sgd": optim.SGD,
    }
    opt_class = opt_map.get(cfg.optimizer.lower(), optim.AdamW)
    optimizer = opt_class(learning_rate=cfg.learning_rate)

    if reader is None:
        # Child path: a throwaway job carries config.iters for the parser.
        reader = _StdoutProgressReader(VLMFineTuneJob(config=cfg), None, emit=emit)

    logger.info(
        f"Starting VL training: iters={cfg.iters} "
        f"batch_size={cfg.batch_size} lr={cfg.learning_rate} "
        f"full_finetune={cfg.full_finetune} train_vision={cfg.train_vision}"
    )

    # mlx_vlm reports progress via print() to stdout. Redirect stdout to a
    # pipe, drain it on a reader thread that parses progress lines and
    # pushes events. tqdm/mlx_vlm Colors write to stdout/stderr; both are
    # captured (stderr discarded to avoid BrokenPipeError under a closed
    # service stderr — same fix as text SFT issue #381).
    read_fd, write_fd = os.pipe()
    reader_thread = threading.Thread(
        target=VLMFineTuneService._drain_pipe, args=(read_fd, reader), daemon=True
    )
    reader_thread.start()
    old_stdout = sys.stdout
    os.dup2(write_fd, 1)
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            train(
                model=model,
                optimizer=optimizer,
                train_dataset=train_ds,
                val_dataset=val_ds,
                args=training_args,
            )
    finally:
        sys.stdout = old_stdout
        os.close(write_fd)
        reader_thread.join(timeout=10.0)

    # Save final adapter weights.
    save_adapter(model, str(adapter_file))
    logger.info(f"VL training complete, adapter saved: {adapter_file}")

    del model
    del optimizer
    gc.collect()
    mx.clear_cache()

    return {"adapter_path": str(adapter_dir), "adapter_file": str(adapter_file)}


def execute_vlm_training_spec(spec: dict, report) -> dict:
    # #1186: child-process entrypoint (module-level so spawn can pickle the
    # target). spec keys: config (VLMFineTuneConfig), model_path, dataset,
    # adapter_path. Progress events stream through `report(event)`.
    cfg: VLMFineTuneConfig = spec["config"]
    return execute_vlm_training_core(
        cfg=cfg,
        model_path=spec["model_path"],
        dataset_path=spec["dataset"],
        adapter_path=spec["adapter_path"],
        emit=report,
    )


def _apply_vlm_progress_event(job: VLMFineTuneJob, event: dict):
    # #1186: parent-side mapping of a child progress event into job state.
    etype = event.get("type")
    if etype == "train_loss":
        job.progress = VLMFineTuneProgress(
            step=event.get("step", 0),
            total_steps=event.get("total_steps", 0),
            train_loss=event.get("train_loss", 0.0),
            val_loss=job.progress.val_loss,
            learning_rate=event.get("learning_rate", 0.0),
            elapsed_seconds=event.get("elapsed_seconds", 0.0),
            eta_seconds=event.get("eta_seconds", 0.0),
        )
    elif etype == "val_loss":
        job.progress.val_loss = event.get("val_loss", 0.0)
    job.events.append(event)
