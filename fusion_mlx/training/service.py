# SPDX-License-Identifier: Apache-2.0
"""Fine-tuning service wrapping mlx_lm.tuner for LoRA/DORA training.

Importers/callers:
  - fusion_mlx.training.__init__ re-exports all 4 symbols
  - fusion_mlx.admin.fine_tune_route imports all 4 for route handlers
  - fusion_mlx.server imports FineTuneService for wiring

Affected API: new /admin/api/fine-tune/* endpoints (POST create, GET list/stream, DELETE cancel/remove)

Data schemas:
  FineTuneConfig  — training hyperparameters (lora_layers, rank, lr, batch_size, etc.)
  FineTuneProgress — per-step metrics (train_loss, val_loss, tokens/sec, eta)
  FineTuneJob     — job record (job_id, model_id, dataset, status, progress, adapter_path, events[], cond)
  FineTuneService — singleton service (queue, CRUD, _execute_training, SSE progress, adapter management)

User verbatim instruction: "开始做，注意设计方案需要有GUI的设计和落地方案，提交给macos app，可以先提pr，晚点在梳理macos app都还需要哪些GUI落地"
"""

from __future__ import annotations

import asyncio
import contextlib
import gc
import io
import json
import logging
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path

import mlx.core as mx

logger = logging.getLogger(__name__)

ADAPTER_BASE_DIR = Path(os.path.expanduser("~/.fusion-mlx/adapters"))


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class FineTuneConfig:
    lora_layers: int = 16
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    fine_tune_type: str = "lora"  # lora | dora | full | qlora
    optimizer: str = "adamw"
    # #402: QLoRA — quantize the frozen base to 4/8-bit, then attach LoRA
    # (mlx-lm has no --quantize-base flag; QLoRA is implicit once the base
    # is a QuantizedLinear, so we quantize on load when quantize_base=True).
    quantize_base: bool = False
    quant_bits: int = 4  # 4 | 8 (ignored unless quantize_base / qlora)
    # #402: MXFP8 mixed-precision training. mlx-lm 0.31.3 has NO fp8 training
    # path (mxfp8 is inference-only weight quant). Field exists to mirror the
    # fusion-trainer API surface; setting it True fails loudly until landed.
    mxfp8: bool = False
    learning_rate: float = 1e-5
    batch_size: int = 4
    iters: int = 100
    val_batches: int = 25
    steps_per_report: int = 10
    steps_per_eval: int = 200
    steps_per_save: int = 100
    max_seq_length: int = 2048
    gradient_checkpointing: bool = False
    grad_accumulation_steps: int = 1
    seed: int = 0
    lr_schedule: dict | None = None
    mask_prompt: bool = False

    def to_mlx_args(self, adapter_path: str, data_path: str, model_path: str):
        lora_params = {
            "rank": self.lora_rank,
            "dropout": self.lora_dropout,
            "scale": self.lora_alpha,
        }
        args_dict = {
            "model": model_path,
            "train": True,
            "fine_tune_type": self.fine_tune_type,
            "optimizer": self.optimizer,
            "data": data_path,
            "seed": self.seed,
            "num_layers": self.lora_layers,
            "batch_size": self.batch_size,
            "iters": self.iters,
            "val_batches": self.val_batches,
            "learning_rate": self.learning_rate,
            "steps_per_report": self.steps_per_report,
            "steps_per_eval": self.steps_per_eval,
            "save_every": self.steps_per_save,
            "max_seq_length": self.max_seq_length,
            "grad_checkpoint": self.gradient_checkpointing,
            "grad_accumulation_steps": self.grad_accumulation_steps,
            "adapter_path": adapter_path,
            "lora_parameters": lora_params,
            "mask_prompt": self.mask_prompt,
            "lr_schedule": self.lr_schedule,
            "hf_dataset": False,
            "test": False,
            "resume_adapter_file": None,
            "report_to": None,
            "project_name": None,
            "adapter_file": str(Path(adapter_path) / "adapters.safetensors"),
            "optimizer_config": {
                "adam": {},
                "adamw": {},
                "muon": {},
                "sgd": {},
                "adafactor": {},
            },
        }
        import types

        return types.SimpleNamespace(**args_dict)


@dataclass
class FineTuneProgress:
    step: int = 0
    total_steps: int = 0
    train_loss: float = 0.0
    val_loss: float | None = None
    learning_rate: float = 0.0
    tokens_per_second: float = 0.0
    iterations_per_second: float = 0.0
    trained_tokens: int = 0
    peak_memory_gb: float = 0.0
    elapsed_seconds: float = 0.0
    eta_seconds: float = 0.0


@dataclass
class FineTuneJob:
    job_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    model_id: str = ""
    dataset: str = ""
    config: FineTuneConfig = field(default_factory=FineTuneConfig)
    status: JobStatus = JobStatus.QUEUED
    progress: FineTuneProgress = field(default_factory=FineTuneProgress)
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    adapter_path: str = ""
    adapter_name: str = ""
    error: str = ""
    events: list[dict] = field(default_factory=list)
    terminal: bool = False
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
        }


class _ProgressCallback:
    """TrainingCallback that pushes progress events into FineTuneJob."""

    def __init__(self, job: FineTuneJob, loop: asyncio.AbstractEventLoop):
        self._job = job
        self._loop = loop
        self._start_time = time.monotonic()

    def on_train_loss_report(self, train_info: dict):
        step = train_info.get("iteration", 0)
        total = self._job.config.iters
        elapsed = time.monotonic() - self._start_time
        it_sec = train_info.get("iterations_per_second", 0.0)
        remaining = (total - step) / it_sec if it_sec > 0 else 0.0

        self._job.progress = FineTuneProgress(
            step=step,
            total_steps=total,
            train_loss=train_info.get("train_loss", 0.0),
            val_loss=self._job.progress.val_loss,
            learning_rate=train_info.get("learning_rate", 0.0),
            tokens_per_second=train_info.get("tokens_per_second", 0.0),
            iterations_per_second=it_sec,
            trained_tokens=train_info.get("trained_tokens", 0),
            peak_memory_gb=train_info.get("peak_memory", 0.0),
            elapsed_seconds=round(elapsed, 1),
            eta_seconds=round(remaining, 1),
        )
        self._push(
            {
                "type": "train_loss",
                "step": step,
                "total_steps": total,
                "train_loss": self._job.progress.train_loss,
                "val_loss": self._job.progress.val_loss,
                "learning_rate": self._job.progress.learning_rate,
                "tokens_per_second": self._job.progress.tokens_per_second,
                "it_sec": it_sec,
                "trained_tokens": self._job.progress.trained_tokens,
                "peak_memory_gb": self._job.progress.peak_memory_gb,
                "elapsed_seconds": self._job.progress.elapsed_seconds,
                "eta_seconds": self._job.progress.eta_seconds,
            }
        )

    def on_val_loss_report(self, val_info: dict):
        val_loss = val_info.get("val_loss", 0.0)
        self._job.progress.val_loss = val_loss
        self._push(
            {
                "type": "val_loss",
                "step": val_info.get("iteration", 0),
                "val_loss": val_loss,
                "val_time": val_info.get("val_time", 0.0),
            }
        )

    def _push(self, event: dict):
        self._job.events.append(event)

        async def _notify():
            async with self._job.cond:
                self._job.cond.notify_all()

        try:
            self._loop.call_soon_threadsafe(lambda: asyncio.ensure_future(_notify()))
        except RuntimeError:
            pass


class FineTuneService:
    """Singleton managing fine-tune job queue and execution.

    Apple Silicon constraint: only 1 concurrent training job because
    training requires the full GPU memory (inference model is evicted).
    """

    def __init__(self):
        self._jobs: dict[str, FineTuneJob] = {}
        self._queue: list[str] = []
        self._current_job_id: str | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._engine_pool = None
        self._running = False
        self._load_jobs()

    def set_engine_pool(self, pool):
        self._engine_pool = pool

    def set_loop(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop

    # =========================================================================
    # Job CRUD
    # =========================================================================

    def create_job(
        self,
        model_id: str,
        dataset: str,
        config: FineTuneConfig | None = None,
        adapter_name: str = "",
    ) -> FineTuneJob:
        cfg = config or FineTuneConfig()
        adapter_name = adapter_name or f"lora-{uuid.uuid4().hex[:6]}"
        adapter_path = str(ADAPTER_BASE_DIR / model_id / adapter_name)

        job = FineTuneJob(
            model_id=model_id,
            dataset=dataset,
            config=cfg,
            adapter_path=adapter_path,
            adapter_name=adapter_name,
        )
        self._jobs[job.job_id] = job
        self._queue.append(job.job_id)
        self._persist_jobs()
        logger.info(
            f"Fine-tune job created: {job.job_id} model={model_id} "
            f"adapter={adapter_name} queued={len(self._queue)}"
        )
        return job

    def get_job(self, job_id: str) -> FineTuneJob | None:
        return self._jobs.get(job_id)

    def list_jobs(self) -> list[FineTuneJob]:
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
            self._persist_jobs()
            logger.info(f"Fine-tune job cancelled (queued): {job_id}")
            return True
        if job.status == JobStatus.RUNNING:
            job.status = JobStatus.CANCELLED
            job.terminal = True
            job.finished_at = time.time()
            self._current_job_id = None
            self._notify_job(job)
            self._persist_jobs()
            self._process_queue()
            logger.info(f"Fine-tune job cancelled (running): {job_id}")
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
        self._persist_jobs()
        logger.info(f"Fine-tune job deleted: {job_id}")
        return True

    # =========================================================================
    # Queue processing
    # =========================================================================

    def start_processing(self):
        # _running is a one-time "initialized" marker, NOT a concurrency gate.
        # Concurrency is guarded by _current_job_id in _process_queue(). Gating
        # start_processing on a sticky _running flag meant jobs submitted after
        # the first one never got processed (issue #361): create_job enqueues,
        # but start_processing bailed out and _process_queue was never called.
        self._running = True
        if self._loop is None:
            try:
                self._loop = asyncio.get_running_loop()
            except RuntimeError:
                self._loop = asyncio.new_event_loop()
        self._process_queue()
        logger.info("Fine-tune service processing queue (jobs=%d)", len(self._queue))

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

        self._current_job_id = job_id
        job.status = JobStatus.RUNNING
        job.started_at = time.time()
        self._persist_jobs()
        self._notify_job(job)

        if self._loop is None:
            logger.error("No event loop for fine-tune execution")
            job.status = JobStatus.FAILED
            job.error = "No event loop available"
            job.terminal = True
            self._current_job_id = None
            return

        asyncio.ensure_future(self._run_job(job), loop=self._loop)

    async def _run_job(self, job: FineTuneJob):
        try:
            await asyncio.to_thread(self._execute_training, job)
        except Exception as exc:
            logger.exception(f"Fine-tune job {job.job_id} failed: {exc}")
            job.status = JobStatus.FAILED
            job.error = str(exc)
            job.terminal = True
            job.finished_at = time.time()
            self._persist_jobs()
            self._notify_job(job)
        finally:
            self._current_job_id = None
            self._process_queue()

    def _execute_training(self, job: FineTuneJob):
        """Run training in a background thread (blocking)."""

        import mlx.optimizers as optim
        from mlx.utils import tree_flatten
        from mlx_lm.tuner.datasets import CacheDataset, load_dataset
        from mlx_lm.tuner.trainer import TrainingArgs, train
        from mlx_lm.tuner.utils import (
            build_schedule,
            linear_to_lora_layers,
            print_trainable_parameters,
        )
        from mlx_lm.utils import load, save_config

        model_id = job.model_id
        dataset_path = job.dataset
        cfg = job.config

        model_path = self._resolve_model_path(model_id)
        if model_path is None:
            raise ValueError(f"Model not found: {model_id}")

        # Evict inference model to free GPU memory
        self._evict_model(model_id)

        # Build adapter output directory
        adapter_path = Path(job.adapter_path)
        adapter_path.mkdir(parents=True, exist_ok=True)
        adapter_file = adapter_path / "adapters.safetensors"

        # Build args for mlx_lm
        args = cfg.to_mlx_args(
            adapter_path=str(adapter_path),
            data_path=dataset_path,
            model_path=model_path,
        )

        # Load model + tokenizer
        logger.info(f"Loading model for training: {model_path}")
        model, tokenizer = load(
            model_path, tokenizer_config={"trust_remote_code": True}
        )

        # Load dataset
        logger.info(f"Loading dataset: {dataset_path}")
        train_set, valid_set, test_set = load_dataset(args, tokenizer)
        if not train_set:
            raise ValueError("Training dataset is empty")

        # Apply LoRA/DORA layers
        mx.random.seed(cfg.seed)
        model.freeze()

        # #402: MXFP8 training is not supported by mlx-lm 0.31.3 — fail loudly
        # instead of silently ignoring the switch (Rule 12: fail visibly).
        if cfg.mxfp8:
            raise ValueError(
                "mxfp8 mixed-precision training is not yet supported "
                "(mlx-lm 0.31.3 has no fp8 training path). Use qlora for "
                "memory savings; mxfp8 will be landed in a later phase."
            )

        # #402: QLoRA — quantize the frozen base to 4/8-bit before attaching
        # LoRA. If the base is already a quantized MLX model, quantize_base is
        # a no-op (LoRALinear.from_base wraps QuantizedLinear as-is). For an
        # unquantized base, nn.quantize() converts Linears in place.
        is_qlora = cfg.fine_tune_type == "qlora"
        if is_qlora or cfg.quantize_base:
            if cfg.quant_bits not in (4, 8):
                raise ValueError(f"quant_bits must be 4 or 8, got {cfg.quant_bits}")
            already_quant = any(
                hasattr(l, "to_quantized") and getattr(l, "bits", None)
                for l in model.layers
            )
            if not already_quant:
                logger.info(
                    "QLoRA: quantizing base to %d-bit (group_size=64)",
                    cfg.quant_bits,
                )
                import mlx.nn as nn

                nn.quantize(model, group_size=64, bits=cfg.quant_bits)
            else:
                logger.info("QLoRA: base already quantized, skipping quantize step")

        if cfg.fine_tune_type == "full":
            for layer in model.layers[-max(cfg.lora_layers, 0) :]:
                layer.unfreeze()
        elif cfg.fine_tune_type in ("lora", "dora", "qlora"):
            # qlora uses plain LoRA on top of the quantized base (not DoRA:
            # DoRALinear support for quantized weights is incomplete).
            use_dora = cfg.fine_tune_type == "dora"
            lora_params = {
                "rank": cfg.lora_rank,
                "dropout": cfg.lora_dropout,
                "scale": cfg.lora_alpha,
            }
            linear_to_lora_layers(
                model,
                cfg.lora_layers,
                lora_params,
                use_dora=use_dora,
            )
        else:
            raise ValueError(f"Unknown fine_tune_type: {cfg.fine_tune_type}")

        print_trainable_parameters(model)

        # Save adapter config
        save_config(vars(args), adapter_path / "adapter_config.json")

        # Build training args
        training_args = TrainingArgs(
            batch_size=cfg.batch_size,
            iters=cfg.iters,
            val_batches=cfg.val_batches,
            steps_per_report=cfg.steps_per_report,
            steps_per_eval=cfg.steps_per_eval,
            steps_per_save=cfg.steps_per_save,
            adapter_file=str(adapter_file),
            max_seq_length=cfg.max_seq_length,
            grad_checkpoint=cfg.gradient_checkpointing,
            grad_accumulation_steps=cfg.grad_accumulation_steps,
        )

        # Build optimizer
        lr = build_schedule(cfg.lr_schedule) if cfg.lr_schedule else cfg.learning_rate
        opt_map = {
            "adam": optim.Adam,
            "adamw": optim.AdamW,
            "muon": optim.Muon,
            "sgd": optim.SGD,
            "adafactor": optim.Adafactor,
        }
        opt_class = opt_map.get(cfg.optimizer.lower(), optim.AdamW)
        optimizer = opt_class(learning_rate=lr)

        # Progress callback
        loop = self._loop or asyncio.get_event_loop()
        callback = _ProgressCallback(job, loop)

        # Run training
        logger.info(
            f"Starting training: job={job.job_id} iters={cfg.iters} "
            f"batch_size={cfg.batch_size} lr={cfg.learning_rate}"
        )
        # tqdm (used inside mlx_lm trainer for eval/train loops) writes to
        # sys.stderr; when fusion-mlx runs as a background service the stderr
        # pipe is closed, and tqdm.status_printer flushes stderr on init →
        # BrokenPipeError kills the job. Redirect stderr to an in-memory
        # buffer for the duration of train(). See issue #381.
        with contextlib.redirect_stderr(io.StringIO()):
            train(
                model=model,
                args=training_args,
                optimizer=optimizer,
                train_dataset=CacheDataset(train_set),
                val_dataset=CacheDataset(valid_set) if valid_set else None,
                training_callback=callback,
            )

        # Save final adapter weights
        adapter_weights = dict(tree_flatten(model.trainable_parameters()))
        mx.save_safetensors(str(adapter_file), adapter_weights)
        logger.info(f"Training complete, adapter saved: {adapter_file}")

        # Mark job completed
        job.status = JobStatus.COMPLETED
        job.terminal = True
        job.finished_at = time.time()
        job.events.append(
            {
                "type": "completed",
                "adapter_path": str(adapter_path),
                "adapter_file": str(adapter_file),
            }
        )
        self._persist_jobs()
        self._notify_job(job)

        # Cleanup: free GPU memory
        del model
        del optimizer
        gc.collect()
        mx.clear_cache()

        logger.info(f"Fine-tune job completed: {job.job_id}")

    # =========================================================================
    # Adapter management
    # =========================================================================

    def list_adapters(self, model_id: str | None = None) -> list[dict]:
        if not ADAPTER_BASE_DIR.exists():
            return []
        adapters = []
        for model_dir in sorted(ADAPTER_BASE_DIR.iterdir()):
            if not model_dir.is_dir():
                continue
            if model_id and model_dir.name != model_id:
                continue
            for adapter_dir in sorted(model_dir.iterdir()):
                if not adapter_dir.is_dir():
                    continue
                config_path = adapter_dir / "adapter_config.json"
                weights_path = adapter_dir / "adapters.safetensors"
                info = {
                    "model_id": model_dir.name,
                    "adapter_name": adapter_dir.name,
                    "adapter_path": str(adapter_dir),
                    "has_weights": weights_path.exists(),
                    "has_config": config_path.exists(),
                }
                if config_path.exists():
                    try:
                        with open(config_path) as f:
                            config = json.load(f)
                        info["lora_layers"] = config.get("num_layers", 0)
                        info["lora_rank"] = config.get("lora_parameters", {}).get(
                            "rank", 0
                        )
                        info["fine_tune_type"] = config.get("fine_tune_type", "lora")
                    except Exception:
                        pass
                adapters.append(info)
        return adapters

    def delete_adapter(self, model_id: str, adapter_name: str) -> bool:
        adapter_dir = ADAPTER_BASE_DIR / model_id / adapter_name
        if not adapter_dir.exists():
            return False
        import shutil

        shutil.rmtree(adapter_dir)
        parent = adapter_dir.parent
        if parent.exists() and not any(parent.iterdir()):
            parent.rmdir()
        logger.info(f"Deleted adapter: {model_id}/{adapter_name}")
        return True

    def get_adapter_path(self, model_id: str, adapter_name: str) -> str | None:
        adapter_dir = ADAPTER_BASE_DIR / model_id / adapter_name
        if adapter_dir.exists():
            return str(adapter_dir)
        return None

    async def serve_adapter(self, model_id: str, adapter_name: str) -> dict:
        adapter_path = self.get_adapter_path(model_id, adapter_name)
        if adapter_path is None:
            raise ValueError(f"Adapter not found: {model_id}/{adapter_name}")
        weights = Path(adapter_path) / "adapters.safetensors"
        if not weights.exists():
            raise ValueError(f"Adapter weights missing: {weights}")
        if self._engine_pool is None:
            raise RuntimeError("Engine pool not initialized")
        engine = await self._engine_pool.get_engine(model_id, adapter_path=adapter_path)
        derived_key = f"{model_id}::lora::{adapter_path}"
        logger.info(f"Serving adapter {model_id}/{adapter_name} as {derived_key}")
        return {
            "served_model_id": derived_key,
            "base_model_id": model_id,
            "adapter_name": adapter_name,
            "adapter_path": adapter_path,
        }

    async def unload_adapter_engine(self, model_id: str, adapter_name: str) -> bool:
        adapter_path = self.get_adapter_path(model_id, adapter_name)
        if adapter_path is None:
            return False
        if self._engine_pool is None:
            return False
        derived_key = f"{model_id}::lora::{adapter_path}"
        entry = self._engine_pool.get_entry(derived_key)
        if entry is None or entry.engine is None:
            return False
        await self._engine_pool.unload_engine_async(derived_key)
        logger.info(f"Unloaded adapter engine: {derived_key}")
        return True

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
                logger.info(f"Evicted inference model for training: {model_id}")
            else:
                logger.warning(
                    f"Could not evict model {model_id} (pinned or in-use). "
                    f"Training will proceed but may OOM."
                )
        except Exception as exc:
            logger.warning(f"Failed to evict model {model_id}: {exc}")

    def _notify_job(self, job: FineTuneJob):
        async def _do():
            async with job.cond:
                job.cond.notify_all()

        if self._loop and self._loop.is_running():
            asyncio.ensure_future(_do(), loop=self._loop)

    # =========================================================================
    # Job persistence
    # =========================================================================

    @property
    def _jobs_file(self) -> Path:
        return ADAPTER_BASE_DIR / "jobs.json"

    def _persist_jobs(self):
        ADAPTER_BASE_DIR.mkdir(parents=True, exist_ok=True)
        data = []
        for job in self._jobs.values():
            data.append(job.to_dict())
        try:
            with open(self._jobs_file, "w") as f:
                json.dump(data, f, indent=2)
            logger.debug(f"Persisted {len(data)} jobs to {self._jobs_file}")
        except Exception as exc:
            logger.warning(f"Failed to persist jobs: {exc}")

    def _load_jobs(self):
        if not self._jobs_file.exists():
            return
        try:
            with open(self._jobs_file) as f:
                data = json.load(f)
        except Exception as exc:
            logger.warning(f"Failed to load jobs: {exc}")
            return
        for item in data:
            try:
                config_data = item.get("config", {})
                config = FineTuneConfig(**config_data)
                progress_data = item.get("progress", {})
                progress = FineTuneProgress(**progress_data)
                status_str = item.get("status", "queued")
                status = JobStatus(status_str)
                if status in (JobStatus.RUNNING, JobStatus.QUEUED):
                    status = JobStatus.CANCELLED
                job = FineTuneJob(
                    job_id=item.get("job_id", uuid.uuid4().hex[:12]),
                    model_id=item.get("model_id", ""),
                    dataset=item.get("dataset", ""),
                    config=config,
                    status=status,
                    progress=progress,
                    created_at=item.get("created_at", 0.0),
                    started_at=item.get("started_at"),
                    finished_at=item.get("finished_at"),
                    adapter_path=item.get("adapter_path", ""),
                    adapter_name=item.get("adapter_name", ""),
                    error=item.get("error", ""),
                    terminal=status
                    in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED),
                )
                self._jobs[job.job_id] = job
            except Exception as exc:
                logger.warning(f"Failed to load job entry: {exc}")
        logger.info(f"Loaded {len(self._jobs)} persisted jobs")
