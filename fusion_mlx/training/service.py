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
    # #425: MXFP8 mixed-precision training. MLX 0.32.0 has no native fp8
    # dtype, so this self-implements by routing to the QLoRA 8-bit path
    # (8-bit frozen base + LoRA). validate() forces quantize_base=True,
    # quant_bits=8, fine_tune_type="qlora" when mxfp8=True. Honest semantics
    # = 8-bit-base LoRA (memory saving), NOT fp8 compute.
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
    # #746: passthrough SFT hyperparameters that mlx-lm 0.31.3 supports but
    # FineTuneConfig previously dropped. Defaults preserve prior behavior
    # (no weight decay, no clipping, all linears in the top-N layers).
    # weight_decay: forwarded to the optimizer ctor (AdamW/SGD/Muon/Adafactor
    #   accept it natively; Adam does NOT — see _build_optimizer).
    # max_grad_norm: global L2 gradient-norm clip applied in the training
    #   step before optimizer.update (mlx-lm trainer has no clip hook, so
    #   we wrap the optimizer's update).
    # lora_target_modules: restrict LoRA to modules whose class-name basename
    #   is in this set (e.g. ["q_proj","v_proj"]). None = all linears in the
    #   top lora_layers (prior behavior).
    weight_decay: float = 0.0
    max_grad_norm: float | None = None
    lora_target_modules: list[str] | None = None

    def to_mlx_args(self, adapter_path: str, data_path: str, model_path: str):
        lora_params = {
            "rank": self.lora_rank,
            "dropout": self.lora_dropout,
            "scale": self.lora_alpha,
        }
        # #746: per-module LoRA targeting. linear_to_lora_layers gates which
        # modules get adapters by `config["keys"]` (a set of full module
        # paths). We resolve the keys from lora_target_modules here is NOT
        # possible — keys depend on the loaded model's module tree, which
        # to_mlx_args doesn't see. _execute_training builds keys post-load
        # and injects into lora_params; to_mlx_args only carries the raw
        # target-module names so the saved adapter_config.json records intent.
        if self.lora_target_modules:
            lora_params["target_modules"] = list(self.lora_target_modules)
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
                "adamw": {"weight_decay": self.weight_decay},
                "muon": {"weight_decay": self.weight_decay},
                "sgd": {"weight_decay": self.weight_decay},
                "adafactor": {"weight_decay": self.weight_decay},
            },
        }
        import types

        return types.SimpleNamespace(**args_dict)

    _VALID_FINE_TUNE_TYPES = ("lora", "dora", "full", "qlora")

    def validate(self):
        # Pure config validation (no I/O, no model load). Called early in
        # _execute_training so invalid configs fail fast before the expensive
        # model+dataset load. Single source of truth for the #402/#425 guards
        # so tests exercise real production code instead of replicating it.
        # #425: mxfp8 self-implement. MLX 0.32.0 has no native fp8 dtype
        # (float8_e4m3fn/e5m2 absent) so real fp8 GEMM compute is impossible
        # on this stack. We honor the downstream fusion-trainer use_mxfp8=True
        # switch by routing to the EXISTING QLoRA 8-bit path: quantize the
        # frozen base to 8-bit (group_size=64) and attach LoRA. Honest
        # semantics = 8-bit-base LoRA (memory saving), NOT fp8 compute. When
        # a future MLX adds real fp8, the mfa/fp8_linear path can activate.
        # mxfp8 + full fine-tune is a contradiction (full unfreezes the base,
        # so there is nothing to quantize) — fail visibly (Rule 12).
        if self.mxfp8:
            if self.fine_tune_type == "full":
                raise ValueError(
                    "mxfp8 is incompatible with fine_tune_type='full' (full "
                    "fine-tuning unfreezes the base, so there is no frozen "
                    "base to quantize); use lora/dora/qlora with mxfp8 "
                    "(issue #425)"
                )
            logger.info(
                "mxfp8=True: routing to QLoRA 8-bit path (8-bit frozen base "
                "+ LoRA). MLX 0.32.0 has no fp8 dtype, so this is 8-bit-base "
                "LoRA for memory saving, NOT fp8 compute (issue #425)."
            )
            self.quantize_base = True
            self.quant_bits = 8
            self.fine_tune_type = "qlora"
        # #402: QLoRA / quantize_base needs 4- or 8-bit base.
        if (self.fine_tune_type == "qlora" or self.quantize_base) and (
            self.quant_bits not in (4, 8)
        ):
            raise ValueError(f"quant_bits must be 4 or 8, got {self.quant_bits}")
        if self.fine_tune_type not in self._VALID_FINE_TUNE_TYPES:
            raise ValueError(f"Unknown fine_tune_type: {self.fine_tune_type}")
        # #746: weight_decay is forwarded to AdamW/SGD/Muon/Adafactor only.
        # mlx.optimizers.Adam has NO weight_decay arg; a non-zero value with
        # the plain Adam optimizer is a config error — fail visibly (Rule 12)
        # rather than silently dropping the knob.
        if self.weight_decay != 0.0 and self.optimizer.lower() == "adam":
            raise ValueError(
                "weight_decay is not supported by the 'adam' optimizer "
                "(mlx.optimizers.Adam has no weight_decay arg); use 'adamw' "
                "or set weight_decay=0.0 (issue #746)"
            )
        # #746: max_grad_norm is global L2 clip — must be positive when set.
        if self.max_grad_norm is not None and self.max_grad_norm <= 0:
            raise ValueError(
                f"max_grad_norm must be a positive float, got {self.max_grad_norm}"
            )
        # #746: lora_target_modules only applies to lora/dora/qlora (not full).
        if self.lora_target_modules and self.fine_tune_type == "full":
            raise ValueError(
                "lora_target_modules has no effect with fine_tune_type='full' "
                "(full fine-tuning unfreezes layers, no LoRA adapters)"
            )


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


class _GradClipOptimizer:
    """Wrap an mlx optimizer to apply global L2 gradient-norm clipping.

    #746: mlx-lm 0.31.3 trainer calls ``optimizer.update(model, grad)`` inside
    the compiled step with no clip hook, and TrainingCallback exposes no grad
    access. Wrapping update is the only seam: clip the global L2 norm of the
    gradient tree, then delegate to the real optimizer. ``state`` proxies to
    the inner optimizer so the trainer's compiled ``state`` list captures the
    real optimizer state (not this wrapper's __dict__).
    """

    def __init__(self, optimizer, max_grad_norm: float):
        self._optimizer = optimizer
        self.max_grad_norm = float(max_grad_norm)

    @property
    def state(self):
        return self._optimizer.state

    def update(self, model, gradients: dict):
        from mlx.utils import tree_flatten

        grads = [g for g in tree_flatten(gradients) if g is not None]
        if not grads:
            return self._optimizer.update(model, gradients)
        total_norm = mx.sqrt(
            mx.sum(mx.stack([mx.sum(g.astype(mx.float32) ** 2) for g in grads]))
        )
        scale = mx.minimum(1.0, self.max_grad_norm / (total_norm + 1e-6))
        from mlx.utils import tree_map

        clipped = tree_map(lambda g: g * scale if g is not None else g, gradients)
        return self._optimizer.update(model, clipped)

    def __getattr__(self, name):
        # Delegate any other attribute (learning_rate, apply_gradients, init,
        # ...) to the wrapped optimizer so the trainer sees a faithful proxy.
        return getattr(self._optimizer, name)


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

        # Fail fast on invalid config before the expensive model+dataset load.
        # #425/#402: validate() is the single source of truth for the
        # mxfp8 / quant_bits / fine_tune_type guards.
        cfg.validate()

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

        # #402: QLoRA — quantize the frozen base to 4/8-bit before attaching
        # LoRA. If the base is already a quantized MLX model, quantize_base is
        # a no-op (LoRALinear.from_base wraps QuantizedLinear as-is). For an
        # unquantized base, nn.quantize() converts Linears in place.
        # quant_bits range already enforced by cfg.validate() above.
        is_qlora = cfg.fine_tune_type == "qlora"
        if is_qlora or cfg.quantize_base:
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
            # #746: per-module LoRA targeting. linear_to_lora_layers gates
            # adapters by config["keys"] (a set of full module paths within
            # the top lora_layers blocks). When lora_target_modules is set,
            # collect the full module paths whose class-name basename is in
            # the target set (e.g. "layers.0.attention.q_proj" → basename
            # "q_proj"). None/empty = all linears (prior behavior).
            if cfg.lora_target_modules:
                targets = {t.strip() for t in cfg.lora_target_modules if t.strip()}
                keys = set()
                for layer in model.layers[-max(cfg.lora_layers, 0) :]:
                    for path, mod in layer.named_modules():
                        basename = path.rsplit(".", 1)[-1]
                        if basename in targets:
                            keys.add(path)
                lora_params["keys"] = keys
                logger.info(
                    "LoRA targeting %d modules matching %s across top %d layers",
                    len(keys),
                    sorted(targets),
                    cfg.lora_layers,
                )
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
        # #746: forward weight_decay to the optimizer ctor for the optimizers
        # that accept it (AdamW/SGD/Muon/Adafactor). Adam has no weight_decay
        # arg — cfg.validate() already rejected that combination, so we only
        # pass it for the others. Default weight_decay=0.0 keeps prior behavior.
        opt_name = cfg.optimizer.lower()
        if opt_name == "adam":
            optimizer = opt_class(learning_rate=lr)
        else:
            optimizer = opt_class(learning_rate=lr, weight_decay=cfg.weight_decay)
        # #746: global gradient-norm clipping. mlx-lm 0.31.3 trainer calls
        # optimizer.update(model, grad) inside the compiled step with no clip
        # hook. We wrap update to clip the global L2 norm of the grad tree
        # before delegating to the real optimizer. The wrapper is applied
        # AFTER construction so optimizer.state is the real one.
        if cfg.max_grad_norm is not None:
            optimizer = _GradClipOptimizer(optimizer, max_grad_norm=cfg.max_grad_norm)
            logger.info(
                "Gradient clipping enabled: max_grad_norm=%.4f",
                cfg.max_grad_norm,
            )

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
