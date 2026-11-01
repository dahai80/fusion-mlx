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
import threading
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from dataclasses import fields as _dc_fields
from enum import Enum
from pathlib import Path

import mlx.core as mx

from fusion_mlx.training.proc import TrainingProcessKilled

logger = logging.getLogger(__name__)

ADAPTER_BASE_DIR = Path(os.path.expanduser("~/.fusion-mlx/adapters"))

_JOB_ERROR_MAX_CHARS = 4000


def format_job_error(exc: BaseException, limit: int = 12) -> str:
    # #1403: job records carried only str(exc), so a missing-module failure
    # ("No module named 'mlx_vlm'") reached the caller with no frame and cost
    # a full CI iteration to localize. Keep the message line first — that is
    # what UIs and grep target — and append the innermost frames behind it.
    try:
        detail = "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__, limit=limit)
        ).strip()
    except Exception:
        detail = ""
    head = f"{type(exc).__name__}: {exc}"
    if not detail:
        return head
    if len(detail) > _JOB_ERROR_MAX_CHARS - len(head) - 1:
        detail = detail[: _JOB_ERROR_MAX_CHARS - len(head) - 4] + "..."
    return f"{head}\n{detail}"


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
    # #1182: resume an interrupted SFT run from a saved adapter checkpoint
    # (adapters.safetensors written every steps_per_save steps). Weights are
    # loaded into the freshly-built LoRA layers before train(); the step
    # counter restarts at 0 (optimizer-state persistence for true step
    # continuation is a follow-up). Same naming as DPOService's
    # init_adapter_path (#1149) for cross-service consistency.
    init_adapter_path: str = ""
    # #1186: process isolation. "process" runs train() in a spawned child
    # with its own Metal memory limit (serving GPU streams no longer share
    # the training workload's command stream); "thread" keeps the legacy
    # in-process path (used by tests / debugging).
    execution_mode: str = "process"
    # #1186: child Metal memory budget in GB. 0 = auto (60% of physical —
    # calibration: observed peak 64.9GB on 128GB while serving was resident).
    memory_limit_gb: float = 0.0
    # #1186: policy when the training child dies without reporting
    # (kill -9 / OOM / crash). "mark_failed" ends the job FAILED (resumable
    # via init_adapter_path); "auto_restart" re-queues up to max_restarts.
    on_process_kill: str = "mark_failed"
    max_restarts: int = 1
    # #1400: sequence packing for short-sample corpora. mlx-lm 0.31.3 has no
    # packing, so training/packing.py supplies the batch iterator and the loss:
    # samples are concatenated into rows of at most max_seq_length tokens with
    # a block-diagonal attention mask (no cross-sample attention) and a
    # per-token loss mask that still honours mask_prompt offsets. Off by
    # default — it changes batch composition, so results are not
    # byte-comparable with unpacked runs.
    pack_sequences: bool = False

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
            "pack_sequences": self.pack_sequences,
            "lr_schedule": self.lr_schedule,
            "hf_dataset": False,
            "test": False,
            "resume_adapter_file": (
                str(Path(self.init_adapter_path) / "adapters.safetensors")
                if self.init_adapter_path
                else None
            ),
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


# #1205: known-field sets for tolerant persisted-job recovery (schema drift
# between releases must drop unknown keys, not the whole record).
_CONFIG_FIELDS = {f.name for f in _dc_fields(FineTuneConfig)}
_PROGRESS_FIELDS = {f.name for f in _dc_fields(FineTuneProgress)}


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


class _BaseProgressCallback:
    """Builds the canonical progress event payloads. Shared by the
    in-process job callback and the process-mode pipe callback (#1186) so
    both paths emit identical event dicts."""

    def __init__(self, total_steps: int):
        self._total = total_steps
        self._start_time = time.monotonic()
        self._val_loss = None

    def on_train_loss_report(self, train_info: dict):
        step = train_info.get("iteration", 0)
        elapsed = time.monotonic() - self._start_time
        it_sec = train_info.get("iterations_per_second", 0.0)
        remaining = (self._total - step) / it_sec if it_sec > 0 else 0.0
        self._emit_train_loss(
            {
                "type": "train_loss",
                "step": step,
                "total_steps": self._total,
                "train_loss": train_info.get("train_loss", 0.0),
                "val_loss": self._val_loss,
                "learning_rate": train_info.get("learning_rate", 0.0),
                "tokens_per_second": train_info.get("tokens_per_second", 0.0),
                "it_sec": it_sec,
                "trained_tokens": train_info.get("trained_tokens", 0),
                "peak_memory_gb": train_info.get("peak_memory", 0.0),
                "elapsed_seconds": round(elapsed, 1),
                "eta_seconds": round(remaining, 1),
            }
        )

    def on_val_loss_report(self, val_info: dict):
        self._val_loss = val_info.get("val_loss", 0.0)
        self._emit_val_loss(
            {
                "type": "val_loss",
                "step": val_info.get("iteration", 0),
                "val_loss": self._val_loss,
                "val_time": val_info.get("val_time", 0.0),
            }
        )

    def _emit_train_loss(self, event: dict):
        raise NotImplementedError

    def _emit_val_loss(self, event: dict):
        raise NotImplementedError


class _ProgressCallback(_BaseProgressCallback):
    """TrainingCallback that pushes progress events into FineTuneJob."""

    def __init__(self, job: FineTuneJob, loop: asyncio.AbstractEventLoop):
        super().__init__(job.config.iters)
        self._job = job
        self._loop = loop
        self._val_loss = job.progress.val_loss

    def _emit_train_loss(self, event: dict):
        self._job.progress = FineTuneProgress(
            step=event["step"],
            total_steps=event["total_steps"],
            train_loss=event["train_loss"],
            val_loss=event["val_loss"],
            learning_rate=event["learning_rate"],
            tokens_per_second=event["tokens_per_second"],
            iterations_per_second=event["it_sec"],
            trained_tokens=event["trained_tokens"],
            peak_memory_gb=event["peak_memory_gb"],
            elapsed_seconds=event["elapsed_seconds"],
            eta_seconds=event["eta_seconds"],
        )
        self._push(event)

    def _emit_val_loss(self, event: dict):
        self._job.progress.val_loss = event["val_loss"]
        self._push(event)

    def _push(self, event: dict):
        # #1480 item 2b: cooperative abort check on every progress event —
        # mlx_lm's train loop calls callbacks unguarded, so raising here
        # propagates and fails the job instead of training through pressure.
        from .abort import check_abort

        check_abort(self._job.job_id)
        self._job.events.append(event)

        async def _notify():
            async with self._job.cond:
                self._job.cond.notify_all()

        try:
            self._loop.call_soon_threadsafe(lambda: asyncio.ensure_future(_notify()))
        except RuntimeError:
            pass


class _PipeProgressCallback(_BaseProgressCallback):
    """Child-side callback (#1186): sends the same event payloads over the
    pipe so the parent can map them into job state."""

    def __init__(self, report, total_steps: int):
        super().__init__(total_steps)
        self._report = report

    def _emit_train_loss(self, event: dict):
        self._report(event)

    def _emit_val_loss(self, event: dict):
        self._report(event)


def _apply_progress_event(job: FineTuneJob, event: dict):
    # #1186: parent-side mapping of a child progress event into job state
    # (mirror of _ProgressCallback's in-process behavior).
    etype = event.get("type")
    if etype == "train_loss":
        job.progress = FineTuneProgress(
            step=event.get("step", 0),
            total_steps=event.get("total_steps", 0),
            train_loss=event.get("train_loss", 0.0),
            val_loss=job.progress.val_loss,
            learning_rate=event.get("learning_rate", 0.0),
            tokens_per_second=event.get("tokens_per_second", 0.0),
            iterations_per_second=event.get("it_sec", 0.0),
            trained_tokens=event.get("trained_tokens", 0),
            peak_memory_gb=event.get("peak_memory_gb", 0.0),
            elapsed_seconds=event.get("elapsed_seconds", 0.0),
            eta_seconds=event.get("eta_seconds", 0.0),
        )
    elif etype == "val_loss":
        job.progress.val_loss = event.get("val_loss", 0.0)
    job.events.append(event)


def _resume_adapter_weights(model, cfg: FineTuneConfig, adapter_path: Path):
    # #1182: load a saved adapter checkpoint into the freshly-built LoRA
    # layers. Must run AFTER linear_to_lora_layers (needs the adapter
    # modules to exist) and BEFORE train(). Warn when resuming in-place —
    # the trainer's first checkpoint save overwrites adapters.safetensors
    # in the same dir.
    if not cfg.init_adapter_path:
        return model
    resume_file = Path(cfg.init_adapter_path) / "adapters.safetensors"
    if not resume_file.exists():
        raise ValueError(
            f"init_adapter_path has no adapters.safetensors: "
            f"{resume_file} (issue #1182)"
        )
    if str(Path(cfg.init_adapter_path).resolve()) == str(adapter_path.resolve()):
        logger.warning(
            "init_adapter_path equals the output adapter_path — the "
            "trainer will overwrite the resumed checkpoint at the "
            "first save (step %d)",
            cfg.steps_per_save,
        )
    # Not the mlx-lm tuner-utils loader: it re-applies
    # linear_to_lora_layers from the saved adapter_config.json,
    # double-applying on our already-built LoRA tree ("Can't convert layer
    # of type LoRALinear to LoRA", found in real-model verification). The
    # layers are already built by this run's cfg (including #746 per-module
    # keys); load only the weights.
    # strict=False: adapter files hold only LoRA params — strict=True would
    # demand the full model tree and fail ("Missing 704 parameters"). Same
    # convention as the mlx-lm tuner loader. Resuming with a different
    # rank/layers cfg than the source run is the caller's responsibility.
    model.load_weights(str(resume_file), strict=False)
    logger.info("Resumed adapter weights from %s (#1182)", cfg.init_adapter_path)
    return model


def execute_training_core(
    cfg: FineTuneConfig,
    model_path: str,
    dataset_path: str,
    adapter_path: str,
    callback,
) -> dict:
    # #1186: training body shared by the in-process thread path and the
    # spawned child process. Holds no FineTuneService state, so it is safe
    # to run in a separate interpreter. Returns the adapter output paths;
    # callers own job-status marking.
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

    # Fail fast on invalid config before the expensive model+dataset load.
    # #425/#402: validate() is the single source of truth for the
    # mxfp8 / quant_bits / fine_tune_type guards.
    cfg.validate()

    adapter_dir = Path(adapter_path)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    adapter_file = adapter_dir / "adapters.safetensors"

    # Build args for mlx_lm
    args = cfg.to_mlx_args(
        adapter_path=str(adapter_dir),
        data_path=dataset_path,
        model_path=model_path,
    )

    # Load model + tokenizer
    logger.info(f"Loading model for training: {model_path}")
    model, tokenizer = load(model_path, tokenizer_config={"trust_remote_code": True})

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

    # #1182: resume weights into the freshly-built LoRA layers. Must run
    # AFTER linear_to_lora_layers (needs the adapter modules to exist) and
    # BEFORE train().
    model = _resume_adapter_weights(model, cfg, adapter_dir)

    # Save adapter config
    save_config(vars(args), adapter_dir / "adapter_config.json")

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

    # #1400: sequence packing swaps the batch iterator and the loss; the model
    # blocks are patched so the packed mask reaches attention. Everything else
    # in mlx-lm's train() (eval, save, reporting, grad accumulation) keeps
    # working because it routes both callables through the same arguments.
    train_kwargs = {}
    if cfg.pack_sequences:
        from . import packing

        packing.install_mask_injection(model)
        train_kwargs["loss"] = packing.packed_loss
        train_kwargs["iterate_batches"] = packing.iterate_packed_batches
        logger.info(
            "Sequence packing enabled: rows up to max_seq_length=%d, "
            "block-diagonal attention mask",
            cfg.max_seq_length,
        )

    # Run training
    logger.info(
        f"Starting training: iters={cfg.iters} "
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
            **train_kwargs,
        )

    # Save final adapter weights
    adapter_weights = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(str(adapter_file), adapter_weights)
    logger.info(f"Training complete, adapter saved: {adapter_file}")

    # Cleanup: free GPU memory
    del model
    del optimizer
    gc.collect()
    mx.clear_cache()

    return {"adapter_path": str(adapter_dir), "adapter_file": str(adapter_file)}


def execute_training_spec(spec: dict, report) -> dict:
    # #1186: child-process entrypoint (module-level so spawn can pickle the
    # target). spec keys: config (FineTuneConfig), model_path, dataset,
    # adapter_path. Progress events stream through `report(event)`.
    cfg: FineTuneConfig = spec["config"]
    callback = _PipeProgressCallback(report, cfg.iters)
    return execute_training_core(
        cfg=cfg,
        model_path=spec["model_path"],
        dataset_path=spec["dataset"],
        adapter_path=spec["adapter_path"],
        callback=callback,
    )


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
        # M-P3-3 (#0908 audit): hold strong refs to ensure_future tasks —
        # asyncio only weak-references Tasks; GC can cancel mid-training.
        self._pending_tasks: set = set()
        # #1205: set when the process must die (fatal_exit path) while a
        # training job is in flight. The training thread can't flush a
        # checkpoint on demand (saves happen inside mlx_lm train() at
        # steps_per_save boundaries), so the graceful-exit contract is:
        # mark the job interrupted in the persisted store NOW, and let the
        # bounded defer window cover the wait for the next trainer-side
        # checkpoint save before the process exits.
        self._interrupted: threading.Event = threading.Event()
        # #1186: handle to the spawned training child (process mode) so
        # cancel/interrupt/fatal paths can terminate it.
        self._proc = None
        self._load_jobs()

    def has_active_job(self) -> bool:
        # #1205: True while a training job is executing (not queued). Used by
        # the memory enforcer to defer a process-fatal exit so the trainer's
        # periodic checkpoint can land.
        return self._current_job_id is not None and not self._interrupted.is_set()

    def _kill_training_process(self) -> bool:
        # #1186: terminate (SIGTERM then SIGKILL) the spawned training child,
        # if one is running. Safe to call when there is none.
        from fusion_mlx.training.proc import kill_training_process

        proc = self._proc
        self._proc = None
        killed = kill_training_process(proc)
        if killed:
            logger.info("Training child process terminated")
        return killed

    def interrupt_active_job(self) -> FineTuneJob | None:
        # #1205: mark the in-flight job interrupted and persist it, so a
        # fatal process exit leaves a resumable record (#1182 resume chain).
        # Returns the job (or None when nothing in flight). Idempotent.
        # #1186: in process mode the training child is killed too — a dying
        # serving process must not leave an orphaned trainer holding GPU
        # memory.
        self._kill_training_process()
        job = self._jobs.get(self._current_job_id or "")
        if job is None:
            self._interrupted.set()
            return None
        if job.status in (JobStatus.RUNNING, JobStatus.QUEUED):
            job.status = JobStatus.FAILED
            job.error = (
                "interrupted by fatal memory pressure before exit; "
                "resume via init_adapter_path (issue #1205)"
            )
            job.terminal = True
            job.finished_at = time.time()
            self._persist_jobs()
            logger.warning(
                "Fine-tune job %s marked interrupted before fatal exit "
                "(adapter checkpoints at steps_per_save cadence survive)",
                job.job_id,
            )
        self._interrupted.set()
        return job

    def set_engine_pool(self, pool):
        self._engine_pool = pool

    def set_loop(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop

    async def shutdown(self):
        # ENG-02 (#0909 audit): cancel pending training tasks on server
        # shutdown so in-flight jobs don't outlive the event loop.
        # #1186: process-mode training child dies with the server too.
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
                "FineTuneService shutdown: cancelled %d pending task(s)", cancelled
            )

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
            # #1186: in process mode the child must die with the job, or the
            # GPU keeps training a cancelled run for hours.
            self._kill_training_process()
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
        # #1473: real per-job budget from the model on disk instead of the
        # blind 60%-of-RAM default; fail-closed preflight against the
        # memory ceiling before any weights load. SFT has no reference
        # model.
        from .mem_preflight import resolve_training_budget_bytes

        _budget, _preflight_reason = resolve_training_budget_bytes(
            job_id=job_id,
            memory_limit_gb=getattr(job.config, "memory_limit_gb", 0.0),
            model_path=self._resolve_model_path(job.model_id),
            needs_reference=False,
        )
        if _preflight_reason is not None:
            job.status = JobStatus.FAILED
            job.error = _preflight_reason
            job.terminal = True
            job.finished_at = time.time()
            self._persist_jobs()
            self._notify_job(job)
            logger.warning(
                "Fine-tune job %s rejected by training memory preflight (#1473)",
                job_id,
            )
            self._process_queue()
            return
        _decision = _gate.admit_training(_budget)
        if not _decision.admitted:
            if _decision.deferred:
                self._queue.insert(0, job_id)
                schedule_recheck(self)
                logger.info(
                    "Fine-tune job %s deferred by admission gate; recheck in %.0fs",
                    job_id,
                    ADMISSION_RECHECK_SECONDS,
                )
                return
            job.status = JobStatus.FAILED
            job.error = _decision.reason
            job.terminal = True
            job.finished_at = time.time()
            self._persist_jobs()
            self._notify_job(job)
            logger.warning("Fine-tune job %s rejected by admission gate", job_id)
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

        # M-P3-3: store strong ref so GC doesn't cancel the training task.
        _t = asyncio.ensure_future(self._run_job(job), loop=self._loop)
        self._pending_tasks.add(_t)
        _t.add_done_callback(self._pending_tasks.discard)

    async def _run_job(self, job: FineTuneJob):
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
            logger.exception(f"Fine-tune job {job.job_id} failed: {exc}")
            job.status = JobStatus.FAILED
            job.error = format_job_error(exc)
            job.terminal = True
            job.finished_at = time.time()
            self._persist_jobs()
            self._notify_job(job)
        finally:
            self._proc = None
            self._current_job_id = None
            self._process_queue()

    async def _run_job_in_process(self, job: FineTuneJob):
        # #1186: spawn train() into a child process with its own Metal
        # memory budget; progress streams back over a pipe. The serving
        # process keeps doing inference-only work.
        from fusion_mlx.training.proc import run_training_process

        cfg = job.config
        cfg.validate()
        model_path = self._resolve_model_path(job.model_id)
        if model_path is None:
            raise ValueError(f"Model not found: {job.model_id}")

        # Evict inference model to free GPU memory
        self._evict_model(job.model_id)

        spec = {
            "config": cfg,
            "model_path": model_path,
            "dataset": job.dataset,
            "adapter_path": job.adapter_path,
            "entrypoint": "fusion_mlx.training.service:execute_training_spec",
        }

        def _on_event(event: dict):
            _apply_progress_event(job, event)
            self._notify_job(job)

        return await asyncio.to_thread(
            run_training_process,
            spec,
            _on_event,
            cfg.memory_limit_gb,
            lambda proc: setattr(self, "_proc", proc),
        )

    def _mark_job_completed(self, job: FineTuneJob):
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
        self._persist_jobs()
        self._notify_job(job)
        logger.info(f"Fine-tune job completed: {job.job_id}")

    def _handle_training_killed(self, job: FineTuneJob, exc: Exception):
        # #1186 kill policy. A user cancel already terminal-marked the job;
        # auto_restart re-queues up to max_restarts, otherwise FAILED
        # (resumable via init_adapter_path per #1182/#1205).
        if job.status == JobStatus.CANCELLED:
            logger.info(
                "Training child death after cancel (job %s) — policy skipped",
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
                "Training child died (%s); auto_restart attempt %d/%d",
                exc,
                job.attempts,
                job.config.max_restarts,
            )
            self._queue.append(job.job_id)
            self._persist_jobs()
            return
        logger.error("Training child died (%s); marking job FAILED", exc)
        job.status = JobStatus.FAILED
        job.error = format_job_error(exc)
        job.terminal = True
        job.finished_at = time.time()
        self._persist_jobs()
        self._notify_job(job)

    def _execute_training(self, job: FineTuneJob):
        """Legacy in-process path (execution_mode="thread"): runs training
        on a background thread. #1186 default is the spawned-process path
        (_run_job_in_process)."""

        from .abort import deregister_job, register_job

        register_job(job.job_id, "sft")
        try:
            return self._run_training_thread(job)
        finally:
            deregister_job(job.job_id)

    def _run_training_thread(self, job: FineTuneJob):
        cfg = job.config
        model_path = self._resolve_model_path(job.model_id)
        if model_path is None:
            raise ValueError(f"Model not found: {job.model_id}")

        # Evict inference model to free GPU memory
        self._evict_model(job.model_id)

        # Progress callback
        loop = self._loop or asyncio.get_event_loop()
        callback = _ProgressCallback(job, loop)

        logger.info(
            f"Starting training (thread mode): job={job.job_id} iters={cfg.iters}"
        )
        return execute_training_core(
            cfg=cfg,
            model_path=model_path,
            dataset_path=job.dataset,
            adapter_path=job.adapter_path,
            callback=callback,
        )

    # =========================================================================
    # Adapter management
    # =========================================================================

    def list_adapters(self, model_id: str | None = None) -> list[dict]:
        if not ADAPTER_BASE_DIR.exists():
            return []
        # #1317: HF-style model ids contain a slash, so adapter dirs nest one
        # level deeper (org/name/<adapter>); walk until a directory carries an
        # adapter marker or is a leaf instead of assuming exactly two levels.
        # Top-level dirs are always model-id components, never adapters.
        marker_names = ("adapter_config.json", "adapters.safetensors")
        adapters = []
        stack: list[tuple[Path, int]] = [(ADAPTER_BASE_DIR, 0)]
        while stack:
            base, depth = stack.pop()
            for entry in sorted(base.iterdir()):
                if not entry.is_dir():
                    continue
                is_adapter = depth >= 1 and (
                    any((entry / name).exists() for name in marker_names)
                    or not any(child.is_dir() for child in entry.iterdir())
                )
                if not is_adapter:
                    if depth < 3:
                        stack.append((entry, depth + 1))
                    continue
                mid = entry.parent.relative_to(ADAPTER_BASE_DIR).as_posix()
                if model_id and mid != model_id:
                    continue
                config_path = entry / "adapter_config.json"
                weights_path = entry / "adapters.safetensors"
                info = {
                    "model_id": mid,
                    "adapter_name": entry.name,
                    "adapter_path": str(entry),
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
        adapters.sort(key=lambda a: (a["model_id"], a["adapter_name"]))
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
                # #1205: persisted configs outlive the dataclass schema — a
                # field renamed/removed between releases made every entry
                # fail FineTuneConfig(**...) ("unexpected keyword"), so
                # recovery silently dropped the whole store ("Loaded 0
                # persisted jobs"). Filter to known fields and warn on the
                # dropped keys instead of losing the record.
                config_data = {
                    k: v
                    for k, v in item.get("config", {}).items()
                    if k in _CONFIG_FIELDS
                }
                unknown_cfg = set(item.get("config", {})) - _CONFIG_FIELDS
                if unknown_cfg:
                    logger.warning(
                        "Job %s: dropping unknown config keys from persisted "
                        "entry: %s",
                        item.get("job_id", "?"),
                        sorted(unknown_cfg),
                    )
                config = FineTuneConfig(**config_data)
                progress_data = {
                    k: v
                    for k, v in item.get("progress", {}).items()
                    if k in _PROGRESS_FIELDS
                }
                progress = FineTuneProgress(**progress_data)
                status_str = item.get("status", "queued")
                try:
                    status = JobStatus(status_str)
                except ValueError:
                    status = JobStatus.FAILED
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
                    attempts=item.get("attempts", 0),
                    adapter_name=item.get("adapter_name", ""),
                    error=item.get("error", ""),
                    terminal=status
                    in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED),
                )
                self._jobs[job.job_id] = job
            except Exception as exc:
                logger.warning(f"Failed to load job entry: {exc}")
        logger.info(f"Loaded {len(self._jobs)} persisted jobs")
