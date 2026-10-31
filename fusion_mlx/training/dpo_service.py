# SPDX-License-Identifier: Apache-2.0
"""DPO / ORPO preference-alignment training service (#399).

Importers/callers:
  - fusion_mlx.admin.fine_tune_route imports DPOService + DPOJob for routes
  - fusion_mlx.server instantiates + wires via set_dpo_context

Affected API: new /admin/api/fine-tune/dpo/jobs + /orpo/jobs endpoints
  (POST create, GET list/stream/{id}, POST cancel, DELETE)

Data schemas:
  DPOJob     — job record (job_id, model_id, preference_pairs, config, events[], cond)
  DPOService — singleton service (queue, CRUD, _execute_dpo, SSE progress)

User verbatim instruction: "启动3个功能issue的修复落地"
"""

from __future__ import annotations

import asyncio
import gc
import json
import logging
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

import mlx.core as mx

from fusion_mlx.training.dpo import DPOConfig, DPOTrainer
from fusion_mlx.training.service import (
    ADAPTER_BASE_DIR,
    JobStatus,
    format_job_error,
)

logger = logging.getLogger(__name__)

# #1469: DPO keeps policy + reference activations per pair; without gradient
# checkpointing a large model's layer intermediates overflow wired memory
# (27B-8bit observed at 203GB, fatal enforcer exit). Auto-enable above this.
_GRAD_CHECKPOINT_AUTO_THRESHOLD_BYTES = 16 * 1024**3


def _chain_adapter_weights(model, cfg: DPOConfig):
    # #1461: chaining must NOT go through mlx_utils.load(adapter_path=...) —
    # that re-applies linear_to_lora_layers from the saved adapter_config.json
    # and double-applies on the tree we just built ("Can't convert layer of
    # type LoRALinear to LoRA"). The layers are already built by this run's
    # cfg; load only the weights (same convention as
    # service._resume_adapter_weights, #1182).
    resume_dir = Path(cfg.init_adapter_path)
    resume_file = resume_dir / "adapters.safetensors"
    if not resume_file.exists():
        raise ValueError(
            f"init_adapter_path has no adapters.safetensors: "
            f"{resume_file} (issue #1461)"
        )
    config_file = resume_dir / "adapter_config.json"
    if config_file.exists():
        try:
            saved = json.loads(config_file.read_text())
        except (OSError, ValueError) as exc:
            raise ValueError(
                f"init_adapter_path adapter_config.json unreadable: "
                f"{config_file} ({exc}) (issue #1461)"
            ) from exc
        saved_rank = (saved.get("lora_parameters") or {}).get("rank")
        if saved_rank is not None and int(saved_rank) != int(cfg.lora_rank):
            # strict=False loads nothing on a rank mismatch and trains from
            # fresh LoRA silently — reject loudly instead (issue #1461).
            raise ValueError(
                f"init_adapter_path adapter rank {saved_rank} != cfg.lora_rank "
                f"{cfg.lora_rank} ({resume_dir}) — a mismatch silently trains "
                "from fresh LoRA (issue #1461)"
            )
        saved_layers = saved.get("num_layers")
        if saved_layers is not None and int(saved_layers) != int(cfg.lora_layers):
            raise ValueError(
                f"init_adapter_path adapter num_layers {saved_layers} != "
                f"cfg.lora_layers {cfg.lora_layers} ({resume_dir}) — a "
                "mismatch loads only part of the adapter (issue #1461)"
            )
    else:
        logger.warning(
            "init_adapter_path has no adapter_config.json (%s) — cannot "
            "validate rank/layers against cfg; loading weights as-is (#1461)",
            resume_dir,
        )
    model.load_weights(str(resume_file), strict=False)
    logger.info(
        "%s: chained adapter weights from %s (#1461)",
        cfg.method.upper(),
        cfg.init_adapter_path,
    )


@dataclass
class DPOJob:
    job_id: str
    model_id: str
    preference_pairs: list = field(default_factory=list)
    config: DPOConfig = field(default_factory=DPOConfig)
    adapter_name: str = ""
    adapter_path: str = ""
    status: JobStatus = JobStatus.QUEUED
    created_at: float = field(default_factory=time.time)
    started_at: float = 0.0
    finished_at: float = 0.0
    error: str = ""
    events: list = field(default_factory=list)
    progress: dict = field(default_factory=dict)
    terminal: bool = False
    cond: asyncio.Condition = field(default_factory=asyncio.Condition, repr=False)

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "model_id": self.model_id,
            "preference_pairs": self.preference_pairs,
            "config": asdict(self.config),
            "adapter_name": self.adapter_name,
            "adapter_path": self.adapter_path,
            "status": self.status.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "progress": self.progress,
        }


class DPOService:
    # Singleton managing DPO/ORPO job queue + execution. One concurrent job
    # (Apple Silicon memory constraint — training evicts inference model).
    # Serves both dpo and orpo via config.method.

    def __init__(self):
        self._jobs: dict[str, DPOJob] = {}
        self._queue: list[str] = []
        self._current_job_id: str | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._engine_pool = None
        self._running = False
        self._pending_tasks: set = set()
        self._load_jobs()

    def set_engine_pool(self, pool):
        self._engine_pool = pool

    def set_loop(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop

    async def shutdown(self):
        # ENG-02 (#0909 audit): cancel pending tasks on server shutdown.
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
            logger.info("DPOService shutdown: cancelled %d pending task(s)", cancelled)

    def _resolve_model_path(self, model_id: str) -> str | None:
        if self._engine_pool is None:
            return model_id
        entry = self._engine_pool.get_entry(model_id)
        if entry is not None and hasattr(entry, "model_path"):
            return entry.model_path
        candidate = Path(os.path.expanduser("~/.fusion-mlx/models")) / model_id
        if candidate.exists():
            return str(candidate)
        return model_id

    # =========================================================================
    # Job CRUD
    # =========================================================================

    def create_job(
        self,
        model_id: str,
        preference_pairs: list,
        config: DPOConfig | None = None,
        adapter_name: str = "",
    ) -> DPOJob:
        cfg = config or DPOConfig()
        method_tag = cfg.method
        adapter_name = adapter_name or f"{method_tag}-{uuid.uuid4().hex[:6]}"
        adapter_path = str(ADAPTER_BASE_DIR / model_id / adapter_name)

        job = DPOJob(
            job_id=uuid.uuid4().hex[:12],
            model_id=model_id,
            preference_pairs=list(preference_pairs),
            config=cfg,
            adapter_name=adapter_name,
            adapter_path=adapter_path,
        )
        self._jobs[job.job_id] = job
        self._queue.append(job.job_id)
        self._persist_jobs()
        logger.info(
            "%s job created: %s model=%s adapter=%s queued=%d",
            method_tag.upper(),
            job.job_id,
            model_id,
            adapter_name,
            len(self._queue),
        )
        return job

    def get_job(self, job_id: str) -> DPOJob | None:
        return self._jobs.get(job_id)

    def list_jobs(self) -> list[DPOJob]:
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
            return True
        if job.status == JobStatus.RUNNING:
            job.status = JobStatus.CANCELLED
            job.terminal = True
            job.finished_at = time.time()
            self._current_job_id = None
            self._notify_job(job)
            self._persist_jobs()
            self._process_queue()
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
        logger.info("DPO service processing queue (jobs=%d)", len(self._queue))

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
                    "DPO job %s deferred by admission gate; recheck in %.0fs",
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
            logger.warning("DPO job %s rejected by admission gate", job_id)
            self._process_queue()
            return

        self._current_job_id = job_id
        job.status = JobStatus.RUNNING
        job.started_at = time.time()
        self._persist_jobs()
        self._notify_job(job)

        if self._loop is None:
            logger.error("No event loop for DPO execution")
            job.status = JobStatus.FAILED
            job.error = "No event loop available"
            job.terminal = True
            self._current_job_id = None
            return

        _t = asyncio.ensure_future(self._run_job(job), loop=self._loop)
        self._pending_tasks.add(_t)
        _t.add_done_callback(self._pending_tasks.discard)

    async def _run_job(self, job: DPOJob):
        try:
            await asyncio.to_thread(self._execute_dpo, job)
        except Exception as exc:
            logger.exception("DPO job %s failed: %s", job.job_id, exc)
            job.status = JobStatus.FAILED
            job.error = format_job_error(exc)
            job.terminal = True
            job.finished_at = time.time()
            self._persist_jobs()
            self._notify_job(job)
        finally:
            self._current_job_id = None
            self._process_queue()

    def _execute_dpo(self, job: DPOJob):
        # Run DPO/ORPO training in a background thread (blocking). Load model,
        # apply LoRA, run preference-pair loop, save adapter, cleanup.
        import mlx_lm.utils as mlx_utils
        from mlx_lm.tuner.utils import linear_to_lora_layers

        model_path = self._resolve_model_path(job.model_id)
        if model_path is None:
            raise ValueError(f"Cannot resolve model path for {job.model_id}")
        cfg = job.config
        logger.info(
            "%s execute: model=%s job=%s", cfg.method.upper(), model_path, job.job_id
        )
        if cfg.init_adapter_path:
            logger.info(
                "%s: chaining on SFT adapter %s (#1149)",
                cfg.method.upper(),
                cfg.init_adapter_path,
            )

        # #1461: load the BASE model — not the adapter. The old
        # mlx_utils.load(..., adapter_path=cfg.init_adapter_path) returned a
        # tree with LoRA layers already built, and the
        # linear_to_lora_layers call below then double-applied ("Can't
        # convert layer of type LoRALinear to LoRA"). Chained weights are
        # loaded into the freshly-built LoRA tree by _chain_adapter_weights.
        model, tokenizer = mlx_utils.load(model_path)

        mx.random.seed(cfg.seed)
        model.freeze()
        lora_params = {
            "rank": cfg.lora_rank,
            "dropout": cfg.lora_dropout,
            "scale": cfg.lora_alpha,
        }
        linear_to_lora_layers(model, cfg.lora_layers, lora_params, use_dora=False)
        if cfg.init_adapter_path:
            _chain_adapter_weights(model, cfg)
        # #1463: hybrid linear-attention models (qwen3_5/qwen3_next) gate
        # gated_delta_update on the module training flag — eval mode routes
        # the forward through the no-vjp metal kernel and the first backward
        # raises "[Primitive::vjp] Not implemented for CustomKernel".
        # model.train() only flips the flag; MLX freeze is separate, so
        # frozen-base + LoRA-trainable semantics are unchanged.
        model.train()
        # #1469: DPO keeps policy + reference activations for the same
        # forward pair, so a large model without gradient checkpointing
        # retains every layer intermediate of BOTH passes — a 27B-8bit job
        # was observed driving wired memory to 203GB (fatal enforcer exit).
        # Auto-enable checkpointing for big models unless the caller set the
        # flag explicitly. Threshold 16GB: covers 27B-8bit (~28GB) and
        # larger, leaves small models on the fast path.
        if not cfg.gradient_checkpointing:
            try:
                from mlx.utils import tree_flatten as _mlx_tree_flatten

                param_bytes = sum(
                    int(p.nbytes)
                    for _path, p in _mlx_tree_flatten(model.parameters())
                    if isinstance(p, mx.array)
                )
            except Exception:
                param_bytes = 0
                logger.warning(
                    "%s: could not estimate param bytes for grad-checkpoint "
                    "auto-enable; leaving gradient_checkpointing=%s",
                    cfg.method.upper(),
                    cfg.gradient_checkpointing,
                )
            if param_bytes >= _GRAD_CHECKPOINT_AUTO_THRESHOLD_BYTES:
                cfg.gradient_checkpointing = True
                logger.info(
                    "%s: auto-enabling gradient checkpointing — "
                    "param_bytes=%.1fGB >= %.0fGB threshold (policy+reference "
                    "activations would overflow wired memory otherwise, #1469)",
                    cfg.method.upper(),
                    param_bytes / 1024**3,
                    _GRAD_CHECKPOINT_AUTO_THRESHOLD_BYTES / 1024**3,
                )
        if cfg.gradient_checkpointing:
            # #1460: same mechanism as the SFT path (mlx_lm tuner
            # grad_checkpoint) — patch the transformer block __call__ with
            # mx.checkpoint to trade compute for activation memory.
            from mlx_lm.tuner.trainer import grad_checkpoint

            grad_checkpoint(model.layers[0])
            logger.info(
                "%s: gradient checkpointing enabled (#1460)", cfg.method.upper()
            )

        trainer = DPOTrainer(model, tokenizer, model_path, cfg)

        pairs = job.preference_pairs
        batch_size = max(cfg.batch_size, 1)
        n_pairs = len(pairs)
        if n_pairs == 0:
            raise ValueError(f"{cfg.method} job has no preference_pairs")

        for it in range(cfg.iters):
            start = (it * batch_size) % n_pairs
            batch = [pairs[(start + i) % n_pairs] for i in range(batch_size)]
            result = trainer.train_step(batch)
            self._push_event(
                job,
                {
                    "type": f"{cfg.method}_step",
                    "iter": it,
                    "total_iters": cfg.iters,
                    "loss": result.loss,
                    "reward_margin": result.reward_margin,
                    "acc_chosen": result.acc_chosen,
                    "n_pairs": len(batch),
                },
            )
            job.progress = {
                "iter": it + 1,
                "total_iters": cfg.iters,
                "loss": result.loss,
                "reward_margin": result.reward_margin,
                "acc_chosen": result.acc_chosen,
            }

        Path(job.adapter_path).mkdir(parents=True, exist_ok=True)
        trainer.save_adapter(str(Path(job.adapter_path) / "adapters.safetensors"))

        adapter_cfg = {
            "adapter_path": job.adapter_path,
            "num_layers": cfg.lora_layers,
            "lora_parameters": {
                "rank": cfg.lora_rank,
                "scale": cfg.lora_alpha,
                "dropout": cfg.lora_dropout,
            },
            "fine_tune_type": "lora",
        }
        with open(Path(job.adapter_path) / "adapter_config.json", "w") as f:
            json.dump(adapter_cfg, f, indent=2)

        del model
        del tokenizer
        gc.collect()
        mx.clear_cache()

        job.status = JobStatus.COMPLETED
        job.terminal = True
        job.finished_at = time.time()
        self._persist_jobs()
        self._push_event(job, {"type": "done", "adapter_path": job.adapter_path})
        logger.info(
            "%s job %s completed: adapter=%s",
            cfg.method.upper(),
            job.job_id,
            job.adapter_path,
        )

    def _push_event(self, job: DPOJob, event: dict):
        job.events.append(event)

        async def _notify():
            async with job.cond:
                job.cond.notify_all()

        try:
            if self._loop and self._loop.is_running():
                self._loop.call_soon_threadsafe(
                    lambda: asyncio.ensure_future(_notify())
                )
        except RuntimeError:
            pass

    def _notify_job(self, job: DPOJob):
        async def _do():
            async with job.cond:
                job.cond.notify_all()

        if self._loop and self._loop.is_running():
            asyncio.ensure_future(_do(), loop=self._loop)

    # =========================================================================
    # Persistence
    # =========================================================================

    @property
    def _jobs_file(self) -> Path:
        return ADAPTER_BASE_DIR / "dpo_jobs.json"

    def _persist_jobs(self):
        ADAPTER_BASE_DIR.mkdir(parents=True, exist_ok=True)
        data = [job.to_dict() for job in self._jobs.values()]
        try:
            with open(self._jobs_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as exc:
            logger.warning("Failed to persist DPO jobs: %s", exc)

    def _load_jobs(self):
        if not self._jobs_file.exists():
            return
        try:
            with open(self._jobs_file) as f:
                data = json.load(f)
        except Exception as exc:
            logger.warning("Failed to load DPO jobs: %s", exc)
            return
        for item in data:
            try:
                cfg = DPOConfig(**item.get("config", {}))
                status = JobStatus(item.get("status", "queued"))
                if status in (JobStatus.RUNNING, JobStatus.QUEUED):
                    status = JobStatus.CANCELLED
                job = DPOJob(
                    job_id=item.get("job_id", uuid.uuid4().hex[:12]),
                    model_id=item.get("model_id", ""),
                    preference_pairs=item.get("preference_pairs", []),
                    config=cfg,
                    adapter_name=item.get("adapter_name", ""),
                    adapter_path=item.get("adapter_path", ""),
                    status=status,
                    created_at=item.get("created_at", 0.0),
                    started_at=item.get("started_at", 0.0),
                    finished_at=item.get("finished_at", 0.0),
                    error=item.get("error", ""),
                    progress=item.get("progress", {}),
                )
                job.terminal = status in (
                    JobStatus.COMPLETED,
                    JobStatus.FAILED,
                    JobStatus.CANCELLED,
                )
                self._jobs[job.job_id] = job
            except Exception as exc:
                logger.warning("Skipping malformed DPO job: %s", exc)
