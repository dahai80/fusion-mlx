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
from fusion_mlx.training.service import ADAPTER_BASE_DIR, JobStatus

logger = logging.getLogger(__name__)


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
        self._load_jobs()

    def set_engine_pool(self, pool):
        self._engine_pool = pool

    def set_loop(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop

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

        asyncio.ensure_future(self._run_job(job), loop=self._loop)

    async def _run_job(self, job: DPOJob):
        try:
            await asyncio.to_thread(self._execute_dpo, job)
        except Exception as exc:
            logger.exception("DPO job %s failed: %s", job.job_id, exc)
            job.status = JobStatus.FAILED
            job.error = str(exc)
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

        model, tokenizer = mlx_utils.load(model_path)

        mx.random.seed(cfg.seed)
        model.freeze()
        lora_params = {
            "rank": cfg.lora_rank,
            "dropout": cfg.lora_dropout,
            "scale": cfg.lora_alpha,
        }
        linear_to_lora_layers(model, cfg.lora_layers, lora_params, use_dora=False)

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
