# SPDX-License-Identifier: Apache-2.0
# Voice fine-tune job service (#1336).
#
# Importers/callers:
#   - fusion_mlx.admin.fine_tune_route (voice routes + set_voice_context)
#   - fusion_mlx.server (startup wiring)
#
# Affected API:
#   POST   /admin/api/fine-tune/voice/jobs
#   GET    /admin/api/fine-tune/voice/jobs
#   GET    /admin/api/fine-tune/voice/jobs/{id}
#   GET    /admin/api/fine-tune/voice/jobs/{id}/stream
#   POST   /admin/api/fine-tune/voice/jobs/{id}/cancel
#   DELETE /admin/api/fine-tune/voice/jobs/{id}
#   GET    /admin/api/fine-tune/voice/backends
#
# Contract mirrors the SFT/DPO/GRPO job surfaces so fusion-trainer can treat
# voice training like any other submit -> poll -> cancel job.

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from fusion_mlx.training.service import ADAPTER_BASE_DIR, JobStatus
from fusion_mlx.training.voice import (
    VOICE_BASE_DIR,
    VoiceTrainConfig,
    train_kokoro_voice,
)

logger = logging.getLogger(__name__)

VOICE_JOBS_FILE = ADAPTER_BASE_DIR / "voice_jobs.json"


@dataclass
class VoiceJob:
    job_id: str
    model_id: str
    dataset_dir: str
    voice_id: str
    tenant_id: str = "default"
    config: VoiceTrainConfig = field(default_factory=VoiceTrainConfig)
    status: JobStatus = JobStatus.QUEUED
    created_at: float = field(default_factory=time.time)
    started_at: float = 0.0
    finished_at: float = 0.0
    error: str = ""
    voice_path: str = ""
    metrics: dict = field(default_factory=dict)
    events: list = field(default_factory=list)
    terminal: bool = False
    cancel_requested: bool = False
    cond: asyncio.Condition = field(default_factory=asyncio.Condition, repr=False)

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "kind": "voice",
            "model_id": self.model_id,
            "dataset_dir": self.dataset_dir,
            "voice_id": self.voice_id,
            "tenant_id": self.tenant_id,
            "config": asdict(self.config),
            "status": self.status.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "voice_path": self.voice_path,
            "voice_ref": self.voice_id if self.status == JobStatus.COMPLETED else "",
            "metrics": self.metrics,
        }


class VoiceService:
    def __init__(self):
        self._jobs: dict[str, VoiceJob] = {}
        self._queue: list[str] = []
        self._current_job_id: str | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._running = False
        self._pending_tasks: set = set()
        self._load_jobs()

    def set_loop(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop

    async def shutdown(self):
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
                "VoiceService shutdown: cancelled %d pending task(s)", cancelled
            )

    def create_job(
        self,
        model_id: str,
        dataset_dir: str,
        voice_id: str = "",
        config: VoiceTrainConfig | None = None,
        tenant_id: str = "default",
    ) -> VoiceJob:
        voice_id = voice_id or f"voice-{uuid.uuid4().hex[:8]}"
        if not voice_id.replace("-", "").replace("_", "").isalnum():
            raise ValueError(
                f"voice_id {voice_id!r} must be alphanumeric (dashes/underscores allowed)"
            )
        if (VOICE_BASE_DIR / voice_id / "voice.safetensors").exists():
            raise ValueError(
                f"voice_id {voice_id!r} already exists; delete it or pick another id"
            )
        job = VoiceJob(
            job_id=uuid.uuid4().hex[:12],
            model_id=model_id,
            dataset_dir=str(Path(dataset_dir).expanduser()),
            voice_id=voice_id,
            tenant_id=tenant_id,
            config=config or VoiceTrainConfig(),
        )
        self._jobs[job.job_id] = job
        self._queue.append(job.job_id)
        self._persist_jobs()
        logger.info(
            "voice job created: %s model=%s voice=%s dataset=%s",
            job.job_id,
            model_id,
            voice_id,
            job.dataset_dir,
        )
        return job

    def get_job(self, job_id: str) -> VoiceJob | None:
        return self._jobs.get(job_id)

    def list_jobs(self) -> list[VoiceJob]:
        return list(self._jobs.values())

    def cancel_job(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if job is None or job.terminal:
            return False
        job.cancel_requested = True
        if job.status == JobStatus.QUEUED:
            job.status = JobStatus.CANCELLED
            job.terminal = True
            job.finished_at = time.time()
            if job_id in self._queue:
                self._queue.remove(job_id)
            self._notify_job(job)
            self._persist_jobs()
        return True

    def delete_job(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if job is None or job.status == JobStatus.RUNNING:
            return False
        self._jobs.pop(job_id, None)
        if job_id in self._queue:
            self._queue.remove(job_id)
        self._persist_jobs()
        return True

    def start_processing(self):
        self._running = True
        if self._loop is None:
            try:
                self._loop = asyncio.get_running_loop()
            except RuntimeError:
                self._loop = None
        self._process_queue()

    def _process_queue(self):
        if not self._running or self._current_job_id is not None or not self._queue:
            return
        job_id = self._queue.pop(0)
        job = self._jobs.get(job_id)
        if job is None or job.status != JobStatus.QUEUED:
            self._process_queue()
            return
        if job.cancel_requested:
            job.status = JobStatus.CANCELLED
            job.terminal = True
            job.finished_at = time.time()
            self._persist_jobs()
            self._notify_job(job)
            self._process_queue()
            return

        self._current_job_id = job_id
        job.status = JobStatus.RUNNING
        job.started_at = time.time()
        self._persist_jobs()
        self._notify_job(job)
        if self._loop is None:
            job.status = JobStatus.FAILED
            job.error = "no event loop available for voice training"
            job.terminal = True
            self._current_job_id = None
            self._persist_jobs()
            self._notify_job(job)
            logger.error("voice job %s failed: no event loop", job_id)
            return
        task = asyncio.ensure_future(self._run_job(job), loop=self._loop)
        self._pending_tasks.add(task)
        task.add_done_callback(self._pending_tasks.discard)

    async def _run_job(self, job: VoiceJob):
        try:
            await asyncio.to_thread(self._execute, job)
        except Exception as exc:
            if job.cancel_requested:
                job.status = JobStatus.CANCELLED
                job.error = "cancelled"
                logger.info("voice job %s cancelled", job.job_id)
            else:
                logger.exception("voice job %s failed: %s", job.job_id, exc)
                job.status = JobStatus.FAILED
                job.error = str(exc)
        else:
            job.status = JobStatus.COMPLETED
            logger.info(
                "voice job %s completed: voice=%s path=%s",
                job.job_id,
                job.voice_id,
                job.voice_path,
            )
        finally:
            job.terminal = True
            job.finished_at = time.time()
            self._persist_jobs()
            self._push_event(
                job,
                {
                    "type": "done",
                    "status": job.status.value,
                    "voice_ref": (
                        job.voice_id if job.status == JobStatus.COMPLETED else ""
                    ),
                    "voice_path": job.voice_path,
                    "error": job.error,
                },
            )
            self._current_job_id = None
            self._process_queue()

    def _execute(self, job: VoiceJob):
        def _on_step(step, total, payload):
            self._push_event(
                job,
                {"type": "progress", "step": step, "total": total, **payload},
            )

        result = train_kokoro_voice(
            dataset_dir=job.dataset_dir,
            model_name=job.model_id,
            voice_id=job.voice_id,
            config=job.config,
            tenant_id=job.tenant_id,
            output_dir=VOICE_BASE_DIR / job.voice_id,
            on_step=_on_step,
            should_cancel=lambda: job.cancel_requested,
        )
        job.voice_path = result["voice_path"]
        job.metrics = result["metrics"]

    def _push_event(self, job: VoiceJob, event: dict):
        job.events.append(event)
        if self._loop is not None and self._loop.is_running():
            try:
                asyncio.run_coroutine_threadsafe(self._notify(job), self._loop)
            except RuntimeError:
                pass

    async def _notify(self, job: VoiceJob):
        async with job.cond:
            job.cond.notify_all()

    def _notify_job(self, job: VoiceJob):
        if self._loop is None or not self._loop.is_running():
            return
        try:
            asyncio.run_coroutine_threadsafe(self._notify(job), self._loop)
        except RuntimeError:
            pass

    def _persist_jobs(self):
        ADAPTER_BASE_DIR.mkdir(parents=True, exist_ok=True)
        data = [job.to_dict() for job in self._jobs.values()]
        try:
            VOICE_JOBS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.warning("failed to persist voice jobs: %s", exc)

    def _load_jobs(self):
        if not VOICE_JOBS_FILE.exists():
            return
        try:
            data = json.loads(VOICE_JOBS_FILE.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("failed to load voice jobs: %s", exc)
            return
        for item in data:
            try:
                status = JobStatus(item.get("status", "queued"))
                if status in (JobStatus.RUNNING, JobStatus.QUEUED):
                    status = JobStatus.CANCELLED
                job = VoiceJob(
                    job_id=item.get("job_id", uuid.uuid4().hex[:12]),
                    model_id=item.get("model_id", ""),
                    dataset_dir=item.get("dataset_dir", ""),
                    voice_id=item.get("voice_id", ""),
                    tenant_id=item.get("tenant_id", "default"),
                    config=VoiceTrainConfig(**item.get("config", {})),
                    status=status,
                    created_at=item.get("created_at", 0.0),
                    started_at=item.get("started_at", 0.0),
                    finished_at=item.get("finished_at", 0.0),
                    error=item.get("error", ""),
                    voice_path=item.get("voice_path", ""),
                    metrics=item.get("metrics", {}),
                )
                job.terminal = status in (
                    JobStatus.COMPLETED,
                    JobStatus.FAILED,
                    JobStatus.CANCELLED,
                )
                self._jobs[job.job_id] = job
            except Exception as exc:
                logger.warning("skipping malformed voice job: %s", exc)


def list_trained_voices() -> list[dict]:
    base = VOICE_BASE_DIR
    if not base.is_dir():
        return []
    out = []
    for manifest_path in sorted(base.glob("*/manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("voice manifest unreadable (%s): %s", manifest_path, exc)
            continue
        voice_file = manifest_path.parent / "voice.safetensors"
        manifest["path"] = str(voice_file)
        manifest["ready"] = voice_file.is_file()
        out.append(manifest)
    return out


def resolve_voice_ref(voice_ref: str) -> str | None:
    if (
        not voice_ref
        or "/" in voice_ref
        or "\\" in voice_ref
        or voice_ref.startswith(".")
    ):
        return None
    candidate = VOICE_BASE_DIR / voice_ref / "voice.safetensors"
    if candidate.is_file():
        return str(candidate)
    return None
