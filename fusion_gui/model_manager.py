"""
Model management bridge — MLX-GUI queue/DB tracking over fusion-mlx EnginePool.
"""

import asyncio
import logging
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, Any, List, Callable

import mlx.core as mx
from sqlalchemy.orm import Session

from fusion_gui.database import get_database_manager, get_db_session
from fusion_gui.models import Model, ModelStatus, InferenceRequest, RequestQueue, QueueStatus

logger = logging.getLogger(__name__)


class LoadingStatus(Enum):
    IDLE = "idle"
    LOADING = "loading"
    LOADED = "loaded"
    UNLOADING = "unloading"
    FAILED = "failed"


@dataclass
class LoadedModel:
    model_id: str
    loaded_at: datetime
    last_used_at: datetime
    memory_usage_gb: float

    def update_last_used(self):
        self.last_used_at = datetime.utcnow()


@dataclass
class LoadRequest:
    model_name: str
    model_path: str
    priority: int = 0
    requester_id: str = "system"
    callback: Optional[Callable] = None
    created_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()


class ModelLoadingQueue:
    """Thread-safe priority queue for model loading requests."""

    def __init__(self):
        self._queue: List[LoadRequest] = []
        self._lock = threading.Lock()
        self._event = threading.Event()

    def add_request(self, request: LoadRequest) -> int:
        with self._lock:
            self._queue.append(request)
            self._queue.sort(key=lambda x: (-x.priority, x.created_at))
            position = self._queue.index(request)
            self._event.set()
            return position

    def get_next_request(self, timeout: Optional[float] = None) -> Optional[LoadRequest]:
        if timeout:
            self._event.wait(timeout)
        with self._lock:
            if self._queue:
                request = self._queue.pop(0)
                if not self._queue:
                    self._event.clear()
                return request
            return None

    def remove_request(self, model_name: str) -> bool:
        with self._lock:
            for i, request in enumerate(self._queue):
                if request.model_name == model_name:
                    self._queue.pop(i)
                    return True
            return False

    def get_queue_status(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [
                {
                    "model_name": req.model_name,
                    "priority": req.priority,
                    "requester_id": req.requester_id,
                    "created_at": req.created_at.isoformat(),
                    "position": i,
                }
                for i, req in enumerate(self._queue)
            ]

    def size(self) -> int:
        with self._lock:
            return len(self._queue)


class ModelManager:
    """
    Bridges MLX-GUI model tracking (SQLite DB + loading queue)
    with fusion-mlx EnginePool as the actual load/unload backend.
    """

    def __init__(self, max_concurrent_models: int = 3):
        self.max_concurrent_models = self._get_max_concurrent_models(max_concurrent_models)

        # Model metadata cache (sync with DB)
        self._loaded_models: Dict[str, LoadedModel] = {}
        self._loading_status: Dict[str, LoadingStatus] = {}
        self._lock = threading.RLock()

        # Queue system
        self._loading_queue = ModelLoadingQueue()
        self._queue_worker_thread = None
        self._cleanup_worker_thread = None
        self._queue_worker_running = False
        self._shutdown_requested = False

        # EnginePool backend (lazy-init)
        self._engine_pool = None
        self._engine_pool_lock = threading.Lock()

        # Auto-unload settings
        self._auto_unload_enabled = True
        self._inactivity_timeout = self._get_inactivity_timeout()
        self._cleanup_interval = 60

        import atexit
        atexit.register(self._force_cleanup)

    # ── EnginePool backend ──

    def _get_engine_pool(self):
        if self._engine_pool is None:
            with self._engine_pool_lock:
                if self._engine_pool is None:
                    from fusion_mlx.pool.engine_pool import EnginePool
                    self._engine_pool = EnginePool()
        return self._engine_pool

    # ── Settings ──

    def _get_max_concurrent_models(self, default_value: int) -> int:
        try:
            db_manager = get_database_manager()
            return db_manager.get_setting("max_concurrent_models", default_value)
        except Exception as e:
            logger.warning(f"Failed to read max concurrent models from database: {e}")
            return default_value

    def _get_inactivity_timeout(self) -> timedelta:
        try:
            db_manager = get_database_manager()
            timeout_minutes = db_manager.get_setting("model_inactivity_timeout_minutes", 5)
            return timedelta(minutes=timeout_minutes)
        except Exception as e:
            logger.warning(f"Failed to read inactivity timeout from database: {e}")
            return timedelta(minutes=5)

    # ── Background workers ──

    def _start_queue_worker(self):
        if not self._queue_worker_running:
            self._queue_worker_running = True
            self._queue_worker_thread = threading.Thread(
                target=self._queue_worker,
                name="model_loader_queue",
                daemon=True,
            )
            self._queue_worker_thread.start()

    def _start_cleanup_worker(self):
        if self._cleanup_worker_thread is None or not self._cleanup_worker_thread.is_alive():
            self._cleanup_worker_thread = threading.Thread(
                target=self._cleanup_worker,
                name="model_cleanup",
                daemon=True,
            )
            self._cleanup_worker_thread.start()

    def _queue_worker(self):
        logger.info("Model loading queue worker started")
        while self._queue_worker_running:
            try:
                request = self._loading_queue.get_next_request(timeout=5.0)
                if request:
                    logger.info(f"Processing load request for {request.model_name}")
                    try:
                        self._load_model_sync(request)
                    except Exception as e:
                        logger.error(f"Error loading model {request.model_name}: {e}")
                        self._set_loading_status(request.model_name, LoadingStatus.FAILED)
            except Exception as e:
                logger.error(f"Queue worker error: {e}")
                time.sleep(1)

    def _cleanup_worker(self):
        logger.info("Model cleanup worker started")
        while not self._shutdown_requested:
            try:
                if self._auto_unload_enabled:
                    self._cleanup_inactive_models()
                for _ in range(self._cleanup_interval):
                    if self._shutdown_requested:
                        break
                    time.sleep(1)
            except Exception as e:
                logger.error(f"Cleanup worker error: {e}")
                time.sleep(1)
        logger.info("Model cleanup worker stopped")

    # ── Lifecycle ──

    def shutdown(self):
        logger.info("Shutting down model manager...")
        self._shutdown_requested = True
        self._queue_worker_running = False
        with self._lock:
            model_names = list(self._loaded_models.keys())
        for model_name in model_names:
            self.unload_model(model_name)
        logger.info("Model manager shutdown complete")

    def _force_cleanup(self):
        self._shutdown_requested = True
        self._queue_worker_running = False
        logger.info("Model manager cleanup completed")

    def _cleanup_inactive_models(self):
        cutoff_time = datetime.utcnow() - self._inactivity_timeout
        models_to_unload = []
        with self._lock:
            for model_name, loaded_model in self._loaded_models.items():
                if loaded_model.last_used_at < cutoff_time:
                    models_to_unload.append(model_name)
        for model_name in models_to_unload:
            logger.info(f"Auto-unloading inactive model: {model_name}")
            self.unload_model(model_name)

    # ── Status tracking ──

    def _set_loading_status(self, model_name: str, status: LoadingStatus):
        with self._lock:
            self._loading_status[model_name] = status
        try:
            db_manager = get_database_manager()
            with db_manager.get_session() as session:
                model = session.query(Model).filter(Model.name == model_name).first()
                if model:
                    if status == LoadingStatus.LOADING:
                        model.status = ModelStatus.LOADING.value
                    elif status == LoadingStatus.LOADED:
                        model.status = ModelStatus.LOADED.value
                        model.last_used_at = datetime.utcnow()
                    elif status == LoadingStatus.FAILED:
                        model.status = ModelStatus.FAILED.value
                    else:
                        model.status = ModelStatus.UNLOADED.value
                    session.commit()
        except Exception as e:
            logger.error(f"Error updating model status in database: {e}")

    # ── Memory helpers ──

    def _calculate_actual_memory_usage(self, model_path: str) -> float:
        total_size_bytes = 0
        try:
            actual_path = self._resolve_model_path(model_path)
            for root, dirs, files in os.walk(actual_path):
                for file in files:
                    if file.endswith(('.safetensors', '.bin', '.pth', '.pt', '.gguf', '.npz')):
                        file_path = os.path.join(root, file)
                        if os.path.exists(file_path):
                            total_size_bytes += os.path.getsize(file_path)
            file_size_gb = total_size_bytes / (1024**3)
            if "whisper" in model_path.lower() or "parakeet" in model_path.lower():
                overhead_multiplier = 1.15
            else:
                overhead_multiplier = 1.25
            actual_memory_gb = round(file_size_gb * overhead_multiplier, 1)
            logger.info(
                f"Model {model_path} -> {actual_path} file size: {file_size_gb:.1f}GB, "
                f"with overhead: {actual_memory_gb:.1f}GB"
            )
            return max(actual_memory_gb, 0.1)
        except Exception as e:
            logger.warning(f"Could not calculate actual memory usage for {model_path}: {e}")
            return 2.0

    def _resolve_model_path(self, model_path: str) -> str:
        if os.path.exists(model_path):
            return model_path
        if "/" in model_path and not os.path.exists(model_path):
            cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "fusion-mlx")
            cache_name = "models--" + model_path.replace("/", "--")
            cache_path = os.path.join(cache_dir, cache_name)
            if os.path.exists(cache_path):
                snapshots_dir = os.path.join(cache_path, "snapshots")
                if os.path.exists(snapshots_dir):
                    snapshot_dirs = [
                        d for d in os.listdir(snapshots_dir)
                        if os.path.isdir(os.path.join(snapshots_dir, d))
                    ]
                    if snapshot_dirs:
                        return os.path.join(snapshots_dir, snapshot_dirs[0])
        return model_path

    def _check_memory_constraints(self, required_memory_gb: float) -> tuple[bool, str]:
        if len(self._loaded_models) >= self.max_concurrent_models:
            return False, f"Maximum concurrent models ({self.max_concurrent_models}) already loaded"
        current_model_memory = sum(m.memory_usage_gb for m in self._loaded_models.values())
        try:
            import psutil
            total_gb = psutil.virtual_memory().total / (1024**3)
        except ImportError:
            total_gb = 16.0
        max_allowed_memory = total_gb * 0.8
        if current_model_memory + required_memory_gb > max_allowed_memory:
            warning_msg = (
                f"Memory warning: Loading this model ({required_memory_gb:.1f}GB) with "
                f"current models ({current_model_memory:.1f}GB) may exceed recommended "
                f"({max_allowed_memory:.1f}GB of {total_gb:.1f}GB total)"
            )
            return True, warning_msg
        return True, ""

    # ── Loading (sync worker calls async EnginePool) ──

    def _load_model_sync(self, request: LoadRequest):
        model_name = request.model_name
        model_path = request.model_path
        try:
            self._set_loading_status(model_name, LoadingStatus.LOADING)
            with self._lock:
                if model_name in self._loaded_models:
                    logger.info(f"Model {model_name} already loaded")
                    self._loaded_models[model_name].update_last_used()
                    return

            db_manager = get_database_manager()
            with db_manager.get_session() as session:
                model_record = session.query(Model).filter(Model.name == model_name).first()
                if not model_record:
                    raise ValueError(f"Model {model_name} not found in database")
                self._ensure_capacity_for_model(model_record.memory_required_gb)
                can_load, memory_warning = self._check_memory_constraints(model_record.memory_required_gb)
                if not can_load:
                    raise RuntimeError(memory_warning)
                elif memory_warning:
                    logger.warning(f"Loading model {model_name} with memory warning: {memory_warning}")

            logger.info(f"Loading model from {model_path} via EnginePool")
            mlx_wrapper = self._run_async(self._load_model_via_pool(model_name, model_path))

            actual_memory = self._calculate_actual_memory_usage(model_path)
            db_manager = get_database_manager()
            with db_manager.get_session() as session:
                model_record = session.query(Model).filter(Model.name == model_name).first()
                if model_record:
                    model_record.memory_required_gb = actual_memory
                    session.commit()
                    logger.info(f"Updated {model_name} memory requirement in DB: {actual_memory:.1f}GB")

            loaded_model = LoadedModel(
                model_id=model_name,
                loaded_at=datetime.utcnow(),
                last_used_at=datetime.utcnow(),
                memory_usage_gb=actual_memory,
            )
            with self._lock:
                self._loaded_models[model_name] = loaded_model
            self._set_loading_status(model_name, LoadingStatus.LOADED)
            logger.info(f"Successfully loaded model {model_name}")
            if request.callback:
                request.callback(model_name, True, None)
        except Exception as e:
            error_msg = f"Failed to load model {model_name}: {e}"
            logger.error(error_msg)
            self._set_loading_status(model_name, LoadingStatus.FAILED)
            try:
                db_manager = get_database_manager()
                with db_manager.get_session() as session:
                    model_record = session.query(Model).filter(Model.name == model_name).first()
                    if model_record:
                        model_record.error_message = str(e)
                        session.commit()
            except Exception as db_error:
                logger.error(f"Error updating database with error message: {db_error}")
            if request.callback:
                request.callback(model_name, False, str(e))

    async def _load_model_via_pool(self, model_name: str, model_path: str):
        pool = self._get_engine_pool()
        try:
            engine = await pool.get_engine(model_name)
            return engine
        except Exception as e:
            logger.error(f"EnginePool failed to load {model_name}: {e}")
            raise

    @staticmethod
    def _run_async(coro):
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    # ── Capacity management ──

    def _ensure_capacity_for_model(self, required_memory_gb: float):
        can_load, message = self._check_memory_constraints(required_memory_gb)
        while not can_load and "Maximum concurrent models" in message:
            if not self._loaded_models:
                break
            lru_model_name = min(
                self._loaded_models.keys(),
                key=lambda x: self._loaded_models[x].last_used_at,
            )
            logger.info(f"Unloading LRU model {lru_model_name} to make space")
            self.unload_model(lru_model_name)
            can_load, message = self._check_memory_constraints(required_memory_gb)
            if len(self._loaded_models) == 0:
                break

    def _free_memory_for_model(self, required_memory_gb: float) -> bool:
        if not self._loaded_models:
            return False
        lru_model_name = min(
            self._loaded_models.keys(),
            key=lambda x: self._loaded_models[x].last_used_at,
        )
        logger.info(
            f"Freeing memory: Unloading LRU model '{lru_model_name}' "
            f"({required_memory_gb:.1f}GB required)"
        )
        success = self.unload_model(lru_model_name)
        if success:
            import gc
            gc.collect()
            try:
                mx.clear_cache()
            except AttributeError:
                mx.metal.clear_cache()
        return success

    def _is_memory_error(self, error: Exception) -> bool:
        error_str = str(error).lower()
        error_type = type(error).__name__
        memory_indicators = [
            "out of memory", "memory allocation", "insufficient memory",
            "memory error", "cuda out of memory", "device out of memory",
            "metal out of memory", "failed to allocate", "allocation failed",
            "not enough memory", "memory limit exceeded",
        ]
        if any(indicator in error_str for indicator in memory_indicators):
            return True
        if error_type in ["OutOfMemoryError", "MemoryError"]:
            return True
        if error_type == "RuntimeError" and any(
            indicator in error_str for indicator in memory_indicators
        ):
            return True
        return False

    # ── Public async API ──

    async def load_model_async(self, model_name: str, model_path: str, priority: int = 0) -> bool:
        with self._lock:
            if model_name in self._loaded_models:
                self._loaded_models[model_name].update_last_used()
                return True
            if self._loading_status.get(model_name) == LoadingStatus.LOADING:
                while self._loading_status.get(model_name) == LoadingStatus.LOADING:
                    await asyncio.sleep(0.1)
                return model_name in self._loaded_models

        if not self._queue_worker_running:
            self._start_queue_worker()
            self._start_cleanup_worker()

        request = LoadRequest(model_name=model_name, model_path=model_path, priority=priority)
        position = self._loading_queue.add_request(request)
        logger.info(f"Added {model_name} to loading queue at position {position}")

        while True:
            await asyncio.sleep(0.5)
            status = self._loading_status.get(model_name, LoadingStatus.IDLE)
            if status == LoadingStatus.LOADED:
                return True
            elif status == LoadingStatus.FAILED:
                return False
            elif status == LoadingStatus.IDLE:
                if not any(req.model_name == model_name for req in self._loading_queue._queue):
                    return False

    def unload_model(self, model_name: str) -> bool:
        try:
            with self._lock:
                if model_name not in self._loaded_models:
                    logger.warning(f"Model {model_name} not loaded")
                    return False
                self._set_loading_status(model_name, LoadingStatus.UNLOADING)

            pool = self._get_engine_pool()
            self._run_async(pool._unload_engine(model_name))

            with self._lock:
                self._loaded_models.pop(model_name, None)
                self._loading_status.pop(model_name, None)

            self._set_loading_status(model_name, LoadingStatus.IDLE)
            logger.info(f"Unloaded model {model_name}")
            return True
        except Exception as e:
            logger.error(f"Error unloading model {model_name}: {e}")
            return False

    # ── Queries ──

    def get_loaded_models(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return {
                name: {
                    "loaded_at": model.loaded_at.isoformat(),
                    "last_used_at": model.last_used_at.isoformat(),
                    "memory_usage_gb": model.memory_usage_gb,
                }
                for name, model in self._loaded_models.items()
            }

    def get_model_status(self, model_name: str) -> Dict[str, Any]:
        with self._lock:
            status = self._loading_status.get(model_name, LoadingStatus.IDLE)
            loaded_model = self._loaded_models.get(model_name)
            result = {
                "name": model_name,
                "status": status.value,
                "loaded": loaded_model is not None,
                "loaded_at": loaded_model.loaded_at.isoformat() if loaded_model else None,
                "last_used_at": loaded_model.last_used_at.isoformat() if loaded_model else None,
                "memory_usage_gb": loaded_model.memory_usage_gb if loaded_model else 0,
                "queue_position": next(
                    (i for i, req in enumerate(self._loading_queue._queue) if req.model_name == model_name),
                    None,
                ),
            }
        return result

    def refresh_settings(self):
        self.max_concurrent_models = self._get_max_concurrent_models(3)
        self._inactivity_timeout = self._get_inactivity_timeout()

    def get_system_status(self) -> Dict[str, Any]:
        self.refresh_settings()
        try:
            import psutil
            vm = psutil.virtual_memory()
            total_gb = vm.total / (1024**3)
            available_gb = vm.available / (1024**3)
        except ImportError:
            total_gb, available_gb = 16.0, 8.0
        with self._lock:
            total_model_memory = sum(m.memory_usage_gb for m in self._loaded_models.values())
        return {
            "loaded_models_count": len(self._loaded_models),
            "max_concurrent_models": self.max_concurrent_models,
            "queue_size": self._loading_queue.size(),
            "total_model_memory_gb": total_model_memory,
            "system_memory_total_gb": total_gb,
            "system_memory_available_gb": available_gb,
            "memory_usage_percent": (total_model_memory / total_gb) * 100 if total_gb else 0,
            "auto_unload_enabled": self._auto_unload_enabled,
            "inactivity_timeout_minutes": self._inactivity_timeout.total_seconds() / 60,
        }

    def get_model_for_inference(self, model_name: str) -> Optional[LoadedModel]:
        with self._lock:
            loaded_model = self._loaded_models.get(model_name)
            if loaded_model:
                loaded_model.update_last_used()
            return loaded_model

    # ── Inference delegation to EnginePool ──

    async def generate_text(self, model_name: str, prompt: str, config: Any) -> Any:
        with self._lock:
            if model_name not in self._loaded_models:
                raise ValueError(f"Model {model_name} is not loaded")
            self._loaded_models[model_name].update_last_used()
        pool = self._get_engine_pool()
        engine = await pool.get_engine(model_name)
        result = await engine.generate(prompt, config)
        self._increment_use_count(model_name)
        return result

    async def generate_text_stream(self, model_name: str, prompt: str, config: Any):
        with self._lock:
            if model_name not in self._loaded_models:
                raise ValueError(f"Model {model_name} is not loaded")
            self._loaded_models[model_name].update_last_used()
        pool = self._get_engine_pool()
        engine = await pool.get_engine(model_name)
        async for token in engine.generate_stream(prompt, config):
            yield token
        self._increment_use_count(model_name)

    @staticmethod
    def _increment_use_count(model_name: str):
        try:
            db_manager = get_database_manager()
            with db_manager.get_session() as session:
                model_record = session.query(Model).filter(Model.name == model_name).first()
                if model_record:
                    model_record.increment_use_count()
                    session.commit()
        except Exception as e:
            logger.error(f"Error updating model usage: {e}")


# ── Global singleton ──

_model_manager: Optional[ModelManager] = None


def get_model_manager() -> ModelManager:
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager


def shutdown_model_manager():
    global _model_manager
    if _model_manager:
        _model_manager.shutdown()
        _model_manager = None