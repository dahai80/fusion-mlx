# SPDX-License-Identifier: Apache-2.0
"""
Engine pool for FusionMLX multi-model serving.

This module manages multiple model engines with LRU-based eviction
when memory limits are exceeded. It supports:

- Pre-load memory checking to ensure models fit before loading
- LRU eviction of least recently used models
- Model pinning to keep specific models always loaded
- BatchedEngine for all LLM models (continuous batching)
"""

from __future__ import annotations

import asyncio
import gc
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .model_settings import ModelSettingsManager

import mlx.core as mx

from ..engine_core import get_mlx_executor
from ..engines.base import BaseEngine
from ..engines.batched import BatchedEngine
from ..engines.embedding import EmbeddingEngine
from ..engines.image_gen import ImageGenEngine
from ..engines.reranker import RerankerEngine
from ..engines.sts import STSEngine
from ..engines.stt import STTEngine
from ..engines.tts import TTSEngine
from ..engines.video import VideoGenEngine
from ..engines.vlm import VLMBatchedEngine
from ..exceptions import (
    AdapterPathError,
    InsufficientMemoryError,
    ModelBusyError,
    ModelLoadingError,
    ModelNotFoundError,
    ModelTooLargeError,
)
from ..scheduler import SchedulerConfig
from ..utils.proc_memory import get_phys_footprint
from .model_discovery import DiscoveredModel, discover_models, format_size

logger = logging.getLogger(__name__)


@dataclass
class EngineEntry:
    """Per-model state in the engine pool."""

    model_id: str  # Directory name (e.g., "llama-3b")
    model_path: str  # Full path to model directory
    model_type: Literal[
        "llm",
        "vlm",
        "embedding",
        "reranker",
        "audio_stt",
        "audio_tts",
        "audio_sts",
        "image",
        "video",
    ]  # Model type
    engine_type: Literal[
        "batched",
        "simple",
        "embedding",
        "reranker",
        "vlm",
        "audio_stt",
        "audio_tts",
        "audio_sts",
        "image_gen",
        "video_gen",
    ]  # Engine type to use
    estimated_size: int  # Pre-calculated from safetensors (bytes)
    actual_size: int | None = None  # Observed process-memory delta after load settles
    config_model_type: str = (
        ""  # Raw model_type from config.json (e.g., "deepseekocr_2")
    )
    thinking_default: bool | None = (
        None  # True if model thinks by default, False if not, None if unknown
    )
    preserve_thinking_default: bool | None = (
        None  # True when template supports preserve_thinking (Qwen 3.6+)
    )
    model_context_length: int | None = (
        None  # Declared context length from config.json (None if unknown)
    )
    source_type: str = "local"
    source_repo_id: str | None = None
    engine: (
        BaseEngine
        | EmbeddingEngine
        | RerankerEngine
        | STTEngine
        | STSEngine
        | TTSEngine
        | VideoGenEngine
        | None
    ) = None  # Loaded engine instance
    last_access: float = 0.0  # Timestamp for LRU (0 if never loaded)
    is_loading: bool = False  # Prevent concurrent loads
    loading_started_at: float | None = None  # Timestamp when current load started
    loading_event: asyncio.Event | None = None  # Signaled when loading completes
    is_pinned: bool = False  # Never evict if True
    abort_loading: bool = False  # Set by memory enforcer to abort in-progress load
    in_use: int = 0  # in-flight acquire/use lease count; never evict while > 0
    abort_requested: bool = False  # Set under hard pressure for leased requests
    pending_unload_reason: str | None = None  # Unload as soon as leases/activity drain
    runtime_settings_signature: tuple[tuple[str, str], ...] | None = None
    adapter_path: str | None = None  # LoRA adapter path for derived adapter entries
    base_model_id: str | None = None  # Base model_id for derived adapter entries


class EnginePool:
    """
    Manages multiple model engines with LRU-based memory management.

    Features:
    - Pre-load memory checking (evict before load, not after)
    - LRU eviction when memory limit is exceeded
    - Model pinning to prevent eviction
    - Automatic engine type selection based on model type
    """

    def __init__(
        self,
        scheduler_config: SchedulerConfig | None = None,
    ):
        """
        Initialize the engine pool.

        Args:
            scheduler_config: Configuration for BatchedEngine schedulers

        Note:
            Pre-load admission consults `enforcer.get_final_ceiling()` via
            the `_get_final_ceiling` callback set by `server.init_server()`.
            Until the callback is wired up the pool admits unconditionally.
        """
        self._entries: dict[str, EngineEntry] = {}
        self._lock = asyncio.Lock()
        self._current_model_memory = 0
        self._scheduler_config = scheduler_config or SchedulerConfig()
        self._process_memory_enforcer: object | None = None  # Set by server
        self._get_final_ceiling: object | None = None  # Set by server
        self._settings_manager: object | None = None  # Set by server
        self._suppress_ttl: bool = False  # Suppress TTL during benchmarks
        self._max_adapter_engines: int = int(
            os.getenv("FUSION_MAX_ADAPTER_ENGINES", "4")
        )
        self._allowed_adapter_dirs: list[str] = self._resolve_allowed_adapter_dirs()
        self._load_seconds_per_gb_ema: float | None = None
        self._load_time_observations: int = 0
        self.configure_hot_cache_budget()

    @property
    def current_model_memory(self) -> int:
        """Current memory used by loaded models in bytes."""
        return self._current_model_memory

    def configure_hot_cache_budget(self) -> None:
        """Ensure loaded schedulers share one process-wide hot cache budget."""
        hot_max = int(getattr(self._scheduler_config, "hot_cache_max_size", 0) or 0)
        if hot_max <= 0:
            self._scheduler_config.hot_cache_budget = None
            return

        current = getattr(self._scheduler_config, "hot_cache_budget", None)
        if current is not None and getattr(current, "max_bytes", None) == hot_max:
            return

        from ..cache.paged_ssd_cache import SharedHotCacheBudget

        self._scheduler_config.hot_cache_budget = SharedHotCacheBudget(hot_max)

    def _current_ceiling(self) -> int:
        """Resolve the current memory ceiling via the enforcer callback.

        Returns 0 when no callback is wired up (treated by callers as
        "no limit").
        """
        cb = self._get_final_ceiling
        if cb is None:
            return 0
        try:
            return int(cb())
        except Exception:  # noqa: BLE001
            return 0

    def _wake_process_memory_enforcer(self, *, active: bool = False) -> None:
        enforcer = self._process_memory_enforcer
        wake = getattr(enforcer, "wake", None) if enforcer is not None else None
        if callable(wake):
            wake(active=active)

    @staticmethod
    def _canonical_signature_value(value: object) -> str:
        if isinstance(value, (dict, list, tuple)):
            return json.dumps(value, sort_keys=True, separators=(",", ":"))
        return repr(value)

    def _engine_runtime_signature(
        self,
        model_id: str,
        runtime_settings: object | None = None,
        *,
        loaded_engine: object | None = None,
    ) -> tuple[tuple[str, str], ...] | None:
        entry = self._entries.get(model_id)
        settings_id = entry.base_model_id if entry and entry.base_model_id else model_id
        settings = runtime_settings
        if settings is None and self._settings_manager is not None:
            get_settings = getattr(self._settings_manager, "get_settings", None)
            if callable(get_settings):
                settings = get_settings(settings_id)
        if settings is None:
            return None

        to_dict = getattr(settings, "to_dict", None)
        data = to_dict() if callable(to_dict) else {}
        is_diffusion = bool(entry and self._entry_is_diffusion_model(entry))
        loaded_engine_name = (
            type(loaded_engine).__name__ if loaded_engine is not None else None
        )

        def has_value(key: str) -> bool:
            value = data.get(key)
            return value is not None and value != ""

        def normalized_index_cache_freq() -> int | None:
            value = data.get("index_cache_freq")
            try:
                freq = int(value) if value is not None else None
            except (TypeError, ValueError):
                return None
            return freq if freq is not None and freq >= 2 else None

        signature: list[tuple[str, str]] = []

        def add(key: str, value: object) -> None:
            signature.append((key, self._canonical_signature_value(value)))

        add("trust_remote_code", bool(data.get("trust_remote_code", False)))
        add("index_cache_freq", normalized_index_cache_freq())

        mtp_active = bool(data.get("mtp_enabled", False))
        add("mtp_enabled", mtp_active)

        turboquant_active = bool(data.get("turboquant_kv_enabled", False))
        add("turboquant_kv_enabled", turboquant_active)
        if turboquant_active:
            add("turboquant_kv_bits", data.get("turboquant_kv_bits", 4))
            add("turboquant_skip_last", data.get("turboquant_skip_last", True))

        specprefill_active = bool(data.get("specprefill_enabled", False)) and has_value(
            "specprefill_draft_model"
        )
        add("specprefill_enabled", specprefill_active)
        if specprefill_active:
            add("specprefill_draft_model", data.get("specprefill_draft_model"))
            add("specprefill_keep_pct", data.get("specprefill_keep_pct", 0.2))
            add("specprefill_threshold", data.get("specprefill_threshold"))

        dflash_active = (
            bool(data.get("dflash_enabled", False))
            and has_value("dflash_draft_model")
            and not is_diffusion
        )
        if loaded_engine_name is not None:
            dflash_active = loaded_engine_name == "DFlashEngine"
        add("dflash_enabled", dflash_active)
        if dflash_active:
            add("dflash_draft_model", data.get("dflash_draft_model"))
            add(
                "dflash_draft_quant_enabled",
                bool(data.get("dflash_draft_quant_enabled", False)),
            )
            if data.get("dflash_draft_quant_enabled", False):
                add(
                    "dflash_draft_quant_weight_bits",
                    data.get("dflash_draft_quant_weight_bits", 4),
                )
                add(
                    "dflash_draft_quant_activation_bits",
                    data.get("dflash_draft_quant_activation_bits", 16),
                )
                add(
                    "dflash_draft_quant_group_size",
                    data.get("dflash_draft_quant_group_size", 64),
                )
            add("dflash_max_ctx", data.get("dflash_max_ctx"))
            add("dflash_in_memory_cache", data.get("dflash_in_memory_cache", True))
            add(
                "dflash_in_memory_cache_max_entries",
                data.get("dflash_in_memory_cache_max_entries", 4),
            )
            add(
                "dflash_in_memory_cache_max_bytes",
                data.get("dflash_in_memory_cache_max_bytes"),
            )
            add("dflash_ssd_cache", bool(data.get("dflash_ssd_cache", False)))
            if data.get("dflash_ssd_cache", False):
                add(
                    "dflash_ssd_cache_max_bytes", data.get("dflash_ssd_cache_max_bytes")
                )
            add("dflash_draft_window_size", data.get("dflash_draft_window_size"))
            add("dflash_draft_sink_size", data.get("dflash_draft_sink_size"))
            add("dflash_verify_mode", data.get("dflash_verify_mode"))

        vlm_mtp_active = bool(data.get("vlm_mtp_enabled", False)) and has_value(
            "vlm_mtp_draft_model"
        )
        if loaded_engine is not None and vlm_mtp_active:
            drafter = getattr(loaded_engine, "vlm_mtp_drafter", None)
            if callable(drafter):
                drafter = drafter()
            vlm_mtp_active = drafter is not None
        add("vlm_mtp_enabled", vlm_mtp_active)
        if vlm_mtp_active:
            add("vlm_mtp_draft_model", data.get("vlm_mtp_draft_model"))
            add("vlm_mtp_draft_block_size", data.get("vlm_mtp_draft_block_size"))

        return tuple(signature)

    def list_models(self) -> list[str]:
        """Return list of all discovered model IDs (loaded or not)."""
        return list(self._entries.keys())

    @property
    def model_count(self) -> int:
        """Total number of discovered models."""
        return len(self._entries)

    @property
    def loaded_model_count(self) -> int:
        """Number of currently loaded models."""
        return sum(1 for e in self._entries.values() if e.engine is not None)

    async def apply_embedding_batch_size(self, batch_size: int) -> None:
        """Apply embedding batch size to future and currently loaded embedding engines."""
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError("embedding batch size must be > 0")

        async with self._lock:
            self._scheduler_config.embedding_batch_size = batch_size
            for entry in list(self._entries.values()):
                engine = entry.engine if entry is not None else None
                if isinstance(engine, EmbeddingEngine):
                    engine._batch_size = batch_size

    def _scan_models(self, model_dirs: str | list[str]) -> dict[str, DiscoveredModel]:
        # Executor-safe filesystem scan: walks model_dirs and reads each
        # model's config. Pure (no self mutation) so it can run off the
        # event loop via run_in_executor (#59).
        from pathlib import Path

        from .model_discovery import discover_models_from_dirs

        if isinstance(model_dirs, str):
            dirs = [Path(model_dirs)]
        else:
            dirs = [Path(d) for d in model_dirs]

        if len(dirs) == 1:
            return discover_models(dirs[0])
        return discover_models_from_dirs(dirs)

    def _merge_discovered(
        self,
        discovered: dict[str, DiscoveredModel],
        pinned_models: list[str] | None = None,
    ) -> None:
        # Mutates self._entries from a scan result. Must run on the loop
        # thread (or under the pool lock) since it touches shared state.
        pinned_set = set(pinned_models or [])

        for model_id, info in discovered.items():
            existing = self._entries.get(model_id)
            if existing is not None and existing.engine is not None:
                # Loaded model: preserve runtime state, only update pinned flag
                existing.is_pinned = model_id in pinned_set
            else:
                # New or unloaded model: create fresh entry
                self._entries[model_id] = EngineEntry(
                    model_id=model_id,
                    model_path=info.model_path,
                    model_type=info.model_type,
                    engine_type=info.engine_type,
                    estimated_size=info.estimated_size,
                    config_model_type=getattr(info, "config_model_type", ""),
                    thinking_default=getattr(info, "thinking_default", None),
                    preserve_thinking_default=getattr(
                        info, "preserve_thinking_default", None
                    ),
                    model_context_length=getattr(info, "model_context_length", None),
                    source_type=getattr(info, "source_type", "local"),
                    source_repo_id=getattr(info, "source_repo_id", None),
                    is_pinned=model_id in pinned_set,
                )

            if model_id in pinned_set:
                logger.info(f"Pinned model: {model_id}")

        # Remove entries no longer discovered and not loaded
        discovered_ids = set(discovered.keys())
        stale = [
            mid
            for mid in self._entries
            if mid not in discovered_ids and self._entries[mid].engine is None
        ]
        for mid in stale:
            del self._entries[mid]

        # Warn about pinned models not found
        found_models = set(self._entries.keys())
        for model_id in pinned_set:
            if model_id not in found_models:
                logger.warning(f"Pinned model not found: {model_id}")

        logger.info(f"Discovered {len(self._entries)} models")

    def discover_models(
        self, model_dirs: str | list[str], pinned_models: list[str] | None = None
    ) -> None:
        # Synchronous discover: scan + merge in one call (back-compat for
        # sync callers and tests). Async callers should use
        # discover_models_async to avoid blocking the event loop (#59).
        discovered = self._scan_models(model_dirs)
        self._merge_discovered(discovered, pinned_models)

    async def discover_models_async(
        self, model_dirs: str | list[str], pinned_models: list[str] | None = None
    ) -> None:
        # Non-blocking discover: run the filesystem scan in an executor so
        # the event loop stays responsive, then merge on the loop thread.
        # Lock the merge to prevent concurrent discover_models_async calls
        # from racing on self._entries mutation (#59).
        loop = asyncio.get_running_loop()
        discovered = await loop.run_in_executor(None, self._scan_models, model_dirs)
        async with self._lock:
            self._merge_discovered(discovered, pinned_models)

    _MODEL_TYPE_TO_ENGINE: dict[str, str] = {
        "llm": "batched",
        "vlm": "vlm",
        "embedding": "embedding",
        "reranker": "reranker",
        "audio_stt": "audio_stt",
        "audio_tts": "audio_tts",
        "audio_sts": "audio_sts",
        "image": "image_gen",
        "video": "video_gen",
        "ti2v": "video_gen",
        # SkyReels-V3 分支类型 (fix #127)
        "r2v_14b": "video_gen",
        "a2v_19b": "video_gen",
        "v2v_14b": "video_gen",
    }

    def apply_settings_overrides(self, settings_manager: ModelSettingsManager) -> None:
        """Apply model_type_override from persisted settings to discovered entries."""
        for model_id, entry in self._entries.items():
            settings = settings_manager.get_settings(model_id)
            if settings.model_type_override:
                entry.model_type = settings.model_type_override
                entry.engine_type = self._MODEL_TYPE_TO_ENGINE.get(
                    settings.model_type_override, "batched"
                )
                logger.info(
                    f"Applied model_type override for {model_id}: "
                    f"type={entry.model_type}, engine={entry.engine_type}"
                )

    def get_model_ids(self) -> list[str]:
        """Get list of all discovered model IDs."""
        return list(self._entries.keys())

    def get_loaded_model_ids(self) -> list[str]:
        """Get list of currently loaded model IDs."""
        return [mid for mid, e in self._entries.items() if e.engine is not None]

    def get_entry(self, model_id: str) -> EngineEntry | None:
        """Get entry for a specific model, or None if not found."""
        return self._entries.get(model_id)

    def set_pinned(self, model_id: str, pinned: bool) -> bool:
        """
        Set the pinned status for a model.

        Args:
            model_id: The model ID to update
            pinned: Whether to pin (True) or unpin (False) the model

        Returns:
            True if successful, False if model not found.
        """
        entry = self._entries.get(model_id)
        if entry is None:
            return False
        entry.is_pinned = pinned
        return True

    def _case_insensitive_entry_match(self, name: str) -> str | None:
        """Find a model entry matching *name* case-insensitively.

        Returns the actual model_id if found, None otherwise.
        """
        lower = name.lower()
        for mid in self._entries:
            if mid.lower() == lower:
                return mid
        return None

    def resolve_model_id(self, model_id_or_alias: str, settings_manager) -> str:
        """Resolve a model alias to its actual model_id (directory name).

        Tries exact match in _entries first, then case-insensitive match,
        then scans model settings for alias match. If those fail and input
        contains a provider prefix (e.g. "fusion-mlx/my-model"), strips the prefix
        and retries. Returns the original string if no match found.
        """
        if model_id_or_alias in self._entries:
            return model_id_or_alias

        # Case-insensitive fallback
        ci_match = self._case_insensitive_entry_match(model_id_or_alias)
        if ci_match is not None:
            return ci_match

        all_settings = None
        if settings_manager is not None:
            all_settings = settings_manager.get_all_settings()
            for mid, ms in all_settings.items():
                if ms.model_alias and ms.model_alias == model_id_or_alias:
                    return mid

        # Strip provider prefix (e.g. "fusion-mlx/qwen3.5-35b" -> "qwen3.5-35b")
        if "/" in model_id_or_alias:
            stripped = model_id_or_alias.split("/", 1)[1]
            if stripped in self._entries:
                return stripped
            ci_match = self._case_insensitive_entry_match(stripped)
            if ci_match is not None:
                return ci_match
            if all_settings is not None:
                for mid, ms in all_settings.items():
                    if ms.model_alias and ms.model_alias == stripped:
                        return mid

        return model_id_or_alias

    @staticmethod
    def _entry_is_diffusion_model(entry: EngineEntry) -> bool:
        model_type = (entry.config_model_type or "").lower().replace("-", "_")
        return model_type == "diffusion_gemma"

    @staticmethod
    def _entry_has_active_requests(entry: EngineEntry) -> bool:
        engine = entry.engine
        if engine is None:
            return False
        has_active_requests = getattr(engine, "has_active_requests", None)
        if not callable(has_active_requests):
            return False
        try:
            return has_active_requests() is True
        except Exception:
            return True

    def _entry_is_busy(self, entry: EngineEntry) -> bool:
        return entry.in_use > 0 or self._entry_has_active_requests(entry)

    def _raise_if_reload_busy(self, entry: EngineEntry, operation: str) -> None:
        if self._entry_is_busy(entry):
            raise ModelBusyError(entry.model_id, operation)

    def _mark_pending_unload_locked(
        self,
        model_id: str,
        reason: str,
        *,
        abort_requested: bool = False,
    ) -> bool:
        """Mark a loaded non-pinned model for unload once it is no longer busy.

        Caller must hold ``self._lock``. Returns True when a pending marker was
        installed. The method deliberately does not unload by itself; call
        ``_unload_pending_if_idle_locked`` after abort/release state changes.
        """
        entry = self._entries.get(model_id)
        if entry is None or entry.engine is None or entry.is_loading or entry.is_pinned:
            return False
        entry.pending_unload_reason = reason
        if abort_requested:
            entry.abort_requested = True
        return True

    def _find_pending_unload_ready_locked(self) -> str | None:
        candidates: list[tuple[float, str]] = []
        for mid, entry in self._entries.items():
            if not entry.pending_unload_reason:
                continue
            if (
                entry.engine is None
                or entry.is_loading
                or entry.is_pinned
                or self._entry_is_busy(entry)
            ):
                continue
            candidates.append((entry.last_access, mid))
        if not candidates:
            return None
        candidates.sort()
        return candidates[0][1]

    async def _unload_pending_if_idle_locked(self, model_id: str) -> bool:
        """Unload a pending model if all leases and active requests have drained.

        Caller must hold ``self._lock``.
        """
        entry = self._entries.get(model_id)
        if (
            entry is None
            or entry.engine is None
            or not entry.pending_unload_reason
            or entry.is_loading
            or entry.is_pinned
            or self._entry_is_busy(entry)
        ):
            return False

        reason = entry.pending_unload_reason
        entry.pending_unload_reason = None
        entry.abort_requested = False
        logger.warning(
            "Unloading pending model '%s' after activity drained (%s)",
            model_id,
            reason,
        )
        await self.unload_engine_async(model_id)
        return True

    def is_abort_requested(self, model_id: str | None) -> bool:
        if model_id is None:
            return False
        entry = self._entries.get(model_id)
        return bool(entry and entry.abort_requested)

    @staticmethod
    def _adapter_key(model_id: str, adapter_path: str | None) -> str:
        if not adapter_path:
            return model_id
        return f"{model_id}::lora::{adapter_path}"

    @staticmethod
    def _resolve_allowed_adapter_dirs() -> list[str]:
        raw = os.getenv("FUSION_LORA_ALLOWED_DIRS", "").strip()
        if not raw:
            return []
        dirs: list[str] = []
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            dirs.append(os.path.realpath(os.path.expanduser(part)))
        return dirs

    @staticmethod
    def _is_within(child: str, base: str) -> bool:
        try:
            return os.path.commonpath([child, base]) == base
        except ValueError:
            return False

    def _validate_adapter_path(self, path: str) -> str:
        if not path or not path.strip():
            raise AdapterPathError(path or "")
        real = os.path.realpath(os.path.expanduser(path))
        if not self._allowed_adapter_dirs:
            raise AdapterPathError(
                path,
                "Per-request LoRA adapters are disabled. Set "
                "FUSION_LORA_ALLOWED_DIRS (comma-separated list of adapter "
                "directories) to enable them.",
            )
        if not any(self._is_within(real, base) for base in self._allowed_adapter_dirs):
            raise AdapterPathError(path)
        return real

    def _make_adapter_entry(
        self,
        base: EngineEntry,
        adapter_path: str,
        entry_key: str,
    ) -> EngineEntry:
        logger.info(
            "Creating derived LoRA entry '%s' from base '%s' (adapter=%s)",
            entry_key,
            base.model_id,
            adapter_path,
        )
        return EngineEntry(
            model_id=entry_key,
            model_path=base.model_path,
            model_type=base.model_type,
            engine_type=base.engine_type,
            estimated_size=base.estimated_size,
            config_model_type=base.config_model_type,
            thinking_default=base.thinking_default,
            preserve_thinking_default=base.preserve_thinking_default,
            model_context_length=base.model_context_length,
            source_type="lora_adapter",
            source_repo_id=base.source_repo_id,
            adapter_path=adapter_path,
            base_model_id=base.model_id,
        )

    def _select_adapter_cap_victims(self, exclude_key: str) -> list[str]:
        cap = self._max_adapter_engines
        if cap <= 0:
            return []
        idle_adapters = [
            (k, e)
            for k, e in self._entries.items()
            if e.source_type == "lora_adapter"
            and e.engine is not None
            and e.in_use == 0
            and k != exclude_key
        ]
        victims: list[str] = []
        while len(idle_adapters) + 1 > cap:
            idle_adapters.sort(key=lambda kv: kv[1].last_access)
            vk, ve = idle_adapters.pop(0)
            ve.pending_unload_reason = "adapter_cap"
            victims.append(vk)
        if victims:
            logger.info(
                "Adapter cap %d exceeded; evicting %d derived " "adapter engine(s): %s",
                cap,
                len(victims),
                victims,
            )
        return victims

    async def get_engine(
        self,
        model_id: str,
        force_lm: bool = False,
        _lease: bool = False,
        runtime_settings: object | None = None,
        adapter_path: str | None = None,
    ) -> (
        BaseEngine
        | EmbeddingEngine
        | RerankerEngine
        | STTEngine
        | STSEngine
        | TTSEngine
    ):
        """
        Get or load engine for the specified model.

        This method implements pre-load memory checking:
        1. Check if model is already loaded → return immediately
        2. Check if model is too large for memory limit → raise error
        3. Evict LRU models until there's enough space
        4. Load the model
        5. Return the engine

        Args:
            model_id: The model ID to get engine for
            force_lm: Force loading as LM (BatchedEngine) even for VLM models.
                Useful for text-only tasks like accuracy benchmarks.
            adapter_path: Optional LoRA adapter path. When set, a derived
                engine entry is lazily created (keyed by model_id::adapter)
                so each adapter gets its own loaded model instance.

        Returns:
            The loaded engine (BaseEngine for LLM, EmbeddingEngine for embeddings)

        Raises:
            ModelNotFoundError: If model is not discovered
            ModelTooLargeError: If model exceeds memory limit
            InsufficientMemoryError: If can't free enough memory (all pinned)
            ModelLoadingError: If model load is aborted by memory enforcer
        """
        # Validate untrusted request-supplied adapter path before any work.
        # Canonicalizes to realpath and enforces the allow-list (default-deny).
        if adapter_path:
            adapter_path = self._validate_adapter_path(adapter_path)
        # Phase 1: Quick check under lock for already-loaded models
        entry_key = self._adapter_key(model_id, adapter_path)
        adapter_victims: list[str] = []
        wait_event: asyncio.Event | None = None
        async with self._lock:
            entry = self._entries.get(entry_key)
            if entry is None:
                if adapter_path:
                    base = self._entries.get(model_id)
                    if base is None:
                        raise ModelNotFoundError(model_id, list(self._entries.keys()))
                    adapter_victims = self._select_adapter_cap_victims(entry_key)
                    entry = self._make_adapter_entry(base, adapter_path, entry_key)
                    self._entries[entry_key] = entry
                else:
                    raise ModelNotFoundError(model_id, list(self._entries.keys()))
            expected_signature = self._engine_runtime_signature(
                entry_key,
                runtime_settings,
            )

            # Already loaded - just update access time (fast path)
            if entry.engine is not None:
                needs_reload = False
                if (
                    expected_signature is not None
                    and entry.runtime_settings_signature is not None
                    and entry.runtime_settings_signature != expected_signature
                ) or (
                    runtime_settings is not None
                    and entry.runtime_settings_signature is None
                ):
                    self._raise_if_reload_busy(
                        entry,
                        "reload runtime settings variant",
                    )
                    needs_reload = True
                if (
                    entry.engine is not None
                    and force_lm
                    and isinstance(entry.engine, VLMBatchedEngine)
                ):
                    self._raise_if_reload_busy(entry, "reload as LM")
                    needs_reload = True

                if not needs_reload:
                    if entry.runtime_settings_signature is None:
                        entry.runtime_settings_signature = expected_signature
                    entry.last_access = time.time()
                    if _lease:
                        entry.in_use += 1
                    return entry.engine

                # Needs reload — mark loading and release lock for slow unload
                if entry.is_loading:
                    wait_event = entry.loading_event or asyncio.Event()
                    entry.loading_event = wait_event
                else:
                    entry.is_loading = True
                    entry.loading_started_at = time.monotonic()
                    entry.abort_loading = False
                    entry.loading_event = asyncio.Event()

            else:
                # Not loaded yet — mark loading
                if entry.is_loading:
                    wait_event = entry.loading_event or asyncio.Event()
                    entry.loading_event = wait_event
                else:
                    entry.is_loading = True
                    entry.loading_started_at = time.monotonic()
                    entry.abort_loading = False
                    entry.loading_event = asyncio.Event()

        # If another coroutine is already loading this model, wait for it
        # OUTSIDE the lock (the loader needs the lock to finish).
        if wait_event is not None:
            logger.info(
                "Model '%s' is already loading — waiting for existing load",
                entry_key,
            )
            await wait_event.wait()
            # Retry from the top — the engine is now loaded (or the load
            # failed and get_engine will re-trigger a fresh attempt).
            return await self.get_engine(
                model_id,
                force_lm=force_lm,
                _lease=_lease,
                runtime_settings=runtime_settings,
                adapter_path=adapter_path,
            )

        # Phase 2: Slow operations OUTSIDE the lock
        # so concurrent requests are not blocked for 5-20 seconds.
        if entry.engine is not None:
            logger.info(
                "Unloading %s before reload (outside lock)",
                entry_key,
            )
            await self.unload_engine_async(entry_key)

        # Evict derived adapter engines over the soft cap (victims selected
        # under the lock in Phase 1). unload_engine_async is slow, so do it here.
        for vk in adapter_victims:
            await self.unload_engine_async(vk)

        # Pre-load admission check (outside lock — memory state is approximate)
        ceiling = self._current_ceiling()
        if ceiling > 0:
            for _ in range(20):
                current = max(
                    mx.get_active_memory(),
                    get_phys_footprint(),
                    self._current_model_memory,
                )
                projected = current + entry.estimated_size
                if projected <= ceiling:
                    break
                victim = self._find_lru_victim()
                if victim is not None:
                    logger.info(
                        f"Evicting '{victim}' to fit '{entry_key}' "
                        f"under memory ceiling "
                        f"({format_size(projected)} > "
                        f"{format_size(ceiling)})"
                    )
                    await self.unload_engine_async(victim)
                    continue
                # Nothing to evict — clean up loading flag and raise
                async with self._lock:
                    entry.is_loading = False
                    loading_event = entry.loading_event
                    entry.loading_event = None
                if loading_event is not None:
                    loading_event.set()
                if entry.estimated_size > ceiling:
                    raise ModelTooLargeError(model_id, entry.estimated_size, ceiling)
                raise InsufficientMemoryError(
                    required=entry.estimated_size,
                    current=current,
                    message=(
                        f"Cannot load {model_id}: projected memory "
                        f"{format_size(projected)} would exceed the memory "
                        f"ceiling {format_size(ceiling)} "
                        f"(current: {format_size(current)}, "
                        f"model: {format_size(entry.estimated_size)}). "
                        "Free system memory or lower memory_guard_tier."
                    ),
                )

        # Now load the model (slow, outside lock)
        await self._load_engine(
            entry_key,
            force_lm=force_lm,
            runtime_settings=runtime_settings,
        )

        async with self._lock:
            loaded = self._entries[entry_key]
            if _lease:
                loaded.in_use += 1
            return loaded.engine

    async def release_engine(
        self, model_id: str, adapter_path: str | None = None
    ) -> None:
        """Release one in-use lease previously taken via get_engine(_lease=True)."""
        entry_key = self._adapter_key(model_id, adapter_path)
        # Detach under the lock (fast), settle outside it. Holding the pool
        # lock across unload_engine_async's ~settle barrier (gc + synchronize +
        # clear_cache x10) blocks every concurrent get_engine for seconds.
        # _detach_engine stops the engine and sets entry.engine=None under the
        # lock (so get_engine's fast-path won't return it), then the slow
        # barrier runs unlocked - mirrors get_engine's "slow work outside lock"
        # pattern. (code-review #74)
        settle_pre: int | None = None
        async with self._lock:
            e = self._entries.get(entry_key)
            if e is not None and e.in_use > 0:
                e.in_use -= 1
            # Inline the pending-unload drain gate (same conditions as
            # _unload_pending_if_idle_locked) so we can split detach/settle
            # across the lock boundary instead of calling the locked helper,
            # which would run the full settle under the lock.
            if (
                e is not None
                and e.engine is not None
                and e.pending_unload_reason
                and not e.is_loading
                and not e.is_pinned
                and not self._entry_is_busy(e)
            ):
                reason = e.pending_unload_reason
                e.pending_unload_reason = None
                e.abort_requested = False
                logger.warning(
                    "Unloading pending model '%s' after activity drained (%s)",
                    entry_key,
                    reason,
                )
                settle_pre = await self._detach_engine(entry_key)
        if settle_pre is not None:
            await self._settle_unloaded_engine(entry_key, settle_pre)

    async def unload_if_idle_unpinned(self, model_id: str) -> bool:
        """Unload a loaded engine only when it is idle and not pinned."""
        # Detach under the lock (fast), settle outside it so the ~settle
        # barrier does not block all get_engine. (code-review #74)
        settle_pre: int | None = None
        async with self._lock:
            entry = self._entries.get(model_id)
            if (
                entry is None
                or entry.engine is None
                or entry.is_loading
                or entry.is_pinned
                or entry.in_use > 0
            ):
                return False

            if self._entry_has_active_requests(entry):
                entry.last_access = time.time()
                return False

            settle_pre = await self._detach_engine(model_id)
        if settle_pre is not None:
            await self._settle_unloaded_engine(model_id, settle_pre)
        return True

    @asynccontextmanager
    async def acquire(self, model_id: str, force_lm: bool = False):
        """Acquire an engine with an atomic in-use lease.

        The lease is taken under the pool lock at acquire time and always
        released in finally, so the engine cannot be evicted mid-request even
        on exception.
        """
        engine = await self.get_engine(model_id, force_lm=force_lm, _lease=True)
        try:
            yield engine
        finally:
            await self.release_engine(model_id)

    def register_engine(self, model_id: str, engine) -> None:
        """Register an externally-created engine in the pool."""
        entry = self._entries.get(model_id)
        if entry is None:
            entry = EngineEntry(
                model_id=model_id,
                model_path="",
                model_type="llm",
                engine_type="batched",
                estimated_size=0,
            )
            self._entries[model_id] = entry
        entry.engine = engine
        entry.last_access = time.monotonic()
        self._current_model_memory += entry.estimated_size
        logger.info(f"Registered engine '{model_id}' in pool")

    def unload_engine(self, model_id: str) -> None:
        """Synchronously remove an engine from pool entries (non-blocking)."""
        entry = self._entries.get(model_id)
        if entry and entry.engine is not None:
            entry.engine = None
            entry.last_access = 0.0
            # Mirror register_engine (+) / unload_engine_async (-): keep the pool's
            # _current_model_memory tracker in sync. Without this the sync
            # unload path (server shutdown / external unregister) leaves the
            # tracker inflated, so pre-load admission over-counts loaded
            # memory and may spuriously evict or reject loads. (code-review #69)
            self._current_model_memory -= entry.estimated_size
            if self._process_memory_enforcer is not None:
                self._process_memory_enforcer.update_loaded_model_bytes(
                    -int(entry.estimated_size)
                )
            logger.info(f"Unregistered engine '{model_id}' from pool")

    def _find_lru_victim(self) -> str | None:
        """
        Find the least recently used non-pinned loaded model.

        Skips models with active inference requests to avoid interrupting
        in-flight generation.

        Returns:
            Model ID of the LRU victim, or None if no evictable model found
        """
        candidates = []
        for mid, e in self._entries.items():
            if e.engine is None or e.is_pinned:
                continue
            if e.in_use > 0:
                continue
            if self._entry_has_active_requests(e):
                logger.debug(f"Skipping victim '{mid}': has active requests")
                continue
            candidates.append((e.last_access, mid))
        if not candidates:
            return None
        candidates.sort()  # Sort by last_access (oldest first)
        return candidates[0][1]

    async def _evict_kv_cache(self, model_id: str) -> bool:
        """Phase 1 eviction — free KV cache only, keep weights in memory.

        Much faster than full unload (~100ms vs ~20s for a 7B model).
        Use this as the first eviction step before unloading weights.

        Returns:
            True if KV cache was freed, False if nothing to free.
        """
        entry = self._entries.get(model_id)
        if not entry or entry.engine is None:
            return False
        if hasattr(entry.engine, "clear_kv_cache"):
            try:
                entry.engine.clear_kv_cache()
                import mx

                loop = asyncio.get_running_loop()
                await loop.run_in_executor(
                    get_mlx_executor(), lambda: (mx.synchronize(), mx.clear_cache())
                )
                logger.info(f"Phase 1 eviction for {model_id}: KV cache freed")
                return True
            except Exception as e:
                logger.warning(f"Phase 1 eviction failed for {model_id}: {e}")
        return False

    async def _detach_engine(self, model_id: str) -> int | None:
        # Stop and detach an engine (set entry.engine = None). Fast teardown:
        # stop + reset activity + drain ready queue + clear entry fields.
        # Returns pre_unload_active for the caller to pass to
        # _settle_unloaded_engine, or None if there was nothing to unload.
        # Caller must hold self._lock (or have set entry.is_loading, as the
        # get_engine reload path does) so a concurrent get_engine fast-path
        # cannot return the engine being stopped. (code-review #74)
        entry = self._entries.get(model_id)
        if not entry or entry.engine is None:
            return None

        logger.info(f"Unloading model: {model_id} (immediate abort)")
        pre_unload_active = mx.get_active_memory()

        try:
            if hasattr(entry.engine, "safe_evict"):
                await entry.engine.safe_evict()
            else:
                await entry.engine.stop()
        except Exception as e:
            logger.warning(f"Error stopping engine for {model_id}: {e}")

        # #1595: the immediate-abort stop() above tears the engine down without the normal
        # per-request completion callbacks, so a non-streaming engine's active_requests
        # counter can leak a phantom count (a stale engine then looks permanently busy).
        # Reset it on teardown so has_active_requests() and the status API stay consistent.
        reset = getattr(entry.engine, "_reset_activity_tracking", None)
        if callable(reset):
            try:
                reset()
            except Exception as e:
                logger.warning(f"Error resetting activity counter for {model_id}: {e}")

        # Yield to the event loop before dropping the engine reference.
        #
        # When abort_all_requests() fires before unload_engine_async(), it sets
        # asyncio Events for each active request.  Server-side streaming
        # generators are then scheduled in the asyncio ready queue, but they
        # cannot run until the event loop gets control.  EngineCore.close()
        # (called inside stop()) blocks the event loop with synchronous
        # .result() calls on the MLX executor -- scheduler.shutdown() and
        # scheduler.deep_reset() -- so those generators are still suspended
        # when stop() returns.
        #
        # If we set entry.engine = None and call gc.collect() immediately,
        # the generators are still alive with a local 'engine' variable
        # referencing the BatchedEngine, keeping its refcount above zero.
        # The model's ~20 GB of MLX weight tensors therefore remain "active"
        # in Metal memory, the settle barrier times out, and subsequent load
        # attempts fail with 507 because the ceiling is still exceeded.
        #
        # A few asyncio.sleep(0) calls drain the ready queue -- generator
        # tear-down is at most a few frames deep -- so that by the time we
        # clear entry.engine and run gc.collect(), no coroutine frame holds
        # a stale engine reference.
        for _ in range(5):
            await asyncio.sleep(0)

        # Clear engine reference before settle barrier
        entry.engine = None
        entry.last_access = 0.0
        entry.actual_size = None
        entry.abort_requested = False
        entry.pending_unload_reason = None
        entry.runtime_settings_signature = None
        return pre_unload_active

    async def unload_engine_async(
        self, model_id: str, with_settle: bool = True
    ) -> None:
        # Full unload (detach + settle). Used by get_engine's reload/evict
        # path (called outside the lock, guarded by entry.is_loading) and by
        # _unload_pending_if_idle_locked (memory_enforcer path). Callers that
        # hold the pool lock (release_engine / unload_if_idle_unpinned) must
        # instead call _detach_engine under the lock and _settle_unloaded_engine
        # outside it, so the ~settle barrier does not block all get_engine.
        # (code-review #74)
        #
        # with_settle=False is the fast teardown path: detach + decrement the
        # memory counter from the estimate + wake the enforcer, WITHOUT the
        # gc/synchronize/clear_cache + 10-round poll barrier. Use it when the
        # caller does not need a precise post-unload memory baseline; this is
        # the same recovery contract as the settle-indeterminate branch (the
        # #1623 max() in get_engine re-reads the live gauge, so estimate drift
        # self-corrects on the next admission).
        pre_unload_active = await self._detach_engine(model_id)
        if pre_unload_active is None:
            return
        if with_settle:
            await self._settle_unloaded_engine(model_id, pre_unload_active)
        else:
            entry = self._entries.get(model_id)
            if entry is not None:
                self._current_model_memory -= entry.estimated_size
            logger.debug(
                f"Fast unload (no settle) for '{model_id}': "
                f"estimate={format_size(entry.estimated_size if entry else 0)}, "
                f"active_memory={format_size(pre_unload_active)}"
            )
            self._wake_process_memory_enforcer()

    async def _settle_unloaded_engine(
        self, model_id: str, pre_unload_active: int
    ) -> None:
        # Memory settle barrier after _detach_engine detached the engine
        # (entry.engine is already None). Polls mx.get_active_memory() to
        # verify Metal buffers are actually reclaimed before updating the
        # memory tracking counter. Safe to run WITHOUT the pool lock: the
        # engine is detached, so get_engine's fast-path sees entry.engine is
        # None and goes to the load path rather than returning a half-stopped
        # engine. (code-review #74)
        entry = self._entries.get(model_id)
        if entry is None:
            return
        # Force garbage collection to release memory.
        # Run mx.clear_cache on the global MLX executor to avoid concurrent
        # Metal operations with running engines. See issue #85.
        # Synchronize before clearing to prevent releasing Metal buffers
        # still referenced by in-flight command buffers. See issue #300.
        gc.collect()
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            get_mlx_executor(), lambda: (mx.synchronize(), mx.clear_cache())
        )
        gc.collect()
        # clear_cache releases C++ Metal buffer wrappers — second GC pass
        # collects the Python-side objects freed by the C++ destructors

        # Memory settle barrier: poll actual freed memory instead of
        # trusting the cumulative _current_model_memory estimate.
        # Scale tolerance with model size: estimated_size includes a 5%
        # overhead factor (model_discovery.py) that may not be reflected in
        # actual freed memory. Use 2 GB floor for small models. See #768.
        settle_tolerance = max(2 * 1024**3, int(entry.estimated_size * 0.05))
        min_expected_freed = max(0, entry.estimated_size - settle_tolerance)
        settled = False
        settle_indeterminate = False
        for _settle_round in range(10):
            active_now = mx.get_active_memory()
            actual_freed = pre_unload_active - active_now
            if actual_freed >= min_expected_freed:
                settled = True
                logger.debug(
                    f"Settle round {_settle_round + 1} for '{model_id}': "
                    f"freed={format_size(actual_freed)} "
                    f"(need>={format_size(min_expected_freed)}) - settled"
                )
                break
            if self._other_entries_serving(model_id):
                # actual_freed is a delta of the process-global MLX gauge,
                # so while another engine allocates (prefill/KV growth) the
                # amount freed by THIS unload is unmeasurable — the delta can
                # even read negative. Burning settle rounds here serializes
                # gc/synchronize/clear_cache against live decode for seconds,
                # under memory pressure, with the enforcer holding the pool
                # lock. Bail out instead: pre-load admission re-reads the
                # live gauge, so nothing downstream trusts this sample.
                settle_indeterminate = True
                logger.info(
                    f"Settle for '{model_id}' indeterminate under concurrent "
                    f"activity (freed={format_size(actual_freed)}, "
                    f"need>={format_size(min_expected_freed)}); skipping "
                    f"settle wait"
                )
                break
            logger.debug(
                f"Settle round {_settle_round + 1} for '{model_id}': "
                f"freed={format_size(actual_freed)} "
                f"(need>={format_size(min_expected_freed)}) - retry"
            )
            await asyncio.sleep(0.5)
            gc.collect()
            await loop.run_in_executor(
                get_mlx_executor(), lambda: (mx.synchronize(), mx.clear_cache())
            )

        # Release memory tracking AFTER barrier
        self._current_model_memory -= entry.estimated_size

        if settled:
            logger.info(
                f"Unloaded model: {model_id}, "
                f"freed={format_size(actual_freed)} "
                f"(expected>={format_size(min_expected_freed)}), "
                f"active_memory: {format_size(active_now)} (settled)"
            )
        elif settle_indeterminate:
            # Settle wait skipped (logged above). Emergency reclaim is
            # deliberately skipped too: its gc + synchronize + clear_cache
            # rounds would stall the live engines that made the measurement
            # indeterminate in the first place. Recovery is not lost:
            # _wake_process_memory_enforcer() below triggers an immediate
            # enforcer re-poll, and pre-load admission re-reads the live gauge
            # alongside the tracked accumulator (the #1623 max() in
            # get_engine), so any unreleased memory stays visible to both.
            pass
        else:
            # Barrier timed out - this is expected for Metal/MLX: the
            # memory is cached internally and WILL be reused on next load.
            logger.info(
                f"Settle barrier: '{model_id}' freed={format_size(actual_freed)} "
                f"(expected>={format_size(min_expected_freed)}), "
                f"Metal cache will reuse on next load"
            )
            for _ in range(3):
                gc.collect()
                await loop.run_in_executor(
                    get_mlx_executor(),
                    lambda: (mx.synchronize(), mx.clear_cache()),
                )
                await asyncio.sleep(1.0)
            active_after = mx.get_active_memory()
            if active_after > self._current_model_memory + 8 * 1024**3:
                logger.warning(
                    f"Emergency reclaim high residual for '{model_id}': "
                    f"active_memory={format_size(active_after)} "
                    f"(expected ≤{format_size(self._current_model_memory + 8 * 1024**3)})"
                )
            else:
                logger.info(
                    f"Emergency reclaim succeeded: "
                    f"active_memory={format_size(active_after)}"
                )

        self._wake_process_memory_enforcer()

    async def _unload_other_dflash_engines(self, model_id: str) -> None:
        """Unload other idle DFlash engines before starting a new one.

        dflash-mlx installs target hooks on shared Python classes and owns a
        process-global runtime cache manager, so multiple loaded DFlash engines
        can leak state across model switches.
        """
        victims: list[str] = []
        blocked: list[str] = []
        for mid, e in self._entries.items():
            if mid == model_id or e.engine is None:
                continue
            if type(e.engine).__name__ != "DFlashEngine":
                continue
            if e.is_loading or e.in_use > 0:
                blocked.append(mid)
                continue
            try:
                if e.engine.has_active_requests():
                    blocked.append(mid)
                    continue
            except AttributeError:
                pass
            if e.is_pinned:
                blocked.append(f"{mid} (pinned)")
                continue
            victims.append(mid)

        if blocked:
            raise RuntimeError(
                "Cannot load DFlash model "
                f"'{model_id}' while another DFlash engine is active: "
                f"{', '.join(blocked)}"
            )

        for victim in victims:
            logger.info(
                "Unloading DFlash model '%s' before loading '%s' because "
                "dflash runtime hooks/cache are process-global",
                victim,
                model_id,
            )
            await self.unload_engine_async(victim)

    @staticmethod
    def _resolve_scheduler_from_engine(engine: object) -> object | None:
        scheduler = getattr(engine, "scheduler", None)
        if scheduler is not None:
            return scheduler
        try:
            return engine._engine.engine.scheduler  # type: ignore[attr-defined]
        except AttributeError:
            return None

    def _is_idle_for_prefill_eviction(self, entry: EngineEntry) -> bool:
        engine = entry.engine
        if engine is None or entry.is_pinned or entry.is_loading or entry.in_use > 0:
            return False
        if self._entry_has_active_requests(entry):
            return False

        scheduler = self._resolve_scheduler_from_engine(engine)
        if scheduler is None:
            return True
        for attr in ("running", "waiting", "prefilling", "requests"):
            value = getattr(scheduler, attr, None)
            if value:
                return False
        return True

    def _find_lru_prefill_eviction_victim(self, *, exclude_model_id: str) -> str | None:
        candidates = []
        for mid, entry in self._entries.items():
            if mid == exclude_model_id:
                continue
            if self._is_idle_for_prefill_eviction(entry):
                candidates.append((entry.last_access, mid))
        if not candidates:
            return None
        candidates.sort()
        return candidates[0][1]

    async def _evict_idle_lru_for_prefill(
        self,
        exclude_model_id: str,
        eviction_request: object,
    ) -> bool:
        """Evict idle LRU models until the requested prefill step should fit."""
        target = int(getattr(eviction_request, "target_cap_bytes", 0) or 0)
        predicted = int(getattr(eviction_request, "predicted_transient_bytes", 0) or 0)
        request_id = str(getattr(eviction_request, "request_id", ""))
        if target <= 0 or predicted <= 0:
            return False

        evicted_any = False
        async with self._lock:
            while True:
                current = max(
                    mx.get_active_memory(),
                    get_phys_footprint(),
                    self._current_model_memory,
                )
                if current + predicted <= target:
                    return evicted_any

                victim = self._find_lru_prefill_eviction_victim(
                    exclude_model_id=exclude_model_id
                )
                if victim is None:
                    if evicted_any:
                        logger.info(
                            "Prefill eviction for request %s stopped with no "
                            "more idle victims (current=%s, predicted=%s, "
                            "target=%s)",
                            request_id,
                            format_size(current),
                            format_size(predicted),
                            format_size(target),
                        )
                    return evicted_any

                logger.info(
                    "Evicting idle model '%s' for prefill headroom on '%s' "
                    "(request=%s, projected=%s > target=%s)",
                    victim,
                    exclude_model_id,
                    request_id,
                    format_size(current + predicted),
                    format_size(target),
                )
                await self.unload_engine_async(victim)
                evicted_any = True

    def _other_entries_serving(self, model_id: str) -> bool:
        """True when any loaded entry other than ``model_id`` is serving.

        Used by the settle barrier in ``unload_engine_async``: the barrier's
        freed-memory check is a delta of the process-global
        ``mx.get_active_memory()`` gauge, which only measures THIS unload
        while no other engine is allocating concurrently.
        """
        # Snapshot the items: admin unload routes call unload_engine_async without
        # the pool lock, so discover_models() can mutate _entries mid-iteration.
        for mid, e in list(self._entries.items()):
            if mid == model_id or e.engine is None:
                continue
            if e.in_use > 0:
                return True
            if self._entry_has_active_requests(e):
                return True
        return False

    async def _load_engine(
        self,
        model_id: str,
        force_lm: bool = False,
        runtime_settings: object | None = None,
    ) -> None:
        """
        Load an engine for the specified model.

        Args:
            model_id: The model ID to load
            force_lm: Force loading as BatchedEngine even for VLM models.

        Raises:
            ModelLoadingError: If model load is aborted by memory enforcer
        """
        entry = self._entries[model_id]
        # get_engine phase 1 reserves the loading slot under the lock before
        # calling us; the admin reload path (models_route) arrives here without
        # a reservation after unload_engine_async. Only reserve when no one has;
        # re-raising here would reject our own phase-1 reservation and every
        # first load would fail with "already being loaded".
        if not entry.is_loading:
            entry.is_loading = True
            entry.loading_started_at = time.monotonic()
            entry.abort_loading = False
        self._wake_process_memory_enforcer(active=True)
        load_started_at = entry.loading_started_at
        load_completed = False
        pre_load_memory = max(mx.get_active_memory(), get_phys_footprint())
        try:
            effective_type = entry.engine_type
            if force_lm and effective_type == "vlm":
                effective_type = "batched"
                logger.info(f"Loading model as LM (force_lm=True): {model_id}")
            else:
                logger.info(f"Loading model: {model_id}")

            # Retrieve per-model settings for post-load transforms.
            # Derived adapter entries reuse the base model's settings so that
            # per-profile defaults (quant, context, etc.) still apply; the
            # adapter path is injected separately via entry.adapter_path.
            settings_id = entry.base_model_id or model_id
            model_settings = runtime_settings
            if model_settings is None and self._settings_manager is not None:
                model_settings = self._settings_manager.get_settings(settings_id)

            # Native MTP forces LM-only dispatch even for VLM models. Vision
            # encoder weights are ignored because the patched mtp_forward only
            # exists on the language model path. mtp_enabled was already
            # validated as mutually exclusive with dflash / turboquant in
            # metal-knowledge: with the mlx-vlm runtime MTP patch (see
            # fusion_mlx/patches/mlx_vlm_mtp/qwen35_moe_vlm_runtime.py) VLM models
            # can run MTP natively while keeping vision intact. The old
            # force-LM-dispatch shortcut here is obsolete for patched
            # model families; let VLMBatchedEngine handle MTP-enabled VLMs.
            pass

            # Check if DFlash is enabled -- takes priority over engine type
            # since DFlash has its own model loading pipeline
            engine = None
            if model_settings is not None:
                dflash_enabled = getattr(model_settings, "dflash_enabled", False)
                dflash_draft = getattr(model_settings, "dflash_draft_model", None)
                if (
                    dflash_enabled
                    and dflash_draft
                    and self._entry_is_diffusion_model(entry)
                ):
                    logger.warning(
                        "DFlash is not supported for diffusion models; "
                        "loading %s with its native VLM engine",
                        model_id,
                    )
                elif dflash_enabled and dflash_draft:
                    try:
                        from .engine.dflash import DFlashEngine

                        engine = DFlashEngine(
                            model_name=entry.model_path,
                            draft_model_path=dflash_draft,
                            draft_quant_enabled=getattr(
                                model_settings, "dflash_draft_quant_enabled", False
                            ),
                            draft_quant_weight_bits=getattr(
                                model_settings, "dflash_draft_quant_weight_bits", 4
                            ),
                            draft_quant_activation_bits=getattr(
                                model_settings, "dflash_draft_quant_activation_bits", 16
                            ),
                            draft_quant_group_size=getattr(
                                model_settings, "dflash_draft_quant_group_size", 64
                            ),
                            model_settings=model_settings,
                            fallback_engine_type=effective_type,
                            scheduler_config=self._scheduler_config,
                            fusion_ssd_cache_dir=getattr(
                                self._scheduler_config, "paged_ssd_cache_dir", None
                            ),
                        )
                        logger.info(
                            f"DFlash enabled for {model_id}, draft={dflash_draft}"
                        )
                    except ImportError:
                        logger.warning(
                            f"DFlash enabled for {model_id} but dflash-mlx is not installed. "
                            f"Falling back to default engine."
                        )
                    except Exception as e:
                        logger.warning(
                            f"DFlash init failed for {model_id}: {e}. "
                            f"Falling back to default engine."
                        )

            # Per-model trust_remote_code (security opt-in, issue #926).
            # When unset, defaults to False -- repos with custom modeling_*.py
            # will fail to load until the user explicitly toggles this on
            # in the admin UI's model settings modal.
            trc = (
                bool(getattr(model_settings, "trust_remote_code", False))
                if model_settings
                else False
            )

            async def prefill_eviction_callback(
                eviction_request: object,
                *,
                _model_id: str = model_id,
            ) -> bool:
                return await self._evict_idle_lru_for_prefill(
                    exclude_model_id=_model_id,
                    eviction_request=eviction_request,
                )

            # Create engine based on engine type (if DFlash not active)
            if engine is None:
                if effective_type == "embedding":
                    engine = EmbeddingEngine(
                        model_name=entry.model_path,
                        trust_remote_code=trc,
                        scheduler_config=self._scheduler_config,
                    )
                elif effective_type == "reranker":
                    engine = RerankerEngine(
                        model_name=entry.model_path,
                        trust_remote_code=trc,
                    )
                elif effective_type == "vlm":
                    engine = VLMBatchedEngine(
                        model_name=entry.model_path,
                        trust_remote_code=trc,
                        scheduler_config=self._scheduler_config,
                        model_settings=model_settings,
                        enable_thinking=entry.thinking_default,
                        preserve_thinking=entry.preserve_thinking_default,
                        prefill_eviction_callback=prefill_eviction_callback,
                    )
                elif entry.engine_type == "audio_stt":
                    engine = STTEngine(model_name=entry.model_path)
                elif entry.engine_type == "audio_tts":
                    engine = TTSEngine(model_name=entry.model_path)
                elif entry.engine_type == "audio_sts":
                    engine = STSEngine(
                        model_name=entry.model_path,
                        config_model_type=entry.config_model_type,
                    )
                elif entry.engine_type == "image_gen":
                    engine = ImageGenEngine(model_name=entry.model_path)
                elif entry.engine_type == "video_gen":
                    engine = VideoGenEngine(model_name=entry.model_path)
                else:
                    engine = BatchedEngine(
                        model_name=entry.model_path,
                        trust_remote_code=trc,
                        scheduler_config=self._scheduler_config,
                        model_settings=model_settings,
                        enable_thinking=entry.thinking_default,
                        preserve_thinking=entry.preserve_thinking_default,
                        prefill_eviction_callback=prefill_eviction_callback,
                        lora_path=entry.adapter_path
                        or getattr(model_settings, "lora_path", None),
                    )

            _is_dflash_engine = (
                engine is not None and type(engine).__name__ == "DFlashEngine"
            )
            if _is_dflash_engine:
                await self._unload_other_dflash_engines(model_id)

            try:
                await engine.start()
            except Exception as start_error:
                if _is_dflash_engine:
                    # DFlash engine failed to start — fall back to the
                    # model's natural engine type (VLM or Batched)
                    logger.warning(
                        f"DFlash start failed for {model_id}: {start_error}. "
                        f"Falling back to {effective_type} engine."
                    )
                    try:
                        await engine.stop()
                    except Exception:
                        logger.debug(
                            "swallowed exception at fusion_mlx/pool/engine_pool.py:693"
                        )

                        pass
                    gc.collect()
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(
                        get_mlx_executor(),
                        lambda: (mx.synchronize(), mx.clear_cache()),
                    )

                    if effective_type == "vlm":
                        engine = VLMBatchedEngine(
                            model_name=entry.model_path,
                            trust_remote_code=trc,
                            scheduler_config=self._scheduler_config,
                            model_settings=model_settings,
                            enable_thinking=entry.thinking_default,
                            preserve_thinking=entry.preserve_thinking_default,
                            prefill_eviction_callback=prefill_eviction_callback,
                        )
                    else:
                        engine = BatchedEngine(
                            model_name=entry.model_path,
                            trust_remote_code=trc,
                            scheduler_config=self._scheduler_config,
                            model_settings=model_settings,
                            enable_thinking=entry.thinking_default,
                            preserve_thinking=entry.preserve_thinking_default,
                            prefill_eviction_callback=prefill_eviction_callback,
                            lora_path=entry.adapter_path
                            or getattr(model_settings, "lora_path", None),
                        )
                    try:
                        await engine.start()
                    except Exception as fallback_error:
                        raise RuntimeError(
                            f"DFlash load failed: {start_error}; "
                            f"{effective_type} fallback also failed: {fallback_error}"
                        ) from start_error
                    logger.info(
                        f"Successfully loaded {model_id} as {effective_type} "
                        f"(fallback from DFlash)"
                    )

                elif force_lm and entry.engine_type == "vlm":
                    # force_lm created a BatchedEngine but mlx-lm can't
                    # load this VLM model — fall back to VLMBatchedEngine.
                    logger.warning(
                        f"LM loading failed for VLM model {model_id} "
                        f"(force_lm=True), falling back to VLM engine: "
                        f"{start_error}"
                    )
                    try:
                        await engine.stop()
                    except Exception:
                        logger.debug(
                            "swallowed exception at fusion_mlx/pool/engine_pool.py:739"
                        )

                        pass
                    gc.collect()
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(
                        get_mlx_executor(),
                        lambda: (mx.synchronize(), mx.clear_cache()),
                    )

                    engine = VLMBatchedEngine(
                        model_name=entry.model_path,
                        trust_remote_code=trc,
                        scheduler_config=self._scheduler_config,
                        model_settings=model_settings,
                        enable_thinking=entry.thinking_default,
                        preserve_thinking=entry.preserve_thinking_default,
                        prefill_eviction_callback=prefill_eviction_callback,
                    )
                    try:
                        await engine.start()
                    except Exception as fallback_error:
                        raise RuntimeError(
                            f"LM load failed (force_lm=True): {start_error}; "
                            f"VLM fallback also failed: {fallback_error}"
                        ) from start_error

                    logger.info(
                        f"Successfully loaded {model_id} as VLM "
                        f"(fallback from force_lm)"
                    )
                elif entry.engine_type == "vlm":
                    # VLM loading failed — fall back to LLM (BatchedEngine)
                    logger.warning(
                        f"VLM loading failed for {model_id}, "
                        f"falling back to LLM: {start_error}"
                    )
                    try:
                        await engine.stop()
                    except Exception:
                        logger.debug(
                            "swallowed exception at fusion_mlx/pool/engine_pool.py:775"
                        )

                        pass
                    gc.collect()
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(
                        get_mlx_executor(),
                        lambda: (mx.synchronize(), mx.clear_cache()),
                    )

                    engine = BatchedEngine(
                        model_name=entry.model_path,
                        trust_remote_code=trc,
                        scheduler_config=self._scheduler_config,
                        model_settings=model_settings,
                        enable_thinking=entry.thinking_default,
                        preserve_thinking=entry.preserve_thinking_default,
                        prefill_eviction_callback=prefill_eviction_callback,
                        lora_path=entry.adapter_path
                        or getattr(model_settings, "lora_path", None),
                    )
                    try:
                        await engine.start()
                    except Exception as fallback_error:
                        raise RuntimeError(
                            f"VLM load failed: {start_error}; "
                            f"LLM fallback also failed: {fallback_error}"
                        ) from start_error

                    entry.model_type = "llm"
                    entry.engine_type = "batched"
                    logger.info(
                        f"Successfully loaded {model_id} as LLM " f"(fallback from VLM)"
                    )
                else:
                    raise

            # Check if memory enforcer requested abort during loading
            if entry.abort_loading:
                logger.warning(f"Model load aborted by memory enforcer: {model_id}")
                try:
                    await engine.stop()
                except Exception as e:
                    logger.warning(f"Error stopping aborted engine for {model_id}: {e}")
                gc.collect()
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(
                    get_mlx_executor(),
                    lambda: (mx.synchronize(), mx.clear_cache()),
                )
                raise ModelLoadingError(
                    model_id,
                    f"Model {model_id} load aborted: " f"process memory limit exceeded",
                )

            entry.engine = engine
            entry.last_access = time.time()
            self._current_model_memory += entry.estimated_size
            load_completed = True
            if self._process_memory_enforcer is not None:
                self._process_memory_enforcer.update_loaded_model_bytes(
                    int(entry.estimated_size)
                )

            # VLM MTP: load gemma4_assistant drafter and attach to engine.
            # Fail-soft — drafter load issues never block the target engine.
            if (
                model_settings is not None
                and getattr(model_settings, "vlm_mtp_enabled", False)
                and getattr(model_settings, "vlm_mtp_draft_model", None)
                and hasattr(engine, "set_vlm_mtp_drafter")
            ):
                drafter_id = model_settings.vlm_mtp_draft_model
                drafter_entry = self._entries.get(drafter_id)
                drafter_path = drafter_entry.model_path if drafter_entry else drafter_id

                def _load_drafter_sync(path: str = drafter_path):
                    from .speculative.vlm_mtp import load_vlm_mtp_drafter

                    return load_vlm_mtp_drafter(path)

                loop = asyncio.get_running_loop()
                try:
                    drafter = await loop.run_in_executor(
                        get_mlx_executor(), _load_drafter_sync
                    )
                except Exception as e:
                    logger.warning(
                        f"VLM MTP drafter load raised for {model_id} "
                        f"(drafter={drafter_id}): {e} — toggle ignored"
                    )
                    drafter = None
                if drafter is not None:
                    engine.set_vlm_mtp_drafter(drafter)
                    logger.info(f"VLM MTP enabled for {model_id}, drafter={drafter_id}")
                else:
                    logger.warning(
                        f"VLM MTP toggle on for {model_id} but drafter "
                        f"load failed; toggle ignored"
                    )

            entry.runtime_settings_signature = self._engine_runtime_signature(
                model_id,
                model_settings,
                loaded_engine=engine,
            )

            # Propagate memory limit to new engine's scheduler
            if self._process_memory_enforcer is not None:
                self._process_memory_enforcer._propagate_memory_limit()

            # Release intermediate Metal buffers from model loading.
            # mlx_lm.load() creates large temporaries (weight transforms,
            # quantization intermediates) that stay in the Metal buffer pool
            # because mx.set_cache_limit(total_mem) prevents automatic release.
            # Without this, memory stays at ~2x model size until the first
            # inference request triggers a clear. (#429)
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                get_mlx_executor(),
                lambda: (mx.synchronize(), mx.clear_cache()),
            )

            post_load_memory = max(mx.get_active_memory(), get_phys_footprint())
            observed_delta = max(0, post_load_memory - pre_load_memory)
            entry.actual_size = observed_delta or entry.estimated_size

            logger.info(
                f"Loaded model: {model_id} "
                f"(actual: {format_size(entry.actual_size)}, "
                f"estimated: {format_size(entry.estimated_size)}, "
                f"total: {format_size(self._current_model_memory)})"
            )
        finally:
            if (
                load_completed
                and load_started_at is not None
                and entry.estimated_size > 0
            ):
                elapsed = max(0.0, time.monotonic() - load_started_at)
                size_gb = entry.estimated_size / (1024**3)
                if size_gb > 0 and elapsed > 0:
                    sample = elapsed / size_gb
                    if self._load_seconds_per_gb_ema is None:
                        self._load_seconds_per_gb_ema = sample
                    else:
                        self._load_seconds_per_gb_ema = (
                            self._load_seconds_per_gb_ema * 0.9 + sample * 0.1
                        )
                    self._load_time_observations += 1
                    logger.debug(
                        f"Observed model load speed: {sample:.2f}s/GB "
                        f"for {model_id} ({elapsed:.1f}s, {format_size(entry.estimated_size)}); "
                        f"EMA={self._load_seconds_per_gb_ema:.2f}s/GB"
                    )
            entry.is_loading = False
            entry.loading_started_at = None
            entry.abort_loading = False
            loading_event = entry.loading_event
            entry.loading_event = None
            if loading_event is not None:
                loading_event.set()
            self._wake_process_memory_enforcer()

    async def preload_pinned_models(self) -> None:
        """
        Preload all pinned models at startup.

        This ensures pinned models are always available.
        """
        pinned_models = [
            model_id for model_id, e in self._entries.items() if e.is_pinned
        ]

        for model_id in pinned_models:
            try:
                logger.info(f"Preloading pinned model: {model_id}")
                await self.get_engine(model_id)
            except Exception as e:
                logger.error(f"Failed to preload pinned model {model_id}: {e}")

    async def shutdown(self) -> None:
        """Shutdown all engines gracefully."""
        async with self._lock:
            for model_id in list(self._entries.keys()):
                entry = self._entries.get(model_id)
                if entry and entry.engine is not None:
                    try:
                        await self.unload_engine_async(model_id)
                    except Exception as e:
                        logger.error(f"Error unloading {model_id} during shutdown: {e}")

        logger.info("Engine pool shutdown complete")

    def get_status(self) -> dict:
        """
        Get pool status for monitoring endpoints.

        Returns:
            Dictionary with pool status information
        """
        return {
            "final_ceiling": self._current_ceiling(),
            "current_model_memory": self._current_model_memory,
            "model_count": len(self._entries),
            "loaded_count": sum(
                1 for e in self._entries.values() if e.engine is not None
            ),
            "load_seconds_per_gb_estimate": self._load_seconds_per_gb_ema,
            "load_time_observations": self._load_time_observations,
            "models": [
                {
                    "id": mid,
                    "model_path": e.model_path,
                    "loaded": e.engine is not None,
                    "is_loading": e.is_loading,
                    "loading_started_at": e.loading_started_at,
                    "estimated_size": e.estimated_size,
                    "actual_size": e.actual_size,
                    "pinned": e.is_pinned,
                    "engine_type": e.engine_type,
                    "model_type": e.model_type,
                    "config_model_type": e.config_model_type,
                    "thinking_default": e.thinking_default,
                    "preserve_thinking_default": e.preserve_thinking_default,
                    "last_access": e.last_access if e.last_access > 0 else None,
                }
                for mid, e in sorted(self._entries.items())
            ],
        }

    async def check_ttl_expirations(
        self,
        settings_manager: ModelSettingsManager,
        global_idle_timeout_seconds: int | None = None,
    ) -> list[str]:
        """Check and unload models that have exceeded their TTL.

        Pinned models are skipped (TTL is ignored for pinned models).
        Models with active requests are skipped and their last_access is refreshed.
        Suppressed during benchmark runs via _suppress_ttl flag.

        Args:
            settings_manager: The settings manager to read TTL values from.
            global_idle_timeout_seconds: Global idle timeout fallback (None = no global TTL).

        Returns:
            List of model IDs that were unloaded.
        """
        if self._suppress_ttl:
            return []

        now = time.time()
        expired: list[str] = []

        async with self._lock:
            for model_id, entry in self._entries.items():
                if entry.engine is None or entry.is_loading or entry.is_pinned:
                    continue

                settings = settings_manager.get_settings(model_id)
                effective_ttl = settings.ttl_seconds
                if effective_ttl is None:
                    effective_ttl = global_idle_timeout_seconds
                if effective_ttl is None:
                    continue

                idle_time = now - entry.last_access
                if idle_time < effective_ttl:
                    continue

                # Check if model has active requests
                has_active = entry.engine.has_active_requests()

                if has_active:
                    entry.last_access = now
                    continue

                logger.info(
                    f"TTL expired for model '{model_id}' "
                    f"(idle {idle_time:.0f}s > ttl {effective_ttl}s)"
                )
                expired.append(model_id)

        # Unload expired models outside the lock to avoid blocking
        for model_id in expired:
            try:
                await self.unload_engine_async(model_id)
            except Exception as e:
                logger.error(f"Failed to unload expired model {model_id}: {e}")

        return expired
