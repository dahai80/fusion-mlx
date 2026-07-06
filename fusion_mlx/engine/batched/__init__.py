# SPDX-License-Identifier: Apache-2.0
"""Batched engine for continuous batching with multiple concurrent users.

Adapted from Rapid-MLX / omlx. Provides:

- ``BatchedEngine``: continuous-batching engine that routes MLLM requests
  through MLLMScheduler and LLM requests through AsyncEngineCore.
- ``_probe_mllm_cache_type``: startup probe that blocks hybrid models
  from --mllm mode (#352).
- ``_resolve_mllm_prefill_step_size``: bump-policy for MLLM prefill
  step size (#682).
"""

import logging
from collections.abc import AsyncIterator
from typing import Any

from ..base import BaseEngine, GenerationOutput

logger = logging.getLogger(__name__)


def _probe_mllm_cache_type(language_model: Any) -> str | None:
    """Return the offending cache type name when ``language_model`` is
    incompatible with MLLM continuous batching, or None if it's fine.

    "Incompatible" means ``make_prompt_cache`` returns something other than
    a list of ``KVCache`` / ``RotatingKVCache`` -- currently ArraysCache (hybrid
    Qwen3.5/3.6, etc.) or MambaCache (Nemotron, Granite4). Returning a name
    instead of a bool lets the caller put the actual class in the error
    message (#352).

    The probe is best-effort; if mlx-lm raises before producing a cache list
    we return None and let the runtime path surface the real error instead
    of masking it with a misleading hybrid-incompat message.
    """
    from mlx_lm.models.cache import KVCache, RotatingKVCache, make_prompt_cache

    try:
        test_cache = make_prompt_cache(language_model)
    except Exception:
        return None
    if not test_cache:
        return None
    sample = test_cache[0]
    if isinstance(sample, (KVCache, RotatingKVCache)):
        return None
    return type(sample).__name__


def _resolve_mllm_prefill_step_size(
    user_value: int | None,
    *,
    text_default: int,
    mllm_default: int,
) -> int:
    """Apply the MLLM ``prefill_step_size`` bump-policy (#682).

    A 1920x1080 screenshot decoded by Qwen3-VL produces ~2200 vision
    tokens -- past the 2048 text-LLM default that ``SchedulerConfig``
    ships with. The per-batch cap in
    ``mllm_batch_generator._process_prompts`` would otherwise fire
    silently and surface as ``finish_reason="length"`` + empty content
    (#682).

    Policy:
    - ``None`` or value equal to ``text_default`` -> ``mllm_default``
      (the Desktop-sidecar happy path).
    - Any other value -> honored as-is (memory-constrained operators
      and high-end deployments keep their explicit choice; codex r2
      MAJOR contract).

    Args:
        user_value: ``getattr(scheduler_config, "prefill_step_size", None)``
        text_default: ``SchedulerConfig.prefill_step_size``'s dataclass default.
        mllm_default: ``MLLMSchedulerConfig.prefill_step_size``'s dataclass default.

    Returns:
        The resolved ``prefill_step_size`` for the MLLM scheduler.
    """
    if user_value is None or user_value == text_default:
        return mllm_default
    return user_value


class BatchedEngine(BaseEngine):
    """Batched engine for continuous batching.

    This engine provides better throughput when serving multiple
    concurrent users by batching requests together.

    For MLLM (multimodal) models, this engine uses MLLMScheduler
    which handles images and videos alongside text generation.
    """

    def __init__(
        self,
        model_name: str,
        trust_remote_code: bool = True,
        scheduler_config: Any | None = None,
        stream_interval: int = 1,
        force_mllm: bool = False,
        gpu_memory_utilization: float = 0.90,
        *,
        force_text: bool = False,
    ):
        self._model_name = model_name
        self._trust_remote_code = trust_remote_code
        self._scheduler_config = scheduler_config
        self._stream_interval = stream_interval
        self._gpu_memory_utilization = gpu_memory_utilization
        if force_text:
            self._is_mllm = False
        else:
            from ...api.utils import is_mllm_model

            self._is_mllm = force_mllm or is_mllm_model(model_name)

        self._model = None
        self._processor = None
        self._tokenizer = None
        self._engine = None
        self._mllm_scheduler = None
        self._loaded = False

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def is_mllm(self) -> bool:
        return self._is_mllm

    @property
    def tokenizer(self) -> Any:
        if self._is_mllm and self._processor:
            return getattr(self._processor, "tokenizer", self._processor)
        return self._tokenizer

    @property
    def model_type(self) -> str | None:
        if self._model is None:
            return None
        try:
            if hasattr(self._model, "config"):
                config = self._model.config
                if hasattr(config, "model_type"):
                    return config.model_type
        except Exception:
            pass
        return None

    @property
    def grammar_compiler(self):
        return None

    @property
    def prefix_cache_enabled(self) -> bool:
        return False

    async def start(self) -> None:
        if self._loaded:
            return
        if self._is_mllm:
            await self._start_mllm()
        else:
            await self._start_llm()
        self._loaded = True
        logger.info(f"BatchedEngine loaded: {self._model_name} (mllm={self._is_mllm})")

    async def _start_mllm(self) -> None:
        import concurrent.futures

        from ...engine_core import _init_mlx_step_thread
        from ...mllm_scheduler import MLLMScheduler, MLLMSchedulerConfig
        from ...models.mllm import MLXMultimodalLM
        from ...scheduler import SchedulerConfig

        _MLLM_DEFAULT_PREFILL_STEP_SIZE = MLLMSchedulerConfig.__dataclass_fields__[
            "prefill_step_size"
        ].default

        self._model_load_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="mllm-step",
            initializer=_init_mlx_step_thread,
        )

        def _load_mllm():
            instance = MLXMultimodalLM(
                self._model_name,
                trust_remote_code=self._trust_remote_code,
            )
            instance.load()
            return instance

        self._mllm_instance = self._model_load_executor.submit(_load_mllm).result()
        self._model = self._mllm_instance.model
        self._processor = self._mllm_instance.processor

        language_model = getattr(self._model, "language_model", self._model)
        cache_type = self._model_load_executor.submit(
            _probe_mllm_cache_type, language_model
        ).result()
        if cache_type is not None:
            raise RuntimeError(
                f"Model '{self._model_name}' uses a hybrid/linear-attention "
                f"language backbone ({cache_type}), which is incompatible "
                f"with --mllm continuous batching (requires standard KVCache "
                f"or RotatingKVCache). Drop --mllm for text-only use, or pick "
                f"a non-hybrid VLM (Qwen3-VL, Gemma-3, etc.). See #352."
            )

        if self._scheduler_config and hasattr(self._scheduler_config, "max_num_seqs"):
            max_num_seqs = self._scheduler_config.max_num_seqs
        else:
            max_num_seqs = 16

        prefill_batch_size = getattr(self._scheduler_config, "prefill_batch_size", 8)
        completion_batch_size = getattr(
            self._scheduler_config, "completion_batch_size", 32
        )
        prefill_step_size = _resolve_mllm_prefill_step_size(
            getattr(self._scheduler_config, "prefill_step_size", None),
            text_default=SchedulerConfig.__dataclass_fields__[
                "prefill_step_size"
            ].default,
            mllm_default=_MLLM_DEFAULT_PREFILL_STEP_SIZE,
        )
        max_concurrent_requests = getattr(
            self._scheduler_config, "max_concurrent_requests", 256
        )

        mllm_config = MLLMSchedulerConfig(
            max_num_seqs=max_num_seqs,
            prefill_batch_size=prefill_batch_size,
            completion_batch_size=completion_batch_size,
            prefill_step_size=prefill_step_size,
            enable_vision_cache=True,
            vision_cache_size=100,
            max_concurrent_requests=max_concurrent_requests,
        )

        self._mllm_scheduler = MLLMScheduler(
            model=self._model,
            processor=self._processor,
            config=mllm_config,
            step_executor=self._model_load_executor,
        )
        await self._mllm_scheduler.start()

    def _ubc_evict_after_load(self) -> None:
        try:
            from ...runtime.ubc_evict import ubc_evict_paths

            model_dir = getattr(self._model, "name_or_path", None) or self._model_name
            if not model_dir:
                return
            from pathlib import Path

            p = Path(model_dir).expanduser()
            if not p.is_dir():
                return
            safetensors_files = sorted(p.glob("*.safetensors"))
            if safetensors_files:
                ubc_evict_paths(str(f) for f in safetensors_files)
        except Exception:
            logger.debug("ubc_evict_after_load failed", exc_info=True)

    def _check_mxfp4_moe_guardrail(self) -> None:
        try:
            from ...mxfp4_moe_guardrail import check_from_profile

            check_from_profile(model_name=self._model_name)
        except Exception:
            logger.debug("mxfp4_moe_guardrail check failed", exc_info=True)

    async def _start_llm(self) -> None:
        import concurrent.futures

        from ...engine_core import AsyncEngineCore, EngineConfig, _init_mlx_step_thread
        from ...scheduler import SchedulerConfig
        from ...utils.tokenizer import load_model_with_fallback

        self._model_load_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="mlx-step",
            initializer=_init_mlx_step_thread,
        )

        tokenizer_config = {"trust_remote_code": self._trust_remote_code}
        if "qwen3" in self._model_name.lower():
            tokenizer_config["eos_token"] = "<|im_end|>"

        self._model, self._tokenizer = self._model_load_executor.submit(
            load_model_with_fallback,
            self._model_name,
            tokenizer_config=tokenizer_config,
        ).result()

        self._ubc_evict_after_load()
        self._check_mxfp4_moe_guardrail()

        scheduler_config = self._scheduler_config or SchedulerConfig()
        engine_config = EngineConfig(
            model_name=self._model_name,
            scheduler_config=scheduler_config,
            stream_interval=self._stream_interval,
            gpu_memory_utilization=self._gpu_memory_utilization,
        )

        self._engine = AsyncEngineCore(
            model=self._model,
            tokenizer=self._tokenizer,
            config=engine_config,
        )
        await self._engine.engine.start(executor=self._model_load_executor)

    async def stop(self) -> None:
        if self._mllm_scheduler:
            await self._mllm_scheduler.stop()
            self._mllm_scheduler = None
            if self._is_mllm and self._model_load_executor is not None:
                self._model_load_executor.shutdown(wait=False)

        if self._engine:
            await self._engine.stop()
            self._engine.engine.close()
            self._engine = None

        self._model_load_executor = None
        self._model = None
        self._tokenizer = None
        self._processor = None
        self._loaded = False
        logger.info("BatchedEngine stopped")

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        stop: list[str] | None = None,
        **kwargs,
    ) -> GenerationOutput:
        if not self._loaded:
            await self.start()

        if self._is_mllm and self._mllm_scheduler:
            _mllm_penalty_kwargs = {}
            if repetition_penalty != 1.0:
                _mllm_penalty_kwargs["repetition_penalty"] = repetition_penalty
            elif "repetition_penalty" in kwargs:
                _mllm_penalty_kwargs["repetition_penalty"] = kwargs.pop(
                    "repetition_penalty"
                )
            if presence_penalty != 0.0:
                _mllm_penalty_kwargs["presence_penalty"] = presence_penalty
            elif "presence_penalty" in kwargs:
                _mllm_penalty_kwargs["presence_penalty"] = kwargs.pop(
                    "presence_penalty"
                )
            if "frequency_penalty" in kwargs:
                _mllm_penalty_kwargs["frequency_penalty"] = kwargs.pop(
                    "frequency_penalty"
                )
            output = await self._mllm_scheduler.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                **_mllm_penalty_kwargs,
            )
            return GenerationOutput(
                text=output.output_text or "",
                prompt_tokens=output.prompt_tokens,
                completion_tokens=output.completion_tokens,
                finish_reason=output.finish_reason,
            )

        from ...request import SamplingParams

        _sp_kwargs = {
            k: kwargs.pop(k)
            for k in (
                "top_k",
                "min_p",
                "repetition_penalty",
                "presence_penalty",
                "frequency_penalty",
                "seed",
            )
            if k in kwargs
        }
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop or [],
            **_sp_kwargs,
        )
        output = await self._engine.generate(
            prompt=prompt,
            sampling_params=sampling_params,
        )
        return GenerationOutput(
            text=output.output_text or "",
            prompt_tokens=output.prompt_tokens,
            completion_tokens=output.completion_tokens,
            finish_reason=output.finish_reason,
        )

    async def stream_generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        stop: list[str] | None = None,
        **kwargs,
    ) -> AsyncIterator[GenerationOutput]:
        if not self._loaded:
            await self.start()

        if self._is_mllm and self._mllm_scheduler:
            _mllm_penalty_kwargs = {}
            if repetition_penalty != 1.0:
                _mllm_penalty_kwargs["repetition_penalty"] = repetition_penalty
            elif "repetition_penalty" in kwargs:
                _mllm_penalty_kwargs["repetition_penalty"] = kwargs.pop(
                    "repetition_penalty"
                )
            if presence_penalty != 0.0:
                _mllm_penalty_kwargs["presence_penalty"] = presence_penalty
            elif "presence_penalty" in kwargs:
                _mllm_penalty_kwargs["presence_penalty"] = kwargs.pop(
                    "presence_penalty"
                )
            if "frequency_penalty" in kwargs:
                _mllm_penalty_kwargs["frequency_penalty"] = kwargs.pop(
                    "frequency_penalty"
                )
            request_id = await self._mllm_scheduler.add_request_async(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                **_mllm_penalty_kwargs,
            )
            async for output in self._mllm_scheduler.stream_outputs(request_id):
                yield GenerationOutput(
                    text=output.output_text or "",
                    new_text=output.new_text,
                    prompt_tokens=output.prompt_tokens,
                    completion_tokens=output.completion_tokens,
                    finished=output.finished,
                    finish_reason=output.finish_reason,
                )
            return

        from ...request import SamplingParams

        _sp_kwargs = {
            k: kwargs.pop(k)
            for k in (
                "top_k",
                "min_p",
                "repetition_penalty",
                "presence_penalty",
                "frequency_penalty",
                "seed",
            )
            if k in kwargs
        }
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop or [],
            **_sp_kwargs,
        )
        request_id = await self._engine.add_request(
            prompt=prompt,
            sampling_params=sampling_params,
        )
        async for output in self._engine.stream_outputs(request_id):
            yield GenerationOutput(
                text=output.output_text or "",
                new_text=output.new_text,
                prompt_tokens=output.prompt_tokens,
                completion_tokens=output.completion_tokens,
                finished=output.finished,
                finish_reason=output.finish_reason,
            )

    async def chat(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        tools: list[dict] | None = None,
        **kwargs,
    ) -> GenerationOutput:
        if not self._loaded:
            await self.start()
        prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        return await self.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs,
        )

    async def stream_chat(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        tools: list[dict] | None = None,
        **kwargs,
    ) -> AsyncIterator[GenerationOutput]:
        if not self._loaded:
            await self.start()
        prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        async for output in self.stream_generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs,
        ):
            yield output

    def has_active_requests(self) -> bool:
        return False

    def get_stats(self) -> dict[str, Any]:
        stats = {
            "engine_type": "batched",
            "model_name": self._model_name,
            "loaded": self._loaded,
        }
        if self._engine:
            stats.update(self._engine.get_stats())
        return stats

    def get_cache_stats(self) -> dict[str, Any] | None:
        if self._engine:
            return self._engine.get_cache_stats()
        return None
