# SPDX-License-Identifier: Apache-2.0
"""Batched engine for continuous batching with multiple concurrent users.

Adapted from Rapid-MLX / omlx. ``BatchedEngine`` routes requests through
``AsyncEngineCore``. Multimodal (VLM) serving is handled by
``VLMBatchedEngine`` (engines/vlm.py); the legacy ``--mllm`` /
``MLLMScheduler`` path was removed as dead code (``is_mllm_model`` was a
``return False`` stub and VLM never routed through BatchedEngine).
"""

import logging
from collections.abc import AsyncIterator
from typing import Any

from ..base import BaseEngine, GenerationOutput

logger = logging.getLogger(__name__)


class BatchedEngine(BaseEngine):
    """Batched engine for continuous batching.

    This engine provides better throughput when serving multiple
    concurrent users by batching requests together.
    """

    def __init__(
        self,
        model_name: str,
        trust_remote_code: bool = True,
        scheduler_config: Any | None = None,
        stream_interval: int = 1,
        gpu_memory_utilization: float = 0.90,
        *,
        force_text: bool = False,
    ):
        # force_text retained as the --no-mllm/--text-only escape-hatch hook;
        # the legacy _is_mllm / MLLMScheduler path was removed (VLM serving
        # routes through VLMBatchedEngine in pool/engine_pool.py).
        _ = force_text
        self._model_name = model_name
        self._trust_remote_code = trust_remote_code
        self._scheduler_config = scheduler_config
        self._stream_interval = stream_interval
        self._gpu_memory_utilization = gpu_memory_utilization
        self._model = None
        self._processor = None
        self._tokenizer = None
        self._engine = None
        self._loaded = False

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def is_mllm(self) -> bool:
        # Legacy MLLM path removed; VLM serving uses VLMBatchedEngine.
        # Kept as False for interface compat (routes/chat.py, tests).
        return False

    @property
    def tokenizer(self) -> Any:
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
        await self._start_llm()
        self._loaded = True
        logger.info(f"BatchedEngine loaded: {self._model_name}")

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
            executor=self._model_load_executor,
        )
        await self._engine.engine.start()

    async def stop(self) -> None:
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
