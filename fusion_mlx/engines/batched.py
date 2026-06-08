# SPDX-License-Identifier: Apache-2.0
"""Batched engine for continuous batching with multiple concurrent users."""

import asyncio
import copy
import logging
from collections.abc import AsyncIterator
from typing import Any
import mlx.core as mx

from ..engine_core import AsyncEngineCore, EngineConfig, get_mlx_executor
from ..request import SamplingParams
from .base import BaseEngine, GenerationOutput

logger = logging.getLogger(__name__)

try:
    from ..adapter.harmony import preprocess_harmony_messages
    HAS_HARMONY_ADAPTER = True
except ImportError:
    HAS_HARMONY_ADAPTER = False
    preprocess_harmony_messages = None     # type: ignore


class BatchedEngine(BaseEngine):
    def __init__(
        self, model_name: str, trust_remote_code: bool = False,
        scheduler_config: Any | None = None, stream_interval: int = 1,
        enable_thinking: bool | None = None, model_settings: Any | None = None,
    ):
        self._model_name = model_name
        self._trust_remote_code = trust_remote_code
        self._scheduler_config = scheduler_config
        self._stream_interval = stream_interval
        self._enable_thinking = enable_thinking
        self._model_settings = model_settings
        self._model = None
        self._tokenizer = None
        self._engine = None
        self._loaded = False
        self._grammar_compiler = None
        self._grammar_compiler_init_attempted = False

    @property
    def model_name(self) -> str:
        return self._model_name

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
                    mt = config.model_type
                    return mt if isinstance(mt, str) else None
                elif isinstance(config, dict):
                    mt = config.get("model_type")
                    return mt if isinstance(mt, str) else None
            if hasattr(self._model, "args"):
                mt = self._model.args.model_type
                return mt if isinstance(mt, str) else None
        except Exception as e:
            logger.debug(f"Error getting model_type: {e}")
        return None

    @property
    def grammar_compiler(self):
        if self._grammar_compiler is not None:
            return self._grammar_compiler
        if self._grammar_compiler_init_attempted:
            return None
        self._grammar_compiler_init_attempted = True
        try:
            from ..api.grammar import create_grammar_compiler
            self._grammar_compiler = create_grammar_compiler(self._tokenizer, self._model)
            logger.info("GrammarCompiler initialized for %s", self._model_name)
        except Exception:
            logger.debug("fusion_mlx/engines/batched.py:83: swallowed exception")
            pass
        return self._grammar_compiler

    @property
    def prefix_cache_enabled(self) -> bool:
        if self._engine is None:
            return False
        try:
            return self._engine.engine.scheduler.block_aware_cache is not None
        except AttributeError:
            return False

    def _preprocess_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self.model_type == "gpt_oss" and HAS_HARMONY_ADAPTER and preprocess_harmony_messages:
            return preprocess_harmony_messages(messages)
        return messages

    async def start(self) -> None:
        if self._loaded:
            return

        from mlx_lm import load
        from ..scheduler import SchedulerConfig

        tokenizer_config = {"trust_remote_code": self._trust_remote_code}

        loop = asyncio.get_running_loop()

        def _load_model_sync():
            return load(self._model_name, tokenizer_config=tokenizer_config)

        self._model, self._tokenizer = await loop.run_in_executor(
            get_mlx_executor(), _load_model_sync
        )

        scheduler_config = (
            copy.copy(self._scheduler_config) if self._scheduler_config else SchedulerConfig()
        )
        scheduler_config.model_name = self._model_name
        engine_config = EngineConfig(
            model_name=self._model_name, scheduler_config=scheduler_config,
            stream_interval=self._stream_interval,
        )
        self._engine = AsyncEngineCore(model=self._model, tokenizer=self._tokenizer, config=engine_config)
        await self._engine.engine.start()

        # TurboQuant KV cache
        if self._model_settings is not None:
            tq_enabled = getattr(self._model_settings, "turboquant_kv_enabled", False)
            if tq_enabled:
                from ..patches.turboquant_attention import apply_turboquant_attention_patch
                apply_turboquant_attention_patch()
                tq_bits = float(getattr(self._model_settings, "turboquant_kv_bits", 4))
                self._engine.engine.scheduler._turboquant_kv_bits = tq_bits
                self._engine.engine.scheduler._turboquant_skip_last = getattr(
                    self._model_settings, "turboquant_skip_last", True
                )

        # SpecPrefill
        if self._model_settings is not None:
            specprefill_draft = getattr(self._model_settings, "specprefill_draft_model", None)
            specprefill_enabled = getattr(self._model_settings, "specprefill_enabled", False)
            if specprefill_enabled and specprefill_draft:
                try:
                    def _load_draft():
                        from ..patches.mlx_lm_mtp import set_mtp_active
                        was_mtp = False
                        try:
                            from ..patches.mlx_lm_mtp import is_mtp_active
                            was_mtp = is_mtp_active()
                        except Exception:
                            logger.debug("fusion_mlx/engines/batched.py:155: swallowed exception")
                            pass
                        set_mtp_active(False)
                        try:
                            draft_model, _ = load(specprefill_draft)
                            return draft_model
                        finally:
                            set_mtp_active(was_mtp)

                    draft_model = await loop.run_in_executor(get_mlx_executor(), _load_draft)
                    self._engine.engine.scheduler.set_specprefill_draft_model(draft_model, draft_model_name=specprefill_draft)
                    logger.info(f"SpecPrefill: draft model loaded ({specprefill_draft})")
                except Exception as e:
                    logger.error(f"SpecPrefill: draft model load failed: {e}")

        self._loaded = True
        logger.info(f"BatchedEngine loaded: {self._model_name}")

    async def stop(self) -> None:
        if self._engine:
            await self._engine.stop()
            if hasattr(self._engine, "engine") and self._engine.engine is not None:
                try:
                    self._engine.engine.close()
                except Exception as e:
                    logger.warning(f"Error closing engine: {e}")
        self._engine = None
        self._model = None
        self._tokenizer = None
        self._loaded = False

    def _apply_chat_template(
        self, messages: list[dict[str, Any]], tools: list[dict] | None = None,
        chat_template_kwargs: dict[str, Any] | None = None, is_partial: bool | None = None,
    ) -> str:
        if hasattr(self._tokenizer, "apply_chat_template"):
            if is_partial is None:
                from ..api.utils import detect_and_strip_partial
                is_partial = detect_and_strip_partial(messages)
            else:
                for msg in messages:
                    msg.pop("partial", None)
            template_kwargs = {"tokenize": False, "add_generation_prompt": not is_partial}
            if is_partial:
                template_kwargs["continue_final_message"] = True
            if tools:
                from ..api.tool_calling import convert_tools_for_template
                template_kwargs["tools"] = convert_tools_for_template(tools)
            if self._enable_thinking is not None:
                template_kwargs["enable_thinking"] = self._enable_thinking
            if chat_template_kwargs:
                template_kwargs.update(chat_template_kwargs)
            try:
                return self._tokenizer.apply_chat_template(messages, **template_kwargs)
            except TypeError:
                if chat_template_kwargs:
                    for key in chat_template_kwargs:
                        template_kwargs.pop(key, None)
                template_kwargs.pop("tools", None)
                template_kwargs.pop("enable_thinking", None)
                return self._tokenizer.apply_chat_template(messages, **template_kwargs)
        else:
            prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
            return prompt + "\nassistant:"

    def count_chat_tokens(
        self, messages: list[dict[str, Any]], tools: list[dict] | None = None,
        chat_template_kwargs: dict[str, Any] | None = None, is_partial: bool | None = None,
    ) -> int:
        messages = self._preprocess_messages(messages)
        template_tools = None
        if tools:
            from ..api.tool_calling import convert_tools_for_template
            template_tools = convert_tools_for_template(tools)
        prompt = self._apply_chat_template(messages, template_tools, chat_template_kwargs=chat_template_kwargs, is_partial=is_partial)
        return len(self._tokenizer.encode(prompt))

    async def generate(
        self, prompt: str, max_tokens: int = 256, temperature: float = 0.7,
        top_p: float = 0.9, top_k: int = 0, min_p: float = 0.0,
        repetition_penalty: float = 1.0, presence_penalty: float = 0.0,
        stop: list[str] | None = None, **kwargs,
    ) -> GenerationOutput:
        if not self._loaded:
            await self.start()
        sampling_params = SamplingParams(
            max_tokens=max_tokens, temperature=temperature, top_p=top_p, top_k=top_k,
            min_p=min_p, xtc_probability=kwargs.get("xtc_probability", 0.0),
            xtc_threshold=kwargs.get("xtc_threshold", 0.1),
            repetition_penalty=repetition_penalty, presence_penalty=presence_penalty,
            frequency_penalty=kwargs.get("frequency_penalty", 0.0),
            stop=stop or [], thinking_budget=kwargs.get("thinking_budget", None),
            compiled_grammar=kwargs.get("compiled_grammar", None), seed=kwargs.get("seed", None),
        )
        output = await self._engine.generate(prompt=prompt, sampling_params=sampling_params)
        from ..api.utils import clean_special_tokens
        text = clean_special_tokens(output.output_text)
        return GenerationOutput(
            text=text, prompt_tokens=output.prompt_tokens,
            completion_tokens=output.completion_tokens, finish_reason=output.finish_reason,
            tool_calls=output.tool_calls, cached_tokens=output.cached_tokens,
        )

    async def stream_generate(
        self, prompt: str, max_tokens: int = 256, temperature: float = 0.7,
        top_p: float = 0.9, top_k: int = 0, min_p: float = 0.0,
        repetition_penalty: float = 1.0, presence_penalty: float = 0.0,
        stop: list[str] | None = None, **kwargs,
    ) -> AsyncIterator[GenerationOutput]:
        if not self._loaded:
            await self.start()
        sampling_params = SamplingParams(
            max_tokens=max_tokens, temperature=temperature, top_p=top_p, top_k=top_k,
            min_p=min_p, xtc_probability=kwargs.get("xtc_probability", 0.0),
            xtc_threshold=kwargs.get("xtc_threshold", 0.1),
            repetition_penalty=repetition_penalty, presence_penalty=presence_penalty,
            frequency_penalty=kwargs.get("frequency_penalty", 0.0),
            stop=stop or [], thinking_budget=kwargs.get("thinking_budget", None),
            compiled_grammar=kwargs.get("compiled_grammar", None), seed=kwargs.get("seed", None),
        )
        specprefill_kwargs = {}
        if kwargs.get("specprefill") is not None:
            specprefill_kwargs["specprefill"] = kwargs.pop("specprefill")
        if kwargs.get("specprefill_keep_pct") is not None:
            specprefill_kwargs["specprefill_keep_pct"] = kwargs.pop("specprefill_keep_pct")
        if kwargs.get("specprefill_threshold") is not None:
            specprefill_kwargs["specprefill_threshold"] = kwargs.pop("specprefill_threshold")
        if kwargs.get("specprefill_system_end") is not None:
            specprefill_kwargs["specprefill_system_end"] = kwargs.pop("specprefill_system_end")

        engine = self._engine
        request_id = await engine.add_request(prompt=prompt, sampling_params=sampling_params, **specprefill_kwargs)
        finished_normally = False
        try:
            async for output in engine.stream_outputs(request_id):
                from ..api.utils import clean_special_tokens
                text = clean_special_tokens(output.output_text)
                if output.finished:
                    finished_normally = True
                yield GenerationOutput(
                    text=text, new_text=output.new_text,
                    prompt_tokens=output.prompt_tokens, completion_tokens=output.completion_tokens,
                    finished=output.finished, finish_reason=output.finish_reason,
                    tool_calls=output.tool_calls, cached_tokens=output.cached_tokens,
                )
        except GeneratorExit:
            logger.info(f"[stream_generate] GeneratorExit for request {request_id}")
        finally:
            if not finished_normally:
                await engine.abort_request(request_id)

    async def chat(
        self, messages: list[dict[str, Any]], max_tokens: int = 256, temperature: float = 0.7,
        top_p: float = 0.9, top_k: int = 0, min_p: float = 0.0,
        repetition_penalty: float = 1.0, presence_penalty: float = 0.0,
        tools: list[dict] | None = None, **kwargs,
    ) -> GenerationOutput:
        if not self._loaded:
            await self.start()
        messages = self._preprocess_messages(messages)
        template_tools = None
        if tools:
            from ..api.tool_calling import convert_tools_for_template
            template_tools = convert_tools_for_template(tools)
        ct_kwargs = kwargs.pop("chat_template_kwargs", None)
        partial = kwargs.pop("is_partial", None)
        prompt = self._apply_chat_template(messages, template_tools, chat_template_kwargs=ct_kwargs, is_partial=partial)
        return await self.generate(
            prompt=prompt, max_tokens=max_tokens, temperature=temperature,
            top_p=top_p, top_k=top_k, min_p=min_p,
            repetition_penalty=repetition_penalty, presence_penalty=presence_penalty, **kwargs,
        )

    async def stream_chat(
        self, messages: list[dict[str, Any]], max_tokens: int = 256, temperature: float = 0.7,
        top_p: float = 0.9, top_k: int = 0, min_p: float = 0.0,
        repetition_penalty: float = 1.0, presence_penalty: float = 0.0,
        tools: list[dict] | None = None, **kwargs,
    ) -> AsyncIterator[GenerationOutput]:
        if not self._loaded:
            await self.start()
        messages = self._preprocess_messages(messages)
        template_tools = None
        if tools:
            from ..api.tool_calling import convert_tools_for_template
            template_tools = convert_tools_for_template(tools)
        ct_kwargs = kwargs.pop("chat_template_kwargs", None)
        partial = kwargs.pop("is_partial", None)
        prompt = self._apply_chat_template(messages, template_tools, chat_template_kwargs=ct_kwargs, is_partial=partial)

        # SpecPrefill system_end
        specprefill_model_enabled = getattr(self._model_settings, "specprefill_enabled", False) if self._model_settings else False
        if specprefill_model_enabled and kwargs.get("specprefill") is not False:
            non_system = [m for m in messages if m.get("role") not in ("system", "developer")]
            if len(non_system) < len(messages) and non_system:
                try:
                    non_system_prompt = self._apply_chat_template(non_system, template_tools, chat_template_kwargs=ct_kwargs)
                    full_tokens = len(self._tokenizer.encode(prompt))
                    non_system_tokens = len(self._tokenizer.encode(non_system_prompt))
                    system_end = full_tokens - non_system_tokens
                    if system_end > 0:
                        kwargs["specprefill_system_end"] = system_end
                except Exception as e:
                    logger.debug(f"SpecPrefill: system_end calc failed: {e}")

        async for output in self.stream_generate(
            prompt=prompt, max_tokens=max_tokens, temperature=temperature,
            top_p=top_p, top_k=top_k, min_p=min_p,
            repetition_penalty=repetition_penalty, presence_penalty=presence_penalty, **kwargs,
        ):
            yield output

    def has_active_requests(self) -> bool:
        ec = getattr(self, "_engine", None)
        if ec is not None:
            inner = getattr(ec, "engine", None)
            if inner is not None:
                return len(getattr(inner, "_output_collectors", {})) > 0
        return False

    def get_stats(self) -> dict[str, Any]:
        stats = {"engine_type": "batched", "model_name": self._model_name, "loaded": self._loaded, "stream_interval": self._stream_interval}
        if self._engine:
            stats.update(self._engine.get_stats())
        return stats

    def get_cache_stats(self) -> dict[str, Any] | None:
        if self._engine:
            return self._engine.get_cache_stats()
        return None

    async def abort_all_requests(self) -> int:
        if self._engine and self._engine.engine:
            return await self._engine.engine.abort_all_requests()
        return 0
