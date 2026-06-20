# SPDX-License-Identifier: Apache-2.0
"""Batched engine for continuous batching with multiple concurrent users."""

import asyncio
import copy
import logging
import time
from collections.abc import AsyncIterator
from typing import Any

from ..engine_core import AsyncEngineCore, EngineConfig, get_executor
from ..request import SamplingParams
from .base import BaseEngine, GenerationOutput, _fallback_parse_tool_calls

logger = logging.getLogger(__name__)


def _human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def _nbytes_per_item(dtype) -> int:
    try:
        return int(dtype.itemsize)
    except AttributeError:
        dtype_map = {
            "float16": 2, "bfloat16": 2, "float32": 4,
            "int32": 4, "int64": 8, "uint8": 1, "bool": 1,
        }
        return dtype_map.get(str(dtype), 2)

try:
    from ..adapter.harmony import preprocess_harmony_messages
    HAS_HARMONY_ADAPTER = True
except ImportError:
    HAS_HARMONY_ADAPTER = False
    preprocess_harmony_messages = None     # type: ignore


def _fallback_parse_tool_calls(gen: GenerationOutput, tokenizer: Any, tools: list[dict]) -> GenerationOutput:
    """Fallback tool call extraction when the scheduler has no parser session.

    Qwen, GLM, and other models that emit XML-based tool call markers
    (e.g. \u241d...\u241e) don't get parsed by the mllm scheduler.
    This runs parse_tool_calls on the final text as a safety net.
    """
    try:
        from ..api.tool_calling import parse_tool_calls
        cleaned, tc_list = parse_tool_calls(gen.text, tokenizer, tools)
        if tc_list:
            tc_dicts = []
            for tc in tc_list:
                tc_dicts.append({
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                })
            gen = copy.deepcopy(gen)
            gen.tool_calls = tc_dicts
            if cleaned.strip() and cleaned.strip() != gen.text.strip():
                gen.text = cleaned
    except Exception as e:
        logger.debug(f"_fallback_parse_tool_calls failed: {e}")
    return gen


class BatchedEngine(BaseEngine):
    def __init__(
        self, model_name: str, trust_remote_code: bool = False,
        scheduler_config: Any | None = None, stream_interval: int = 1,
        enable_thinking: bool | None = None,
        preserve_thinking: bool | None = None, model_settings: Any | None = None,
    ):
        super().__init__()
        self._model_name = model_name
        self._trust_remote_code = trust_remote_code
        self._scheduler_config = scheduler_config
        self._stream_interval = stream_interval
        self._enable_thinking = enable_thinking
        self._preserve_thinking = preserve_thinking
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
    def is_mllm(self) -> bool:
        return False

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
    def supports_prefill_only(self) -> bool:
        return True

    @property
    def supports_kv_handoff(self) -> bool:
        return True

    @property
    def prefix_cache_enabled(self) -> bool:
        if self._engine is None:
            return False
        try:
            return self._engine.engine.scheduler.block_aware_cache is not None
        except AttributeError:
            return False

    def _sort_system_first(self, messages: list[dict]) -> list[dict]:
        """Ensure exactly one system message at the beginning."""
        systems = [m for m in messages if m.get("role") in ("system", "developer")]
        others = [m for m in messages if m.get("role") not in ("system", "developer")]
        if systems:
            sys_text = "\n\n".join(m.get("content", "") for m in systems if m.get("content"))
            systems = [{"role": "system", "content": sys_text}]
        elif others:
            systems = [{"role": "system", "content": "You are a helpful assistant."}]
        return systems + others

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
            start = time.monotonic()
            logger.info("Loading model: %s", self._model_name)
            model, tokenizer = load(self._model_name, tokenizer_config=tokenizer_config)
            elapsed = time.monotonic() - start
             # Estimate model size from loaded weights
            total_params = 0
            try:
                from mlx.utils import tree_flatten
                flat = tree_flatten(model.parameters())
                total_params = sum(arr.size * arr.itemsize for _, arr in flat)
            except Exception:
                total_params = 0
            size_str = _human_size(total_params)
            logger.info(
                "Model loaded in %.1fs | %s | %s",
                elapsed, size_str, self._model_name,
            )
            return model, tokenizer

        self._model, self._tokenizer = await asyncio.wait_for(
            loop.run_in_executor(get_executor("io"), _load_model_sync), timeout=120.0)

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
                from ..patches.turboquant_attention import (
                    apply_turboquant_attention_patch,
                )
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

                    draft_model = await asyncio.wait_for(
                        loop.run_in_executor(get_executor("io"), _load_draft), timeout=120.0)
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
            if self._preserve_thinking is not None:
                template_kwargs["preserve_thinking"] = self._preserve_thinking
            if chat_template_kwargs:
                template_kwargs.update(chat_template_kwargs)
            # Ensure system message exists at start
            if not messages or messages[0].get("role") not in ("system", "developer"):
                logger.warning(
                    "No system message at start, inserting. first_role=%s, total=%d",
                    messages[0].get("role", "empty") if messages else "empty", len(messages),
                )
                messages.insert(0, {"role": "system", "content": "You are a helpful assistant."})
            logger.info(
                "Chat template: roles=%s",
                [m.get("role") for m in messages[:5]],
            )
            try:
                return self._tokenizer.apply_chat_template(messages, **template_kwargs)
            except Exception as e:
                if "system message" in str(e).lower():
                    logger.error("Template demands system message, retrying with fallback insert")
                    messages.insert(0, {"role": "system", "content": "You are a helpful assistant."})
                    return self._tokenizer.apply_chat_template(messages, **template_kwargs)
                if isinstance(e, TypeError):
                    if chat_template_kwargs:
                        for key in chat_template_kwargs:
                            template_kwargs.pop(key, None)
                    template_kwargs.pop("tools", None)
                    template_kwargs.pop("enable_thinking", None)
                    template_kwargs.pop("preserve_thinking", None)
                    return self._tokenizer.apply_chat_template(messages, **template_kwargs)
                logger.error("apply_chat_template failed: %s, roles=%s", e, [m.get("role") for m in messages[:3]])
                raise
        else:
            prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
            return prompt + "\nassistant:"

    def count_chat_tokens(
        self, messages: list[dict[str, Any]], tools: list[dict] | None = None,
        chat_template_kwargs: dict[str, Any] | None = None, is_partial: bool | None = None,
    ) -> int:
        messages = self._preprocess_messages(messages)
        messages = self._sort_system_first(messages)
        template_tools = None
        if tools:
            from ..api.tool_calling import convert_tools_for_template
            template_tools = convert_tools_for_template(tools)
        prompt = self._apply_chat_template(messages, template_tools, chat_template_kwargs=chat_template_kwargs, is_partial=is_partial)
        return len(self._tokenizer.encode(prompt))

    async def generate(
        self, prompt: str, max_tokens: int = 4096, temperature: float = 0.7,
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
            stop=stop or [], thinking_budget=kwargs.get("thinking_budget"),
            compiled_grammar=kwargs.get("compiled_grammar"), seed=kwargs.get("seed"),
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
        self, prompt: str, max_tokens: int = 4096, temperature: float = 0.7,
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
            stop=stop or [], thinking_budget=kwargs.get("thinking_budget"),
            compiled_grammar=kwargs.get("compiled_grammar"), seed=kwargs.get("seed"),
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
        self, messages: list[dict[str, Any]], max_tokens: int = 4096, temperature: float = 0.7,
        top_p: float = 0.9, top_k: int = 0, min_p: float = 0.0,
        repetition_penalty: float = 1.0, presence_penalty: float = 0.0,
        tools: list[dict] | None = None, **kwargs,
    ) -> GenerationOutput:
        if not self._loaded:
            await self.start()
        messages = self._preprocess_messages(messages)
        messages = self._sort_system_first(messages)
        template_tools = None
        if tools:
            from ..api.tool_calling import convert_tools_for_template
            template_tools = convert_tools_for_template(tools)
        ct_kwargs = kwargs.pop("chat_template_kwargs", None)
        partial = kwargs.pop("is_partial", None)
        prompt = self._apply_chat_template(messages, template_tools, chat_template_kwargs=ct_kwargs, is_partial=partial)
        prefill_only = kwargs.pop("prefill_only", False)
        kv_handoff = kwargs.pop("kv_handoff", None)
        if prefill_only:
            sampling_params = SamplingParams(
                max_tokens=max_tokens, temperature=temperature, top_p=top_p, top_k=top_k,
                min_p=min_p, repetition_penalty=repetition_penalty, presence_penalty=presence_penalty,
                stop=kwargs.pop("stop", []),
            )
            result = await self._engine.prefill(prompt, sampling_params)
            return GenerationOutput(
                text="", prompt_tokens=0, completion_tokens=0, finish_reason=None,
                tool_calls=[], cached_tokens=0,
                kv_state=result.get("kv_state", {}),
            )
        if kv_handoff is not None:
            sampling_params = SamplingParams(
                max_tokens=max_tokens, temperature=temperature, top_p=top_p, top_k=top_k,
                min_p=min_p, repetition_penalty=repetition_penalty, presence_penalty=presence_penalty,
                stop=kwargs.pop("stop", []),
            )
            token_ids = self._tokenizer.encode(prompt)
            ro = await self._engine.decode_with_handoff(token_ids, sampling_params, kv_handoff.kv_buffers)
            from ..api.utils import clean_special_tokens
            return GenerationOutput(
                text=clean_special_tokens(ro.output_text), prompt_tokens=ro.prompt_tokens,
                completion_tokens=ro.completion_tokens, finish_reason=ro.finish_reason,
                tool_calls=ro.tool_calls, cached_tokens=ro.cached_tokens,
            )
        gen = await self.generate(
            prompt=prompt, max_tokens=max_tokens, temperature=temperature,
            top_p=top_p, top_k=top_k, min_p=min_p,
            repetition_penalty=repetition_penalty, presence_penalty=presence_penalty, **kwargs,
             )
        if tools and not gen.tool_calls:
            gen = _fallback_parse_tool_calls(gen, self._tokenizer, tools)
        return gen

    async def stream_chat(
        self, messages: list[dict[str, Any]], max_tokens: int = 4096, temperature: float = 0.7,
        top_p: float = 0.9, top_k: int = 0, min_p: float = 0.0,
        repetition_penalty: float = 1.0, presence_penalty: float = 0.0,
        tools: list[dict] | None = None, **kwargs,
    ) -> AsyncIterator[GenerationOutput]:
        if not self._loaded:
            await self.start()
        messages = self._preprocess_messages(messages)
        messages = self._sort_system_first(messages)
        template_tools = None
        if tools:
            from ..api.tool_calling import convert_tools_for_template
            template_tools = convert_tools_for_template(tools)
        ct_kwargs = kwargs.pop("chat_template_kwargs", None)
        partial = kwargs.pop("is_partial", None)
        prompt = self._apply_chat_template(messages, template_tools, chat_template_kwargs=ct_kwargs, is_partial=partial)

        kv_handoff = kwargs.pop("kv_handoff", None)
        prefill_only = kwargs.pop("prefill_only", False)
        if prefill_only:
     # prefill_only on streaming path: yield single result from non-streaming
            result = await self.chat(messages, max_tokens=max_tokens,
                          temperature=temperature, top_p=top_p, top_k=top_k,
                          min_p=min_p, repetition_penalty=repetition_penalty,
                          presence_penalty=presence_penalty, tools=tools,
                          prefill_only=True, **kwargs)
            yield result
            return
        if kv_handoff is not None:
             # Decode with handoff — stream outputs
            sampling_params = SamplingParams(
                max_tokens=max_tokens, temperature=temperature, top_p=top_p, top_k=top_k,
                min_p=min_p, repetition_penalty=repetition_penalty, presence_penalty=presence_penalty,
                stop=[],
             )
            token_ids = self._tokenizer.encode(prompt)
            request_id = await self._engine.add_request(
                prompt=token_ids, sampling_params=sampling_params
             )
            self._engine.scheduler.import_kv_state(request_id, kv_handoff.kv_buffers)
            async for output in self._engine.stream_outputs(request_id):
                from ..api.utils import clean_special_tokens
                text = clean_special_tokens(output.output_text)
                yield GenerationOutput(
                    text=text, new_text=output.new_text,
                    prompt_tokens=output.prompt_tokens, completion_tokens=output.completion_tokens,
                    finished=output.finished, finish_reason=output.finish_reason,
                    tool_calls=output.tool_calls, cached_tokens=output.cached_tokens,
                 )
            return

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
            if output.finished and tools and not output.tool_calls:
                output = _fallback_parse_tool_calls(output, self._tokenizer, tools)
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
