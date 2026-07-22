from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
from huggingface_hub import snapshot_download
from mlx_lm import load
from mlx_lm.models import cache as cache_lib
from mlx_lm.models import qwen3

logger = logging.getLogger(__name__)


def resolve_model_path(path_or_repo: str) -> Path:
    path = Path(path_or_repo)
    if path.exists():
        return path
    return Path(snapshot_download(path_or_repo))


class MLXTargetAdapter:
    family: str = "unknown"

    def resolve_target_model_path(self, path_or_repo: str) -> Path:
        return resolve_model_path(path_or_repo)

    def build_prompt(
        self, tokenizer, prompt_text: str, enable_thinking: bool = False
    ) -> mx.array:
        raise NotImplementedError

    def stop_token_ids(self, tokenizer) -> set[int]:
        raise NotImplementedError

    def make_cache(self, model) -> list[Any]:
        return model.make_cache()

    def embed_tokens(self, model, tokens: mx.array) -> mx.array:
        raise NotImplementedError

    def lm_head_logits(self, model, hidden_states: mx.array) -> mx.array:
        raise NotImplementedError

    def lm_head_argmax(self, model, hidden_states: mx.array) -> mx.array:
        # Hook for greedy verifier experiments; architecture-specific adapters
        # can replace this with a fused top-1 LM-head kernel.
        logits = self.lm_head_logits(model, hidden_states)
        return mx.argmax(logits, axis=-1).astype(mx.uint32)

    def forward_with_hidden_states(
        self,
        model,
        inputs: mx.array,
        cache: list[Any],
        layer_ids: list[int],
    ) -> tuple[mx.array, mx.array]:
        raise NotImplementedError

    def forward_verifier_states(
        self,
        model,
        inputs: mx.array,
        cache: list[Any],
        layer_ids: list[int],
    ) -> tuple[mx.array, mx.array]:
        raise NotImplementedError(
            f"{self.family} does not expose verifier states before the LM head."
        )

    def forward_accept_all_block(
        self,
        model,
        inputs: mx.array,
        cache: list[Any],
        layer_ids: list[int],
    ) -> tuple[mx.array, mx.array]:
        logits, target_hidden = self.forward_with_hidden_states(
            model,
            inputs,
            cache,
            layer_ids,
        )
        return logits[:, -1:, :], target_hidden

    def rewind_kv_caches(self, cache: list[Any], num_tokens: int) -> None:
        raise NotImplementedError

    def cache_summary(self, cache: list[Any]) -> str:
        raise NotImplementedError

    def set_vision_inputs(self, **kwargs) -> None:
        raise NotImplementedError(f"{self.family} does not support vision inputs.")

    def clear_vision_inputs(self) -> None:
        pass


class Qwen3TargetAdapter(MLXTargetAdapter):
    family = "qwen3"

    def build_prompt(
        self, tokenizer, prompt_text: str, enable_thinking: bool = False
    ) -> mx.array:
        # Thinking defaults OFF: the released DSpark drafts were trained on
        # non-thinking data, and <think> traces roughly halve acceptance
        # length (see README "Thinking mode"). Opt in with enable_thinking.
        messages = [{"role": "user", "content": prompt_text}]
        try:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        return mx.array(tokens, dtype=mx.uint32)

    def stop_token_ids(self, tokenizer) -> set[int]:
        eos_token_ids = tokenizer.eos_token_ids
        if isinstance(eos_token_ids, int):
            return {eos_token_ids}
        return set(eos_token_ids)

    def make_cache(self, model) -> list[Any]:
        return [cache_lib.KVCache() for _ in model.layers]

    def embed_tokens(self, model, tokens: mx.array) -> mx.array:
        return model.model.embed_tokens(tokens)

    def lm_head_logits(self, model, hidden_states: mx.array) -> mx.array:
        if model.args.tie_word_embeddings:
            return model.model.embed_tokens.as_linear(hidden_states)
        return model.lm_head(hidden_states)

    def _forward_states(
        self,
        model,
        inputs: mx.array,
        cache: list[Any],
        layer_ids: list[int],
    ) -> tuple[mx.array, mx.array]:
        # Tap convention (reference extract_context_feature): layer_id maps to
        # the *output of* decoder layer layer_id (HF hidden_states[layer_id+1]);
        # -1 means the embedding output. layer_ids are strictly increasing, so
        # collecting in forward order preserves the concat order.
        text_model = model.model
        hidden_states = text_model.embed_tokens(inputs)
        mask = qwen3.create_attention_mask(hidden_states, cache[0])

        selected_hidden_states: list[mx.array] = []
        target_layer_ids = set(layer_ids)
        if -1 in target_layer_ids:
            selected_hidden_states.append(hidden_states)
        for idx, (layer, layer_cache) in enumerate(zip(text_model.layers, cache)):
            hidden_states = layer(hidden_states, mask=mask, cache=layer_cache)
            if idx in target_layer_ids:
                selected_hidden_states.append(hidden_states)

        norm_hidden_states = text_model.norm(hidden_states)
        target_hidden = mx.concatenate(selected_hidden_states, axis=-1)
        return norm_hidden_states, target_hidden

    def forward_with_hidden_states(
        self,
        model,
        inputs: mx.array,
        cache: list[Any],
        layer_ids: list[int],
    ) -> tuple[mx.array, mx.array]:
        norm_hidden_states, target_hidden = self._forward_states(
            model,
            inputs,
            cache,
            layer_ids,
        )
        logits = self.lm_head_logits(model, norm_hidden_states)
        return logits, target_hidden

    def forward_verifier_states(
        self,
        model,
        inputs: mx.array,
        cache: list[Any],
        layer_ids: list[int],
    ) -> tuple[mx.array, mx.array]:
        return self._forward_states(model, inputs, cache, layer_ids)

    def forward_accept_all_block(
        self,
        model,
        inputs: mx.array,
        cache: list[Any],
        layer_ids: list[int],
    ) -> tuple[mx.array, mx.array]:
        norm_hidden_states, target_hidden = self._forward_states(
            model,
            inputs,
            cache,
            layer_ids,
        )
        return self.lm_head_logits(model, norm_hidden_states[:, -1:, :]), target_hidden

    def rewind_kv_caches(self, cache: list[Any], num_tokens: int) -> None:
        for layer_cache in cache:
            if isinstance(layer_cache, cache_lib.KVCache):
                layer_cache.trim(num_tokens)

    def cache_summary(self, cache: list[Any]) -> str:
        return " ".join(
            f"{idx}:kv={layer_cache.offset}"
            for idx, layer_cache in enumerate(cache)
            if isinstance(layer_cache, cache_lib.KVCache)
        )


class Qwen3VLTargetAdapter(MLXTargetAdapter):
    family = "qwen3_vl"

    def __init__(self):
        self._vision_inputs: dict | None = None

    def set_vision_inputs(self, **kwargs) -> None:
        self._vision_inputs = dict(kwargs) if kwargs else None

    def clear_vision_inputs(self) -> None:
        self._vision_inputs = None

    @staticmethod
    def _tokenizer_of(tokenizer):
        return getattr(tokenizer, "tokenizer", tokenizer)

    def build_prompt(
        self, tokenizer, prompt_text: str, enable_thinking: bool = False
    ) -> mx.array:
        tok = self._tokenizer_of(tokenizer)
        messages = [{"role": "user", "content": prompt_text}]
        try:
            prompt = tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            prompt = tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        tokens = tok.encode(prompt, add_special_tokens=False)
        return mx.array(tokens, dtype=mx.uint32)

    def stop_token_ids(self, tokenizer) -> set[int]:
        tok = self._tokenizer_of(tokenizer)
        eos = getattr(tok, "eos_token_ids", None)
        if eos is None:
            eos = getattr(tok, "eos_token_id", None)
        if isinstance(eos, int):
            return {eos}
        return set(eos) if eos is not None else set()

    def make_cache(self, model) -> list[Any]:
        from mlx_vlm.models.cache import KVCache

        layers = model.language_model.model.layers
        return [KVCache() for _ in layers]

    def embed_tokens(self, model, tokens: mx.array) -> mx.array:
        return model.language_model.model.embed_tokens(tokens)

    def _lm_head(self, model):
        if model.language_model.args.tie_word_embeddings:
            return model.language_model.model.embed_tokens.as_linear
        return model.language_model.lm_head

    def lm_head_logits(self, model, hidden_states: mx.array) -> mx.array:
        return self._lm_head(model)(hidden_states)

    def _decode_position_ids(self, model, seq_len: int, cache: list[Any]) -> mx.array:
        cache_offset = 0
        if cache and cache[0] is not None:
            c0 = cache[0]
            cache_offset = c0._idx if hasattr(c0, "_idx") else c0.offset
        cache_offset_arr = (
            cache_offset
            if isinstance(cache_offset, mx.array)
            else mx.array(int(cache_offset))
        )
        rope_delta = getattr(model.language_model, "_rope_deltas", None)
        if rope_delta is not None:
            delta = cache_offset_arr + rope_delta.reshape(-1)[0]
        else:
            delta = cache_offset_arr
        pos = mx.arange(seq_len) + delta
        pos = pos.reshape(1, 1, seq_len)
        return mx.broadcast_to(pos, (3, 1, seq_len))

    def _vlm_layer_forward(
        self,
        model,
        inputs: mx.array,
        cache: list[Any],
        layer_ids: list[int],
        *,
        inputs_embeds: mx.array | None = None,
        position_ids: mx.array | None = None,
        visual_pos_masks: mx.array | None = None,
        deepstack_visual_embeds=None,
    ) -> tuple[mx.array, mx.array]:
        from mlx_vlm.models.base import create_attention_mask

        text_model = model.language_model.model
        h = (
            inputs_embeds
            if inputs_embeds is not None
            else text_model.embed_tokens(inputs)
        )
        mask = create_attention_mask(
            h, cache[0] if cache and cache[0] is not None else None
        )
        position_embeddings = None
        if (
            position_ids is not None
            and text_model.layers
            and not text_model.layers[0].self_attn.rotary_emb.fused_apply
        ):
            position_embeddings = text_model.layers[0].self_attn.rotary_emb(
                h, position_ids
            )
        selected: list[mx.array] = []
        target_layer_ids = set(layer_ids)
        if -1 in target_layer_ids:
            selected.append(h)
        for idx, (layer, c) in enumerate(zip(text_model.layers, cache)):
            h = layer(h, mask, c, position_ids, position_embeddings)
            if deepstack_visual_embeds is not None and idx in range(
                len(deepstack_visual_embeds)
            ):
                h = text_model._deepstack_process(
                    h, visual_pos_masks, deepstack_visual_embeds[idx]
                )
            if idx in target_layer_ids:
                selected.append(h)
        norm_hidden = text_model.norm(h)
        target_hidden = mx.concatenate(selected, axis=-1) if selected else norm_hidden
        return norm_hidden, target_hidden

    @staticmethod
    def _text_only_taps(
        target_hidden: mx.array, visual_pos_masks: mx.array | None
    ) -> mx.array:
        if visual_pos_masks is None:
            return target_hidden
        import numpy as np

        mask = visual_pos_masks
        if mask.ndim == 3:
            mask = mask[..., 0]
        text_idx = mx.array(np.where(~np.asarray(mask[0]))[0], dtype=mx.uint32)
        if int(text_idx.size) == 0:
            return target_hidden[:, -1:, :]
        return mx.take(target_hidden, text_idx, axis=1)

    def _prefill_with_vision(
        self, model, inputs: mx.array, cache: list[Any], layer_ids: list[int]
    ) -> tuple[mx.array, mx.array]:
        import numpy as np

        vi = self._vision_inputs or {}
        features = model.get_input_embeddings(
            inputs,
            pixel_values=vi.get("pixel_values"),
            image_grid_thw=vi.get("image_grid_thw"),
            video_grid_thw=vi.get("video_grid_thw"),
        )
        model.language_model._rope_deltas = features.rope_deltas
        model.language_model._position_ids = features.position_ids
        norm_hidden, target_hidden = self._vlm_layer_forward(
            model,
            inputs,
            cache,
            layer_ids,
            inputs_embeds=features.inputs_embeds,
            position_ids=features.position_ids,
            visual_pos_masks=features.visual_pos_masks,
            deepstack_visual_embeds=features.deepstack_visual_embeds,
        )
        vpm = features.visual_pos_masks
        target_hidden = self._text_only_taps(target_hidden, vpm)
        n_vision = 0
        if vpm is not None:
            m = vpm[..., 0] if vpm.ndim == 3 else vpm
            n_vision = int(np.asarray(m[0]).sum())
        seq_len = int(inputs.shape[1])
        logger.info(
            "dspark-vlm prefill: seq=%d vision=%d text=%d taps=%d",
            seq_len,
            n_vision,
            seq_len - n_vision,
            int(target_hidden.shape[1]),
        )
        self._vision_inputs = None
        return norm_hidden, target_hidden

    def _decode_text(
        self, model, inputs: mx.array, cache: list[Any], layer_ids: list[int]
    ) -> tuple[mx.array, mx.array]:
        seq_len = int(inputs.shape[1])
        position_ids = self._decode_position_ids(model, seq_len, cache)
        return self._vlm_layer_forward(
            model,
            inputs,
            cache,
            layer_ids,
            inputs_embeds=None,
            position_ids=position_ids,
            visual_pos_masks=None,
            deepstack_visual_embeds=None,
        )

    def forward_verifier_states(
        self,
        model,
        inputs: mx.array,
        cache: list[Any],
        layer_ids: list[int],
    ) -> tuple[mx.array, mx.array]:
        vi = self._vision_inputs
        if vi is not None and vi.get("pixel_values") is not None:
            return self._prefill_with_vision(model, inputs, cache, layer_ids)
        return self._decode_text(model, inputs, cache, layer_ids)

    def forward_with_hidden_states(
        self,
        model,
        inputs: mx.array,
        cache: list[Any],
        layer_ids: list[int],
    ) -> tuple[mx.array, mx.array]:
        norm_hidden, target_hidden = self._decode_text(model, inputs, cache, layer_ids)
        logits = self.lm_head_logits(model, norm_hidden)
        return logits, target_hidden

    def rewind_kv_caches(self, cache: list[Any], num_tokens: int) -> None:
        for layer_cache in cache:
            if layer_cache is not None and hasattr(layer_cache, "trim"):
                layer_cache.trim(num_tokens)

    def cache_summary(self, cache: list[Any]) -> str:
        parts = []
        for idx, c in enumerate(cache):
            if c is None:
                continue
            off = getattr(c, "_idx", getattr(c, "offset", "?"))
            parts.append(f"{idx}:kv={off}")
        return " ".join(parts)


ADAPTERS: dict[str, type[MLXTargetAdapter]] = {
    "qwen3": Qwen3TargetAdapter,
    "qwen3_vl": Qwen3VLTargetAdapter,
}


def adapter_for_model_type(model_type: str) -> type[MLXTargetAdapter] | None:
    return ADAPTERS.get(model_type)


@dataclass
class LoadedTargetModel:
    requested_model: str
    resolved_model_path: Path
    model: Any
    tokenizer: Any
    adapter: MLXTargetAdapter

    def build_prompt(self, prompt_text: str, enable_thinking: bool = False) -> mx.array:
        return self.adapter.build_prompt(
            self.tokenizer, prompt_text, enable_thinking=enable_thinking
        )

    def stop_token_ids(self) -> set[int]:
        return self.adapter.stop_token_ids(self.tokenizer)

    def make_cache(self) -> list[Any]:
        return self.adapter.make_cache(self.model)

    def embed_tokens(self, tokens: mx.array) -> mx.array:
        return self.adapter.embed_tokens(self.model, tokens)

    def lm_head_logits(self, hidden_states: mx.array) -> mx.array:
        return self.adapter.lm_head_logits(self.model, hidden_states)

    def lm_head_argmax(self, hidden_states: mx.array) -> mx.array:
        return self.adapter.lm_head_argmax(self.model, hidden_states)

    def forward_with_hidden_states(
        self,
        inputs: mx.array,
        cache: list[Any],
        layer_ids: list[int],
    ) -> tuple[mx.array, mx.array]:
        return self.adapter.forward_with_hidden_states(
            self.model,
            inputs,
            cache,
            layer_ids,
        )

    def forward_verifier_states(
        self,
        inputs: mx.array,
        cache: list[Any],
        layer_ids: list[int],
    ) -> tuple[mx.array, mx.array]:
        return self.adapter.forward_verifier_states(
            self.model,
            inputs,
            cache,
            layer_ids,
        )

    def forward_accept_all_block(
        self,
        inputs: mx.array,
        cache: list[Any],
        layer_ids: list[int],
    ) -> tuple[mx.array, mx.array]:
        return self.adapter.forward_accept_all_block(
            self.model,
            inputs,
            cache,
            layer_ids,
        )

    def rewind_kv_caches(self, cache: list[Any], num_tokens: int) -> None:
        self.adapter.rewind_kv_caches(cache, num_tokens)

    def cache_summary(self, cache: list[Any]) -> str:
        return self.adapter.cache_summary(cache)

    def set_vision_inputs(self, **kwargs) -> None:
        self.adapter.set_vision_inputs(**kwargs)

    def clear_vision_inputs(self) -> None:
        self.adapter.clear_vision_inputs()


def load_target_model(
    path_or_repo: str,
    model_config: dict[str, Any] | None = None,
) -> LoadedTargetModel:
    base_path = resolve_model_path(path_or_repo)
    config = json.loads((base_path / "config.json").read_text())
    model_type = config.get("model_type")
    adapter_cls = adapter_for_model_type(model_type)
    if adapter_cls is None:
        registered = ", ".join(sorted(ADAPTERS))
        raise NotImplementedError(
            f"Unsupported MLX DSpark target model_type={model_type!r} for "
            f"{path_or_repo!r}. A matching DSpark draft checkpoint is not enough; "
            "the target family also needs an MLX adapter for hidden-state "
            "extraction and exact cache rollback. Current adapters: "
            f"{registered}."
        )

    adapter = adapter_cls()
    resolved_model_path = adapter.resolve_target_model_path(path_or_repo)
    if adapter.family == "qwen3_vl":
        from mlx_vlm import load as vlm_load

        if model_config:
            logger.info(
                "dspark-vlm: model_config overrides ignored for VLM target %s",
                path_or_repo,
            )
        model, processor = vlm_load(str(resolved_model_path))
        tokenizer = processor
    else:
        # model_config entries (e.g. a rope_scaling YaRN override for
        # exploratory long-context benchmarks) are merged over the
        # checkpoint's config.json.
        model, tokenizer = load(str(resolved_model_path), model_config=model_config)
    return LoadedTargetModel(
        requested_model=path_or_repo,
        resolved_model_path=resolved_model_path,
        model=model,
        tokenizer=tokenizer,
        adapter=adapter,
    )
