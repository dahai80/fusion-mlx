# SPDX-License-Identifier: Apache-2.0
# Vendored from mflux 0.20.0 (https://github.com/filipstrand/mflux) PR#736
# MIT-licensed pure-MLX Qwen-Image-2.1 reference (txt2img + 4-ch RGBA VAE).
# Pinned to fusion-mlx's mlx stack. Upstream-tracked; do not modify forward
# numerics without parity-testing against mflux qwen21.

import mlx.core as mx

from mflux.models.common.tokenizer import Tokenizer
from .qwen21_text_encoder import Qwen21TextEncoder


class Qwen21PromptEncoder:
    SYSTEM_PROMPT = "Comprehend and analyze the provided prompt."
    # Raw template string, not apply_chat_template: the checkpoint was trained on this form.
    PROMPT_TEMPLATE_T2I = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{{}}<|im_end|>\n<|im_start|>assistant\n"
    )
    SYSTEM_PREFIX = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"

    @staticmethod
    def encode_prompt(
        prompt: str,
        prompt_cache: dict[str, tuple[mx.array, mx.array]],
        tokenizer: Tokenizer,
        text_encoder: Qwen21TextEncoder,
    ) -> tuple[mx.array, mx.array]:
        if prompt in prompt_cache:
            return prompt_cache[prompt]
        if not prompt or not prompt.strip():
            prompt = " "
        tokens = tokenizer.tokenize(prompt)
        hidden_states = text_encoder(input_ids=tokens.input_ids, attention_mask=tokens.attention_mask)
        drop_idx = Qwen21PromptEncoder._system_prefix_length(tokenizer)

        prompt_embeds = hidden_states[:, drop_idx:, :]
        prompt_mask = tokens.attention_mask[:, drop_idx:]
        prompt_cache[prompt] = (prompt_embeds, prompt_mask)
        return prompt_embeds, prompt_mask

    @staticmethod
    def _system_prefix_length(tokenizer: Tokenizer) -> int:
        prefix_tokens = tokenizer.tokenizer(Qwen21PromptEncoder.SYSTEM_PREFIX, add_special_tokens=False)["input_ids"]
        return len(prefix_tokens)
