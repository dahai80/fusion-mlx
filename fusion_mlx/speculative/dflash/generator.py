# SPDX-License-Identifier: Apache-2.0
import logging
import math
from collections.abc import Generator
from typing import Any

import mlx.core as mx

from .accept_counter import get_global_counter
from .drafter import BlockDiffusionDrafter
from .verifier import verify_block

logger = logging.getLogger(__name__)


def _validate_block_size(block_size: int) -> None:
    if not isinstance(block_size, int):
        raise TypeError(f"block_size must be int; got {type(block_size).__name__}")
    if block_size < 1:
        raise ValueError(f"block_size must be >= 1; got {block_size}")


def _validate_temperature(temperature: float) -> None:
    if not math.isfinite(float(temperature)):
        raise ValueError(
            f"temperature={temperature!r} is not finite (NaN / Inf not allowed)"
        )
    if temperature != 0.0:
        raise NotImplementedError(
            "DFlash generator currently only supports temperature=0.0."
        )


def dflash_generate_step(
    prompt: mx.array,
    model: Any,
    drafter: BlockDiffusionDrafter,
    *,
    block_size: int = 16,
    max_tokens: int = 256,
    temperature: float = 0.0,
    accept_counter: Any = None,
) -> Generator[tuple[int, mx.array | None, bool], None, None]:
    _validate_block_size(block_size)
    _validate_temperature(temperature)
    if drafter.block_size != block_size:
        raise ValueError(
            f"drafter.block_size={drafter.block_size} does not match "
            f"requested block_size={block_size}"
        )
    if prompt.ndim != 1:
        raise ValueError(f"prompt must be 1-D; got shape {tuple(prompt.shape)}")
    if prompt.shape[0] == 0:
        raise ValueError("prompt must contain at least one token")
    if max_tokens < 1:
        raise ValueError(f"max_tokens must be >= 1; got {max_tokens}")

    if accept_counter is None:
        accept_counter = get_global_counter()

    drafter.reset()

    from mlx_lm.models import cache as _cache_module

    cache: list[Any] = _cache_module.make_prompt_cache(model)

    prefill_logits = model(prompt[None], cache=cache)
    last_logit = prefill_logits[:, -1, :]
    primary_token = int(mx.argmax(last_logit, axis=-1).item())

    emitted = 0
    yield primary_token, None, False
    emitted += 1
    if emitted >= max_tokens:
        return

    last_confirmed = primary_token
    current_offset = int(prompt.shape[0])

    while emitted < max_tokens:
        prefix_so_far = mx.array([last_confirmed], dtype=mx.uint32)
        try:
            draft_block = drafter.draft_block(prefix_so_far, current_offset)
        except IndexError:
            return
        accept_counter.record_attempt()

        result = verify_block(
            model,
            draft_block,
            last_confirmed_token=last_confirmed,
            cache=cache,
            block_size=block_size,
            current_offset=current_offset,
            temperature=temperature,
        )

        if result.accepted_len > 0:
            bonus_saved = max(0, result.accepted_len - 1)
            accept_counter.record_accept(tokens_saved=bonus_saved)
        else:
            accept_counter.record_reject()

        for tok in result.accepted_tokens:
            yield int(tok), None, True
            emitted += 1
            if emitted >= max_tokens:
                return

        if result.bonus_token >= 0:
            yield int(result.bonus_token), None, False
            emitted += 1
            if emitted >= max_tokens:
                return

        last_confirmed = int(result.bonus_token)
        current_offset = result.verify_offset_after + 1
