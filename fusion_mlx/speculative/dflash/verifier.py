# SPDX-License-Identifier: Apache-2.0
import logging
import math
from dataclasses import dataclass
from typing import Any

import mlx.core as mx

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VerifyResult:
    accepted_tokens: tuple[int, ...]
    bonus_token: int
    accepted_len: int
    block_was_full: bool
    verify_offset_after: int


def _argmax_per_position(logits: mx.array) -> mx.array:
    if logits.ndim != 3:
        raise ValueError(
            f"verify logits must be 3-D (B, S, V); got shape {tuple(logits.shape)}"
        )
    return mx.argmax(logits, axis=-1).astype(mx.uint32)


def _decide_accepted_prefix(
    verify_argmax: list[int],
    draft_tokens: list[int],
) -> tuple[int, int]:
    if len(verify_argmax) != len(draft_tokens):
        raise ValueError(
            f"verify_argmax length {len(verify_argmax)} must equal "
            f"draft_tokens length {len(draft_tokens)}"
        )
    accepted_len = 0
    for v, d in zip(verify_argmax, draft_tokens, strict=True):
        if v == d:
            accepted_len += 1
        else:
            break
    if accepted_len == len(draft_tokens):
        return accepted_len, -1
    return accepted_len, verify_argmax[accepted_len]


def _isfinite_or_raise(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name}={value!r} is not finite (NaN / Inf not allowed)")


def verify_block(
    model: Any,
    draft_block: mx.array,
    *,
    last_confirmed_token: int,
    cache: list[Any],
    block_size: int,
    current_offset: int,
    temperature: float = 0.0,
) -> VerifyResult:
    _isfinite_or_raise("temperature", float(temperature))
    if temperature != 0.0:
        raise NotImplementedError(
            "DFlash verifier currently only supports greedy decoding "
            "(temperature=0.0)."
        )
    if draft_block.ndim != 1:
        raise ValueError(
            f"draft_block must be 1-D; got shape {tuple(draft_block.shape)}"
        )
    if int(draft_block.shape[0]) != block_size:
        raise ValueError(
            f"draft_block length {int(draft_block.shape[0])} does not "
            f"match block_size={block_size}"
        )
    if current_offset < 0:
        raise ValueError(f"current_offset must be >= 0; got {current_offset}")

    inputs = mx.concatenate(
        [
            mx.array([last_confirmed_token], dtype=mx.uint32),
            draft_block,
        ]
    )

    logits = model(inputs[None], cache=cache)

    verify_argmax_arr = _argmax_per_position(logits)
    mx.eval(verify_argmax_arr)
    verify_argmax = [int(x) for x in verify_argmax_arr[0, :].tolist()]
    draft_list = [int(x) for x in draft_block.tolist()]

    block_argmax = verify_argmax[:block_size]
    accepted_len, bonus_token = _decide_accepted_prefix(block_argmax, draft_list)

    block_was_full = accepted_len == block_size
    if block_was_full:
        bonus_token = verify_argmax[block_size]

    verify_offset_after = current_offset + accepted_len
    if not block_was_full:
        _rewind_cache_to(cache, verify_offset_after)

    accepted_tokens = tuple(draft_list[:accepted_len])
    return VerifyResult(
        accepted_tokens=accepted_tokens,
        bonus_token=bonus_token,
        accepted_len=accepted_len,
        block_was_full=block_was_full,
        verify_offset_after=verify_offset_after,
    )


def _rewind_cache_to(cache: list[Any], target_offset: int) -> None:
    for c in cache:
        if hasattr(c, "offset"):
            try:
                if c.offset > target_offset:
                    c.offset = target_offset
            except AttributeError:
                logger.warning(
                    "[dflash.verifier] cache %s has read-only offset; "
                    "skipping rewind.",
                    type(c).__name__,
                )
