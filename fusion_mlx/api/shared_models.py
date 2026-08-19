# SPDX-License-Identifier: Apache-2.0
"""Shared models and utilities for API responses."""

import math
import time
import uuid
from enum import Enum

from pydantic import BaseModel


class IDPrefix(str, Enum):
    """Prefixes for generated IDs."""

    CHAT_COMPLETION = "chatcmpl"
    COMPLETION = "cmpl"
    MESSAGE = "msg"
    EMBEDDING = "emb"
    RERANK = "rerank"
    RESPONSE = "resp"
    FUNCTION_CALL = "fc"
    REASONING = "rs"


def generate_id(prefix: IDPrefix, length: int = 8) -> str:
    """Generate a unique ID with the given prefix.

    Args:
        prefix: The ID prefix to use
        length: Length of the random suffix (default 8)

    Returns:
        Generated ID string (e.g., "chatcmpl-abc12345")
    """
    if prefix == IDPrefix.MESSAGE:
        # Anthropic style: msg_<24-char-hex>
        return f"msg_{uuid.uuid4().hex[:24]}"
    if prefix == IDPrefix.RESPONSE:
        return f"resp_{uuid.uuid4().hex[:24]}"
    if prefix == IDPrefix.FUNCTION_CALL:
        return f"fc_{uuid.uuid4().hex[:8]}"
    if prefix == IDPrefix.REASONING:
        return f"rs_{uuid.uuid4().hex[:24]}"
    return f"{prefix.value}-{uuid.uuid4().hex[:length]}"


def get_unix_timestamp() -> int:
    """Get current Unix timestamp.

    Returns:
        Current time as Unix timestamp (integer seconds since epoch)
    """
    return int(time.time())


class BaseUsage(BaseModel):
    """Base class for token usage statistics.

    This provides a foundation for both OpenAI-style (prompt_tokens/completion_tokens)
    and Anthropic-style (input_tokens/output_tokens) usage tracking.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def model_post_init(self, __context) -> None:
        """Calculate total_tokens and sync Anthropic-style aliases."""
        if self.total_tokens == 0 and (
            self.prompt_tokens > 0 or self.completion_tokens > 0
        ):
            object.__setattr__(
                self,
                "total_tokens",
                self.prompt_tokens + self.completion_tokens,
            )
        if self.input_tokens == 0 and self.prompt_tokens > 0:
            object.__setattr__(self, "input_tokens", self.prompt_tokens)
        if self.output_tokens == 0 and self.completion_tokens > 0:
            object.__setattr__(self, "output_tokens", self.completion_tokens)


# r5-E B-7 / R6-H8: shared sampling-param validators used across all
# three OpenAI-shape surfaces (/v1/chat/completions, /v1/responses,
# /v1/completions). Single source of truth for thresholds so no surface
# has an escape hatch.
_TOP_K_SENTINEL_CAP = 1 << 20  # 1,048,576 — inclusive upper bound


def validate_top_k(v, field_name: str = "top_k"):
    # Reject bool (Python bool is an int subclass -> would coerce True->1
    # silently), non-int wire shapes, negatives, and pathological values
    # past the sentinel cap. None is the server-default sentinel and stays
    # valid. top_k=0 is the documented mlx-lm "disabled" sentinel (legal).
    if v is None:
        return v
    if isinstance(v, bool):
        raise ValueError(f"{field_name} must be an integer, got boolean")
    if not isinstance(v, int):
        raise ValueError(f"{field_name} must be an integer")
    if v < 0:
        raise ValueError(f"{field_name} must be >= 0")
    if v > _TOP_K_SENTINEL_CAP:
        raise ValueError(
            f"{field_name}={v} exceeds the sentinel cap "
            f"({_TOP_K_SENTINEL_CAP}); use a smaller value or 0 to disable"
        )
    return v


def validate_seed(v, field_name: str = "seed"):
    # Reject bool (would coerce True->1) and negatives. seed=0 is a
    # legitimate PRNG key (eval harnesses) and stays valid. None is the
    # server-default sentinel.
    if v is None:
        return v
    if isinstance(v, bool):
        raise ValueError(f"{field_name} must be an integer, got boolean")
    if not isinstance(v, int):
        raise ValueError(f"{field_name} must be an integer")
    if v < 0:
        raise ValueError(f"{field_name} must be >= 0")
    return v


def _validate_finite_in_range(
    v,
    min_value: float | None = None,
    max_value: float | None = None,
    field_name: str = "value",
    min_inclusive: bool = True,
    max_inclusive: bool = True,
):
    # H-10 shared float gate: reject nan/inf + enforce range bounds. None
    # is the server-default sentinel and passes through unchanged. Used as
    # a Pydantic field_validator on every sampling-float param so the
    # rejection envelope is uniform across chat/completions/anthropic.
    if v is None:
        return v
    if isinstance(v, bool):
        raise ValueError(f"{field_name} must be a finite number, got boolean")
    if not isinstance(v, (int, float)):
        raise ValueError(f"{field_name} must be a finite number")
    if not math.isfinite(float(v)):
        raise ValueError(f"{field_name} must be a finite number")
    fv = float(v)
    if min_value is not None:
        if min_inclusive:
            if fv < min_value:
                raise ValueError(f"{field_name} must be >= {min_value}")
        else:
            if fv <= min_value:
                raise ValueError(f"{field_name} must be > {min_value}")
    if max_value is not None:
        if max_inclusive:
            if fv > max_value:
                raise ValueError(f"{field_name} must be <= {max_value}")
        else:
            if fv >= max_value:
                raise ValueError(f"{field_name} must be < {max_value}")
    return v


def _validate_nonnegative_int(v, field_name: str = "value"):
    # H-10 shared int gate: reject bool, non-integer wire shapes, negatives.
    # Accept integer-valued floats (64.0 -> 64) to mirror Pydantic v2 lax
    # coercion so this gate is purely additive over the legacy path. None is
    # the server-default sentinel and passes through.
    if v is None:
        return v
    if isinstance(v, bool):
        raise ValueError(f"{field_name} must be an integer, got boolean")
    if isinstance(v, float):
        if not math.isfinite(v) or v != int(v):
            raise ValueError(f"{field_name} must be an integer")
        v = int(v)
    if not isinstance(v, int):
        raise ValueError(f"{field_name} must be an integer")
    if v < 0:
        raise ValueError(f"{field_name} must be >= 0")
    return v


def _validate_logit_bias_finite(v):
    # H-10 logit_bias value gate (chat + legacy completions; Anthropic has
    # no logit_bias). None / empty dict pass through. Each value must be a
    # finite number; bool values are rejected defensively (a wire bool is
    # almost certainly a serialization bug, not a +1.0 bias).
    if v is None:
        return v
    if not isinstance(v, dict):
        return v
    for key, bias in v.items():
        if isinstance(bias, bool):
            raise ValueError(
                f"logit_bias[{key!r}] must be a finite number, got boolean"
            )
        if not isinstance(bias, (int, float)) or not math.isfinite(float(bias)):
            raise ValueError(f"logit_bias[{key!r}] must be a finite number")
    return v
