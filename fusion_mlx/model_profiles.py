"""Model profile and template primitives for per-model settings."""

from __future__ import annotations

import logging
import re
import unicodedata
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)

UNIVERSAL_PROFILE_FIELDS = (
    "max_context_window",
    "max_tokens",
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "repetition_penalty",
    "presence_penalty",
    "force_sampling",
    "enable_thinking",
    "preserve_thinking",
    "thinking_budget_enabled",
    "thinking_budget_tokens",
    "reasoning_parser",
    "guided_grammar_enabled",
    "guided_grammar",
    "max_tool_result_tokens",
    "chat_template_kwargs",
    "forced_ct_kwargs",
)

MODEL_SPECIFIC_PROFILE_FIELDS = (
    "turboquant_kv_enabled",
    "turboquant_kv_bits",
    "turboquant_skip_last",
    "dflash_enabled",
    "dflash_draft_model",
    "dflash_draft_quant_enabled",
    "dflash_draft_quant_weight_bits",
    "dflash_draft_quant_activation_bits",
    "dflash_draft_quant_group_size",
    "dflash_max_ctx",
    "dflash_in_memory_cache",
    "dflash_in_memory_cache_max_entries",
    "dflash_in_memory_cache_max_bytes",
    "dflash_ssd_cache",
    "dflash_ssd_cache_max_bytes",
    "dflash_draft_window_size",
    "dflash_draft_sink_size",
    "dflash_verify_mode",
    "mtp_enabled",
    "vlm_mtp_enabled",
    "vlm_mtp_draft_model",
    "vlm_mtp_draft_block_size",
    "specprefill_enabled",
    "specprefill_draft_model",
    "specprefill_keep_pct",
    "specprefill_threshold",
    "index_cache_freq",
    "ngram_spec_enabled",
    "ngram_spec_order",
    "ngram_spec_num_draft",
    "ngram_spec_break_even",
)

EXCLUDED_FROM_PROFILES = frozenset(
    {
        "is_pinned",
        "is_default",
        "display_name",
        "description",
        "model_alias",
        "model_type_override",
        "active_profile_name",
        "ttl_seconds",
        "trust_remote_code",
    }
)


def filter_universal_fields(data: dict[str, Any]) -> dict[str, Any]:
    allowed = set(UNIVERSAL_PROFILE_FIELDS)
    return {k: v for k, v in data.items() if k in allowed}


def filter_profile_fields(data: dict[str, Any]) -> dict[str, Any]:
    allowed = set(UNIVERSAL_PROFILE_FIELDS) | set(MODEL_SPECIFIC_PROFILE_FIELDS)
    return {k: v for k, v in data.items() if k in allowed}


def utcnow() -> datetime:
    return datetime.now(UTC)


class InvalidProfileNameError(ValueError):
    pass


_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")


def validate_profile_name(name: str) -> None:
    if not isinstance(name, str) or not _NAME_RE.match(name):
        raise InvalidProfileNameError(
            f"Invalid profile/template name: {name!r}. "
            f"Must match ^[a-z0-9][a-z0-9_-]{{0,31}}$"
        )


def slugify_profile_api_name(value: str | None, fallback: str = "profile") -> str:
    text = unicodedata.normalize("NFKD", value or "")
    text = text.encode("ascii", "ignore").decode("ascii").lower()
    text = re.sub(r"[^a-z0-9_-]+", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-_")
    if not text or not re.match(r"^[a-z0-9]", text):
        text = fallback
    text = text[:32].rstrip("-_")
    if not text or not _NAME_RE.match(text):
        text = fallback[:32].rstrip("-_") or "profile"
    return text
