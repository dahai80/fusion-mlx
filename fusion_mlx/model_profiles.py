"""Model profile definitions for fusion-mlx."""

import logging

logger = logging.getLogger(__name__)

# Models that should not be included in auto-profile detection
EXCLUDED_FROM_PROFILES: list[str] = [
    "google/t5-v1_1-xxl",
    "OpenSuper/CLIP-ViT-bigG-14",
    "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
]

# Default model profiles: model_id -> {params, type, context}
DEFAULT_PROFILES: dict[str, dict[str, str]] = {}

# Profile fields that apply to ALL models regardless of architecture
UNIVERSAL_PROFILE_FIELDS: list[str] = [
    "model_alias",
    "max_context_window",
    "max_tokens",
    "temperature",
    "top_p",
    "top_k",
    "repetition_penalty",
    "min_p",
    "presence_penalty",
    "force_sampling",
    "max_tool_result_tokens",
    "chat_template_kwargs",
    "forced_ct_kwargs",
    "ttl_seconds",
    "index_cache_freq",
    "enable_thinking",
    "thinking_budget_enabled",
    "thinking_budget_tokens",
    "turboquant_kv_enabled",
    "turboquant_kv_bits",
    "reasoning_parser",
    "is_pinned",
    "is_default",
    "trust_remote_code",
]

# Profile fields only relevant for specific model architectures
MODEL_SPECIFIC_PROFILE_FIELDS: list[str] = [
    "model_type_override",
    "specprefill_enabled",
    "specprefill_draft_model",
    "specprefill_keep_pct",
    "specprefill_threshold",
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
    "ngram_spec_enabled",
    "ngram_spec_order",
    "ngram_spec_num_draft",
    "ngram_spec_break_even",
]
