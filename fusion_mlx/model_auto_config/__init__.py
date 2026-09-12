# SPDX-License-Identifier: Apache-2.0
"""model_auto_config package — per-model profile registry.

Split from the original model_auto_config.py monolith (62K) into:
- core.py: ModelConfig + detect_model_config + family detection + enrich
- profile.py: format_profile_table/summary + get_profile + suffix tier

Public surface preserved.
"""

from .core import (
    _DEEPSEEK_V3_BODY_PARSERS,
    _DEEPSEEK_V3_FAMILY_PARSERS,
    _DEEPSEEK_V31_BODY_PARSERS,
    _FAMILY_FROM_PATH,
    _HF_CACHE_INTERMEDIATE_SEGMENTS,
    _MODEL_PATTERNS,
    ModelConfig,
    _classify_deepseek_template_name,
    _deepseek_template_family,
    _detect_family_from_path,
    _extract_model_name_segment,
    _log_resolution_once,
    _reset_resolution_log_cache,
    detect_model_config,
    enrich_model_config,
    model_has_recurrent_cache,
    warn_misbound_deepseek_v3_parser,
)
from .profile import (
    _arch_label,
    _suffix_tier_cell,
    _truncate_tier_note,
    classify_suffix_decoding_tier,
    format_profile_summary,
    format_profile_table,
    get_profile,
    suffix_decoding_hint,
)

__all__ = [
    "ModelConfig",
    "_DEEPSEEK_V31_BODY_PARSERS",
    "_DEEPSEEK_V3_BODY_PARSERS",
    "_DEEPSEEK_V3_FAMILY_PARSERS",
    "_FAMILY_FROM_PATH",
    "_HF_CACHE_INTERMEDIATE_SEGMENTS",
    "_MODEL_PATTERNS",
    "_arch_label",
    "_classify_deepseek_template_name",
    "_deepseek_template_family",
    "_detect_family_from_path",
    "_extract_model_name_segment",
    "_log_resolution_once",
    "_reset_resolution_log_cache",
    "_suffix_tier_cell",
    "_truncate_tier_note",
    "classify_suffix_decoding_tier",
    "detect_model_config",
    "enrich_model_config",
    "format_profile_summary",
    "format_profile_table",
    "get_profile",
    "model_has_recurrent_cache",
    "suffix_decoding_hint",
    "warn_misbound_deepseek_v3_parser",
]
