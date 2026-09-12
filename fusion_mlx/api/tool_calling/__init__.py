# SPDX-License-Identifier: Apache-2.0
"""
Tool calling parsing and conversion utilities.

This package re-exports all symbols from the original tool_calling.py module.
Downstream code imports from fusion_mlx.api.tool_calling as before.
"""

from jsonschema import ValidationError, validate

# Re-export imported names for backward compatibility
from ..openai_models import FunctionCall, ResponseFormat, ToolCall
from .parse import (
    ToolCallExtraction,
    _decode_json_like,
    _extract_tool_names,
    _gemma4_args_to_json_robust,
    _parse_bracket_tool_calls,
    _parse_gemma4_tool_call_fallback,
    _parse_namespaced_tool_calls,
    _parse_xml_tool_calls,
    _serialize_tool_call_arguments,
    _tool_parser_result_to_dict,
    extract_tool_calls_with_thinking,
    parse_tool_calls,
    parse_tool_calls_with_thinking_fallback,
    sanitize_tool_call_markup,
)
from .schema_helpers import (
    _coerce_schema_value,
    _get_tool_param_config,
    _schema_type,
    build_json_system_prompt,
    check_schema_validity,
    extract_json_from_text,
    extract_json_schema_for_guided,
    is_strict_json_schema,
    parse_json_output,
    validate_json_schema,
    validate_output_against_schema,
)
from .stream_filter import (
    _GEMMA4_COLLIDING_PARAMS,
    _GEMMA4_RENAME_PREFIX,
    ToolCallStreamFilter,
    convert_tools_for_template,
    enrich_tool_params_for_gemma4,
    format_tool_call_for_message,
    restore_gemma4_param_names,
)
