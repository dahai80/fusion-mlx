# SPDX-License-Identifier: Apache-2.0
"""
Structured output (JSON Schema) utilities.

Functions for validating JSON against schemas, extracting JSON from model
output, and building system prompts for structured output.
"""

import json
import logging
import re
from typing import Any

from jsonschema import ValidationError, validate

from ..openai_models import ResponseFormat
from .parse import _decode_json_like

logger = logging.getLogger(__name__)


def _get_tool_param_config(
    tool_name: str | None, request: dict[str, Any] | None
) -> dict[str, Any]:
    if not tool_name or not isinstance(request, dict):
        return {}
    tools = request.get("tools")
    if not isinstance(tools, list):
        return {}
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function")
        if not isinstance(function, dict) or function.get("name") != tool_name:
            continue
        parameters = function.get("parameters")
        if not isinstance(parameters, dict):
            return {}
        properties = parameters.get("properties")
        if isinstance(properties, dict):
            return properties
        return parameters
    return {}


def _schema_type(schema: Any) -> str | None:
    if isinstance(schema, str):
        return schema.strip().lower()
    if not isinstance(schema, dict):
        return None
    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        schema_type = next((item for item in schema_type if item != "null"), None)
    if isinstance(schema_type, str):
        return schema_type.strip().lower()
    for key in ("anyOf", "oneOf", "allOf"):
        options = schema.get(key)
        if isinstance(options, list):
            for option in options:
                option_type = _schema_type(option)
                if option_type and option_type != "null":
                    return option_type
    if "items" in schema:
        return "array"
    if "properties" in schema or "additionalProperties" in schema:
        return "object"
    if "enum" in schema:
        return "string"
    return None


def _coerce_schema_value(value: Any, schema: Any) -> Any:
    value = _decode_json_like(value)
    schema_type = _schema_type(schema)
    if schema_type is None:
        return value
    if value is None:
        return None
    if schema_type in ("string", "str", "text", "varchar", "char", "enum"):
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False)
    if schema_type in ("array", "object"):
        return value
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    try:
        if schema_type in ("integer", "int"):
            return int(stripped)
        if schema_type in ("number", "float"):
            return float(stripped)
    except (TypeError, ValueError):
        return value
    if schema_type in ("boolean", "bool"):
        if stripped.lower() == "true":
            return True
        if stripped.lower() == "false":
            return False
    return value


# =============================================================================
# Structured Output (JSON Schema) Utilities
# =============================================================================


def validate_json_schema(data: Any, schema: dict[str, Any]) -> tuple[bool, str | None]:
    """
    Validate JSON data against a JSON Schema.

    Args:
        data: The JSON data to validate (dict, list, etc.)
        schema: JSON Schema specification

    Returns:
        Tuple of (is_valid, error_message)
        - is_valid: True if data matches schema
        - error_message: Error description if invalid, None if valid
    """
    try:
        validate(instance=data, schema=schema)
        return True, None
    except ValidationError as e:
        return False, str(e.message)


def extract_json_from_text(text: str) -> dict[str, Any] | None:
    """
    Extract JSON from model output text.

    Tries multiple strategies:
    1. Parse entire text as JSON
    2. Extract JSON from markdown code blocks
    3. Find JSON object/array in text

    Args:
        text: Raw model output text

    Returns:
        Parsed JSON data, or None if no valid JSON found
    """
    text = text.strip()

    # Strategy 1: Try to parse entire text as JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: Extract from markdown code blocks
    # Match ```json ... ``` or ``` ... ```
    code_block_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
    matches = re.findall(code_block_pattern, text)
    for match in matches:
        try:
            return json.loads(match.strip())
        except json.JSONDecodeError:
            continue

    # Strategy 3: Find JSON object or array in text
    # Look for { ... } or [ ... ]
    json_patterns = [
        r"(\{[\s\S]*\})",  # Object
        r"(\[[\s\S]*\])",  # Array
    ]
    for pattern in json_patterns:
        match = re.search(pattern, text)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                continue

    return None


def parse_json_output(
    text: str, response_format: ResponseFormat | dict[str, Any] | None = None
) -> tuple[str, dict[str, Any] | None, bool, str | None]:
    """
    Parse JSON from model output when response_format is set.

    Args:
        text: Raw model output text
        response_format: ResponseFormat specification (optional)
            - If type="json_object", extracts any valid JSON
            - If type="json_schema", extracts and validates against schema

    Returns:
        Tuple of (cleaned_text, parsed_json, is_valid, error_message)
        - cleaned_text: Original text (preserved for reference)
        - parsed_json: Extracted JSON data, or None if extraction failed
        - is_valid: True if JSON is valid (and matches schema if specified)
        - error_message: Error description if invalid, None if valid
    """
    # Handle None or text format - just return original
    if response_format is None:
        return text, None, True, None

    # Normalize response_format to dict. ``models.py`` and
    # ``openai_models.py`` define identical ``ResponseFormat`` duplicates;
    # duck-type via ``model_dump`` so an isinstance against one class does
    # not miss an instance of the other (fixes ``'ResponseFormat' object has
    # no attribute 'get'`` when the request model uses the models.py variant).
    if hasattr(response_format, "model_dump"):
        rf_dict = {"type": response_format.type, "json_schema": None}
        if response_format.json_schema:
            rf_dict["json_schema"] = {
                "name": response_format.json_schema.name,
                "description": response_format.json_schema.description,
                "schema": response_format.json_schema.schema_,
                "strict": response_format.json_schema.strict,
            }
    else:
        rf_dict = response_format

    format_type = rf_dict.get("type", "text")

    # text format - no JSON extraction
    if format_type == "text":
        return text, None, True, None

    # json_object or json_schema - extract JSON
    parsed = extract_json_from_text(text)

    if parsed is None:
        return text, None, False, "Failed to extract valid JSON from output"

    # json_object - just verify it's valid JSON (already done by extraction)
    if format_type == "json_object":
        return text, parsed, True, None

    # json_schema - validate against schema
    if format_type == "json_schema":
        json_schema_spec = rf_dict.get("json_schema", {})
        schema = json_schema_spec.get("schema", {})

        if schema:
            is_valid, error = validate_json_schema(parsed, schema)
            if not is_valid:
                return text, parsed, False, f"JSON Schema validation failed: {error}"

        return text, parsed, True, None

    # Unknown format type - treat as text
    return text, None, True, None


def build_json_system_prompt(
    response_format: ResponseFormat | dict[str, Any] | None = None,
) -> str | None:
    """
    Build a system prompt instruction for JSON output.

    For models without native JSON mode support, this adds instructions
    to the prompt to encourage proper JSON formatting.

    Args:
        response_format: ResponseFormat specification

    Returns:
        System prompt instruction string, or None if not needed
    """
    if response_format is None:
        return None

    # Normalize to dict. Two ResponseFormat pydantic classes coexist
    # (api.models.ResponseFormat vs api.openai_models.ResponseFormat);
    # isinstance only catches the openai_models one, so duck-type on
    # model_dump to route both into the instance arm - mirrors
    # extract_json_schema_for_guided / is_strict_json_schema below.
    if hasattr(response_format, "model_dump"):
        rf_dict = {"type": response_format.type, "json_schema": None}
        if response_format.json_schema:
            rf_dict["json_schema"] = {
                "name": response_format.json_schema.name,
                "description": response_format.json_schema.description,
                "schema": response_format.json_schema.schema_,
                "strict": response_format.json_schema.strict,
            }
    else:
        rf_dict = response_format

    format_type = rf_dict.get("type", "text")

    if format_type == "text":
        return None

    if format_type == "json_object":
        return (
            "You must respond with valid JSON only. "
            "Do not include any explanation or text outside the JSON object."
        )

    if format_type == "json_schema":
        json_schema_spec = rf_dict.get("json_schema", {})
        schema = json_schema_spec.get("schema", {})
        name = json_schema_spec.get("name", "response")
        description = json_schema_spec.get("description", "")

        prompt = f"You must respond with valid JSON matching the '{name}' schema."
        if description:
            prompt += f" {description}"
        prompt += (
            f"\n\nJSON Schema:\n```json\n{json.dumps(schema, indent=2)}\n```\n\n"
            "Respond with only the JSON object, no additional text or explanation."
        )
        return prompt

    return None


def check_schema_validity(json_schema: dict) -> tuple[bool, str | None]:
    try:
        from jsonschema.exceptions import SchemaError
        from jsonschema.validators import validator_for
    except ImportError:
        return True, None
    try:
        validator_cls = validator_for(json_schema)
    except TypeError as exc:
        return False, f"{type(exc).__name__}: {exc}"
    try:
        validator_cls.check_schema(json_schema)
    except SchemaError as exc:
        return False, f"{type(exc).__name__}: {exc}"
    except TypeError as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return True, None


def extract_json_schema_for_guided(response_format) -> dict | None:
    if response_format is None:
        return None
    if hasattr(response_format, "model_dump"):
        rf_dict = response_format.model_dump(by_alias=True)
    elif isinstance(response_format, dict):
        rf_dict = response_format
    else:
        return None
    format_type = rf_dict.get("type", "text")
    if format_type != "json_schema":
        return None
    json_schema_spec = rf_dict.get("json_schema", {})
    schema = json_schema_spec.get("schema", {})
    if not schema:
        return None
    return schema


def is_strict_json_schema(response_format) -> bool:
    if response_format is None:
        return False
    if hasattr(response_format, "model_dump"):
        rf_dict = response_format.model_dump(by_alias=True)
    elif isinstance(response_format, dict):
        rf_dict = response_format
    else:
        return False
    if rf_dict.get("type") != "json_schema":
        return False
    if rf_dict.get("strict") is True:
        return True
    spec = rf_dict.get("json_schema") or {}
    if not isinstance(spec, dict):
        return False
    return spec.get("strict") is True


def validate_output_against_schema(
    output_text: str, json_schema: dict[str, Any]
) -> tuple[bool, str | None]:
    text = (output_text or "").strip()
    if not text:
        return False, "empty output"
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        return False, f"invalid JSON: {exc}"
    try:
        validate(instance=parsed, schema=json_schema)
    except ValidationError as exc:
        return False, f"schema violation: {exc.message}"
    except Exception as exc:
        return False, f"validator error: {type(exc).__name__}: {exc}"
    return True, None
