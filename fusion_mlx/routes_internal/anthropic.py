import json
import logging

logger = logging.getLogger(__name__)

from ..api._anthropic_helpers import (  # noqa: F401
    _enforce_named_tool_choice_present,
    _filter_tool_calls_by_tool_choice,
)
from ..api.anthropic_routes import router  # noqa: F401


def _split_tool_input_json(tool_input: object) -> list[str]:
    if not isinstance(tool_input, dict) or not tool_input:
        return [json.dumps(tool_input)]
    if not all(isinstance(k, str) for k in tool_input):
        return [json.dumps(tool_input)]
    keys = list(tool_input.keys())
    fragments: list[str] = []
    for i, key in enumerate(keys):
        value_repr = json.dumps(tool_input[key])
        key_repr = json.dumps(key)
        opener = "{" if i == 0 else ", "
        closer = "}" if i == len(keys) - 1 else ""
        fragments.append(f"{opener}{key_repr}: {value_repr}{closer}")
    return fragments
