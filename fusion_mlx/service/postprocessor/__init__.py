# SPDX-License-Identifier: Apache-2.0
"""Streaming post-processor — unified reasoning + tool call + sanitization pipeline.

Replaces 500+ lines of duplicated logic across stream_chat_completion,
_stream_anthropic_messages, and stream_completion. NOT a filter chain —
one cohesive orchestrator, because reasoning/tool/sanitize are tightly coupled.
"""

from .core import StreamingPostProcessor
from .formatters import (
    _find_json_start,
    _find_json_fence_opener,
    _json_fence_suffix_hold_len,
)
from .parsers import (
    _create_reasoning_parser,
    _create_tool_parser,
    _clone_injected_tool_parser,
    _forced_tool_choice_arguments_violate_object_root,
    _continuation_arguments_definitively_non_object,
)

# Re-export names that external code patches via
# "fusion_mlx.service.postprocessor.<name>"
from ...api.tool_calling import parse_tool_calls
from ...api.utils import sanitize_output, strip_special_tokens

__all__ = ["StreamingPostProcessor"]
