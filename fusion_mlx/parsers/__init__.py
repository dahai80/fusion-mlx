"""mlx-tool-parsers — Tool call format parsers.

Auto-detects and parses tool call formats for Qwen, Gemma, Harmony, Llama, etc.
Merged from omlx/tool_parser + Rapid-MLX tool logits bias.
"""

from .gemma4 import Gemma4OutputParserSession  # noqa: F401
from .harmony import HarmonyStreamingParser  # noqa: F401
from .output_parser import (
    OutputParserFactory,
    OutputParserSession,
    detect_output_parser,
)

__all__ = [
    "OutputParserFactory",
    "OutputParserSession",
    "detect_output_parser",
]
