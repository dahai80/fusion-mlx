"""mlx-tool-parsers — Tool call format parsers.

Auto-detects and parses tool call formats for Qwen, Gemma, Harmony, Llama, etc.
Merged from omlx/tool_parser + Rapid-MLX tool logits bias.
"""

from .output_parser import OutputParserFactory, OutputParserSession, detect_output_parser
from .gemma4 import Gemma4Parser
from .harmony import HarmonyParser

__all__ = [
        "OutputParserFactory",
        "OutputParserSession",
        "detect_output_parser",
        "Gemma4Parser",
        "HarmonyParser",
]
