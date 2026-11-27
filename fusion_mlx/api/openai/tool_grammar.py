# SPDX-License-Identifier: Apache-2.0
# Default-on grammar forcing for tool-bearing requests (#1730).
#
# For any chat request that carries tools and has no explicit grammar
# constraint, compile an llguidance structural-tag grammar from the tool
# definitions so tool-call arguments are emitted as schema-valid JSON
# instead of degrading to plain text (the #1694 failure mode).
#
# Mechanism: llguidance structural tags - free text until the tokenizer's
# tool-call start marker appears, then marker + function name (literal per
# tool) + arguments JSON are constrained and the end marker forced.
# Composes with compiled_grammar plumbing (GrammarConstraintProcessor ->
# scheduler). GaaD-style MTP draft merge is a follow-up.
#
# Scope: Hermes-style markup only - Qwen/Qwen3/Qwen2.5 family. Other
# markups (minimax XML, gemma4, mistral) are skipped with a log line
# rather than constrained with a wrong grammar.

import logging
import os

logger = logging.getLogger(__name__)

NL = chr(10)
HERMES_START = "<tool_call>"
HERMES_END = "</tool_call>"
STRUCT_END = "}" + NL + HERMES_END


def tool_grammar_enabled():
    raw = os.environ.get("FUSION_MLX_TOOL_GRAMMAR")
    if raw is None:
        return True
    return raw.strip().lower() not in ("0", "false", "off", "no")


def _tool_entries(tools):
    entries = []
    for tool in tools:
        fn = getattr(tool, "function", None)
        if fn is None and isinstance(tool, dict):
            fn = tool.get("function")
        if isinstance(fn, dict):
            name = fn.get("name")
            params = fn.get("parameters")
        else:
            name = getattr(fn, "name", None)
            params = getattr(fn, "parameters", None)
        if not name:
            logger.debug("tool grammar: tool without name skipped")
            continue
        if not isinstance(params, dict):
            params = {"type": "object", "properties": {}}
        entries.append((str(name), params))
    return entries


def build_tool_grammar_spec(tools, tokenizer):
    if not getattr(tokenizer, "has_tool_calling", False):
        logger.debug("tool grammar: tokenizer has no tool-call markers")
        return None
    start = tokenizer.tool_call_start
    end = tokenizer.tool_call_end
    if start != HERMES_START or end != HERMES_END:
        logger.info(
            "tool grammar: unsupported tool-call markup (start=%r end=%r), skipping",
            [hex(ord(c)) for c in (start or "")],
            [hex(ord(c)) for c in (end or "")],
        )
        return None
    entries = _tool_entries(tools)
    if not entries:
        return None
    import llguidance as lg

    tags = []
    for name, params in entries:
        begin = HERMES_START + NL + '{"name": "' + name + '", "arguments": '
        tags.append(
            lg.StructTag(
                trigger=HERMES_START, begin=begin, grammar=params, end=STRUCT_END
            )
        )
    try:
        spec = lg.StructTag.to_grammar(tags, assume_special=False)
    except Exception as exc:
        logger.warning("tool grammar: structural tag build failed: %s", exc)
        return None
    logger.info(
        "tool grammar: structural tag compiled for %d tool(s): %s",
        len(entries),
        ", ".join(name for name, _ in entries),
    )
    return spec


def compile_tool_grammar(engine, request):
    if not tool_grammar_enabled():
        return None
    tools = getattr(request, "tools", None)
    if not tools:
        return None
    if getattr(request, "code_mode", False):
        return None
    if getattr(request, "tool_choice", None) == "none":
        return None
    if getattr(request, "structured_outputs", None) is not None:
        return None
    if getattr(request, "response_format", None) is not None:
        return None
    tokenizer = getattr(engine, "_tokenizer", None)
    if tokenizer is None:
        logger.debug("tool grammar: engine has no tokenizer, skipping")
        return None
    spec = build_tool_grammar_spec(tools, tokenizer)
    if spec is None:
        return None
    from fusion_mlx.api.grammar import create_llguidance_matcher

    matcher = create_llguidance_matcher(tokenizer, {"grammar": spec, "format": "lark"})
    if matcher is None:
        logger.warning("tool grammar: llguidance compilation failed, skipping")
    else:
        logger.info("tool grammar: matcher compiled, constraining tool-call path")
    return matcher
