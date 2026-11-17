# SPDX-License-Identifier: Apache-2.0
#
# Code-mode tool interface (M10, arXiv 2611.19123). Instead of the
# OpenAI tool-call JSON round-trip, the model writes a Python script
# against stub functions generated from the tool definitions. The
# controlled study found code mode cuts tool-related tokens by
# 70-90% while raising utility, and that declaring return shapes in
# the stubs is a first-order variable (up to 39.5pt swing).
#
# V1 is prompt-side only: build the stub module, inject it as a
# system message, and return the model's code block in the message
# content for the caller's harness to execute. No execution happens
# in this server.
from __future__ import annotations

import json
import logging
import re

logger = logging.getLogger(__name__)

CODE_MODE_SYSTEM_PROMPT = """You write Python code to call tools.
A stub module for the available tools is provided below. Rules:
- Call stub functions exactly as declared; do not invent functions or parameters.
- Return values have the documented shapes; do not reformat them.
- Respond with a single ```python code block and nothing else.
- Assign the final answer to a variable named `result`."""

_PY_TYPES = {
    "string": "str",
    "integer": "int",
    "number": "float",
    "boolean": "bool",
    "array": "list",
    "object": "dict",
    "null": "None",
}


def _py_type(schema: dict) -> str:
    t = schema.get("type")
    if isinstance(t, list):
        t = next((x for x in t if x != "null"), t[0] if t else None)
    if t == "array":
        item = schema.get("items")
        inner = _py_type(item) if isinstance(item, dict) else "Any"
        return f"list[{inner}]"
    if t == "object":
        return "dict"
    return _PY_TYPES.get(t, "Any")


def _shape_hint(function: dict) -> str:
    # Return-shape hint: explicit x-return-shape wins (first-order
    # variable in 2611.19123); otherwise synthesize from the declared
    # output schema if present, else a conservative opaque-dict note.
    explicit = function.get("x-return-shape")
    if explicit is not None:
        return json.dumps(explicit, ensure_ascii=False)
    returns = function.get("returns") or function.get("x-returns")
    if isinstance(returns, dict):
        props = returns.get("properties")
        if isinstance(props, dict):
            return "dict with keys " + json.dumps(sorted(props), ensure_ascii=False)
    return "JSON value (shape not declared by the tool author)"


def build_code_mode_stubs(tools: list[dict] | None) -> str:
    """Render tool definitions as a Python stub module with signatures
    and return-shape hints. Degenerate inputs yield an empty module
    rather than raising — code_mode is best-effort prompt shaping."""
    if not tools:
        return ""
    lines = ["# Tool stubs (do not edit; shapes are binding)"]
    for tool in tools:
        fn = tool.get("function") if isinstance(tool, dict) else None
        if not isinstance(fn, dict) or not fn.get("name"):
            continue
        name = fn["name"]
        desc = " ".join(str(fn.get("description", "")).split())
        params = fn.get("parameters") or {}
        props = params.get("properties") if isinstance(params, dict) else None
        required = set()
        if isinstance(params, dict):
            required = set(params.get("required") or [])
        sig_parts = []
        if isinstance(props, dict) and props:
            for pname, pschema in props.items():
                ptype = _py_type(pschema) if isinstance(pschema, dict) else "Any"
                sig_parts.append(
                    f"{pname}: {ptype}" + ("" if pname in required else " = None")
                )
        sig = ", ".join(sig_parts)
        lines.append(f"def {name}({sig}) -> Any:")
        if desc:
            lines.append(f'    """{desc}"""')
        lines.append(f"    # returns: {_shape_hint(fn)}")
        lines.append("    ...")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def extract_code_blocks(text: str | None) -> list[str]:
    """Extract fenced python blocks from a model response, in order.
    Unfenced text is ignored — callers execute only declared blocks."""
    if not text:
        return []
    blocks = []
    for m in re.finditer(r"```(?:python|py)?\s*\n(.*?)```", text, re.S):
        blocks.append(m.group(1).strip())
    return blocks


def inject_code_mode_messages(
    messages: list[dict], tools: list[dict] | None
) -> list[dict]:
    """Return a new message list with the code-mode system prompt +
    stub module prepended. No-op (same content, new list) when tools
    are absent."""
    stubs = build_code_mode_stubs(tools)
    if not stubs:
        return list(messages)
    system = f"{CODE_MODE_SYSTEM_PROMPT}\n\n```python\n{stubs}```"
    return [{"role": "system", "content": system}, *messages]
