# SPDX-License-Identifier: Apache-2.0
"""Deterministic prompt-class classifier for speculative-decode gating.

Measured acceptance (P2-2, #1549/#1577 chain): math/code prompts sustain
1.5x+ with DSpine at T=0.7, while open chat sits at ~15% acceptance
(≈ parity — draft overhead cancels the accept wins). The auto-router's
abandon/keep hysteresis reacts only AFTER a request paid the draft cost;
this classifier gates UP FRONT so losing classes never speculate.

Pure function, no model call, no tokenizer required for text prompts
(Rule 5: decide with code). Token-id prompts return "other" (speculate —
current behavior) unless the caller decodes first.
"""

from __future__ import annotations

import re

CLASS_MATH = "math"
CLASS_CODE = "code"
CLASS_CHAT = "chat"
CLASS_OTHER = "other"

_CODE_MARKERS = (
    "```",
    "def ",
    "class ",
    "import ",
    "from ",
    "function ",
    "const ",
    "let ",
    "var ",
    "return ",
    "if __name__",
    "print(",
    "console.log",
    "SELECT ",
    "INSERT ",
    "#!/",
    "</",
    "->",
    "&&",
    "||",
    "===",
)
_CODE_HINTS = (
    "write a function",
    "write a python",
    "write a sql",
    "implement ",
    "refactor ",
    "debug ",
    "code review",
    "unit test",
    "regular expression",
    "regex",
    "写一个函数",
    "实现一个",
    "写代码",
    "写一个脚本",
)
_MATH_HINTS = (
    "solve",
    "compute",
    "calculate",
    "prove",
    "derivative",
    "integral",
    "equation",
    "probability",
    "matrix",
    "vector",
    "sum of",
    "how many",
    "gcd",
    "lcm",
    "modulo",
    "factorial",
    "geometric",
    "arithmetic",
    "求解",
    "计算",
    "证明",
    "方程",
    "概率",
    "几何",
    "微积分",
)
_OPERATOR_RE = re.compile(r"[+\-*/^=<>]|\d")
_DIGIT_RE = re.compile(r"\d")
_MATH_TOKEN_RE = re.compile(r"[+\-*/^=<>]")
_CODE_SYMBOL_RE = re.compile(r"[{}()\[\];=]|_{2,}")
_MIN_LEN = 24


_USER_TURN_RE = re.compile(
    r"<\|im_start\|>user\r?\n(.*?)(?:<\|im_end\|>|<\|im_start\|>|$)", re.DOTALL
)


def extract_last_user_turn(rendered_prompt: str) -> str:
    # The scheduler hands us the rendered chat template. Classifying the
    # whole render is wrong twice over: system-prompt prose and special-token
    # markers both pollute marker-based classification. Extract the last
    # user turn; if the template is not Qwen-style, return the render minus
    # special-token spans so marker noise stays bounded.
    if "<|im_start|>user" in rendered_prompt:
        matches = _USER_TURN_RE.findall(rendered_prompt)
        if matches:
            return matches[-1].strip()
    cleaned = re.sub(r"<\|[^>]*\|>", " ", rendered_prompt)
    return cleaned.strip()


def classify_prompt(prompt: str | list[int] | None) -> str:
    """Return one of math / code / chat / other for a prompt.

    Text prompts are classified by deterministic marker/signal scoring;
    token-id prompts and empty/short inputs return "other" (never gates).
    """
    if not isinstance(prompt, str) or len(prompt) < _MIN_LEN:
        return CLASS_OTHER
    text = prompt[:4096]
    lowered = text.lower()

    for marker in _CODE_MARKERS:
        if marker in text:
            return CLASS_CODE
    if any(h in lowered for h in _CODE_HINTS):
        return CLASS_CODE

    math_hits = sum(1 for h in _MATH_HINTS if h in lowered)
    op_density = len(_OPERATOR_RE.findall(text)) / max(len(text), 1)
    digit_density = len(_DIGIT_RE.findall(text)) / max(len(text), 1)
    if math_hits >= 1 and (op_density > 0.01 or digit_density > 0.05):
        return CLASS_MATH
    if math_hits >= 2:
        return CLASS_MATH
    if math_hits == 1 and any(h in lowered for h in _MATH_HINTS[:6]):
        # Strong verbs (solve/compute/calculate/prove/derivative/integral)
        # alone are enough.
        return CLASS_MATH

    if math_hits == 0 and digit_density < 0.02 and op_density < 0.005:
        return CLASS_CHAT
    return CLASS_OTHER


def spec_gate_allows(prompt_class: str, temperature: float) -> bool:
    """Gate: speculate only when the class historically wins.

    math/code speculate at any temperature; chat/other speculate greedy
    (T=0 verify is cheap and chat acceptance at T=0 still helps) but skip
    sampled speculation at T>0 where open-chat acceptance measured ~15%.
    """
    if prompt_class in (CLASS_MATH, CLASS_CODE):
        return True
    if prompt_class == CLASS_CHAT:
        return temperature <= 1e-6
    return True
