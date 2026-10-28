# SPDX-License-Identifier: Apache-2.0

import bisect
import json
import logging

import regex

from ..openai_models import ToolCall

logger = logging.getLogger(__name__)


# Gemma 4 robust fallback parser
# ---------------------------------------------------------------------------

# Gemma 4's non-standard string delimiter (mlx_lm.tool_parsers.gemma4 uses
# the same literal in its regex).
_GEMMA4_STR_DELIM = '<|"|>'

# Bounds for parsing model-emitted arguments.  Model output is untrusted and
# attacker-influenceable (prompt injection can steer emissions verbatim), so
# parsing must stay linear-time and bounded: breaching a bound is a clean
# parse failure that flows into the existing drop-with-warning path, never an
# exception that escapes the parse chain.
_GEMMA4_MAX_ARGS_LEN = 262_144
_GEMMA4_MAX_DEPTH = 64


class _Gemma4ArgsTooComplexError(ValueError):
    pass


# A tool-call head: the name plus its opening ``{``.  Only the head is matched
# by regex; the argument span is found by _scan_gemma4_args_span, not by a
# recursive pattern (see that function).  The name segment captures namespaced
# MCP names (colon/dot/hyphen separated, e.g.
# call:google:mcp:text_generation:create-pdf-file, #1830).  The ``call:``
# opener is made optional and tolerant — ``(?:call)?:?`` — so the diffusion
# lane's degenerate prefixes (``calldone{`` missing the colon, ``:done{``
# missing ``call``, #1837) still match; the fallback only runs on
# marker-delimited content, so a permissive prefix cannot misfire on prose.
#
# Compiled with the ``regex`` module, NOT ``re``: once the ``call:`` literal
# anchor became optional (above), ``re``'s engine restarts the greedy
# ``[\w.-]+`` match at every position of a long bare argument value and
# backtracks O(n^2) hunting an opening ``{`` that never comes, hanging on
# adversarial output (a 300 KB bare value pegs a core indefinitely).  The
# ``regex`` engine fails that same partial match fast, so finditer stays
# linear.  See test_oversized_args_fail_cleanly.
_GEMMA4_CALL_HEAD = regex.compile(r"(?:call)?:?([\w.-]+(?::[\w.-]+)*)\{")

# The parenthesized variant of the head (#1846): under instruction-dense
# agentic load (large system prompt + many tool schemas + a big tool result)
# Gemma 4 26B reproducibly degrades from ``call:name{...}`` to a Python-kwargs
# shell ``call:name(key=value, ...)`` — same name grammar and same ``<|"|>``
# string quoting, only the OUTER ``{}`` becomes ``()`` and the top-level
# ``:`` separator becomes ``=``.  The nested grammar (objects, arrays, strings)
# is unchanged, so the hardened transcoder is reused for the inner content; only
# the outer shell needs a separate head + span scan.  Compiled with ``regex``,
# not ``re``, for the SAME reason as ``_GEMMA4_CALL_HEAD``: the optional/tolerant
# prefix makes ``re`` backtrack O(n^2) on a long bare value hunting an opening
# ``(`` that never comes (the #1854 ReDoS).  See test_oversized_paren_args.
_GEMMA4_CALL_HEAD_PAREN = regex.compile(r"(?:call)?:?([\w.-]+(?::[\w.-]+)*)\(")

# Matching close character for each balanced opener the parsers track.
_GEMMA4_CLOSE_CHAR = {"{": "}", "[": "]", "(": ")"}


def _squote_close_positions(s: str) -> list:
    closes: list[int] = []
    next_sig = ""  # next non-whitespace char AFTER the current index
    for idx in range(len(s) - 1, -1, -1):
        ch = s[idx]
        if ch == "'" and (next_sig == "" or next_sig in ",}])"):
            closes.append(idx)
        if not ch.isspace():
            next_sig = ch
    closes.reverse()
    return closes


def _scan_gemma4_args_span(
    text: str,
    open_idx: int,
    squote_closes: list,
    open_ch: str = "{",
    close_ch: str = "}",
) -> int:
    n = len(text)
    depth = 0
    last_sig = ""
    i = open_idx
    limit = min(n, open_idx + _GEMMA4_MAX_ARGS_LEN)
    while i < limit:
        if text.startswith(_GEMMA4_STR_DELIM, i):
            close = text.find(_GEMMA4_STR_DELIM, i + len(_GEMMA4_STR_DELIM))
            if close == -1:
                return -1  # unterminated string: malformed, give up cleanly
            i = close + len(_GEMMA4_STR_DELIM)
            last_sig = '"'
            continue
        ch = text[i]
        if ch == '"':
            # Standard JSON double-quoted string. The <|"|> delimiter is
            # matched by the startswith branch above before we reach here, so
            # a bare ``"`` is an ordinary JSON string open: skip to its closing
            # unescaped quote so a ``}`` inside the value cannot truncate the
            # span (#1854 — without this the suffix remap turned the corrupted
            # parse into an executable call with silently mangled arguments).
            # Honors ``\"`` and ``\\`` so an escaped quote never closes early.
            j = i + 1
            while j < limit:
                if text[j] == "\\":
                    j += 2  # escaped char is literal, never closes the string
                    continue
                if text[j] == '"':
                    break
                j += 1
            else:
                return -1  # unterminated string within bounds: drop cleanly
            i = j + 1
            last_sig = '"'
            continue
        if ch == "'" and last_sig in ":=,[":
            k = bisect.bisect_right(squote_closes, i)
            if k < len(squote_closes):
                i = squote_closes[k] + 1
                last_sig = "'"
                continue
            # No valid close ahead: treat the quote as ordinary content.
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return i + 1
        if not ch.isspace():
            last_sig = ch
        i += 1
    return -1


def _gemma4_args_to_json_robust(args_str: str) -> dict:
    try:
        return _gemma4_transcode_to_json(args_str)
    except _Gemma4ArgsTooComplexError:
        # Defensive bound breached: reject hard.  The legacy parser ignores
        # these bounds and would parse the input anyway, defeating the DoS
        # guard, so it must NOT see oversized/deeply-nested args.
        raise
    except (ValueError, json.JSONDecodeError):
        # The legacy path's NUL-placeholder forge vector is reintroduced
        # ONLY for ambiguous input the strict transcoder could not parse
        # (e.g. bare multi-comma markdown values, #1837); the common path
        # keeps the transcoder's no-placeholder, injection-safe guarantee.
        return _gemma4_args_to_json_legacy(args_str)


def _gemma4_transcode_to_json(args_str: str) -> dict:
    if len(args_str) > _GEMMA4_MAX_ARGS_LEN:
        raise _Gemma4ArgsTooComplexError("Gemma 4 args too large to parse")

    squote_closes = _squote_close_positions(args_str)
    n = len(args_str)
    out: list[str] = []  # JSON text fragments
    stack: list[str] = []  # open containers: "{" or "["
    expect = "object"  # object | key | value | delim
    i = 0

    def _skip_ws(i: int) -> int:
        while i < n and args_str[i].isspace():
            i += 1
        return i

    def _read_marked_string(i: int):
        if args_str.startswith(_GEMMA4_STR_DELIM, i):
            close = args_str.find(_GEMMA4_STR_DELIM, i + len(_GEMMA4_STR_DELIM))
            if close == -1:
                raise ValueError("unterminated Gemma 4 string")
            return (
                args_str[i + len(_GEMMA4_STR_DELIM) : close],
                close + len(_GEMMA4_STR_DELIM),
            )
        if args_str[i] == "'":
            k = bisect.bisect_right(squote_closes, i)
            if k < len(squote_closes):
                close = squote_closes[k]
                return args_str[i + 1 : close], close + 1
            # No anchored close ahead: not a string, treat as bare content.
        return None

    def _read_json_string(i: int):
        j = i + 1
        while j < n:
            if args_str[j] == "\\":
                j += 2
                continue
            if args_str[j] == '"':
                return args_str[i : j + 1], j + 1
            j += 1
        raise ValueError("unterminated double-quoted string")

    while True:
        i = _skip_ws(i)
        if expect == "object":
            # The outer container is ``{`` for the canonical form and ``(`` for
            # the parenthesized variant (#1846).  Either way it is an object;
            # we always emit ``{`` to JSON and remember the real opener on the
            # stack so its matching closer (``}`` or ``)``) is accepted below.
            if i >= n or args_str[i] not in "{(":
                raise ValueError("Gemma 4 args must start with '{' or '('")
            out.append("{")
            stack.append(args_str[i])
            i += 1
            expect = "key"
        elif expect == "key":
            if i >= n:
                raise ValueError("unterminated object")
            if args_str[i] == _GEMMA4_CLOSE_CHAR[stack[-1]]:
                # Empty object, or tolerated trailing comma.  Closer is ``}``
                # for a ``{`` opener and ``)`` for the paren variant's ``(``.
                if out and out[-1] == ", ":
                    out.pop()
                out.append("}")
                stack.pop()
                i += 1
                expect = "delim"
                continue
            if args_str[i] == '"':
                tok, i = _read_json_string(i)
                key = json.loads(tok)
            else:
                marked = _read_marked_string(i)
                if marked is not None:
                    key, i = marked
                else:
                    # Bare key: everything up to the separator.  ``=`` is a
                    # separator too for the parenthesized kwargs variant
                    # (#1846), so it bounds the key just like ``:``.
                    j = i
                    while j < n and args_str[j] not in ":=,{}[]'\"":
                        j += 1
                    key = args_str[i:j].strip()
                    if not key:
                        raise ValueError("malformed object key")
                    i = j
            i = _skip_ws(i)
            # Accept ``=`` as well as ``:`` — the parenthesized variant (#1846)
            # uses ``key=value``.  This is applied universally (not only at top
            # level), so it does widen the curly grammar to also accept ``=``
            # separators; that is a strict superset, so every valid ``:``-based
            # curly parse is unchanged and a previously-rejected ``{a = 1}`` now
            # succeeds rather than corrupting anything.  ``=`` inside a value is
            # untouched: strings are read atomically and bare values stop only
            # at ``,``/``}``/``]`` (plus ``)`` when a paren container is open).
            if i >= n or args_str[i] not in ":=":
                raise ValueError("expected ':' or '=' after object key")
            out.append(json.dumps(key))
            out.append(": ")
            i += 1
            expect = "value"
        elif expect == "value":
            if i >= n:
                raise ValueError("unterminated value")
            ch = args_str[i]
            if ch == "{" or ch == "[":
                # Depth bound, not recursion: a breach must surface as a
                # clean parse failure on the existing drop path, never as a
                # RecursionError (uncaught by the parse chain's excepts).
                if len(stack) >= _GEMMA4_MAX_DEPTH:
                    raise _Gemma4ArgsTooComplexError("Gemma 4 args nested too deeply")
                out.append(ch)
                stack.append(ch)
                i += 1
                expect = "key" if ch == "{" else "value"
                continue
            if ch == "]" and stack and stack[-1] == "[":
                # Empty array, or tolerated trailing comma.
                if out and out[-1] == ", ":
                    out.pop()
                out.append("]")
                stack.pop()
                i += 1
                expect = "delim"
                continue
            if ch == '"':
                tok, i = _read_json_string(i)
                out.append(tok)
                expect = "delim"
                continue
            marked = _read_marked_string(i)
            if marked is not None:
                content, i = marked
                out.append(json.dumps(content))
                expect = "delim"
                continue
            # Bare value: runs to the next structural separator.  When the
            # call uses the parenthesized outer shell (#1846), ``)`` also
            # terminates a bare value so the closing paren of the call is not
            # swallowed (``call:f(units=metric)`` — the value is ``metric``,
            # not ``metric)``).  For the curly form ``)`` stays ordinary
            # content so a value may legitimately contain parentheses
            # (``{expr: f(x)}``).  ``(`` is only ever the outermost container,
            # so its presence on the stack is the reliable signal.
            stops = ",)}]" if "(" in stack else ",}]"
            j = i
            while j < n and args_str[j] not in stops:
                j += 1
            value = args_str[i:j].strip()
            i = j
            if not value:
                raise ValueError("empty value")
            low = value.lower()
            if low in ("true", "false", "null"):
                out.append(low)  # normalize case (models emit True/False)
            else:
                try:
                    json.loads(value)  # already a valid scalar (number, ...)
                    out.append(value)
                except (json.JSONDecodeError, ValueError):
                    out.append(json.dumps(value))
            expect = "delim"
        else:  # expect == "delim"
            if not stack:
                if i < n:
                    raise ValueError("trailing data after args object")
                break
            if i >= n:
                raise ValueError("unterminated args")
            ch = args_str[i]
            if ch == ",":
                out.append(", ")
                i += 1
                # After a comma, an object (``{`` or the paren outer ``(``)
                # expects a key; an array expects a value.
                expect = "key" if stack[-1] in "{(" else "value"
            elif stack[-1] in "{(" and ch == _GEMMA4_CLOSE_CHAR[stack[-1]]:
                # Object close: ``}`` for ``{``, ``)`` for the paren outer.
                out.append("}")
                stack.pop()
                i += 1
            elif ch == "]" and stack[-1] == "[":
                out.append("]")
                stack.pop()
                i += 1
            else:
                raise ValueError("malformed args structure")

    result = json.loads("".join(out))
    if not isinstance(result, dict):
        raise ValueError("Gemma 4 args did not parse to an object")
    return result


def _gemma4_args_to_json_legacy(args_str: str) -> dict:
    import regex

    # 1. Extract <|"|>-delimited strings and replace with placeholders
    strings: list[str] = []

    def _capture(m):
        strings.append(m.group(1))
        return f"\x00{len(strings) - 1}\x00"

    text = regex.sub(r'<\|"\|>(.*?)<\|"\|>', _capture, args_str, flags=regex.DOTALL)

    # 2. Quote bare keys (allow whitespace after { or ,)
    text = regex.sub(r"(?<=[{,])\s*(\w+)\s*:", r' "\1":', text)

    # 3. Restore captured strings as properly escaped JSON strings
    for i, s in enumerate(strings):
        text = text.replace(f"\x00{i}\x00", json.dumps(s))

    # 4. Try json.loads — works when all values are already valid JSON primitives
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 5. Quote bare string values that are not numbers, booleans, or null
    def _quote_bare(m):
        value = m.group(2).strip()
        suffix = m.group(3)
        if value.lower() in ("true", "false", "null"):
            return f": {value}{suffix}"
        try:
            json.loads(value)
            return f": {value}{suffix}"
        except (json.JSONDecodeError, ValueError):
            return f": {json.dumps(value)}{suffix}"

    # Keep the pre-step-5 text: if step 5 fails, its partial quoting has
    # corrupted multi-line bare values and step 6 must start clean.
    pre_quote_text = text
    text = regex.sub(r"(:\s*)([^\",\[\]{}\s][^,}]*?)(\s*[,}])", _quote_bare, text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 6. Last resort: key-anchored value capture. Bare values that
    # themselves contain commas, braces, or newlines (e.g. long markdown
    # emitted into an ``answer:`` argument — observed live on the
    # diffusion lane) defeat the per-pair regex in step 5. Anchor on the
    # quoted keys produced by step 2 and treat everything between a
    # key's colon and the next key (or the end) as that key's value.
    # Operates on the pre-step-5 text so step 5's partial quoting cannot
    # corrupt the captured values.
    inner = pre_quote_text.strip()
    if inner.startswith("{") and inner.endswith("}"):
        inner = inner[1:-1]
    key_pat = regex.compile(r'"([A-Za-z_]\w*)"\s*:')
    key_matches = list(key_pat.finditer(inner))
    if not key_matches:
        return json.loads(text)  # re-raise original-style error
    result: dict = {}
    for i, km in enumerate(key_matches):
        value_start = km.end()
        value_end = (
            key_matches[i + 1].start() if i + 1 < len(key_matches) else len(inner)
        )
        raw_value = inner[value_start:value_end].strip()
        if i + 1 < len(key_matches):
            raw_value = raw_value.rstrip().rstrip(",").rstrip()
        try:
            result[km.group(1)] = json.loads(raw_value)
        except (json.JSONDecodeError, ValueError):
            result[km.group(1)] = raw_value
    return result


def _parse_gemma4_tool_call_fallback(text: str) -> dict | list:
    squote_closes = _squote_close_positions(text)

    # Collect heads from both the canonical curly form and the parenthesized
    # variant (#1846), then process them in document order so the consumed-span
    # dedup below holds across BOTH forms: a curly head that falls inside an
    # already-consumed paren span (the inner ``{...}`` of ``call:f(a={...})``)
    # is string/structure content, not a sibling call, and must be skipped.
    heads = [(m.start(), m, "{", "}") for m in _GEMMA4_CALL_HEAD.finditer(text)]
    heads += [(m.start(), m, "(", ")") for m in _GEMMA4_CALL_HEAD_PAREN.finditer(text)]
    heads.sort(key=lambda h: h[0])

    results = []
    consumed_until = 0
    for start, m, open_ch, close_ch in heads:
        # A head inside an already-consumed args span is string content
        # (e.g. quoted prose mentioning a tool call), not a sibling call.
        if start < consumed_until:
            continue
        open_idx = m.end() - 1
        end = _scan_gemma4_args_span(text, open_idx, squote_closes, open_ch, close_ch)
        if end == -1:
            continue
        args_str = text[open_idx:end]
        try:
            arguments = _gemma4_args_to_json_robust(args_str)
        except (ValueError, json.JSONDecodeError, RecursionError):
            continue  # one malformed call must not drop its siblings
        if not isinstance(arguments, dict):
            continue
        results.append({"name": m.group(1), "arguments": arguments})
        consumed_until = end

    if not results:
        raise ValueError("No function call found in Gemma 4 format")
    return results[0] if len(results) == 1 else results


def _extract_tool_names(tools: list) -> set:
    names = set()
    for tool in tools:
        if isinstance(tool, dict):
            func = tool.get("function", {})
            if isinstance(func, dict):
                name = func.get("name")
                if name:
                    names.add(name)
    return names


def _remap_tool_call_names(tool_calls: list[ToolCall], tools: list | None) -> None:
    if not tool_calls or not tools:
        return
    valid_names = _extract_tool_names(tools)
    if not valid_names:
        return
    for tc in tool_calls:
        name = tc.function.name if tc.function else ""
        if not name or name in valid_names or ":" not in name:
            continue
        parts = name.split(":")
        suffixes = {":".join(parts[i:]) for i in range(1, len(parts))}
        candidates = suffixes & valid_names
        if len(candidates) == 1:
            target = next(iter(candidates))
            logger.info(
                "Remapped namespaced tool call name %r to registered " "tool %r",
                name[:200],
                target,
            )
            tc.function.name = target
