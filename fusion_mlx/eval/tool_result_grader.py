# SPDX-License-Identifier: Apache-2.0
"""G1/S1: Tool-call result grader (ADVISORY).

Grades whether a model's tool-call output is well-formed and correct
against expected structure. Checks: valid JSON args, expected tool name
invoked, expected keys present.

CLI: python -m fusion_mlx.eval.tool_result_grader --model <alias>
"""

import argparse
import asyncio
import json
import logging
import os
from dataclasses import asdict, dataclass

logger = logging.getLogger(__name__)

_PROBES = [
    {
        "prompt": "Use the get_weather tool for Paris. Respond with a tool call only.",
        "expected_tool": "get_weather",
        "expected_keys": ["location"],
        "expected_values": {"location": "Paris"},
    },
    {
        "prompt": "Call the search tool with query 'test'. Respond with a tool call only.",
        "expected_tool": "search",
        "expected_keys": ["query"],
        "expected_values": {"query": "test"},
    },
]


@dataclass
class ToolGradeSample:
    prompt: str
    tool_calls: list
    expected_tool: str
    tool_name_match: bool
    args_valid_json: bool
    keys_present: bool
    values_match: bool
    passed: bool
    reason: str = ""


@dataclass
class ToolGradeResult:
    model: str
    samples: list[ToolGradeSample]
    pass_rate: float
    all_passed: bool

    def to_dict(self) -> dict:
        return asdict(self)


def _grade(tool_calls: list, probe: dict) -> ToolGradeSample:
    prompt = probe["prompt"]
    expected = probe["expected_tool"]
    if not tool_calls:
        return ToolGradeSample(
            prompt=prompt,
            tool_calls=[],
            expected_tool=expected,
            tool_name_match=False,
            args_valid_json=False,
            keys_present=False,
            values_match=False,
            passed=False,
            reason="no tool_calls in response",
        )
    tc = tool_calls[0]
    name = (
        tc.get("function", {}).get("name", "")
        if isinstance(tc, dict)
        else getattr(tc, "function", None).name
    )
    args_str = (
        tc.get("function", {}).get("arguments", "{}")
        if isinstance(tc, dict)
        else getattr(tc.function, "arguments", "{}")
    )
    name_match = name == expected
    try:
        args = json.loads(args_str) if isinstance(args_str, str) else args_str
        args_valid = isinstance(args, dict)
    except (json.JSONDecodeError, ValueError):
        args = {}
        args_valid = False
    keys_ok = all(k in args for k in probe["expected_keys"])
    values_ok = all(
        str(args.get(k)) == str(v) for k, v in probe["expected_values"].items()
    )
    passed = name_match and args_valid and keys_ok
    reasons = []
    if not name_match:
        reasons.append(f"name={name}≠{expected}")
    if not args_valid:
        reasons.append("args not valid JSON object")
    if not keys_ok:
        reasons.append(f"missing keys {probe['expected_keys']}")
    return ToolGradeSample(
        prompt=prompt,
        tool_calls=tool_calls,
        expected_tool=expected,
        tool_name_match=name_match,
        args_valid_json=args_valid,
        keys_present=keys_ok,
        values_match=values_ok,
        passed=passed,
        reason="; ".join(reasons) if reasons else "ok",
    )


def probe_one(host: str, api_key: str, model: str, probe: dict) -> ToolGradeSample:
    import httpx

    url = f"{host}/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        resp = httpx.post(
            url,
            json={
                "model": model,
                "messages": [{"role": "user", "content": probe["prompt"]}],
                "max_tokens": 128,
                "temperature": 0.0,
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "description": "Get weather for a location",
                            "parameters": {
                                "type": "object",
                                "properties": {"location": {"type": "string"}},
                                "required": ["location"],
                            },
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "search",
                            "description": "Search the web",
                            "parameters": {
                                "type": "object",
                                "properties": {"query": {"type": "string"}},
                                "required": ["query"],
                            },
                        },
                    },
                ],
            },
            headers=headers,
            timeout=httpx.Timeout(60.0, connect=10.0),
        )
        if resp.status_code != 200:
            return _grade([], probe)
        data = resp.json()
        msg = data.get("choices", [{}])[0].get("message", {})
        tool_calls = msg.get("tool_calls", [])
    except (httpx.HTTPError, OSError, json.JSONDecodeError) as exc:
        logger.warning("tool_grader: request failed: %s", exc)
        return _grade([], probe)
    return _grade(tool_calls, probe)


async def run_gate(host: str, api_key: str, model: str) -> ToolGradeResult:
    samples: list[ToolGradeSample] = []
    for probe in _PROBES:
        s = await asyncio.to_thread(probe_one, host, api_key, model, probe)
        samples.append(s)
    passed = sum(1 for s in samples if s.passed)
    rate = passed / len(samples) if samples else 0.0
    return ToolGradeResult(
        model=model, samples=samples, pass_rate=rate, all_passed=passed == len(samples)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Tool-call result grader (ADVISORY)")
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--host", default=os.environ.get("FUSION_HOST", "http://127.0.0.1:11434")
    )
    parser.add_argument("--api-key", default=os.environ.get("FUSION_MLX_API_KEY", ""))
    args = parser.parse_args()
    result = asyncio.run(run_gate(args.host, args.api_key, args.model))
    print(json.dumps(result.to_dict(), indent=2))
    return 0  # ADVISORY: never fail CI


if __name__ == "__main__":
    raise SystemExit(main())
