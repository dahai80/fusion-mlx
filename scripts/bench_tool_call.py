#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Tool-call accuracy benchmark for fusion-mlx.

Tests N JSON-schema-constrained tool calls against the local fusion-mlx
server and reports success rate, JSON validity, and schema compliance.

Usage:
    # Start server first:
    fusion-mlx serve --model Qwen3.6-27B-mxfp8

    # Run benchmark (default 100 calls):
    python scripts/bench_tool_call.py

    # Custom count and endpoint:
    python scripts/bench_tool_call.py --n 50 --base-url http://localhost:8897

    # Specific model:
    python scripts/bench_tool_call.py --model Qwen3.5-9B-6bit
"""

import argparse
import json
import logging
import sys
import time
from typing import Any

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Test schemas — diverse shapes to exercise different constrained-decode paths
# ---------------------------------------------------------------------------

TOOL_SCHEMAS = [
    {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "schema": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location"],
        },
    },
    {
        "name": "search_web",
        "description": "Search the web for information",
        "schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "max_results": {"type": "integer", "minimum": 1, "maximum": 20},
                "safe_search": {"type": "boolean"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "create_calendar_event",
        "description": "Create a new calendar event",
        "schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "start_time": {"type": "string", "description": "ISO 8601 datetime"},
                "end_time": {"type": "string", "description": "ISO 8601 datetime"},
                "location": {"type": "string"},
                "attendees": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["title", "start_time", "end_time"],
        },
    },
    {
        "name": "calculate",
        "description": "Perform a mathematical calculation",
        "schema": {
            "type": "object",
            "properties": {
                "expression": {"type": "string"},
                "precision": {"type": "integer", "minimum": 0, "maximum": 15},
            },
            "required": ["expression"],
        },
    },
    {
        "name": "send_email",
        "description": "Send an email to recipients",
        "schema": {
            "type": "object",
            "properties": {
                "to": {
                    "type": "array",
                    "items": {"type": "string", "format": "email"},
                },
                "subject": {"type": "string"},
                "body": {"type": "string"},
                "cc": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "priority": {"type": "string", "enum": ["low", "normal", "high"]},
            },
            "required": ["to", "subject", "body"],
        },
    },
]

PROMPTS = [
    "What's the weather like in Tokyo?",
    "Search for recent advances in quantum computing",
    "Schedule a team meeting tomorrow at 2pm for 1 hour in conference room B",
    "Calculate the compound interest on $10000 at 5% for 3 years",
    "Send an email to alice@example.com about the project deadline",
    "Is it going to rain in London this weekend?",
    "Find information about the Python requests library",
    "Create a reminder for my dentist appointment next Monday at 10am",
    "What is 2^32 minus 1?",
    "Email the team about the upcoming sprint review",
]


def build_request(
    prompt: str,
    tool_schema: dict[str, Any],
    model: str,
) -> dict[str, Any]:
    tools = [
        {
            "type": "function",
            "function": {
                "name": tool_schema["name"],
                "description": tool_schema["description"],
                "parameters": tool_schema["schema"],
            },
        }
    ]
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "tools": tools,
        "tool_choice": {"type": "function", "function": {"name": tool_schema["name"]}},
        "max_tokens": 256,
        "temperature": 0.0,
    }


def validate_tool_call(
    response_json: dict,
    expected_function: str,
    expected_schema: dict,
) -> dict[str, Any]:
    result = {
        "has_tool_call": False,
        "function_name_match": False,
        "json_valid": False,
        "schema_compliant": False,
        "required_fields_present": False,
        "error": None,
    }
    try:
        choice = response_json["choices"][0]["message"]
        tool_calls = choice.get("tool_calls", [])
        if not tool_calls:
            result["error"] = "no tool_calls in response"
            return result
        result["has_tool_call"] = True

        tc = tool_calls[0]
        fn = tc.get("function", {})
        result["function_name_match"] = fn.get("name") == expected_function

        args_str = fn.get("arguments", "")
        try:
            args = json.loads(args_str)
            result["json_valid"] = True
        except json.JSONDecodeError as e:
            result["error"] = f"JSON decode error: {e}"
            return result

        required = expected_schema.get("required", [])
        result["required_fields_present"] = all(r in args for r in required)

        for key, val in args.items():
            props = expected_schema.get("properties", {}).get(key, {})
            if "enum" in props and val not in props["enum"]:
                result["error"] = f"enum violation: {key}={val} not in {props['enum']}"
                return result
            if "type" in props:
                type_map = {
                    "string": str,
                    "integer": int,
                    "boolean": bool,
                    "array": list,
                    "number": (int, float),
                }
                expected_type = type_map.get(props["type"])
                if expected_type and not isinstance(val, expected_type):
                    result["error"] = f"type violation: {key}={val} expected {props['type']}"
                    return result

        result["schema_compliant"] = True
    except (KeyError, IndexError, TypeError) as e:
        result["error"] = f"response structure error: {e}"
    return result


def run_benchmark(
    base_url: str,
    model: str,
    n_calls: int,
) -> dict[str, Any]:
    url = f"{base_url}/v1/chat/completions"
    results = []

    logger.info("Starting tool-call accuracy benchmark: %d calls to %s", n_calls, url)
    logger.info("Model: %s", model)

    total = 0
    tool_call_ok = 0
    function_match = 0
    json_valid = 0
    schema_compliant = 0
    required_present = 0
    errors = {}

    start_time = time.time()

    for i in range(n_calls):
        schema_idx = i % len(TOOL_SCHEMAS)
        prompt_idx = i % len(PROMPTS)
        tool_schema = TOOL_SCHEMAS[schema_idx]
        prompt = PROMPTS[prompt_idx]

        req = build_request(prompt, tool_schema, model)

        try:
            resp = requests.post(url, json=req, timeout=30)
            resp.raise_for_status()
            resp_json = resp.json()
        except requests.RequestException as e:
            logger.warning("Call %d: request failed: %s", i + 1, e)
            total += 1
            errors.setdefault("request_failed", 0)
            errors["request_failed"] += 1
            results.append({"call": i + 1, "status": "request_failed", "error": str(e)})
            continue

        validation = validate_tool_call(resp_json, tool_schema["name"], tool_schema["schema"])
        total += 1

        if validation["has_tool_call"]:
            tool_call_ok += 1
        if validation["function_name_match"]:
            function_match += 1
        if validation["json_valid"]:
            json_valid += 1
        if validation["required_fields_present"]:
            required_present += 1
        if validation["schema_compliant"]:
            schema_compliant += 1

        if validation["error"]:
            errors.setdefault(validation["error"].split(":")[0], 0)
            errors[validation["error"].split(":")[0]] += 1

        status = "ok" if validation["schema_compliant"] else validation.get("error", "unknown")
        results.append({"call": i + 1, "function": tool_schema["name"], "status": status})

        if (i + 1) % 10 == 0:
            logger.info(
                "  %d/%d calls — schema_compliant=%.1f%%",
                i + 1,
                n_calls,
                100 * schema_compliant / total,
            )

    elapsed = time.time() - start_time

    report = {
        "model": model,
        "total_calls": total,
        "elapsed_s": round(elapsed, 1),
        "avg_latency_s": round(elapsed / max(total, 1), 2),
        "tool_call_generated": {
            "count": tool_call_ok,
            "rate": round(tool_call_ok / max(total, 1), 4),
        },
        "function_name_match": {
            "count": function_match,
            "rate": round(function_match / max(total, 1), 4),
        },
        "json_valid": {
            "count": json_valid,
            "rate": round(json_valid / max(total, 1), 4),
        },
        "required_fields_present": {
            "count": required_present,
            "rate": round(required_present / max(total, 1), 4),
        },
        "schema_compliant": {
            "count": schema_compliant,
            "rate": round(schema_compliant / max(total, 1), 4),
        },
        "errors": errors,
    }
    return report


def main():
    parser = argparse.ArgumentParser(description="Tool-call accuracy benchmark")
    parser.add_argument("--base-url", default="http://localhost:8897", help="Server URL")
    parser.add_argument("--model", default=None, help="Model name (auto-detect if omitted)")
    parser.add_argument("--n", type=int, default=100, help="Number of test calls")
    args = parser.parse_args()

    model = args.model
    if not model:
        try:
            resp = requests.get(f"{args.base_url}/v1/models", timeout=5)
            models = resp.json().get("data", [])
            if models:
                model = models[0]["id"]
                logger.info("Auto-detected model: %s", model)
            else:
                logger.error("No models loaded on server")
                sys.exit(1)
        except Exception as e:
            logger.error("Cannot reach server at %s: %s", args.base_url, e)
            sys.exit(1)

    report = run_benchmark(args.base_url, model, args.n)

    print("\n" + "=" * 60)
    print("Tool-Call Accuracy Benchmark Report")
    print("=" * 60)
    print(f"Model:          {report['model']}")
    print(f"Total calls:    {report['total_calls']}")
    print(f"Elapsed:        {report['elapsed_s']}s ({report['avg_latency_s']}s/call)")
    print()
    print(f"Tool call generated:   {report['tool_call_generated']['count']}/{report['total_calls']}  ({100*report['tool_call_generated']['rate']:.1f}%)")
    print(f"Function name match:   {report['function_name_match']['count']}/{report['total_calls']}  ({100*report['function_name_match']['rate']:.1f}%)")
    print(f"JSON valid:            {report['json_valid']['count']}/{report['total_calls']}  ({100*report['json_valid']['rate']:.1f}%)")
    print(f"Required fields:       {report['required_fields_present']['count']}/{report['total_calls']}  ({100*report['required_fields_present']['rate']:.1f}%)")
    print(f"Schema compliant:      {report['schema_compliant']['count']}/{report['total_calls']}  ({100*report['schema_compliant']['rate']:.1f}%)")
    if report["errors"]:
        print("\nError breakdown:")
        for err_type, count in sorted(report["errors"].items(), key=lambda x: -x[1]):
            print(f"  {err_type}: {count}")

    out_path = f"bench_tool_call_{model.replace('/', '_')}_{report['total_calls']}.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nFull report saved to: {out_path}")


if __name__ == "__main__":
    main()
