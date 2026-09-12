# SPDX-License-Identifier: Apache-2.0
"""G1/S1: Output coherence gate (ADVISORY).

Checks model output is non-empty, non-degenerate (not stuck repeating),
and topically responsive to the prompt. Heuristic, not semantic — catches
gross regressions (empty output, repetition collapse, wrong-language drift).

CLI: python -m fusion_mlx.eval.coherence_gate --model <alias>
"""

import argparse
import asyncio
import json
import logging
import os
import re
from dataclasses import asdict, dataclass

logger = logging.getLogger(__name__)

_PROBES = [
    ("What is 2+2?", lambda r: "4" in r),
    ("Say hello in English.", lambda r: "hello" in r.lower() or "hi" in r.lower()),
    (
        "List one fruit.",
        lambda r: bool(
            re.search(r"\b(apple|orange|banana|mango|pear|grape|peach)\b", r.lower())
        ),
    ),
]
_MAX_REPEAT = 8


@dataclass
class CoherenceSample:
    prompt: str
    response: str
    passed: bool
    reason: str = ""


@dataclass
class CoherenceGateResult:
    model: str
    samples: list[CoherenceSample]
    pass_rate: float
    all_passed: bool

    def to_dict(self) -> dict:
        return asdict(self)


def _has_repetition_collapse(text: str) -> bool:
    words = text.split()
    if len(words) < _MAX_REPEAT * 2:
        return False
    tail = words[-_MAX_REPEAT:]
    if len(set(tail)) <= 2:
        return True
    return False


def _check_response(prompt: str, response: str, checker) -> tuple[bool, str]:
    if not response.strip():
        return False, "empty response"
    if _has_repetition_collapse(response):
        return False, "repetition collapse in tail"
    if len(response.strip()) < 3:
        return False, "response too short"
    if not checker(response):
        return False, "checker predicate failed"
    return True, ""


def probe_one(
    host: str, api_key: str, model: str, prompt: str, checker
) -> CoherenceSample:
    import httpx

    url = f"{host}/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        resp = httpx.post(
            url,
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 64,
                "temperature": 0.0,
            },
            headers=headers,
            timeout=httpx.Timeout(60.0, connect=10.0),
        )
        if resp.status_code != 200:
            return CoherenceSample(
                prompt=prompt,
                response="",
                passed=False,
                reason=f"HTTP {resp.status_code}",
            )
        data = resp.json()
        response = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    except (httpx.HTTPError, OSError, json.JSONDecodeError) as exc:
        return CoherenceSample(
            prompt=prompt, response="", passed=False, reason=f"request error: {exc}"
        )
    passed, reason = _check_response(prompt, response, checker)
    return CoherenceSample(
        prompt=prompt, response=response[:200], passed=passed, reason=reason
    )


async def run_gate(host: str, api_key: str, model: str) -> CoherenceGateResult:
    samples: list[CoherenceSample] = []
    for prompt, checker in _PROBES:
        s = await asyncio.to_thread(probe_one, host, api_key, model, prompt, checker)
        samples.append(s)
    passed = sum(1 for s in samples if s.passed)
    rate = passed / len(samples) if samples else 0.0
    return CoherenceGateResult(
        model=model,
        samples=samples,
        pass_rate=rate,
        all_passed=passed == len(samples),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Output coherence gate (ADVISORY)")
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--host", default=os.environ.get("FUSION_HOST", "http://127.0.0.1:11434")
    )
    parser.add_argument("--api-key", default=os.environ.get("FUSION_MLX_API_KEY", ""))
    args = parser.parse_args()
    result = asyncio.run(run_gate(args.host, args.api_key, args.model))
    print(json.dumps(result.to_dict(), indent=2))
    return 0 if result.all_passed else 0  # ADVISORY: never fail CI


if __name__ == "__main__":
    raise SystemExit(main())
