# SPDX-License-Identifier: Apache-2.0
"""End-to-end test: ask Qwen3.6-27B-mxfp8 to design a Python program.

Purpose: Verify the model can generate a complete, long response
(500+ tokens) without being truncated after 1-2 minutes.

Usage:
    .venv/bin/python3 tests/e2e_test_qwen36_generation.py
"""

import argparse
import json
import time

import httpx

MODEL_ID = "Qwen3.6-27B-mxfp8"
BASE_URL = "http://localhost:11435"

PROMPT = """You are a senior Python developer. Design and implement a comprehensive performance benchmark tool for a local AI model serving framework called "fusion-mlx".

The tool should:
1. Measure Time To First Token (TTFT) for different prompt lengths
2. Measure Tokens Per Output Token (TPOT) latency
3. Test concurrent request handling with 2, 4, and 8 simultaneous clients
4. Measure memory usage before, during, and after generation
5. Generate a detailed report with statistics (mean, median, p95, p99)

Write the complete Python code with proper error handling, logging, and a clean CLI interface using argparse. Include type hints throughout. The code should be production-ready and well-structured."""


def run_e2e_test(max_tokens: int = 4096, temperature: float = 0.7):
    """Run the end-to-end generation test."""
    print("=" * 60)
    print("E2E Test: Qwen3.6-27B-mxfp8 Long Generation")
    print("=" * 60)

    client = httpx.Client(timeout=None, trust_env=False)
    results = {"passed": [], "failed": [], "stats": {}}

    # Step 1: Load model
    print("\n[1/4] Loading model...")
    load_start = time.time()
    try:
        resp = client.post(
            f"{BASE_URL}/v1/models/load",
            json={"model": MODEL_ID},
        )
        if resp.status_code in (200, 202):
            print(f"  Model loaded in {time.time() - load_start:.1f}s")
        else:
            print(f"  Load response: {resp.status_code} {resp.text[:200]}")
    except Exception as e:
        print(f"  Model load error (may already be loaded): {e}")

    # Step 2: Generate
    print("\n[2/4] Generating response (max_tokens=4096)...")
    gen_start = time.time()
    first_token_time = None
    tokens = []
    token_times = []

    try:
        with client.stream(
            "POST",
            f"{BASE_URL}/v1/chat/completions",
            json={
                "model": MODEL_ID,
                "messages": [{"role": "user", "content": PROMPT}],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": True,
            },
        ) as resp:
            for line in resp.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[6:]  # strip "data: "
                if data_str == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                    content = (
                        data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                    )
                    if content:
                        if first_token_time is None:
                            first_token_time = time.time() - gen_start
                        token_times.append(time.time() - gen_start)
                        tokens.append(content)
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"  Generation error: {e}")
        results["failed"].append(f"Generation failed: {e}")
        return results

    gen_elapsed = time.time() - gen_start
    full_text = "".join(tokens)

    # Step 3: Analyze
    print("\n[3/4] Analyzing results...")
    print(f"  Total tokens (chars): {len(full_text)}")
    print(f"  Total tokens (estimated words*1.5): {len(full_text.split()) * 1.5:.0f}")
    print(f"  Generation time: {gen_elapsed:.1f}s")

    if first_token_time is not None:
        print(f"  TTFT (Time To First Token): {first_token_time:.2f}s")
        results["stats"]["ttft"] = first_token_time

    # Calculate TPOT from token times
    if len(token_times) >= 2:
        inter_token = [
            token_times[i] - token_times[i - 1] for i in range(1, len(token_times))
        ]
        avg_tpot = sum(inter_token) / len(inter_token)
        sorted_tpot = sorted(inter_token)
        p95_idx = int(len(sorted_tpot) * 0.95)
        p99_idx = int(len(sorted_tpot) * 0.99)
        print(f"  TPOT average: {avg_tpot:.4f}s")
        print(f"  TPOT p95: {sorted_tpot[p95_idx]:.4f}s")
        print(f"  TPOT p99: {sorted_tpot[p99_idx]:.4f}s")
        results["stats"]["tpot_avg"] = avg_tpot
        results["stats"]["tpot_p95"] = sorted_tpot[p95_idx]
        results["stats"]["tpot_p99"] = sorted_tpot[p99_idx]

    # Step 4: Verify completeness
    print("\n[4/4] Verifying response completeness...")
    checks = {
        "Has imports": "import" in full_text,
        "Has class definition": "class " in full_text,
        "Has function definition": "def " in full_text,
        "Has argparse": "argparse" in full_text,
        "Has type hints": "-> " in full_text or ": " in full_text,
        "Has error handling": "try:" in full_text or "except" in full_text,
        "Has logging": "logging" in full_text or "logger" in full_text,
        "Has TTFT measurement": "ttft" in full_text.lower()
        or "first token" in full_text.lower(),
        "Has TPOT measurement": "tpot" in full_text.lower()
        or "token per" in full_text.lower(),
        "Has concurrency test": "concurrent" in full_text.lower()
        or "thread" in full_text.lower()
        or "asyncio" in full_text,
        "Has statistics": "mean" in full_text.lower()
        or "median" in full_text.lower()
        or "percentile" in full_text.lower(),
        "Minimum length (1000 chars)": len(full_text) >= 1000,
        "Minimum length (3000 chars)": len(full_text) >= 3000,
        "Minimum length (5000 chars)": len(full_text) >= 5000,
    }

    passed_checks = 0
    for check_name, result in checks.items():
        status = "PASS" if result else "FAIL"
        if result:
            passed_checks += 1
        print(f"  [{status}] {check_name}")

    total_checks = len(checks)
    print(f"\n  Summary: {passed_checks}/{total_checks} checks passed")

    if passed_checks >= total_checks * 0.8:
        results["passed"].append(f"Completed {passed_checks}/{total_checks} checks")
        print(
            "\n  >>> TEST PASSED: Model generated a complete, comprehensive response <<<"
        )
    else:
        results["failed"].append(f"Only {passed_checks}/{total_checks} checks passed")
        print("\n  >>> TEST FAILED: Response incomplete or truncated <<<")

    # Print first 500 chars of response
    print("\n  Response preview (first 500 chars):")
    print("  " + "-" * 78)
    print(full_text[:500].replace("\n", "\n  "))
    print("  " + "-" * 78)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="E2E generation test for Qwen3.6-27B")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    results = run_e2e_test(args.max_tokens, args.temperature)

    exit_code = 0 if results["passed"] else 1
    print(f"\nExit code: {exit_code}")
    exit(exit_code)
