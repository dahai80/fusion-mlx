# SPDX-License-Identifier: Apache-2.0
"""Prefix-reuse retention benchmark for harness compaction (#1559 / P3-1).

Replays an agent-style trajectory against a live fusion-mlx serve,
compacting the history at a configured token threshold with two policies:

- fixed:   CompactionConfig.keep_recent_rounds=<keep>, no advisor
- advisor: same config + PrefixCacheAdvisor from fusion_mlx.harness_bridge

After every turn (and after every compaction) the script probes the
engine's chain-hashed cache state and sends one real chat completion so
the next turn's probe reflects what a live session would hit. Metrics
recorded per run: cumulative cached-prefix tokens at request time, TTFT
per turn, compaction event count. The P3-1 acceptance question is whether
the advisor policy retains more prefix reuse over the same trajectory.

Nightly-only: never wire this into PR CI (plan P2-4). See
scripts/nightly_compaction_bench.sh for the driver.
"""

import argparse
import json
import logging
import statistics
import sys
import time
from pathlib import Path

import httpx

logger = logging.getLogger("compaction_retention_bench")

DEFAULT_BASE_URL = "http://127.0.0.1:11434"


def _probe_cache(client: httpx.Client, messages: list[dict], model: str) -> dict:
    response = client.post(
        "/harness/v1/cache/probe",
        json={"messages": messages, "model_id": model},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def _chat_turn(
    client: httpx.Client,
    base_url: str,
    messages: list[dict],
    model: str,
    api_key: str | None,
    max_tokens: int = 8,
) -> tuple[float, str]:
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    t0 = time.perf_counter()
    response = httpx.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0,
        },
        headers=headers,
        timeout=300,
    )
    ttft = time.perf_counter() - t0
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"] or ""
    return ttft, content


def _estimate_tokens(client: httpx.Client, messages: list[dict], model: str) -> int:
    texts = [str(m.get("content") or "") for m in messages]
    response = client.post(
        "/harness/v1/tokens/count",
        json={"texts": texts, "model_id": model},
        timeout=60,
    )
    response.raise_for_status()
    return sum(response.json()["counts"])


def _compact_fixed(messages: list[dict], keep_rounds: int) -> list[dict]:
    # Deterministic fixed policy: drop all but the last keep_rounds rounds,
    # mirroring harness _split_rounds semantics.
    system_end = 0
    for m in messages:
        if m.get("role") == "system":
            system_end += 1
        else:
            break
    rest = messages[system_end:]
    starts = [0] + [
        i for i, m in enumerate(rest, start=system_end) if m.get("role") == "user"
    ][1:]
    if len(starts) <= keep_rounds:
        return messages
    kept_start = starts[len(starts) - keep_rounds]
    dropped = messages[system_end:kept_start]
    summary = "[compacted] " + " | ".join(
        str(m.get("content") or "")[:80] for m in dropped[:20]
    )
    return (
        messages[:system_end]
        + [{"role": "system", "content": summary}]
        + messages[kept_start:]
    )


def run_policy(
    policy: str,
    client: httpx.Client,
    base_url: str,
    model: str,
    api_key: str | None,
    turns: list[list[dict]],
    compact_threshold_tokens: int,
    keep_rounds: int,
) -> dict:
    from fusion_mlx.harness_bridge import PrefixCacheAdvisor

    advisor = None
    if policy == "advisor":
        advisor = PrefixCacheAdvisor(base_url, model_id=model, api_key=api_key)

    history: list[dict] = []
    compactions = 0
    cached_samples = []
    ttfts = []
    for turn in turns:
        history.extend(turn)
        tokens = _estimate_tokens(client, history, model)
        if tokens >= compact_threshold_tokens:
            keep = keep_rounds
            if advisor is not None:
                suggested = advisor.suggest_keep_rounds(history)
                keep = max(keep, suggested)
            history = _compact_fixed(history, keep)
            compactions += 1
            logger.info(
                "policy=%s compaction #%d keep=%d tokens_before=%d",
                policy,
                compactions,
                keep,
                tokens,
            )
        probe = _probe_cache(client, history, model)
        cached_samples.append(probe.get("cached_prefix_tokens", 0))
        ttft, answer = _chat_turn(client, base_url, history, model, api_key)
        ttfts.append(ttft)
        history.append({"role": "assistant", "content": answer})
    if advisor is not None:
        advisor.close()
    return {
        "policy": policy,
        "compactions": compactions,
        "cached_prefix_samples": cached_samples,
        "cached_prefix_total": sum(cached_samples),
        "ttfts_s": [round(t, 4) for t in ttfts],
        "ttft_mean_s": round(statistics.mean(ttfts), 4) if ttfts else None,
    }


def _build_turns(rounds: int, tool_blocks: int) -> list[list[dict]]:
    turns = []
    for i in range(rounds):
        user = {
            "role": "user",
            "content": f"Round {i}: inspect module {i} and summarize findings.",
        }
        tool = {
            "role": "tool",
            "content": f"log-line-{i}-" + "x" * (200 + tool_blocks * 32),
        }
        assistant = {
            "role": "assistant",
            "content": f"Round {i} summary: module {i} behaves as expected.",
        }
        turns.append([user, tool, assistant])
    return turns


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--rounds", type=int, default=40)
    parser.add_argument("--compact-threshold-tokens", type=int, default=6000)
    parser.add_argument("--keep-rounds", type=int, default=3)
    parser.add_argument(
        "--policy",
        choices=["fixed", "advisor"],
        default="fixed",
        help="Single policy to run; a fresh server per policy avoids "
        "cross-policy cache leakage (see nightly driver).",
    )
    parser.add_argument(
        "--output",
        default="benchmarks/reports/compaction_retention.json",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    turns = _build_turns(args.rounds, tool_blocks=8)
    results = []
    with httpx.Client(base_url=args.base_url) as client:
        health = client.get("/health", timeout=10)
        health.raise_for_status()
        logger.info("running policy=%s", args.policy)
        results.append(
            run_policy(
                args.policy,
                client,
                args.base_url,
                args.model,
                args.api_key,
                turns,
                args.compact_threshold_tokens,
                args.keep_rounds,
            )
        )

    report = {
        "model": args.model,
        "rounds": args.rounds,
        "compact_threshold_tokens": args.compact_threshold_tokens,
        "keep_rounds": args.keep_rounds,
        "policy": args.policy,
        "runs": results,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    run = results[0]
    logger.info(
        "result: policy=%s compactions=%d cached_total=%d ttft_mean=%s report=%s",
        run["policy"],
        run["compactions"],
        run["cached_prefix_total"],
        run["ttft_mean_s"],
        out_path,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
