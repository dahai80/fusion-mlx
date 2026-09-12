#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Partition tests/unit by modality for CI sharding (L18 / PR-D11.3).

Usage:
    python scripts/shard_tests.py <shard-name>

Prints a space-separated list of test file paths for the named shard.
Shards are keyword-prefix groups over test filenames. A file matches
the first shard whose keyword list contains a prefix found in the
filename. Files matching no shard fall into ``rest`` (shard-rest),
so no test is silently dropped.

Shards:
    core   — server/config/cli/request/public_api/middleware/auth/route/surface
    llm    — engine/pool/scheduler/cache/tool/chat/stream/anthropic/openai/ollama/responses/dispatch
    modal  — audio/image/video/vlm/embed/rerank/ner/ocr/latentsync
    infra  — model/admin/telemetry/issue/memory/doctor/upgrade/log/metric
    exp    — minimax/ltx/dflash/spec/quant/migrate/gguf/llama/ane/chip/profile/cache-internals/...
    rest   — everything else (uncategorized)

Exit 0 with a file list; exit 1 with a message if the shard name is
unknown. Designed for CI:
    pytest $(python scripts/shard_tests.py core)
"""

from __future__ import annotations

import sys
from pathlib import Path

SHARDS: dict[str, list[str]] = {
    "core": [
        "server",
        "config",
        "cli",
        "request",
        "public_api",
        "middleware",
        "auth",
        "route",
        "surface",
        "no_mllm",
        "parent_watchdog",
        "startup",
        "lifespan",
        "probe",
        "health",
    ],
    "llm": [
        "engine",
        "pool",
        "scheduler",
        "cache",
        "tool",
        "chat",
        "stream",
        "anthropic",
        "openai",
        "ollama",
        "responses",
        "dispatch",
        "batched",
        "spec_decode",
        "grammar",
        "guided",
        "thinking",
        "strict_json",
        "mcp",
        "agent",
        "watermark",
        "convert",
        "distributed",
        "session",
    ],
    "modal": [
        "audio",
        "image",
        "video",
        "vlm",
        "embed",
        "rerank",
        "ner",
        "ocr",
        "latentsync",
        "vision",
        "musetalk",
        "pulid",
        "tts",
        "stt",
        "sts",
    ],
    "infra": [
        "model",
        "admin",
        "telemetry",
        "issue",
        "memory",
        "doctor",
        "upgrade",
        "log",
        "metric",
        "memory_monitor",
        "proc_memory",
    ],
    "exp": [
        "minimax",
        "ltx",
        "dflash",
        "spec",
        "quant",
        "turbo",
        "oq",
        "migrate",
        "gguf",
        "llama",
        "ane",
        "chip",
        "profile",
        "radix",
        "paged",
        "tiered",
        "response_cache",
        "boundary",
        "kv",
        "dflash2",
        "dspark",
        "eagle",
        "ngram",
        "mtp",
        "denoise",
        "auto_config",
        "alias",
        "checksum",
        "patch",
        "monkeypatch",
        "w4a8",
        "mxfp",
        "kv_cache",
        "head_dim",
        "sdpa",
        "gemm",
        "kernel",
        "metal",
        "prefill",
        "eviction",
        "admission",
        "scheduling",
        "priority",
        "preempt",
        "watchdog",
        "poison",
        "recovery",
        "snapshot",
        "atomic",
        "rollback",
        "converse",
        "inpaint",
        "img2img",
        "safety",
        "sanitize",
        "compat",
        "shim",
        "stub",
        "torch",
        "feature",
        "cache_recovery",
        "kv_resume",
        "disconnect",
        "postprocess",
        "helper",
        "rate",
        "cors",
        "body_limit",
        "request_id",
        "exception",
        "probe_fastpath",
        "route_guard",
        "context",
        "setting",
        "mirror",
        "preflight",
        "download",
        "submit",
        "bench",
        "usage",
        "runtime_config",
        "doctor_config",
        "cluster",
        "node",
        "mdns",
        "peer",
        "failover",
        "router",
        "shard",
        "pipeline",
        "ablation",
        "eval",
        "perf_gate",
        "coherence",
        "grader",
        "perplexity",
    ],
}

# "rest" is implicit — files not claimed by any shard above.
KNOWN_SHARDS = list(SHARDS) + ["rest"]

TESTS_DIR = Path(__file__).resolve().parent.parent / "tests" / "unit"


def _shard_for(filename: str) -> str:
    # filename like "test_audio_routes.py" — strip prefix, match keyword.
    stem = filename
    if stem.startswith("test_"):
        stem = stem[len("test_") :]
    for shard, keywords in SHARDS.items():
        for kw in keywords:
            if stem.startswith(kw):
                return shard
    return "rest"


def files_for(shard: str) -> list[str]:
    if shard not in KNOWN_SHARDS:
        print(
            f"unknown shard: {shard!r} (known: {', '.join(KNOWN_SHARDS)})",
            file=sys.stderr,
        )
        return []
    files: list[str] = []
    for p in sorted(TESTS_DIR.glob("test_*.py")):
        if _shard_for(p.name) == shard:
            files.append(str(p))
    return files


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"usage: {argv[0]} <shard-name>", file=sys.stderr)
        print(f"shards: {', '.join(KNOWN_SHARDS)}", file=sys.stderr)
        return 2
    shard = argv[1]
    files = files_for(shard)
    if not files and shard not in KNOWN_SHARDS:
        return 1
    print(" ".join(files))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
