# SPDX-License-Identifier: Apache-2.0
"""Community benchmark submission — build payload + upload to bench.dpdns.org."""

from __future__ import annotations

import hashlib
import logging
import uuid
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)

BENCH_API_URL = "http://bench.dpdns.org/api/benchmarks"


def _compute_owner_hash(io_uuid: str, chip: str, gpu_cores: int, ram_gb: float) -> str:
    raw = f"{io_uuid}:{chip}:{gpu_cores}:{ram_gb}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _get_io_uuid() -> str | None:
    import subprocess

    try:
        result = subprocess.run(
            ["ioreg", "-rd1", "-c", "IOPlatformExpertDevice"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        for line in result.stdout.splitlines():
            if "IOPlatformUUID" in line:
                return line.split('"')[-2]
    except Exception:
        pass
    return None


def _detect_quantization(hf_path: str) -> str:
    path_lower = hf_path.lower()
    if "4bit" in path_lower or "q4" in path_lower:
        return "4bit"
    if "8bit" in path_lower or "q8" in path_lower:
        return "8bit"
    if "fp16" in path_lower:
        return "fp16"
    if "bf16" in path_lower:
        return "bf16"
    return "unknown"


def _clean_model_name(hf_path: str) -> str:
    parts = hf_path.split("/")
    name = parts[-1] if parts else hf_path
    name = name.replace("mlx-community--", "").replace("mlx_community__", "")
    return name


def build_submission_payload(
    hardware: Any,
    software: Any,
    alias: str,
    hf_path: str,
    bench: Any,
    notes: str | None = None,
    tier: str | None = None,
    smoke_result: dict | None = None,
    harness_result: dict | None = None,
) -> list[dict[str, Any]]:
    chip_str = getattr(hardware, "chip", "")
    parts = chip_str.split()
    chip_name = parts[0] if parts else chip_str
    chip_variant = " ".join(parts[1:]) if len(parts) > 1 else ""

    io_uuid = _get_io_uuid()
    owner_hash = None
    if io_uuid:
        owner_hash = _compute_owner_hash(
            io_uuid,
            chip_name,
            getattr(hardware, "gpu_cores", 0),
            getattr(hardware, "ram_gb", 0),
        )

    quantization = _detect_quantization(hf_path)
    model_name = _clean_model_name(hf_path)
    submission_group = str(uuid.uuid4())

    payloads = []
    for bucket_name, bucket in [("short", bench.short), ("long", bench.long)]:
        context_length = 128 if bucket_name == "short" else 1024
        payload = {
            "chip_name": chip_name,
            "chip_variant": chip_variant,
            "memory_gb": round(getattr(hardware, "ram_gb", 0)),
            "gpu_cores": getattr(hardware, "gpu_cores", 0),
            "fusionmlx_version": getattr(software, "fusion_mlx", "unknown"),
            "os_version": getattr(software, "macos", "unknown"),
            "model_name": model_name,
            "quantization": quantization,
            "context_length": context_length,
            "pp_tps": round(bucket.prefill_stat.median, 2),
            "tg_tps": round(bucket.decode_stat.median, 2),
            "ttft_ms": round(bucket.ttft_stat.median, 1),
            "submission_group": submission_group,
            "alias": alias,
        }
        if owner_hash:
            payload["owner_hash"] = owner_hash
        if notes:
            payload["notes"] = notes
        if tier:
            payload["tier"] = tier
        if smoke_result:
            payload["smoke_result"] = smoke_result
        if harness_result:
            payload["harness_result"] = harness_result
        payloads.append(payload)

    return payloads


def submit_interactive(payloads: list[dict[str, Any]], repo_root: Path) -> int:
    if not payloads:
        print("  Error: no payload to submit.")
        return 1

    p = payloads[0]
    print()
    print("  ── Submission Preview ──")
    print(f"  Model:      {p.get('model_name')} ({p.get('alias')})")
    print(f"  Quant:      {p.get('quantization')}")
    print(f"  Chip:       {p.get('chip_name')} {p.get('chip_variant')}")
    print(f"  RAM:        {p.get('memory_gb')} GB")
    print(f"  GPU cores:  {p.get('gpu_cores')}")
    print(
        f"  Short pp128:  decode={p.get('tg_tps')} tok/s, prefill={p.get('pp_tps')} tok/s, ttft={p.get('ttft_ms')} ms"
    )
    if len(payloads) > 1:
        p2 = payloads[1]
        print(
            f"  Long pp1024:  decode={p2.get('tg_tps')} tok/s, prefill={p2.get('pp_tps')} tok/s, ttft={p2.get('ttft_ms')} ms"
        )
    if p.get("notes"):
        print(f"  Notes:      {p.get('notes')}")
    print()

    import sys

    if sys.stdin.isatty():
        answer = input("  Submit to bench.dpdns.org? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            print("  Submission cancelled.")
            return 0
    else:
        print("  (non-interactive mode — auto-submitting)")

    success = 0
    failed = 0
    for i, payload in enumerate(payloads):
        context = payload.get("context_length", "?")
        print(f"  Uploading pp{context}…", end=" ", flush=True)
        try:
            resp = requests.post(BENCH_API_URL, json=payload, timeout=30)
            if resp.status_code in (200, 201):
                data = resp.json()
                url = data.get("url", "")
                print(f"✓ ({url})")
                success += 1
            elif resp.status_code == 409:
                data = resp.json()
                url = data.get("existing_url", "")
                print(f"≈ duplicate ({url})")
                success += 1
            else:
                print(f"✗ HTTP {resp.status_code}: {resp.text[:200]}")
                failed += 1
        except Exception as e:
            print(f"✗ {e}")
            failed += 1

    print(f"\n  Done: {success} submitted, {failed} failed.")
    return 0 if failed == 0 else 1


def submit_benchmark(*args: Any, **kwargs: Any) -> dict[str, Any]:
    logger.info("community_bench: submit_benchmark called")
    return {"submitted": False, "reason": "use submit_interactive instead"}


def submit(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return submit_benchmark(*args, **kwargs)
