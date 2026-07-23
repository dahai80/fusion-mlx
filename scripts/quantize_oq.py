#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""oQ streaming quantization driver for Qwen3.6-27B-mxfp8.

Usage: python scripts/quantize_oq.py <oq_level> [output_suffix]
e.g.   python scripts/quantize_oq.py 4 oq4
"""
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("quantize_oq")

SOURCE = "/Users/dahai/.fusion-mlx/models/mlx-community/Qwen3.6-27B-mxfp8"
MODELS_DIR = Path("/Users/dahai/.fusion-mlx/models")


def main():
    if len(sys.argv) < 2:
        print("usage: quantize_oq.py <oq_level> [output_suffix]", file=sys.stderr)
        sys.exit(2)
    oq_level = float(sys.argv[1])
    suffix = sys.argv[2] if len(sys.argv) > 2 else f"oq{int(oq_level) if oq_level.is_integer() else oq_level}"
    output = MODELS_DIR / f"Qwen3.6-27B-{suffix}"
    if output.exists():
        log.error("output already exists: %s", output)
        sys.exit(1)

    from fusion_mlx.oq import quantize_oq_streaming, estimate_bpw_and_size

    est = estimate_bpw_and_size(SOURCE, oq_level, 64, False)
    log.info("source=%s", SOURCE)
    log.info("oq_level=%s output=%s", oq_level, output)
    log.info("estimate: bpw=%.2f size=%s peak_mem=%s",
             est["effective_bpw"],
             est["output_size_formatted"],
             est["memory_streaming_formatted"])

    t0 = time.time()
    last_pct = [-1.0]

    def cb(phase, pct):
        if pct - last_pct[0] >= 5.0 or phase != "quantizing":
            log.info("phase=%s pct=%.0f%% elapsed=%.0fs", phase, pct, time.time() - t0)
            last_pct[0] = pct

    quantize_oq_streaming(
        model_path=SOURCE,
        output_path=str(output),
        oq_level=oq_level,
        group_size=64,
        progress_callback=cb,
        text_only=False,
        dtype="bfloat16",
        preserve_mtp=False,
        auto_proxy_sensitivity=True,
    )
    dt = time.time() - t0
    log.info("DONE oQ%s -> %s in %.1f min", oq_level, output, dt / 60.0)


if __name__ == "__main__":
    main()
