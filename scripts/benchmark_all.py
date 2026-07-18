#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
fusion-mlx 全方位性能基准测试 (Comprehensive Benchmark)
======================================================

测试融合推理引擎的各个特性角度，覆盖：
  A. LLM 核心吞吐量 (单请求 / 连续批处理 / TTFT)
  B. KV Cache 优化 (TurboQuant / 量化 / 分页)
  C. 提示压缩 (PFlash)
  D. 推测解码 (DFlash / n-gram)
  E. 调度策略 (分块预填充 / 优先级)
  F. 多引擎 (VLM / Embedding / Reranker / ImageGen / VideoGen / STT / TTS)
  G. 内存管理 (峰值 / 活跃 / 缓存)
  H. API 兼容性 (OpenAI / Anthropic)

用法:
    python scripts/benchmark_all.py [model] [--quick] [--output results.json]

示例:
    python scripts/benchmark_all.py Qwen3.5-9B-4bit
    python scripts/benchmark_all.py --quick                              # 默认模型快速测试
    python scripts/benchmark_all.py --output /tmp/bench.json
"""

import argparse
import logging
import sys

# ─── 日志 ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("benchmark")


def main():
    """委托到 benchmark_full.py 的新版基准测试"""
    parser = argparse.ArgumentParser(
        description="fusion-mlx 全方位基准测试 (委托到 v2 引擎)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "model",
        nargs="?",
        default="mlx-community/Qwen3.5-9B-4bit",
        help="要测试的模型 (默认: mlx-community/Qwen3.5-9B-4bit)",
    )
    parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="快速模式 (较少的测试点和较短的生成长度)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="输出 JSON 报告文件路径",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="详细日志输出",
    )

    # Deprecation notice
    logger.warning(
        "scripts/benchmark_all.py 已弃用，请使用 scripts/benchmark_full.py\n"
        "  委托执行 benchmark_full.py ..."
    )

    # 委托到 benchmark_full.py
    from benchmark_full import main as bm_main

    sys.argv = [sys.argv[0]]
    if args.model:
        sys.argv.append(args.model)
    if args.quick:
        sys.argv.append("--quick")
    if args.output:
        sys.argv.extend(["--output", args.output])
    if args.verbose:
        sys.argv.append("--verbose")

    bm_main()


if __name__ == "__main__":
    main()