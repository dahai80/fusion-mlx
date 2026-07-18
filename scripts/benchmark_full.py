#!/usr/bin/env python3
"""
fusion-mlx 全方位基准测试 (v2 — 适配当前代码库 API)

覆盖所有特性:
  A. LLM 单请求吞吐量 (不同预填充/生成长度)
  B. 连续批处理吞吐量 (不同批量大小)
  C. TTFT 与内存分析 (长预填充)
  D. KV Cache 优化对比 (标准 + 前缀缓存)
  E. PFlash 提示压缩 (如可用)
  F. 推测解码 (如可用)
  G. 内存管理概览
  H. 多引擎支持

输出: 终端表格 + JSON 报告
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator

logger = logging.getLogger("benchmark")


# ─── 数据结构 ───────────────────────────────────────────────────────────────

@dataclass
class BenchResult:
    category: str
    name: str
    metrics: dict
    model: str = ""
    config: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BenchReport:
    model: str
    hardware: dict = field(default_factory=dict)
    started_at: str = ""
    finished_at: str = ""
    results: list[BenchResult] = field(default_factory=list)
    summary: dict = field(default_factory=dict)

    def add(self, result: BenchResult) -> None:
        self.results.append(result)

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "hardware": self.hardware,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "results": [r.to_dict() for r in self.results],
            "summary": self.summary,
        }


# ─── 辅助函数 ───────────────────────────────────────────────────────────────

def _human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


def _human_memory(n: int) -> str:
    return _human_size(n)


def _generate_prompt(tokenizer: Any, target_tokens: int) -> str:
    """生成指定 token 数量的提示文本"""
    text = "The quick brown fox jumps over the lazy dog. " * 50
    encoded = tokenizer.encode(text)
    repeats = max(1, target_tokens // len(encoded) + 1)
    return (text * repeats)[: target_tokens * 6]


def _compute_single_metrics(
    prompt_tokens: int,
    completion_tokens: int,
    start_time: float,
    first_token_time: float,
    end_time: float,
    peak_memory: int = 0,
    cached_tokens: int = 0,
) -> dict:
    ttft_s = first_token_time - start_time
    gen_s = end_time - first_token_time
    total_s = end_time - start_time

    ttft_ms = ttft_s * 1000
    gen_tps = completion_tokens / max(gen_s, 1e-9)
    processing_tps = (prompt_tokens + completion_tokens) / max(total_s, 1e-9)
    pp_tps = prompt_tokens / max(ttft_s, 1e-9)

    return {
        "ttft_ms": round(ttft_ms, 1),
        "gen_tps": round(gen_tps, 1),
        "processing_tps": round(processing_tps, 1),
        "pp_tps": round(pp_tps, 1),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_time_s": round(total_s, 3),
        "gen_time_s": round(gen_s, 3),
        "peak_memory_bytes": peak_memory,
        "peak_memory_human": _human_memory(peak_memory) if peak_memory else "?",
        "cached_tokens": cached_tokens,
    }


def detect_hardware() -> dict:
    info = {}
    try:
        import platform
        info["platform"] = platform.platform()
        info["processor"] = platform.processor()
        info["machine"] = platform.machine()
        info["python_version"] = platform.python_version()
    except Exception:
        pass
    try:
        import mlx.core as mx
        info["device_name"] = str(mx.default_device())
        info["mlx_version"] = getattr(mx, "__version__", "?")
        try:
            info["metal_device"] = str(mx.metal.device_info())
        except Exception:
            pass
    except Exception:
        pass
    try:
        import psutil
        info["memory_total_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
        info["memory_available_gb"] = round(psutil.virtual_memory().available / 1e9, 1)
    except Exception:
        pass
    return info


# ─── 基准测试器 ─────────────────────────────────────────────────────────────

class FusionMLXBenchmarkV2:
    """fusion-mlx 全方位基准测试器 (v2)"""

    def __init__(self, model_path: str, quick: bool = False):
        self.model_path = model_path
        self.quick = quick
        self.report = BenchReport(model=model_path)
        self._engine = None
        self._tokenizer = None
        self._model = None

        # 测试规模
        if quick:
            self.prompt_lengths = [128, 512, 1024]
            self.generation_lengths = [64, 128]
            self.batch_sizes = [2, 4]
            self.long_prompt_lengths = [1024, 4096]
        else:
            self.prompt_lengths = [128, 512, 1024, 2048, 4096]
            self.generation_lengths = [64, 128, 256]
            self.batch_sizes = [1, 2, 4, 8]
            self.long_prompt_lengths = [1024, 4096, 8192]

    async def _load_engine(self) -> None:
        """加载模型引擎 (直接使用 mlx_lm + AsyncEngineCore)"""
        from fusion_mlx.engine_core import AsyncEngineCore, EngineConfig
        from fusion_mlx.scheduler import SchedulerConfig
        from mlx_lm import load

        logger.info("加载模型: %s ...", self.model_path)

        # 模型路径处理 — 如果是 HF repo 名，尝试本地缓存
        model_path = self.model_path
        local_path = Path(model_path)
        if not local_path.exists():
            hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
            safe_name = model_path.replace("/", "--")
            cache_dir = hf_cache / f"models--{safe_name}"
            if cache_dir.exists():
                snapshots = list((cache_dir / "snapshots").iterdir())
                if snapshots:
                    model_path = str(snapshots[0])
                    logger.info("使用本地缓存路径: %s", model_path)

        self._model, self._tokenizer = load(
            model_path,
            tokenizer_config={"trust_remote_code": True, "eos_token": "<|im_end|>"},
        )
        logger.info("模型加载完成: %s", type(self._model).__name__)

        # 创建 EngineCore
        scheduler_config = SchedulerConfig()
        engine_config = EngineConfig(
            model_name=self.model_path,
            scheduler_config=scheduler_config,
            stream_interval=1,
        )
        self._engine = AsyncEngineCore(
            model=self._model,
            tokenizer=self._tokenizer,
            config=engine_config,
        )
        await self._engine.engine.start()
        logger.info("引擎启动完成")

    async def _unload_engine(self) -> None:
        """卸载引擎"""
        if self._engine:
            try:
                await self._engine.stop()
            except Exception as e:
                logger.warning("卸载引擎失败: %s", e)
            self._engine = None
        self._tokenizer = None
        self._model = None

    async def _stream_generate(
        self, prompt: str, max_tokens: int, **kwargs
    ) -> AsyncIterator[Any]:
        """使用 AsyncEngineCore API 进行流式生成"""
        from fusion_mlx.request import SamplingParams

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=kwargs.get("temperature", 0.0),
            top_p=kwargs.get("top_p", 1.0),
            stop=kwargs.get("stop", []),
        )
        request_id = await self._engine.add_request(
            prompt=prompt, sampling_params=sampling_params,
        )
        async for output in self._engine.stream_outputs(request_id):
            yield output

    # ── A: LLM 单请求吞吐量 ─────────────────────────────────────────────────

    async def bench_llm_single(self) -> list[BenchResult]:
        """测试不同预填充/生成长度下的单请求吞吐量"""
        logger.info("━" * 60)
        logger.info("A. LLM 核心吞吐量 — 单请求")
        results = []

        engine = self._engine
        tokenizer = self._tokenizer

        # 预热
        logger.info("  预热中...")
        warmup_prompt = _generate_prompt(tokenizer, 32)
        async for _ in self._stream_generate(
            warmup_prompt, max_tokens=8, temperature=0.0
        ):
            pass

        # 生成提示缓存
        prompts_cache = {}
        for pp_len in self.prompt_lengths:
            prompts_cache[pp_len] = _generate_prompt(tokenizer, pp_len)

        for gen_len in self.generation_lengths:
            for pp_len in self.prompt_lengths:
                try:
                    import mlx.core as mx
                    mx.reset_peak_memory()
                except Exception:
                    pass

                start = time.perf_counter()
                first_token = None
                prev_tokens = 0
                last_output = None

                async for output in self._stream_generate(
                    prompts_cache[pp_len],
                    max_tokens=gen_len,
                    temperature=0.0,
                    top_p=1.0,
                ):
                    if first_token is None and output.completion_tokens > prev_tokens:
                        first_token = time.perf_counter()
                    prev_tokens = output.completion_tokens
                    last_output = output

                end = time.perf_counter()
                if first_token is None:
                    first_token = end

                peak = 0
                try:
                    peak = mx.get_peak_memory()
                except Exception:
                    pass

                pt = last_output.prompt_tokens if last_output else 0
                ct = last_output.completion_tokens if last_output else 0
                cached = last_output.cached_tokens if last_output else 0

                metrics = _compute_single_metrics(
                    prompt_tokens=pt,
                    completion_tokens=ct,
                    start_time=start,
                    first_token_time=first_token,
                    end_time=end,
                    peak_memory=peak,
                    cached_tokens=cached,
                )

                result = BenchResult(
                    category="llm_single",
                    name=f"pp{pp_len}_tg{gen_len}",
                    metrics=metrics,
                    model=self.model_path,
                    config={"prompt_tokens": pp_len, "generation_tokens": gen_len},
                )
                results.append(result)

                logger.info(
                    "  pp=%-6d tg=%-4d  TTFT=%-8.1fms  gen=%-8.1ftok/s  "
                    "proc=%-8.1ftok/s  mem=%s",
                    pp_len, gen_len,
                    metrics["ttft_ms"],
                    metrics["gen_tps"],
                    metrics["processing_tps"],
                    metrics.get("peak_memory_human", "?"),
                )

        return results

    # ── B: 连续批处理吞吐量 ─────────────────────────────────────────────────

    async def bench_continuous_batching(self) -> list[BenchResult]:
        """测试不同批量大小下的连续批处理吞吐量"""
        logger.info("━" * 60)
        logger.info("B. 连续批处理吞吐量")
        results = []

        engine = self._engine
        tokenizer = self._tokenizer

        from fusion_mlx.request import SamplingParams

        pp_len = 512
        gen_len = 128 if not self.quick else 64
        prompt = _generate_prompt(tokenizer, pp_len)

        for batch_size in self.batch_sizes:
            if batch_size == 1:
                continue

            try:
                wall_start = time.perf_counter()
                first_tokens = []
                end_times = []
                total_completion = 0

                prompts = [prompt] * batch_size
                sampling_params = SamplingParams(
                    max_tokens=gen_len,
                    temperature=0.0,
                    top_p=1.0,
                )

                async def _run_single(p: str, idx: int) -> dict:
                    nonlocal total_completion
                    first = None
                    prev_t = 0
                    ct = 0
                    request_id = await engine.add_request(
                        prompt=p, sampling_params=sampling_params,
                    )
                    async for output in engine.stream_outputs(request_id):
                        if first is None and output.completion_tokens > prev_t:
                            first = time.perf_counter()
                        prev_t = output.completion_tokens
                        if output.finished:
                            ct = output.completion_tokens
                    end = time.perf_counter()
                    if first is None:
                        first = end
                    return {
                        "ttft_s": first - wall_start,
                        "first_abs": first,
                        "end_abs": end,
                        "tokens": ct,
                    }

                sub_results = await asyncio.gather(
                    *[_run_single(prompts[i], i) for i in range(batch_size)]
                )
                wall_end = time.perf_counter()

                total_gen = sum(r["tokens"] for r in sub_results)
                total_prompt = pp_len * batch_size
                wall_time = wall_end - wall_start
                avg_ttft = (sum(r["ttft_s"] for r in sub_results) / batch_size) * 1000
                max_first = max(r["first_abs"] for r in sub_results)
                prefill_wall = max_first - wall_start
                pp_tps = total_prompt / max(prefill_wall, 1e-9)
                gen_wall = wall_end - max_first
                tg_tps = total_gen / max(gen_wall, 1e-9)

                metrics = {
                    "batch_size": batch_size,
                    "pp_tps": round(pp_tps, 1),
                    "tg_tps": round(tg_tps, 1),
                    "avg_ttft_ms": round(avg_ttft, 1),
                    "e2e_latency_s": round(wall_time, 3),
                    "total_prompt_tokens": total_prompt,
                    "total_gen_tokens": total_gen,
                    "throughput_per_user": round(tg_tps / batch_size, 1),
                }

                result = BenchResult(
                    category="continuous_batching",
                    name=f"batch{batch_size}_pp{pp_len}_tg{gen_len}",
                    metrics=metrics,
                    model=self.model_path,
                    config={"batch_size": batch_size, "pp": pp_len, "tg": gen_len},
                )
                results.append(result)

                logger.info(
                    "  batch=%-2d  pp=%-6d tg=%-4d  pp_tps=%-8.1f  tg_tps=%-8.1f  "
                    "avg_ttft=%-6.1fms",
                    batch_size, pp_len, gen_len, pp_tps, tg_tps, avg_ttft,
                )

            except Exception as e:
                logger.warning("  批处理测试 batch=%d 失败: %s", batch_size, e)

        return results

    # ── C: TTFT 与内存分析 ─────────────────────────────────────────────────

    async def bench_ttft_memory(self) -> list[BenchResult]:
        """测试不同预填充长度下的 TTFT 和内存峰值"""
        logger.info("━" * 60)
        logger.info("C. TTFT 与内存分析")
        results = []

        engine = self._engine
        tokenizer = self._tokenizer

        for pp_len in self.long_prompt_lengths:
            if pp_len > 8192 and self.quick:
                continue

            prompt = _generate_prompt(tokenizer, pp_len)
            gen_len = 32

            try:
                import mlx.core as mx
                mx.reset_peak_memory()
            except Exception:
                pass

            start = time.perf_counter()
            first_token = None
            prev_tokens = 0
            last_output = None

            mem_samples = []

            async for output in self._stream_generate(
                prompt=prompt, max_tokens=gen_len, temperature=0.0, top_p=1.0,
            ):
                if first_token is None and output.completion_tokens > prev_tokens:
                    first_token = time.perf_counter()
                prev_tokens = output.completion_tokens
                last_output = output

                try:
                    import mlx.core as mx
                    mem_samples.append({
                        "time_s": time.perf_counter() - start,
                        "active": mx.get_active_memory(),
                        "cache": mx.get_cache_memory(),
                        "peak": mx.get_peak_memory(),
                    })
                except Exception:
                    pass

            end = time.perf_counter()
            if first_token is None:
                first_token = end

            peak = 0
            try:
                peak = mx.get_peak_memory()
            except Exception:
                pass

            base_mem = 0
            try:
                base_mem = mx.get_active_memory() - (peak - mx.get_peak_memory())
            except Exception:
                pass

            ttft = (first_token - start) * 1000
            ct = last_output.completion_tokens if last_output else 0

            metrics = {
                "prompt_tokens": pp_len,
                "ttft_ms": round(ttft, 1),
                "peak_memory_bytes": peak,
                "peak_memory_human": _human_memory(peak),
                "base_memory_estimate": base_mem,
                "gen_tokens": ct,
                "memory_per_token": round(peak / max(ct + pp_len, 1)),
            }

            result = BenchResult(
                category="ttft_memory",
                name=f"pp{pp_len}",
                metrics=metrics,
                model=self.model_path,
                config={"prompt_tokens": pp_len, "generation_tokens": gen_len},
            )
            results.append(result)

            logger.info(
                "  pp=%-6d  TTFT=%-8.1fms  峰值内存=%-12s",
                pp_len, ttft,
                _human_memory(peak),
            )

        return results

    # ── D: KV Cache 优化 ───────────────────────────────────────────────────

    async def bench_kv_cache(self) -> list[BenchResult]:
        """测试不同 KV Cache 配置下的性能"""
        logger.info("━" * 60)
        logger.info("D. KV Cache 优化对比")
        results = []

        engine = self._engine
        tokenizer = self._tokenizer

        pp_len = 1024
        gen_len = 128 if not self.quick else 64
        prompt = _generate_prompt(tokenizer, pp_len)

        # D1: 标准 (无量化)
        logger.info("  D1. 标准 KV Cache (无量化) ...")
        try:
            start = time.perf_counter()
            first_token = None
            prev_tokens = 0
            last_output = None
            async for output in self._stream_generate(
                prompt=prompt, max_tokens=gen_len, temperature=0.0, top_p=1.0,
            ):
                if first_token is None and output.completion_tokens > prev_tokens:
                    first_token = time.perf_counter()
                prev_tokens = output.completion_tokens
                last_output = output
            end = time.perf_counter()
            if first_token is None:
                first_token = end
            pt = last_output.prompt_tokens if last_output else 0
            ct = last_output.completion_tokens if last_output else 0
            metrics = _compute_single_metrics(pt, ct, start, first_token, end)
            result = BenchResult(
                category="kv_cache",
                name="standard",
                metrics=metrics,
                model=self.model_path,
                config={"kv_cache_mode": "standard"},
            )
            results.append(result)
            logger.info("    TTFT=%.1fms  gen=%.1ftok/s", metrics["ttft_ms"], metrics["gen_tps"])
        except Exception as e:
            logger.warning("    D1 失败: %s", e)

        # D2: 前缀缓存
        logger.info("  D2. 前缀缓存命中率...")
        try:
            shared_prefix = "The quick brown fox jumps over the lazy dog. " * 100
            prompt_a = shared_prefix + "What is the capital of France?"
            prompt_b = shared_prefix + "What is the capital of Germany?"

            shared_prefix_tokens = len(tokenizer.encode(shared_prefix))
            # 第一次请求
            async for _ in self._stream_generate(
                prompt_a, max_tokens=16, temperature=0.0,
            ):
                pass

            # 第二次请求 (应命中前缀缓存)
            start = time.perf_counter()
            async for output in self._stream_generate(
                prompt_b, max_tokens=16, temperature=0.0,
            ):
                last_output = output
            end = time.perf_counter()

            cached = last_output.cached_tokens if last_output else 0
            ttft = (end - start) * 1000

            metrics = {
                "shared_prefix_tokens": shared_prefix_tokens,
                "cached_tokens": cached,
                "cache_hit_rate": round(cached / max(shared_prefix_tokens, 1), 2),
                "ttft_with_cache_ms": round(ttft, 1),
            }
            result = BenchResult(
                category="kv_cache",
                name="prefix_cache",
                metrics=metrics,
                model=self.model_path,
                config={"test": "prefix_cache_hit"},
            )
            results.append(result)
            logger.info(
                "    共享前缀 %d tokens, 缓存命中 %d, 命中率 %.0f%%, TTFT=%.1fms",
                shared_prefix_tokens, cached,
                metrics["cache_hit_rate"] * 100,
                ttft,
            )
        except Exception as e:
            logger.warning("    D2 失败: %s", e)

        return results

    # ── E: PFlash 提示压缩 ─────────────────────────────────────────────────

    async def bench_pflash(self) -> list[BenchResult]:
        """测试 PFlash 提示压缩对 TTFT 的影响"""
        logger.info("━" * 60)
        logger.info("E. PFlash 提示压缩")
        results = []

        tokenizer = self._tokenizer
        pp_len = 4096 if not self.quick else 1024
        prompt = _generate_prompt(tokenizer, pp_len)

        try:
            from fusion_mlx.pflash import PFlashConfig, compress_request_tokens
            has_pflash = True
        except ImportError:
            has_pflash = False

        if not has_pflash:
            logger.info("  PFlash 不可用，跳过")
            return results

        # E1: 基线
        logger.info("  E1. 基线 (无压缩)...")
        engine = self._engine
        try:
            start = time.perf_counter()
            first_token = None
            prev_tokens = 0
            async for output in self._stream_generate(
                prompt=prompt, max_tokens=32, temperature=0.0, top_p=1.0,
            ):
                if first_token is None and output.completion_tokens > prev_tokens:
                    first_token = time.perf_counter()
                prev_tokens = output.completion_tokens
            end = time.perf_counter()
            if first_token is None:
                first_token = end
            base_ttft = (first_token - start) * 1000
            metrics = {
                "prompt_tokens": pp_len,
                "ttft_baseline_ms": round(base_ttft, 1),
                "pflash_enabled": False,
            }
            result = BenchResult(
                category="pflash",
                name=f"baseline_pp{pp_len}",
                metrics=metrics,
                model=self.model_path,
                config={"pflash": False, "pp": pp_len, "tg": 32},
            )
            results.append(result)
            logger.info("    TTFT 基线: %.1fms", base_ttft)
        except Exception as e:
            logger.warning("    E1 失败: %s", e)

        # E2: 压缩比分析
        logger.info("  E2. 压缩比分析...")
        try:
            config = PFlashConfig(mode="always")
            prompt_ids = tokenizer.encode(prompt)
            compressed_ids, metadata = compress_request_tokens(prompt_ids, config)
            original = metadata.get("original_tokens", len(prompt_ids))
            compressed_t = metadata.get("kept_tokens", len(compressed_ids))
            ratio = metadata.get("compression_ratio", 0.0)
            if metadata.get("compressed", False):
                logger.info(
                    "    原始 %d → 压缩 %d (压缩比 %.2f%%)",
                    original, compressed_t, ratio * 100,
                )
            else:
                logger.info(
                    "    未压缩 (原因: %s) 原始 %d",
                    metadata.get("reason", "unknown"), original,
                )
            metrics = {
                "original_tokens": original,
                "compressed_tokens": compressed_t,
                "compression_ratio": round(ratio, 4),
                "reduction_pct": round((1 - ratio) * 100, 1) if ratio > 0 else 0.0,
                "compressed": metadata.get("compressed", False),
                "reason": metadata.get("reason", ""),
            }
            result = BenchResult(
                category="pflash",
                name="compression_ratio",
                metrics=metrics,
                model=self.model_path,
                config={"pflash_mode": "always"},
            )
            results.append(result)
        except Exception as e:
            logger.warning("    E2 失败: %s", e)

        return results

    # ── F: 推测解码 ────────────────────────────────────────────────────────

    async def bench_spec_decode(self) -> list[BenchResult]:
        """测试推测解码的各模式性能"""
        logger.info("━" * 60)
        logger.info("F. 推测解码")
        results = []

        try:
            from fusion_mlx.scheduler.spec_decode import SPEC_DRAFT_MODEL_ENABLED
            from fusion_mlx.scheduler.ngram_spec import NGRAM_SPEC_ENABLED
        except ImportError:
            logger.warning("  推测解码模块不可用，跳过")
            return results

        tokenizer = self._tokenizer
        engine = self._engine
        engine_core = getattr(engine, "engine", None)
        if engine_core is None:
            logger.warning("  引擎无核心调度器，跳过")
            return results

        scheduler = getattr(engine_core, "scheduler", None)
        if scheduler is None:
            logger.warning("  调度器不可用，跳过")
            return results

        # F1: n-gram 推测解码状态
        ngram_state = getattr(scheduler, "_ngram_spec_state", None)
        if ngram_state:
            predictor = getattr(ngram_state, "predictor", None)
            if predictor:
                order = getattr(predictor, "order", 0)
                num_draft = getattr(predictor, "num_draft", 0)
                logger.info("  F1. n-gram 推测解码: order=%d, num_draft=%d", order, num_draft)
                metrics = {"ngram_order": order, "ngram_num_draft": num_draft, "enabled": True}
                results.append(BenchResult(
                    category="spec_decode",
                    name="ngram_status",
                    metrics=metrics,
                    model=self.model_path,
                ))
        else:
            logger.info("  F1. n-gram 推测解码: 未启用")

        # F2: 推测解码生成速度
        pp_len = 512
        gen_len = 256 if not self.quick else 64
        prompt = _generate_prompt(tokenizer, pp_len)
        logger.info("  F2. 推测解码生成速度...")
        try:
            start = time.perf_counter()
            first_token = None
            prev_tokens = 0
            last_output = None
            async for output in self._stream_generate(
                prompt=prompt, max_tokens=gen_len, temperature=0.0, top_p=1.0,
            ):
                if first_token is None and output.completion_tokens > prev_tokens:
                    first_token = time.perf_counter()
                prev_tokens = output.completion_tokens
                last_output = output
            end = time.perf_counter()
            if first_token is None:
                first_token = end
            pt = last_output.prompt_tokens if last_output else 0
            ct = last_output.completion_tokens if last_output else 0
            gen_duration = end - first_token
            gen_tps = ct / max(gen_duration, 1e-9)
            ttft = (first_token - start) * 1000

            logger.info("    gen=%.1ftok/s  TTFT=%.1fms  tokens=%d", gen_tps, ttft, ct)
            metrics = {
                "gen_tps": round(gen_tps, 1),
                "ttft_ms": round(ttft, 1),
                "gen_tokens": ct,
                "prompt_tokens": pt,
                "spec_decode_enabled": ngram_state is not None,
            }
            results.append(BenchResult(
                category="spec_decode",
                name=f"spec_generation_pp{pp_len}_tg{gen_len}",
                metrics=metrics,
                model=self.model_path,
                config={"pp": pp_len, "tg": gen_len},
            ))
        except Exception as e:
            logger.warning("    F2 失败: %s", e)

        return results

    # ── G: 内存管理 ───────────────────────────────────────────────────────

    async def bench_memory(self) -> list[BenchResult]:
        """测试内存使用概览"""
        logger.info("━" * 60)
        logger.info("G. 内存管理概览")
        results = []

        try:
            import mlx.core as mx
            active = mx.get_active_memory()
            cache = mx.get_cache_memory()
            peak = mx.get_peak_memory()

            metrics = {
                "active_memory_bytes": active,
                "active_memory_human": _human_memory(active),
                "cache_memory_bytes": cache,
                "cache_memory_human": _human_memory(cache),
                "peak_memory_bytes": peak,
                "peak_memory_human": _human_memory(peak),
                "total_allocated_bytes": active + cache,
                "total_allocated_human": _human_memory(active + cache),
            }

            results.append(BenchResult(
                category="memory",
                name="memory_overview",
                metrics=metrics,
                model=self.model_path,
            ))

            logger.info(
                "  活跃=%-12s  缓存=%-12s  峰值=%-12s  合计=%-12s",
                _human_memory(active),
                _human_memory(cache),
                _human_memory(peak),
                _human_memory(active + cache),
            )
        except Exception as e:
            logger.warning("  G 测试失败: %s", e)

        return results

    # ── H: 多引擎 ─────────────────────────────────────────────────────────

    async def bench_multi_engine(self) -> list[BenchResult]:
        """测试其他引擎类型 (如可用)"""
        logger.info("━" * 60)
        logger.info("H. 多引擎支持")
        results = []

        try:
            from fusion_mlx.engines import (
                EmbeddingEngine, RerankerEngine, STSEngine,
                ImageGenEngine, VideoGenEngine, STTEngine, TTSEngine,
            )
            engine_types = {
                "EmbeddingEngine": EmbeddingEngine,
                "RerankerEngine": RerankerEngine,
                "STSEngine": STSEngine,
                "ImageGenEngine": ImageGenEngine,
                "VideoGenEngine": VideoGenEngine,
                "STTEngine": STTEngine,
                "TTSEngine": TTSEngine,
            }
            for name, cls in engine_types.items():
                try:
                    inst = cls(model_name="test")
                    loaded = inst._loaded if hasattr(inst, "_loaded") else False
                    logger.info("  %s: 可用 (%s)", name, "已加载" if loaded else "未加载")
                except Exception as e:
                    logger.info("  %s: 不可用 (%s)", name, e)

            metrics = {
                "available_engines": list(engine_types.keys()),
                "engine_count": len(engine_types),
            }
            results.append(BenchResult(
                category="multi_engine",
                name="engine_availability",
                metrics=metrics,
                model=self.model_path,
            ))
        except Exception as e:
            logger.warning("  H 测试失败: %s", e)

        return results

    # ── 运行所有测试 ──────────────────────────────────────────────────────

    async def run_all(self) -> BenchReport:
        """运行所有基准测试"""
        self.report.started_at = datetime.now().isoformat()
        self.report.hardware = detect_hardware()

        logger.info("═" * 60)
        logger.info("fusion-mlx 全方位基准测试 v2")
        logger.info("═" * 60)
        logger.info("模型: %s", self.model_path)
        logger.info("模式: %s", "快速" if self.quick else "完整")
        logger.info("硬件: %s", self.report.hardware.get("device_name", "?"))
        logger.info("")

        try:
            await self._load_engine()

            test_suites = [
                ("A. LLM 单请求", self.bench_llm_single()),
                ("B. 连续批处理", self.bench_continuous_batching()),
                ("C. TTFT & 内存", self.bench_ttft_memory()),
                ("D. KV Cache", self.bench_kv_cache()),
                ("E. PFlash", self.bench_pflash()),
                ("F. 推测解码", self.bench_spec_decode()),
                ("G. 内存管理", self.bench_memory()),
                ("H. 多引擎", self.bench_multi_engine()),
            ]

            for name, coro in test_suites:
                logger.info("")
                try:
                    suite_results = await coro
                    for r in suite_results:
                        self.report.add(r)
                except Exception as e:
                    logger.error("  %s 测试套件失败: %s", name, e)
                    import traceback
                    traceback.print_exc()

            # 生成摘要
            await self._generate_summary()

        except Exception as e:
            logger.error("基准测试失败: %s", e)
            import traceback
            traceback.print_exc()
        finally:
            await self._unload_engine()

        self.report.finished_at = datetime.now().isoformat()
        return self.report

    async def _generate_summary(self) -> None:
        """生成基准测试摘要"""
        summary = {
            "total_tests": len(self.report.results),
            "categories": {},
            "highlights": {},
        }

        for r in self.report.results:
            cat = r.category
            if cat not in summary["categories"]:
                summary["categories"][cat] = {"count": 0, "tests": []}
            summary["categories"][cat]["count"] += 1
            summary["categories"][cat]["tests"].append(r.name)

        highlights = {}

        llm_results = [
            r for r in self.report.results
            if r.category == "llm_single" and "gen_tps" in r.metrics
        ]
        if llm_results:
            best = max(llm_results, key=lambda r: r.metrics["gen_tps"])
            highlights["best_gen_tps"] = {
                "value": best.metrics["gen_tps"],
                "unit": "tok/s",
                "test": best.name,
            }
            best_ttft = min(llm_results, key=lambda r: r.metrics.get("ttft_ms", 9999))
            highlights["best_ttft"] = {
                "value": best_ttft.metrics.get("ttft_ms", 0),
                "unit": "ms",
                "test": best_ttft.name,
            }

        batch_results = [
            r for r in self.report.results
            if r.category == "continuous_batching" and "tg_tps" in r.metrics
        ]
        if batch_results:
            best_batch = max(batch_results, key=lambda r: r.metrics["tg_tps"])
            highlights["best_batch_tg_tps"] = {
                "value": best_batch.metrics["tg_tps"],
                "unit": "tok/s",
                "batch_size": best_batch.metrics.get("batch_size"),
                "test": best_batch.name,
            }

        memory_results = [
            r for r in self.report.results
            if r.category == "memory" and "peak_memory_bytes" in r.metrics
        ]
        if memory_results:
            for r in memory_results:
                highlights["peak_memory"] = {
                    "value": r.metrics.get("peak_memory_human", "?"),
                    "bytes": r.metrics.get("peak_memory_bytes", 0),
                }
                highlights["active_memory"] = {
                    "value": r.metrics.get("active_memory_human", "?"),
                    "bytes": r.metrics.get("active_memory_bytes", 0),
                }

        summary["highlights"] = highlights
        self.report.summary = summary

        logger.info("")
        logger.info("═" * 60)
        logger.info("基准测试完成 — 摘要")
        logger.info("═" * 60)
        logger.info("总测试数: %d", summary["total_tests"])
        logger.info("类别:")
        for cat, info in summary["categories"].items():
            logger.info("  %-20s %d 项测试", cat, info["count"])
        logger.info("")
        if highlights:
            for key, val in highlights.items():
                logger.info("  %s: %s", key, val)


# ─── 报告输出 ───────────────────────────────────────────────────────────────

def print_report_table(report: BenchReport) -> None:
    """打印人类可读的基准测试结果表"""
    print("\n")
    print("=" * 100)
    print(f"  fusion-mlx 基准测试报告")
    print(f"  模型: {report.model}")
    print(f"  硬件: {report.hardware.get('device_name', '?')}")
    print(f"  时间: {report.started_at} → {report.finished_at}")
    print("=" * 100)

    categories = {}
    for r in report.results:
        categories.setdefault(r.category, []).append(r)

    for cat, tests in sorted(categories.items()):
        print(f"\n  [{cat.upper()}]")
        all_keys = set()
        for t in tests:
            all_keys.update(t.metrics.keys())
        display_keys = [k for k in all_keys if not k.endswith("_bytes") and not k.endswith("_human")]
        display_keys = sorted(display_keys)

        header = f"  {'测试':<30}"
        for k in display_keys[:8]:
            header += f" {k:<18}"
        print(header)
        print("  " + "-" * len(header))

        for t in tests:
            row = f"  {t.name:<30}"
            for k in display_keys[:8]:
                val = t.metrics.get(k, "")
                row += f" {str(val):<18}"
            print(row)
        print()

    if report.summary.get("highlights"):
        print("\n  [亮点摘要]")
        for key, val in report.summary["highlights"].items():
            print(f"    {key:<25} {val}")
    print()


def save_report(report: BenchReport, path: str) -> None:
    """保存报告为 JSON"""
    with open(path, "w") as f:
        json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
    logger.info("报告已保存到: %s", path)


# ─── 主入口 ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="fusion-mlx 全方位基准测试 v2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "model",
        nargs="?",
        default="mlx-community/Qwen3.5-9B-4bit",
        help="要测试的模型路径或 HuggingFace repo (默认: mlx-community/Qwen3.5-9B-4bit)",
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

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        for mod in ("fusion_mlx",):
            logging.getLogger(mod).setLevel(logging.DEBUG)

    # 检测 MLX
    try:
        import mlx.core as mx
        logger.info("MLX 版本: %s", mx.__version__ if hasattr(mx, "__version__") else "?")
        logger.info("MLX 设备: %s", mx.default_device())
    except ImportError:
        logger.error("MLX 未安装，无法运行基准测试")
        sys.exit(1)

    # 解析模型路径 — 如果是本地路径且存在，直接使用
    model_path = args.model
    local_path = Path(model_path)
    if not local_path.exists():
        # 尝试从 HuggingFace 缓存查找
        hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
        safe_name = model_path.replace("/", "--")
        cache_dir = hf_cache / f"models--{safe_name}"
        if cache_dir.exists():
            snapshots = list((cache_dir / "snapshots").iterdir())
            if snapshots:
                model_path = str(snapshots[0])
                logger.info("使用本地缓存路径: %s", model_path)

    # 运行基准测试
    bench = FusionMLXBenchmarkV2(model_path=model_path, quick=args.quick)
    report = asyncio.run(bench.run_all())

    # 打印报告
    print_report_table(report)

    # 保存报告
    if args.output:
        save_report(report, args.output)

    # 最终输出
    print(f"\n  基准测试完成: {len(report.results)} 项测试")
    if report.summary.get("highlights"):
        h = report.summary["highlights"]
        if "best_gen_tps" in h:
            print(f"  最佳生成速度: {h['best_gen_tps']['value']} {h['best_gen_tps']['unit']}")
        if "best_ttft" in h:
            print(f"  最低 TTFT: {h['best_ttft']['value']} {h['best_ttft']['unit']}")
        if "best_batch_tg_tps" in h:
            print(f"  最佳批处理吞吐量: {h['best_batch_tg_tps']['value']} {h['best_batch_tg_tps']['unit']} (batch={h['best_batch_tg_tps']['batch_size']})")
        if "peak_memory" in h:
            print(f"  峰值内存: {h['peak_memory']['value']}")
        if "active_memory" in h:
            print(f"  活跃内存: {h['active_memory']['value']}")


if __name__ == "__main__":
    main()