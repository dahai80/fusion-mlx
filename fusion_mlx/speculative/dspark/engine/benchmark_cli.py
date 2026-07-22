#!/usr/bin/env python3
"""DSpark benchmark suite (U7): DSpark vs vanilla mlx-lm decode throughput.

Two run modes:

- ``dspark``  — the DSpark loop (``dspark_metal.runtime.dspark_generate_stream``)
  with temperature / confidence threshold / verify mode / seed configurable.
- ``baseline`` — vanilla mlx-lm ``stream_generate`` with NO draft model, the
  plan's R5 baseline, same prompts / dtype / sampler settings.

Measurement rigor:

- a warmup generation (default 128 tokens, >= 64 enforced unless disabled)
  runs before the measured generation and is excluded from all stats;
- ``mlx_lm.generate.wired_limit`` wraps the dspark generation (baseline
  ``stream_generate`` applies it internally);
- ``mx.synchronize()`` brackets every window boundary and the run start/end,
  so window wall times only cover fully-retired GPU work;
- tokens/sec is sampled per generated-token window (default 1024) to build
  the throughput-vs-length curve; dspark windows also record tau (mean
  committed tokens per verify round) and peak memory (``mx.get_peak_memory``,
  reset at measurement start);
- run records are appended as JSONL under ``benchmarks/raw/`` (gitignored)
  with the full config, machine info, package versions, and git state;
- the ``report`` subcommand renders markdown tables (per-window curves,
  dspark/baseline cumulative ratios at checkpoints) from those JSONL files.

Timing convention (identical in both modes): the first generated token is
produced by the prompt prefill forward, so the window clock starts when that
first token arrives and the curve measures steady-state decode from token 2
onward. Long-curve runs use ``--ignore-eos`` (EOS is generated and committed
but never treated as a stop), the standard practice for length-controlled
throughput benchmarks.

Position-range guard: runs whose prompt + max-new-tokens exceed
min(32768, target ``max_position_embeddings``) are refused unless
``--allow-beyond-rope`` is passed (used for the exploratory YaRN run together
with ``--rope-scaling-json``).
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .history import prompt_sha256, run_metadata

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = REPO_ROOT / "benchmarks" / "raw"
DEFAULT_OUT = DEFAULT_RAW_DIR / "runs.jsonl"
DEFAULT_DRAFT_MODEL = "models/dspark_qwen3_4b_block7-mlx"

DEFAULT_WINDOW_SIZE = 1024
DEFAULT_WARMUP_TOKENS = 128
MIN_WARMUP_TOKENS = 64
# MLX's buffer cache accumulates freed buffers keyed by size; a decode loop
# whose attention buffers grow by a few tokens per round allocates a *new*
# size every round, so the cache grows quadratically with generation length
# (~11 GB by 5.6k tokens on Qwen3-8B, measured at both temp 0 and temp 1) and
# long runs can die with a Metal "Insufficient Memory" command-buffer error
# once eviction loses to the wired working set. Bounding the cache keeps 32k+
# runs safe; it applies identically to both modes (0 disables the cap).
DEFAULT_CACHE_LIMIT_GB = 8.0
ROPE_SAFE_TOTAL_TOKENS = 32768
DEFAULT_CHECKPOINTS = (1024, 2048, 4096, 8192, 16384, 32768)

WARMUP_PROMPT = (
    "Explain, step by step and in detail, how a refrigerator keeps food cold."
)

# Fixed long-generation prompt set. The long-curve protocol needs prompts that
# elicit long continuations; combined with --ignore-eos they reach 32k tokens.
# "mtbench-travel" is row 0 of refs/deepspec/eval_datasets/mt-bench.jsonl
# (first turn), inlined here so the benchmark CLI has no dependency on the
# gitignored refs/ checkout.
BENCH_PROMPTS: dict[str, str] = {
    "book-systems": (
        "Write a very long, extremely detailed technical book chapter about the "
        "design and implementation of a distributed key-value store. Cover, in "
        "order and in depth: consistent hashing and virtual nodes, replication "
        "and quorum reads/writes, hinted handoff, anti-entropy repair with "
        "Merkle trees, gossip-based membership and failure detection, "
        "log-structured storage engines and LSM-tree compaction strategies, "
        "write-ahead logging and crash recovery, snapshot and backup design, "
        "multi-datacenter operation, and operational war stories with "
        "post-mortems. Include code sketches, concrete numbers, and worked "
        "examples for every section. Do not summarize; keep expanding each "
        "topic with subsections until every topic is covered in full depth."
    ),
    "book-compilers": (
        "Write a very long, exhaustive technical book chapter that teaches how "
        "to build an optimizing compiler for a small systems language. Walk "
        "through lexing, recursive-descent parsing, AST design, name "
        "resolution, a Hindley-Milner-style type checker, lowering to a typed "
        "IR, SSA construction, classic optimizations (constant propagation, "
        "dead code elimination, common subexpression elimination, loop "
        "invariant code motion, inlining heuristics), register allocation via "
        "graph coloring, instruction selection, and runtime/ABI concerns. Give "
        "concrete data structure definitions, pseudocode for every algorithm, "
        "worked examples on a running sample program, and discussions of "
        "engineering trade-offs. Do not stop early; expand every phase into "
        "detailed subsections."
    ),
    "mtbench-travel": (
        "Compose an engaging travel blog post about a recent trip to Hawaii, "
        "highlighting cultural experiences and must-see attractions."
    ),
}


# ---------------------------------------------------------------------------
# Window bookkeeping (pure Python; unit-tested without models)
# ---------------------------------------------------------------------------


@dataclass
class Window:
    """One measurement window over the generation stream.

    ``start_token``/``end_token`` are cumulative *measured* generated-token
    counts (the prefill-produced first token is excluded by the runners).
    ``rounds`` counts dspark verify rounds inside the window (0 for baseline,
    where ``tau`` is None).
    """

    index: int
    start_token: int
    end_token: int
    tokens: int
    wall_s: float
    tps: float
    rounds: int
    tau: float | None
    peak_memory_gb: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "start_token": self.start_token,
            "end_token": self.end_token,
            "tokens": self.tokens,
            "wall_s": self.wall_s,
            "tps": self.tps,
            "rounds": self.rounds,
            "tau": self.tau,
            "peak_memory_gb": self.peak_memory_gb,
        }


class WindowTracker:
    """Accumulates per-window stats over a generation stream.

    Protocol: call ``begin(t)`` once the first (prefill) token has arrived,
    then ``add(tokens, rounds)`` per stream event. When ``add`` returns True a
    window boundary was reached: the caller synchronizes the device and calls
    ``mark(now, peak)``. ``finalize(now, peak)`` closes any partial window.

    Boundaries are anchored at absolute multiples of ``window_size``. A dspark
    round commits up to block_size tokens, so a window may overshoot its
    boundary by a few tokens; ``end_token`` records the actual count and
    ``tps`` always uses actual tokens / actual wall time.
    """

    def __init__(self, window_size: int):
        if window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}")
        self.window_size = int(window_size)
        self.windows: list[Window] = []
        self._tokens = 0
        self._rounds = 0
        self._window_start_tokens = 0
        self._window_start_rounds = 0
        self._t0: float | None = None
        self._window_start_t: float | None = None
        self._next_boundary = self.window_size

    @property
    def started(self) -> bool:
        return self._t0 is not None

    @property
    def total_tokens(self) -> int:
        return self._tokens

    @property
    def total_rounds(self) -> int:
        return self._rounds

    def begin(self, now: float) -> None:
        if self.started:
            raise RuntimeError("WindowTracker.begin called twice")
        self._t0 = now
        self._window_start_t = now

    def add(self, tokens: int, rounds: int = 0) -> bool:
        if not self.started:
            raise RuntimeError("WindowTracker.add before begin")
        self._tokens += int(tokens)
        self._rounds += int(rounds)
        return self._tokens >= self._next_boundary

    def _close_window(self, now: float, peak_memory_gb: float | None) -> Window:
        tokens = self._tokens - self._window_start_tokens
        rounds = self._rounds - self._window_start_rounds
        wall_s = now - self._window_start_t
        window = Window(
            index=len(self.windows),
            start_token=self._window_start_tokens,
            end_token=self._tokens,
            tokens=tokens,
            wall_s=wall_s,
            tps=tokens / max(wall_s, 1e-9),
            rounds=rounds,
            tau=(tokens / rounds) if rounds > 0 else None,
            peak_memory_gb=peak_memory_gb,
        )
        self.windows.append(window)
        self._window_start_tokens = self._tokens
        self._window_start_rounds = self._rounds
        self._window_start_t = now
        self._next_boundary = (self._tokens // self.window_size + 1) * self.window_size
        return window

    def mark(self, now: float, peak_memory_gb: float | None = None) -> Window:
        if not self.started:
            raise RuntimeError("WindowTracker.mark before begin")
        return self._close_window(now, peak_memory_gb)

    def finalize(
        self, now: float, peak_memory_gb: float | None = None
    ) -> Window | None:
        if not self.started:
            raise RuntimeError("WindowTracker.finalize before begin")
        if self._tokens == self._window_start_tokens:
            return None
        return self._close_window(now, peak_memory_gb)

    @property
    def measured_wall_s(self) -> float:
        if not self.windows:
            return 0.0
        return sum(window.wall_s for window in self.windows)

    @property
    def cumulative_tps(self) -> float:
        return self.total_tokens / max(self.measured_wall_s, 1e-9)


def cumulative_stats_at(
    windows: list[dict[str, Any]], checkpoint_tokens: int
) -> dict[str, float] | None:
    """Cumulative tokens/sec (and tau) over the windows up to ``checkpoint``.

    Uses every window up to and including the first whose ``end_token``
    reaches the checkpoint; dspark windows overshoot boundaries by at most a
    block, so the cumulative token count matches the checkpoint to within a
    few tokens and the tps is computed from actual tokens / actual time.
    Returns None when the run never reached the checkpoint.
    """
    tokens = 0
    rounds = 0
    wall_s = 0.0
    for window in windows:
        tokens = window["end_token"]
        rounds += window["rounds"]
        wall_s += window["wall_s"]
        if window["end_token"] >= checkpoint_tokens:
            return {
                "tokens": float(tokens),
                "wall_s": wall_s,
                "tps": tokens / max(wall_s, 1e-9),
                "tau": (tokens / rounds) if rounds > 0 else float("nan"),
            }
    return None


# ---------------------------------------------------------------------------
# Machine / environment metadata
# ---------------------------------------------------------------------------


def _sysctl(name: str) -> str:
    try:
        return subprocess.check_output(
            ["sysctl", "-n", name], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def machine_info() -> dict[str, Any]:
    import mlx.core as mx
    import mlx_lm

    info: dict[str, Any] = {
        "platform": platform.platform(),
        "chip": _sysctl("machdep.cpu.brand_string"),
        "memory_gb": None,
        "macos": platform.mac_ver()[0],
        "mlx_version": mx.__version__,
        "mlx_lm_version": mlx_lm.__version__,
    }
    memsize = _sysctl("hw.memsize")
    if memsize.isdigit():
        info["memory_gb"] = round(int(memsize) / 2**30, 1)
    try:
        info["device"] = dict(mx.device_info())
    except Exception:
        pass
    return info


# ---------------------------------------------------------------------------
# Guards and shared helpers
# ---------------------------------------------------------------------------


def target_max_position_embeddings(target: str) -> int | None:
    from .adapters import resolve_model_path

    config_path = resolve_model_path(target) / "config.json"
    if not config_path.exists():
        return None
    config = json.loads(config_path.read_text())
    value = config.get("max_position_embeddings")
    return int(value) if value is not None else None


def enforce_rope_guard(
    total_tokens: int,
    max_position_embeddings: int | None,
    allow_beyond_rope: bool,
) -> None:
    """Refuse runs beyond the validated position range unless overridden."""
    limit = ROPE_SAFE_TOTAL_TOKENS
    if max_position_embeddings is not None:
        limit = min(limit, max_position_embeddings)
    if total_tokens > limit and not allow_beyond_rope:
        raise SystemExit(
            f"Refusing to run {total_tokens} total tokens: beyond the "
            f"validated position range (limit {limit}; target "
            f"max_position_embeddings={max_position_embeddings}, safe cap "
            f"{ROPE_SAFE_TOTAL_TOKENS}). Pass --allow-beyond-rope (with a "
            "--rope-scaling-json YaRN override) for exploratory runs."
        )


def build_chat_prompt_tokens(tokenizer, prompt_text: str) -> list[int]:
    """Chat-template a user prompt exactly like Qwen3TargetAdapter.build_prompt."""
    messages = [{"role": "user", "content": prompt_text}]
    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return tokenizer.encode(prompt, add_special_tokens=False)


def distinct_ngram_ratio(tokens: list[int], n: int = 4) -> float:
    """Repetition sanity metric: fraction of distinct n-grams (1.0 = no reuse)."""
    if len(tokens) < n:
        return 1.0
    grams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    return len(set(grams)) / len(grams)


def resolve_prompt(args: argparse.Namespace) -> tuple[str, str]:
    if args.prompt is not None:
        return "custom", args.prompt
    if args.prompt_id not in BENCH_PROMPTS:
        raise SystemExit(
            f"Unknown --prompt-id {args.prompt_id!r}; available: "
            f"{sorted(BENCH_PROMPTS)}"
        )
    return args.prompt_id, BENCH_PROMPTS[args.prompt_id]


def parse_rope_scaling(raw: str | None) -> dict[str, Any] | None:
    if raw is None:
        return None
    override = json.loads(raw)
    if not isinstance(override, dict):
        raise SystemExit("--rope-scaling-json must be a JSON object")
    return override


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(record) + "\n")


def _text_excerpt(text: str, chars: int = 400) -> dict[str, str]:
    return {"head": text[:chars], "tail": text[-chars:] if len(text) > chars else ""}


def apply_cache_limit(cache_limit_gb: float) -> None:
    """Bound the MLX buffer cache (see DEFAULT_CACHE_LIMIT_GB rationale)."""
    if cache_limit_gb and cache_limit_gb > 0:
        import mlx.core as mx

        mx.set_cache_limit(int(cache_limit_gb * 1e9))


# ---------------------------------------------------------------------------
# dspark runner
# ---------------------------------------------------------------------------


def run_dspark_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    import mlx.core as mx
    from mlx_lm.generate import wired_limit

    from .api import DSparkGenerator
    from .runtime import dspark_generate, dspark_generate_stream

    prompt_id, prompt_text = resolve_prompt(args)
    rope_scaling = parse_rope_scaling(args.rope_scaling_json)
    model_config: dict[str, Any] | None = None
    if rope_scaling is not None:
        model_config = {"rope_scaling": rope_scaling}

    apply_cache_limit(args.cache_limit_gb)
    print(f"[load] target={args.target} draft={args.draft}", flush=True)
    generator = DSparkGenerator(
        target_model=args.target,
        draft_model=args.draft,
        draft_quant_bits=args.draft_quant_bits,
        draft_quant_group_size=args.draft_quant_group_size,
        draft_reuse_target_embeddings=args.draft_reuse_target_embeddings,
        seed=args.seed,
        target_model_config=model_config,
    )
    if generator.draft_quantization is not None:
        print(f"[quant] draft quantization: {generator.draft_quantization}", flush=True)
    prompt_tokens = generator.encode_prompt(prompt_text)
    prompt_len = int(prompt_tokens.shape[0])
    enforce_rope_guard(
        prompt_len + args.max_new_tokens,
        target_max_position_embeddings(args.target),
        args.allow_beyond_rope,
    )
    stop_ids = set() if args.ignore_eos else generator.target.stop_token_ids()
    layer_ids = generator.draft.target_layer_ids
    sts_temperatures = generator.draft.sts_temperatures

    # --- warmup (excluded from all stats) ---------------------------------
    warmup_info: dict[str, Any] = {"tokens": 0, "wall_s": 0.0}
    if args.warmup_tokens > 0:
        warmup_start = time.perf_counter()
        _, warmup_metrics = dspark_generate(
            target=generator.target,
            draft=generator.draft,
            prompt_tokens=generator.encode_prompt(WARMUP_PROMPT),
            max_new_tokens=args.warmup_tokens,
            temperature=args.temperature,
            stop_token_ids=set(),  # warmup always runs its full length
            layer_ids=layer_ids,
            confidence_threshold=args.threshold,
            verify_mode=args.verify_mode,
            verify_chunk_size=args.verify_chunk_size,
            seed=args.seed,
        )
        warmup_info = {
            "tokens": warmup_metrics["num_output_tokens"],
            "wall_s": time.perf_counter() - warmup_start,
        }
        print(
            f"[warmup] {warmup_info['tokens']} tokens in "
            f"{warmup_info['wall_s']:.1f}s",
            flush=True,
        )
        mx.clear_cache()

    # --- measured generation ----------------------------------------------
    tracker = WindowTracker(args.window_size)
    mx.synchronize()
    mx.reset_peak_memory()
    final_metrics: dict[str, Any] | None = None
    output_tokens: list[int] = []
    run_start = time.perf_counter()
    with wired_limit(generator.target.model):
        stream = dspark_generate_stream(
            target=generator.target,
            draft=generator.draft,
            prompt_tokens=prompt_tokens,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            stop_token_ids=stop_ids,
            layer_ids=layer_ids,
            confidence_threshold=args.threshold,
            verify_mode=args.verify_mode,
            verify_chunk_size=args.verify_chunk_size,
            seed=args.seed,
            profile=True,
        )
        for event in stream:
            if event.finished:
                final_metrics = event.metrics
                output_tokens = event.output_tokens
                break
            if not tracker.started:
                # First event = the prefill-sampled first token; start the
                # steady-state decode clock here (token excluded from windows).
                mx.synchronize()
                tracker.begin(time.perf_counter())
                continue
            if tracker.add(len(event.token_ids), rounds=1):
                mx.synchronize()
                tracker.mark(
                    time.perf_counter(),
                    peak_memory_gb=mx.get_peak_memory() / 1e9,
                )
        mx.synchronize()
        end_time = time.perf_counter()
        if tracker.started:
            tracker.finalize(end_time, peak_memory_gb=mx.get_peak_memory() / 1e9)

    if final_metrics is None:
        raise RuntimeError("dspark stream produced no final metrics event")

    generated = output_tokens[prompt_len:]
    text = generator.target.tokenizer.decode(generated)
    rounds = len(final_metrics["acceptance_lengths"])
    summary: dict[str, Any] = {
        "generated_tokens": final_metrics["num_output_tokens"],
        "measured_tokens": tracker.total_tokens,
        "measured_wall_s": tracker.measured_wall_s,
        "cumulative_tps": tracker.cumulative_tps,
        "prefill_s": final_metrics["prefill_time_s"],
        "total_wall_s": end_time - run_start,
        "tau_overall": final_metrics["avg_acceptance_length"],
        "rounds": rounds,
        "mean_proposal_len": (
            sum(final_metrics["proposal_lengths"]) / rounds if rounds else None
        ),
        "peak_memory_gb": mx.get_peak_memory() / 1e9,
        "finish_reason": (
            "length"
            if final_metrics["num_output_tokens"] >= args.max_new_tokens
            else "stop"
        ),
        "distinct_4gram_ratio": distinct_ngram_ratio(generated),
        "profile": final_metrics.get("profile"),
        "sts_temperatures_applied": sts_temperatures is not None,
        "text_excerpt": _text_excerpt(text),
    }
    return build_run_record(
        args,
        "dspark",
        prompt_id,
        prompt_text,
        prompt_len,
        tracker,
        summary,
        warmup_info,
    )


# ---------------------------------------------------------------------------
# baseline runner (vanilla mlx-lm, no draft)
# ---------------------------------------------------------------------------


def run_baseline_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.generate import stream_generate
    from mlx_lm.sample_utils import make_sampler

    prompt_id, prompt_text = resolve_prompt(args)
    rope_scaling = parse_rope_scaling(args.rope_scaling_json)
    model_config = {"rope_scaling": rope_scaling} if rope_scaling is not None else None

    apply_cache_limit(args.cache_limit_gb)
    print(f"[load] baseline target={args.target}", flush=True)
    model, tokenizer = load(args.target, model_config=model_config)
    prompt_tokens = build_chat_prompt_tokens(tokenizer, prompt_text)
    prompt_len = len(prompt_tokens)
    enforce_rope_guard(
        prompt_len + args.max_new_tokens,
        target_max_position_embeddings(args.target),
        args.allow_beyond_rope,
    )
    if args.ignore_eos:
        # TokenizerWrapper keeps the stop set on _eos_token_ids; emptying it
        # makes stream_generate run to max_tokens (EOS still gets committed).
        tokenizer._eos_token_ids = set()

    def make_run_sampler():
        mx.random.seed(args.seed)
        return make_sampler(temp=args.temperature)

    # --- warmup (excluded from all stats) ---------------------------------
    warmup_info: dict[str, Any] = {"tokens": 0, "wall_s": 0.0}
    if args.warmup_tokens > 0:
        warmup_start = time.perf_counter()
        warmup_count = 0
        warmup_prompt = build_chat_prompt_tokens(tokenizer, WARMUP_PROMPT)
        for response in stream_generate(
            model,
            tokenizer,
            warmup_prompt,
            max_tokens=args.warmup_tokens,
            sampler=make_run_sampler(),
        ):
            warmup_count = response.generation_tokens
        warmup_info = {
            "tokens": warmup_count,
            "wall_s": time.perf_counter() - warmup_start,
        }
        print(
            f"[warmup] {warmup_info['tokens']} tokens in "
            f"{warmup_info['wall_s']:.1f}s",
            flush=True,
        )
        mx.clear_cache()

    # --- measured generation ----------------------------------------------
    tracker = WindowTracker(args.window_size)
    mx.synchronize()
    mx.reset_peak_memory()
    last_count = 0
    prefill_s: float | None = None
    finish_reason = "length"
    token_ids: list[int] = []
    text_chunks: list[str] = []
    run_start = time.perf_counter()
    # stream_generate applies wired_limit(model, [generation_stream]) itself.
    for response in stream_generate(
        model,
        tokenizer,
        prompt_tokens,
        max_tokens=args.max_new_tokens,
        sampler=make_run_sampler(),
    ):
        text_chunks.append(response.text)
        if response.finish_reason is not None:
            finish_reason = response.finish_reason
        delta = response.generation_tokens - last_count
        if delta <= 0:
            continue
        last_count = response.generation_tokens
        token_ids.append(int(response.token))
        if not tracker.started:
            # First yielded token comes out of the prefill forward: start the
            # steady-state decode clock here (token excluded from windows).
            prefill_s = response.prompt_tokens / max(response.prompt_tps, 1e-9)
            mx.synchronize()
            tracker.begin(time.perf_counter())
            continue
        if tracker.add(delta, rounds=0):
            mx.synchronize()
            tracker.mark(
                time.perf_counter(),
                peak_memory_gb=mx.get_peak_memory() / 1e9,
            )
    mx.synchronize()
    end_time = time.perf_counter()
    if tracker.started:
        tracker.finalize(end_time, peak_memory_gb=mx.get_peak_memory() / 1e9)

    summary: dict[str, Any] = {
        "generated_tokens": last_count,
        "measured_tokens": tracker.total_tokens,
        "measured_wall_s": tracker.measured_wall_s,
        "cumulative_tps": tracker.cumulative_tps,
        "prefill_s": prefill_s,
        "total_wall_s": end_time - run_start,
        "tau_overall": None,
        "rounds": 0,
        "peak_memory_gb": mx.get_peak_memory() / 1e9,
        "finish_reason": finish_reason,
        "distinct_4gram_ratio": distinct_ngram_ratio(token_ids),
        "text_excerpt": _text_excerpt("".join(text_chunks)),
    }
    return build_run_record(
        args,
        "baseline",
        prompt_id,
        prompt_text,
        prompt_len,
        tracker,
        summary,
        warmup_info,
    )


# ---------------------------------------------------------------------------
# Run record assembly
# ---------------------------------------------------------------------------


def build_run_record(
    args: argparse.Namespace,
    mode: str,
    prompt_id: str,
    prompt_text: str,
    prompt_len: int,
    tracker: WindowTracker,
    summary: dict[str, Any],
    warmup_info: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "benchmark_run",
        "mode": mode,
        "tag": args.tag,
        **run_metadata("dspark-metal-bench", experiment_tag=args.tag),
        "machine": machine_info(),
        "config": {
            "target": args.target,
            "draft": args.draft if mode == "dspark" else None,
            "prompt_id": prompt_id,
            "prompt_sha256": prompt_sha256(prompt_text),
            "prompt_tokens": prompt_len,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "confidence_threshold": args.threshold if mode == "dspark" else None,
            "verify_mode": args.verify_mode if mode == "dspark" else None,
            "verify_chunk_size": args.verify_chunk_size if mode == "dspark" else None,
            "draft_quant_bits": (args.draft_quant_bits if mode == "dspark" else None),
            "draft_quant_group_size": (
                args.draft_quant_group_size if mode == "dspark" else None
            ),
            "draft_reuse_target_embeddings": (
                args.draft_reuse_target_embeddings if mode == "dspark" else None
            ),
            "seed": args.seed,
            "ignore_eos": args.ignore_eos,
            "window_size": args.window_size,
            "warmup_tokens": args.warmup_tokens,
            "rope_scaling": parse_rope_scaling(args.rope_scaling_json),
            "allow_beyond_rope": args.allow_beyond_rope,
            "cache_limit_gb": getattr(args, "cache_limit_gb", None),
        },
        "warmup": warmup_info,
        "windows": [window.to_dict() for window in tracker.windows],
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Markdown report emitters
# ---------------------------------------------------------------------------


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        if value != value:  # NaN
            return "—"
        return f"{value:.{digits}f}"
    return str(value)


def run_label(run: dict[str, Any]) -> str:
    config = run["config"]
    parts = [
        run["mode"],
        Path(str(config["target"])).name,
        config["prompt_id"],
        f"temp{config['temperature']:g}",
    ]
    if run["mode"] == "dspark":
        parts.append(f"thr{config['confidence_threshold']:g}")
        if config.get("draft_quant_bits"):
            parts.append(f"draft-q{config['draft_quant_bits']}")
        if config.get("draft_reuse_target_embeddings"):
            parts.append("reuse-embed")
    if run.get("tag"):
        parts.append(run["tag"])
    return " ".join(parts)


def format_runs_summary_table(runs: list[dict[str, Any]]) -> str:
    lines = [
        "| mode | target | prompt | temp | thr | tokens | tok/s | tau | peak GB | 4-gram | finish |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for run in runs:
        config = run["config"]
        summary = run["summary"]
        lines.append(
            "| {mode} | {target} | {prompt} | {temp:g} | {thr} | {tokens} "
            "| {tps} | {tau} | {peak} | {ngram} | {finish} |".format(
                mode=run["mode"],
                target=Path(str(config["target"])).name,
                prompt=config["prompt_id"],
                temp=config["temperature"],
                thr=_fmt(config["confidence_threshold"], 2),
                tokens=summary["measured_tokens"],
                tps=_fmt(summary["cumulative_tps"], 2),
                tau=_fmt(summary["tau_overall"], 3),
                peak=_fmt(summary["peak_memory_gb"], 2),
                ngram=_fmt(summary.get("distinct_4gram_ratio"), 3),
                finish=summary["finish_reason"],
            )
        )
    return "\n".join(lines)


def format_pair_curve_table(
    dspark_run: dict[str, Any], baseline_run: dict[str, Any]
) -> str:
    """Per-window curve table for a matched dspark/baseline pair."""
    dspark_windows = dspark_run["windows"]
    baseline_windows = baseline_run["windows"]
    lines = [
        "| window (tokens) | baseline tok/s | dspark tok/s | ratio | tau | dspark peak GB |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for idx in range(max(len(dspark_windows), len(baseline_windows))):
        dspark_window = dspark_windows[idx] if idx < len(dspark_windows) else None
        baseline_window = baseline_windows[idx] if idx < len(baseline_windows) else None
        anchor = dspark_window or baseline_window
        label = f"{anchor['start_token']}–{anchor['end_token']}"
        ratio = None
        if dspark_window and baseline_window:
            ratio = dspark_window["tps"] / max(baseline_window["tps"], 1e-9)
        lines.append(
            f"| {label} "
            f"| {_fmt(baseline_window['tps'] if baseline_window else None)} "
            f"| {_fmt(dspark_window['tps'] if dspark_window else None)} "
            f"| {_fmt(ratio)} "
            f"| {_fmt(dspark_window['tau'] if dspark_window else None, 3)} "
            f"| {_fmt(dspark_window['peak_memory_gb'] if dspark_window else None)} |"
        )
    return "\n".join(lines)


def format_pair_cumulative_table(
    dspark_run: dict[str, Any],
    baseline_run: dict[str, Any],
    checkpoints: tuple[int, ...] = DEFAULT_CHECKPOINTS,
) -> str:
    """Cumulative tok/s + ratio at token checkpoints (the acceptance metric).

    The cumulative ratio at checkpoint C is (dspark tokens / dspark wall time)
    divided by (baseline tokens / baseline wall time), each accumulated over
    all measurement windows up to C. A final "full" row accumulates every
    window of each run (the headline number for ~32k runs, whose generated
    count sits just under 32768 because of the total-position guard).
    """

    def full_stats(windows: list[dict[str, Any]]) -> dict[str, float] | None:
        if not windows:
            return None
        return cumulative_stats_at(windows, windows[-1]["end_token"])

    lines = [
        "| tokens | baseline cum tok/s | dspark cum tok/s | cumulative ratio | dspark cum tau |",
        "|---:|---:|---:|---:|---:|",
    ]
    rows: list[tuple[str, dict | None, dict | None]] = []
    for checkpoint in checkpoints:
        rows.append(
            (
                str(checkpoint),
                cumulative_stats_at(dspark_run["windows"], checkpoint),
                cumulative_stats_at(baseline_run["windows"], checkpoint),
            )
        )
    dspark_full = full_stats(dspark_run["windows"])
    baseline_full = full_stats(baseline_run["windows"])
    full_label = "full"
    if dspark_full and baseline_full:
        full_label = (
            f"full ({int(dspark_full['tokens'])}/{int(baseline_full['tokens'])})"
        )
    rows.append((full_label, dspark_full, baseline_full))
    for label, dspark_stats, baseline_stats in rows:
        ratio = None
        if dspark_stats and baseline_stats:
            ratio = dspark_stats["tps"] / max(baseline_stats["tps"], 1e-9)
        lines.append(
            f"| {label} "
            f"| {_fmt(baseline_stats['tps'] if baseline_stats else None)} "
            f"| {_fmt(dspark_stats['tps'] if dspark_stats else None)} "
            f"| {_fmt(ratio)} "
            f"| {_fmt(dspark_stats['tau'] if dspark_stats else None, 3)} |"
        )
    return "\n".join(lines)


def pair_key(run: dict[str, Any]) -> tuple:
    config = run["config"]
    return (
        config["target"],
        config["prompt_id"],
        config["temperature"],
        config["ignore_eos"],
        config["max_new_tokens"],
    )


def load_runs(paths: list[Path]) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for path in paths:
        with Path(path).open() as handle:
            for line in handle:
                line = line.strip()
                if line:
                    runs.append(json.loads(line))
    return runs


def report_main(args: argparse.Namespace) -> None:
    runs = load_runs(args.runs)
    if args.tag:
        runs = [run for run in runs if run.get("tag") == args.tag]
    if not runs:
        raise SystemExit("No runs matched.")
    checkpoints = tuple(int(c) for c in args.checkpoints.split(","))

    print("## All runs\n")
    print(format_runs_summary_table(runs))

    groups: dict[tuple, dict[str, list[dict[str, Any]]]] = {}
    for run in runs:
        group = groups.setdefault(pair_key(run), {"dspark": [], "baseline": []})
        group[run["mode"]].append(run)
    for key, group in groups.items():
        for dspark_run in group["dspark"]:
            for baseline_run in group["baseline"]:
                print(f"\n## {run_label(dspark_run)}  vs  {run_label(baseline_run)}\n")
                print(
                    format_pair_cumulative_table(dspark_run, baseline_run, checkpoints)
                )
                print()
                print(format_pair_curve_table(dspark_run, baseline_run))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dspark-metal-bench",
        description="DSpark vs vanilla mlx-lm decode benchmark suite.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="Run one benchmark generation.")
    run_parser.add_argument("--mode", choices=("dspark", "baseline"), required=True)
    run_parser.add_argument("--target", required=True, help="Target model repo/path.")
    run_parser.add_argument(
        "--draft",
        default=DEFAULT_DRAFT_MODEL,
        help="Converted DSpark draft dir (dspark mode only).",
    )
    run_parser.add_argument(
        "--prompt-id",
        default="book-systems",
        help=f"Fixed benchmark prompt id: {sorted(BENCH_PROMPTS)}.",
    )
    run_parser.add_argument("--prompt", default=None, help="Custom prompt text.")
    run_parser.add_argument("--max-new-tokens", type=int, default=2048)
    run_parser.add_argument("--temperature", type=float, default=0.0)
    run_parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Confidence threshold (dspark mode).",
    )
    run_parser.add_argument(
        "--verify-mode", choices=("full", "lazy-logits"), default="full"
    )
    run_parser.add_argument(
        "--draft-quant-bits",
        type=int,
        choices=(4, 8),
        default=None,
        help="Quantize the draft at load time to this many bits (dspark mode).",
    )
    run_parser.add_argument(
        "--draft-quant-group-size",
        type=int,
        default=64,
        help="Quantization group size for --draft-quant-bits.",
    )
    run_parser.add_argument(
        "--draft-reuse-target-embeddings",
        action="store_true",
        help="Rebind draft embed_tokens/lm_head to the target's bf16 tensors "
        "(audit-verified identical); with --draft-quant-bits, quantizes only "
        "backbone+heads and keeps the shared embeddings bf16.",
    )
    run_parser.add_argument("--verify-chunk-size", type=int, default=4)
    run_parser.add_argument("--seed", type=int, default=0)
    run_parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE)
    run_parser.add_argument(
        "--warmup-tokens",
        type=int,
        default=DEFAULT_WARMUP_TOKENS,
        help=f"Warmup generation length (>= {MIN_WARMUP_TOKENS}; 0 disables).",
    )
    run_parser.add_argument(
        "--ignore-eos",
        action="store_true",
        help="Never treat EOS as a stop (long-curve runs).",
    )
    run_parser.add_argument(
        "--allow-beyond-rope",
        action="store_true",
        help="Override the 32k/max_position_embeddings guard (exploratory).",
    )
    run_parser.add_argument(
        "--rope-scaling-json",
        default=None,
        help='Target rope_scaling override, e.g. \'{"rope_type":"yarn",...}\'.',
    )
    run_parser.add_argument(
        "--cache-limit-gb",
        type=float,
        default=DEFAULT_CACHE_LIMIT_GB,
        help="MLX buffer-cache cap in GB (0 disables; see source rationale).",
    )
    run_parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    run_parser.add_argument("--tag", default="")

    report_parser = sub.add_parser(
        "report", help="Render markdown tables from run JSONL files."
    )
    report_parser.add_argument("--runs", type=Path, nargs="+", required=True)
    report_parser.add_argument(
        "--checkpoints",
        default=",".join(str(c) for c in DEFAULT_CHECKPOINTS),
    )
    report_parser.add_argument("--tag", default=None, help="Only runs with this tag.")
    return parser


def run_main(args: argparse.Namespace) -> dict[str, Any]:
    if 0 < args.warmup_tokens < MIN_WARMUP_TOKENS:
        raise SystemExit(
            f"--warmup-tokens must be 0 or >= {MIN_WARMUP_TOKENS} "
            f"(got {args.warmup_tokens})"
        )
    if args.mode == "dspark":
        record = run_dspark_benchmark(args)
    else:
        record = run_baseline_benchmark(args)
    append_jsonl(args.out, record)
    summary = record["summary"]
    tau = summary.get("tau_overall")
    print(
        f"[done] mode={record['mode']} prompt={record['config']['prompt_id']} "
        f"tokens={summary['measured_tokens']} "
        f"tok/s={summary['cumulative_tps']:.2f} "
        f"tau={tau if tau is None else f'{tau:.3f}'} "
        f"peak={summary['peak_memory_gb']:.2f}GB "
        f"finish={summary['finish_reason']} -> {args.out}",
        flush=True,
    )
    return record


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        run_main(args)
    else:
        report_main(args)


if __name__ == "__main__":
    main()
