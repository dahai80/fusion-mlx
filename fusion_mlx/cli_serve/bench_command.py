# SPDX-License-Identifier: Apache-2.0
"""bench_command — fusion-mlx bench."""

import logging
import sys

from fusion_mlx._cli_base import _print_unknown_model_help

from .config_resolve import _build_benchmark_context
from .preflight import _check_disk_space, _check_memory_capacity
from .submit_flow import _run_submit_flow, _run_tier_submit_flow

logger = logging.getLogger(__name__)


def bench_command(args):
    """Run benchmark."""
    import asyncio
    import time

    # Install the MLX hardware-compat shim BEFORE `from mlx_lm import load`.
    # `mlx_lm/__init__.py` re-exports from `mlx_lm.generate`, which captures
    # `mx.new_thread_local_stream(mx.default_device())` at module-import time;
    # on M5 single-stream GPUs that stream is unusable (#404). Bench is a
    # separate entry point from `serve` so it doesn't inherit the
    # scheduler-side install — wire the shim here directly. Idempotent, no-op
    # on hardware where the original API works.
    from .. import _mlx_compat as _mlx_compat

    _mlx_compat.install()

    # --tier routes through the user-facing tier dispatcher (PR #2 of
    # the bench-consolidation series). PR #5 unified --tier with
    # --submit: when both flags are set the dispatcher runs the
    # requested smoke/harness work for the schema-v2 sub-objects and
    # ALSO runs the locked B=1 ``run_standardized_bench`` so the
    # required ``buckets`` field carries comparable numbers (the
    # lightweight tier-speed probe is NEVER submitted — its results
    # aren't apples-to-apples with the community DB).
    if getattr(args, "tier", None) and getattr(args, "submit", False):
        sys.exit(_run_tier_submit_flow(args))

    if getattr(args, "tier", None):
        from ..bench.tier_runner import TierRunnerUnavailable, run_tier

        try:
            sys.exit(
                run_tier(
                    model=args.model,
                    tier=args.tier,
                    base_url=getattr(args, "base_url", None),
                    sampled=getattr(args, "sampled", False),
                )
            )
        except TierRunnerUnavailable as e:
            print(f"\n  {e}", file=sys.stderr)
            sys.exit(2)

    # --submit routes through the standardized community-bench runner,
    # which locks the comparability knobs the freeform path exposes.
    # Keep the branch high in this function so the rest of bench_command
    # doesn't accidentally read --submit-only args.
    if getattr(args, "submit", False):
        sys.exit(_run_submit_flow(args))

    from mlx_lm import load

    from ..api.utils import is_mllm_model as _bench_is_mllm_model
    from ..engine_core import AsyncEngineCore, EngineConfig
    from ..pflash import config_from_args as _pflash_config_from_args
    from ..pflash import resolve_pflash_mode_default as _pflash_resolve_default
    from ..pflash import validate_model_support as _bench_pflash_validate
    from ..request import SamplingParams
    from ..scheduler import SchedulerConfig

    _check_disk_space(args.model, force=getattr(args, "force_disk_check", False))
    _check_memory_capacity(args.model)

    # Handle prefix cache flags
    enable_prefix_cache = args.enable_prefix_cache and not args.disable_prefix_cache

    # PFlash for the bench command — same per-alias default as serve:
    # verified Qwen3.5 / Qwen3.6 aliases switch to ``always``, everything
    # else stays ``off``. Resolves before config_from_args so the
    # validate path sees the final mode, then runs the MLLM-rejection
    # gate ``serve``/``server.py`` already enforce (codex r3 BLOCKING:
    # bench previously skipped this check, so ``fusion-mlx bench
    # --pflash always <mllm-alias>`` would admit a combo PFlash
    # explicitly rejects elsewhere).
    args.pflash = _pflash_resolve_default(args, model_name=args.model)
    try:
        bench_pflash_config = _pflash_config_from_args(args)
        _bench_pflash_validate(
            bench_pflash_config,
            model_name=args.model,
            is_mllm=_bench_is_mllm_model(args.model),
        )
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    async def run_benchmark():
        print(f"Loading model: {args.model}")
        try:
            model, tokenizer = load(args.model)
        except Exception as e:
            # Mirror serve_command: clean message instead of a 30-line
            # traceback when the user typed a missing repo / bad alias.
            from huggingface_hub.utils import RepositoryNotFoundError

            is_404 = isinstance(e, RepositoryNotFoundError) or (
                "404" in str(e) or "not found" in str(e).lower()
            )
            if is_404:
                shown = getattr(args, "_original_alias", args.model)
                print(f"\n  Error: Model '{shown}' not found on HuggingFace.")
                _print_unknown_model_help(
                    shown, full_path_example="mlx-community/Qwen3.5-9B-4bit"
                )
            else:
                print(f"\n  Error loading model: {e}")
            sys.exit(1)

        scheduler_config = SchedulerConfig(
            max_num_seqs=args.max_num_seqs,
            max_concurrent_requests=getattr(args, "max_concurrent_requests", 256),
            prefill_batch_size=args.prefill_batch_size,
            completion_batch_size=args.completion_batch_size,
            enable_prefix_cache=enable_prefix_cache,
            prefix_cache_size=args.prefix_cache_size,
            # R15-P1 (task #303): radix-tree prefix-cache index. Same
            # default as the main serve path so benches reflect the
            # production index choice.
            prefix_cache_index=getattr(args, "prefix_cache_index", "radix"),
            # Memory-aware cache options
            use_memory_aware_cache=not args.no_memory_aware_cache,
            cache_memory_mb=args.cache_memory_mb,
            cache_memory_percent=args.cache_memory_percent,
            # Paged cache options
            use_paged_cache=args.use_paged_cache,
            paged_cache_block_size=args.paged_cache_block_size,
            max_cache_blocks=args.max_cache_blocks,
            # KV cache quantization
            kv_cache_quantization=args.kv_cache_quantization,
            kv_cache_quantization_bits=args.kv_cache_quantization_bits,
            kv_cache_quantization_group_size=args.kv_cache_quantization_group_size,
            kv_cache_min_quantize_tokens=args.kv_cache_min_quantize_tokens,
            # R15-P1 (task #296): disk-backed KV checkpointing. Bench
            # path mirrors serve so a regression in the boundary trigger
            # surfaces in `fusion-mlx bench` numbers too.
            kv_disk_checkpoint_interval=getattr(
                args, "kv_disk_checkpoint_interval", 256
            ),
            # PFlash long-prompt compression (#287)
            pflash_config=bench_pflash_config,
        )
        engine_config = EngineConfig(
            model_name=args.model,
            scheduler_config=scheduler_config,
        )

        if args.use_paged_cache:
            print(
                f"Paged cache: block_size={args.paged_cache_block_size}, max_blocks={args.max_cache_blocks}"
            )

        # Generate prompts
        prompts = [
            f"Write a short poem about {topic}."
            for topic in [
                "nature",
                "love",
                "technology",
                "space",
                "music",
                "art",
                "science",
                "history",
                "food",
                "travel",
            ][: args.num_prompts]
        ]
        # Prepend a deterministic long context when the user asks for
        # one — primarily for PFlash TTFT replication runs (#287).
        long_prompt_tokens = getattr(args, "long_prompt_tokens", 0)
        long_context = _build_benchmark_context(long_prompt_tokens)
        if long_context:
            prompts = [
                f"{long_context}\n\nUser request:\n{prompt}" for prompt in prompts
            ]

        params = SamplingParams(
            max_tokens=args.max_tokens,
            temperature=0.7,
        )

        print(
            f"\nRunning benchmark with {len(prompts)} prompts, max_tokens={args.max_tokens}"
        )
        if long_prompt_tokens > 0:
            print(f"Long prompt target: ~{long_prompt_tokens} tokens")
        print("-" * 50)

        total_prompt_tokens = 0
        total_completion_tokens = 0

        async with AsyncEngineCore(model, tokenizer, engine_config) as engine:
            await asyncio.sleep(0.1)  # Warm up

            start_time = time.perf_counter()

            # Add all requests
            request_ids = []
            for prompt in prompts:
                rid = await engine.add_request(prompt, params)
                request_ids.append(rid)

            # Collect all outputs
            async def get_output(rid):
                async for out in engine.stream_outputs(rid, timeout=120):
                    if out.finished:
                        return out
                return None

            results = await asyncio.gather(*[get_output(r) for r in request_ids])

            total_time = time.perf_counter() - start_time

        # Calculate stats
        for r in results:
            if r:
                total_prompt_tokens += r.prompt_tokens
                total_completion_tokens += r.completion_tokens

        total_tokens = total_prompt_tokens + total_completion_tokens

        print("\nResults:")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Prompts: {len(prompts)}")
        print(f"  Prompts/second: {len(prompts) / total_time:.2f}")
        print(f"  Total prompt tokens: {total_prompt_tokens}")
        print(f"  Total completion tokens: {total_completion_tokens}")
        print(f"  Total tokens: {total_tokens}")
        print(f"  Tokens/second: {total_completion_tokens / total_time:.2f}")
        print(f"  Throughput: {total_tokens / total_time:.2f} tok/s")

    asyncio.run(run_benchmark())


if __name__ == "__main__":
    # Delegate to the canonical argparse dispatcher in fusion_mlx.cli so
    # `python -m fusion_mlx.cli_serve <subcommand>` works - this module only
    # hosts the serve/bench handlers, not the parser. One parser, not two.
    from fusion_mlx.cli import main

    main()
