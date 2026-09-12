# SPDX-License-Identifier: Apache-2.0
"""Benchmark tier/standard submit flows."""

import logging
import os
import sys

from .preflight import _check_disk_space, _check_memory_capacity

logger = logging.getLogger(__name__)


def _run_tier_submit_flow(args) -> int:
    """``fusion-mlx bench <model> --tier <T> --submit`` — PR #5 unification.

    Three-phase pipeline:

    1. Run the requested tier's smoke / harness work through the
       existing HTTP-server-backed dispatcher (``run_tier`` with
       ``return_results=True``). For ``tier='all'`` we pass
       ``skip_speed=True`` because phase 2 will produce the comparable
       speed numbers directly from the engine; running the lightweight
       HTTP-speed probe too would just double-cost the bench AND
       produce a second set of non-comparable numbers next to it.
       For ``tier='speed'`` phase 1 is a no-op — straight to phase 2.
    2. Run the locked B=1 ``run_standardized_bench`` against the same
       model so the schema-required ``buckets`` field carries the
       comparable numbers the community-benchmarks corpus expects.
       This phase IS what plain ``--submit`` (no ``--tier``) has
       always done; the tier kwargs just decorate the payload.
    3. Build the schema-v2 payload and run the standard interactive
       submit flow (consent → write → commit → push → gh pr create).

    Tier-failure handling: if phase 1's smoke probe FAILS, abort
    before phase 2 — there's no point benching a model that can't
    answer "what is 2+2?". A phase 1 harness failure does NOT abort:
    submitting a failure row IS the point of the harness tier (the
    aggregator wants visibility into "this combo doesn't pass the
    gauntlet"), so we proceed and let the payload carry the per-
    adapter failure flags.
    """
    tier = args.tier
    # Validate the tier even though argparse's ``choices=`` should
    # have rejected anything else — a programmatic Namespace (e.g.
    # someone constructing args directly) could bypass argparse, and
    # the previous ``assert`` would be stripped under ``python -O``
    # (Codex PR #623 review NIT-1). Explicit guard returns 2 with a
    # readable error rather than blowing up later inside the submit
    # flow with a less targeted traceback.
    if tier not in ("smoke", "speed", "harness", "all"):
        print(
            f"  Error: unknown tier {tier!r}; expected one of "
            "smoke / speed / harness / all",
            file=sys.stderr,
        )
        return 2

    # Reject --base-url for the --submit combo (Codex PR #623
    # BLOCKING-1). The community-bench corpus aggregates by
    # (chip, model, version) — every submission MUST reflect the
    # contributor's actual hardware booting their actual model. Two
    # gaps if we allowed --base-url:
    #
    # 1. ``smoke_result.boot_time_ms`` is meaningless when the
    #    server was already up (we didn't measure the user's boot);
    #    the producer would have to invent a ``0.0`` placeholder
    #    that downstream consumers can't distinguish from "machine
    #    boots the model in zero ms" — a misleading row in the DB.
    # 2. Phase 2 runs ``run_standardized_bench`` IN PROCESS against
    #    a freshly-loaded engine, so the buckets numbers would NOT
    #    match the server the user pointed at. We'd publish a
    #    payload labelling itself as the user's setup while the
    #    speed numbers came from a separate engine init.
    #
    # The narrow --tier (no --submit) --base-url path is still
    # supported — that's the gauntlet/release_check use case where
    # we WANT to validate against an already-running server.
    # Belt-and-braces: an active ``FUSION_MLX_HARNESS_PROFILES_FILTER``
    # produces a partial harness payload (only the filtered keys), which
    # would fail the schema-v2 ``required`` set at submission time
    # downstream. The G12 gauntlet path only sets this env when calling
    # ``--tier harness --base-url`` (no --submit) — but a future caller
    # combining ``--submit`` with the filter would silently break here.
    # Refuse loudly instead.
    if os.environ.get("FUSION_MLX_HARNESS_PROFILES_FILTER"):
        print(
            "  Error: --submit is incompatible with "
            "FUSION_MLX_HARNESS_PROFILES_FILTER. The filter scopes the "
            "sweep to a subset of harnesses, producing a payload that "
            "would fail the community-bench schema's required-keys check "
            "(all 5 harnesses must be present). Unset the env var or "
            "drop --submit.",
            file=sys.stderr,
        )
        return 2

    if getattr(args, "base_url", None):
        print(
            "  Error: --base-url is incompatible with --submit. "
            "Community-bench submissions must reflect a fresh boot of "
            "your model on your hardware — smoke_result.boot_time_ms "
            "and the standardized B=1 buckets are both measured "
            "in-process. Drop --base-url and let bench --tier "
            "--submit boot the server itself.",
            file=sys.stderr,
        )
        return 2

    # tier='speed' --submit is the historical --submit path with a
    # new ``tier='speed'`` tag on the payload. No phase 1 needed.
    if tier == "speed":
        return _run_submit_flow(args, tier="speed")

    # Phase 1: run the tier dispatcher to capture smoke/harness data.
    # Speed bucket is intentionally skipped (see docstring); ``run_tier``
    # only honours ``skip_speed`` when tier=='all'.
    from ..bench.tier_runner import TierRunnerUnavailable, run_tier

    try:
        rc, tier_results = run_tier(
            model=args.model,
            tier=tier,
            base_url=getattr(args, "base_url", None),
            sampled=getattr(args, "sampled", False),
            return_results=True,
            skip_speed=True,
        )
    except TierRunnerUnavailable as e:
        print(f"\n  {e}", file=sys.stderr)
        return 2
    smoke_result = tier_results.get("smoke_result")
    harness_result = tier_results.get("harness_result")

    # Abort gating. The smoke probe is a hard prerequisite for ANY
    # submission: if the model can't say "4" the speed numbers we'd
    # collect in phase 2 would be misleading at best and a fork-and-
    # burn of the user's compute at worst. Harness failures are
    # surfaced THROUGH the payload (the schema's per-adapter
    # ``passed: false`` carries the signal); we DON'T abort there.
    if tier in ("smoke", "all") and smoke_result is not None:
        if not smoke_result.get("first_prompt_ok", False):
            print(
                "\n  Submission aborted: smoke probe failed. The model "
                "couldn't answer the boot prompt cleanly — submitting "
                "speed/harness numbers from this run would be "
                "misleading. Re-check the model + environment with "
                "`fusion-mlx bench <model> --tier smoke` first.",
                file=sys.stderr,
            )
            return 1

    if tier == "smoke" and smoke_result is None:
        # Phase 1 errored before producing smoke_result (e.g. server
        # boot failure). The exit code from ``run_tier`` is already
        # the right thing to return — don't try to phase 2 without
        # the required smoke_result data.
        print(
            "\n  Submission aborted: smoke phase did not produce a "
            "result (server boot likely failed). Nothing was sent.",
            file=sys.stderr,
        )
        return rc or 1
    if tier == "harness" and harness_result is None:
        print(
            "\n  Submission aborted: harness phase did not produce a "
            "result. Nothing was sent.",
            file=sys.stderr,
        )
        return rc or 1
    if tier == "all" and (smoke_result is None or harness_result is None):
        print(
            "\n  Submission aborted: --tier all did not produce both "
            "smoke and harness results. Nothing was sent.",
            file=sys.stderr,
        )
        return rc or 1

    # Phase 2 + 3 reuse the existing standardized + submit path; the
    # tier kwargs decorate the payload built inside ``_run_submit_flow``.
    return _run_submit_flow(
        args,
        tier=tier,
        smoke_result=smoke_result,
        harness_result=harness_result,
    )


def _run_submit_flow(
    args,
    *,
    tier: str | None = None,
    smoke_result: dict | None = None,
    harness_result: dict | None = None,
) -> int:
    """Execute the standardized B=1 community-bench + PR-open flow.

    Routed-to from ``bench_command`` whenever ``--submit`` is set.
    Kept as a separate function so the freeform bench path stays
    completely untouched — the standardized path imports its own
    deps lazily so that users who never touch ``--submit`` don't pay
    the import cost of the community_bench module.

    PR #5 added the schema-v2 tier-tagging kwargs:

    - ``tier`` — string copied verbatim into the ``tier`` field of the
      payload (``"speed"`` | ``"smoke"`` | ``"harness"`` | ``"all"``).
      ``None`` (the default, used by ``--submit`` without ``--tier``)
      omits the field, preserving byte-for-byte equivalence with the
      v1 ``--submit`` payload shape.
    - ``smoke_result`` / ``harness_result`` — schema-v2 sub-objects
      from the tier dispatcher. The builder enforces the
      tier↔result coupling so passing the wrong combo here ``ValueError``s
      at the payload-build line rather than landing a half-shaped row
      in the submissions corpus.
    """
    import asyncio
    from pathlib import Path

    from huggingface_hub.utils import RepositoryNotFoundError
    from mlx_lm import load

    from ..community_bench.hardware import collect as collect_hw
    from ..community_bench.hardware import is_apple_silicon
    from ..community_bench.runner import run_standardized_bench
    from ..community_bench.submission import (
        build_submission_payload,
        submit_interactive,
    )
    from ..engine_core import AsyncEngineCore, EngineConfig
    from ..model_aliases import resolve_profile
    from ..scheduler import SchedulerConfig

    if not is_apple_silicon():
        print(
            "  Error: --submit only runs on Apple Silicon (arm64 Darwin). "
            "The community database is Apple-Silicon-specific."
        )
        return 2

    # Whitelist gate. ``model.alias`` in the payload is the bucketing
    # key, so we require the user to type the canonical alias *key*
    # rather than a raw HF path — accepting both forms would let a
    # contributor's typo silently shift their submission into a
    # different bucket via the reverse-lookup. (Codex PR #582 BLOCKING:
    # silent alias coercion bypasses the intended "must be a whitelist
    # key" contract.) The GHA validator re-checks the alias against
    # aliases.json, so this guard is layered. ``args._original_alias``
    # holds the user-typed value when the dispatcher resolved an alias
    # to an HF path; if it's absent (HF path passed directly, or any
    # other no-resolution case) we fall back to ``args.model``, which
    # this guard then re-checks for the ``/`` HF-path signature.
    user_typed = getattr(args, "_original_alias", None) or args.model
    if "/" in user_typed:
        print(
            f"  Error: --submit requires the canonical alias key "
            f"(e.g. 'qwen3.5-9b-4bit'), not the resolved HF path "
            f"'{user_typed}'. Run `fusion-mlx models` for the whitelist."
        )
        return 2
    profile = resolve_profile(user_typed)
    if profile is None:
        print(
            f"  Error: '{user_typed}' is not a registered alias. "
            f"Only models listed in fusion_mlx/aliases.json can be submitted "
            f"(this keeps the comparison apples-to-apples)."
        )
        print("  Run `fusion-mlx models` to see the full whitelist.")
        return 2
    alias = user_typed
    hf_path = profile.hf_path

    notes = args.notes or None
    if notes is not None:
        if len(notes) > 200:
            print("  Error: --notes must be <= 200 chars (schema cap).")
            return 2
        # Reject control characters in --notes. Newlines/CR/terminal
        # escapes would land in the PR body, the JSON file, and any
        # future renderer — the schema's free-form ``notes`` field
        # invites contributor commentary, but it does not invite
        # ``\x1b]0;owned\x07`` terminal-title-set sequences.
        # (Codex PR #582 round-7 NIT.)
        if any(ord(c) < 0x20 or ord(c) == 0x7F for c in notes):
            print(
                "  Error: --notes contains control characters; only "
                "printable ASCII/UTF-8 is permitted."
            )
            return 2

    _check_disk_space(hf_path, force=getattr(args, "force_disk_check", False))
    _check_memory_capacity(hf_path)

    # ``--sampled`` runs a SECOND submission (with sampling="sampled")
    # in addition to the always-on greedy run. The README contract is
    # "two rows when --sampled is set, one row otherwise" — a previous
    # version replaced greedy with sampled, breaking that contract and
    # silently losing the greedy comparison line. (Codex PR #582
    # round-7 BLOCKING.) Greedy goes first so the contributor can
    # still cancel the sampled half during its consent prompt.
    sampling_modes: list[str] = ["greedy"]
    if getattr(args, "sampled", False):
        sampling_modes.append("sampled")

    async def _run() -> int:
        import concurrent.futures

        from ..engine_core import _init_mlx_step_thread

        # Load model on the future mlx-step worker thread (#170). mlx-lm
        # 0.31.3+ binds module-level ``generation_stream`` and any
        # auto-default stream to the thread that triggers them. If the
        # model weights or ``mx.compile``-cached graphs are touched on
        # the asyncio loop thread first, every later eval on the step
        # worker raises "There is no Stream(gpu, N) in current thread."
        # Spinning the worker BEFORE load and reusing it for
        # AsyncEngineCore keeps every MLX op on a single owning thread.
        # Mirrors the pattern in ``BatchedEngine._start_llm`` (which is
        # why ``fusion-mlx serve`` works but the unfixed ``bench`` path
        # doesn't).
        print(f"  Loading model {alias} ({hf_path})…")
        model_load_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="mlx-step",
            initializer=_init_mlx_step_thread,
        )
        try:
            model, tokenizer = model_load_executor.submit(load, hf_path).result()
        except (ValueError, ModuleNotFoundError) as e:
            # mlx-lm raises ``ValueError: Model type X not supported`` plus an
            # internal ``ModuleNotFoundError: No module named 'mlx_lm.models.X'``
            # for any architecture it can't import. The Gemma 4 family lives
            # in mlx-vlm (the model classes are vision-aware even for the
            # text-only checkpoints), so a bare ``pip install fusion-mlx``
            # without the ``[vision]`` extras hits this every time. The
            # README still recommends ``gemma-4-*`` aliases so newcomers
            # would otherwise see a raw traceback and conclude the model
            # is broken — translate to an actionable hint. Placed BEFORE
            # the broader ``OSError`` clause so a future maintainer can't
            # accidentally make the broad branch swallow it (Codex PR
            # #600 round-1 BLOCKING).
            msg = str(e)
            needs_vision = (
                "gemma4_unified" in msg
                or "gemma4" in msg
                or "mlx_vlm" in msg
                or "mlx-vlm" in msg
            )
            if needs_vision:
                print()
                print(
                    "  Error: this model needs the vision extras (Gemma 4 "
                    "architecture classes live in mlx-vlm)."
                )
                print("  Install them and re-run:")
                print()
                print("    pip install 'fusion-mlx[vision]'")
                print()
                print(
                    "  Or, if you only need text inference (smaller "
                    "footprint, ~16 MB vs ~450 MB):"
                )
                print("    pip install --no-deps 'mlx-vlm>=0.6.1'")
                print()
            else:
                print(f"  Error loading model: {e}")
            model_load_executor.shutdown(wait=False)
            return 2
        except (RepositoryNotFoundError, OSError) as e:
            print(f"  Error loading model: {e}")
            model_load_executor.shutdown(wait=False)
            return 2

        # Standardized config: B=1, no batching, prefix-cache off so the
        # numbers reflect cold prefill on each round (which is what the
        # tg/pp metrics are supposed to measure).
        scheduler_config = SchedulerConfig(
            max_num_seqs=1,
            max_concurrent_requests=1,
            prefill_batch_size=1,
            completion_batch_size=1,
            enable_prefix_cache=False,
        )
        engine_config = EngineConfig(
            model_name=hf_path,
            scheduler_config=scheduler_config,
        )

        print("  Collecting hardware fingerprint…")
        hardware, software = collect_hw()
        print(
            f"    chip={hardware.chip}, ram={hardware.ram_gb} GB, "
            f"cpu_cores={hardware.cpu_cores}, gpu_cores={hardware.gpu_cores}"
        )
        print(
            f"    macos={software.macos}, fusion_mlx={software.fusion_mlx}, "
            f"mlx={software.mlx}, python={software.python}"
        )

        repo_root = Path(args.repo_root) if args.repo_root else Path.cwd()
        # Pass the EXISTING executor to AsyncEngineCore so the engine
        # loop, BatchGenerator construction, and every forward pass run
        # on the same thread that owns the model weights.
        async with AsyncEngineCore(
            model, tokenizer, engine_config, executor=model_load_executor
        ) as engine:
            for mode in sampling_modes:
                print(
                    f"  Running standardized bench "
                    f"(sampling={mode}, 2 buckets × 5 rounds + 1 warmup)…"
                )
                try:
                    bench = await run_standardized_bench(
                        engine, tokenizer, sampling=mode
                    )
                except RuntimeError as exc:
                    # Friendly surface for the bench's "exactly N tokens"
                    # guard. As of #567's fix this branch is engine-bug
                    # territory (sampling sets ``ignore_eos=True`` so the
                    # model's EOS shouldn't fire); previously it blamed
                    # the user's model alias. Print a clear summary so
                    # contributors aren't dumped into a raw traceback.
                    msg = str(exc)
                    if "standardized bench requires exactly" in msg:
                        print()
                        print(
                            "  Bench round aborted (engine bug — NOT your model's fault):"
                        )
                        for line in msg.split(". "):
                            line = line.strip()
                            if line:
                                print(f"    {line}")
                        print()
                        return 1
                    raise

                print(
                    f"    short: decode={bench.short.decode_stat['median']:.2f} tok/s, "
                    f"prefill={bench.short.prefill_stat['median']:.2f} tok/s, "
                    f"ttft={bench.short.ttft_stat['median']:.1f} ms"
                )
                print(
                    f"    long:  decode={bench.long.decode_stat['median']:.2f} tok/s, "
                    f"prefill={bench.long.prefill_stat['median']:.2f} tok/s, "
                    f"ttft={bench.long.ttft_stat['median']:.1f} ms"
                )

                payload = build_submission_payload(
                    hardware=hardware,
                    software=software,
                    alias=alias,
                    hf_path=hf_path,
                    bench=bench,
                    notes=notes,
                    # v2 tier-tagging: pass through only when the caller
                    # supplied them. The builder validates the tier ↔
                    # smoke_result/harness_result coupling — passing
                    # ``smoke_result`` for ``tier=speed`` would
                    # ``ValueError`` here rather than land a half-shaped
                    # row in the corpus.
                    tier=tier,
                    smoke_result=smoke_result,
                    harness_result=harness_result,
                )
                rc = submit_interactive(payload, repo_root)
                if rc != 0:
                    # Setup error (not a "user said no") — bail out
                    # before kicking off the second submission so the
                    # contributor sees the failure clearly.
                    return rc
        return 0

    return asyncio.run(_run())
