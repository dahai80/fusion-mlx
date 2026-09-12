# SPDX-License-Identifier: Apache-2.0
"""serve_command — main boot path for fusion-mlx serve."""

import logging

from fusion_mlx._cli_base import (
    _port_preflight_or_die,
    _print_unknown_model_help,
    _resolve_audio_model_for_serve,
    _run_uvicorn,
    _uds_path_from_host,
)

from .audio_mode import (
    _display_host,
    _load_embedding_model_or_exit,
    _serve_audio_mode,
)
from .config_resolve import (
    _apply_mtp_cli_model_type_reconciliation,
    _autoconfig_parsers,
    _boot_guard_checks,
    _print_profile_banner,
    _print_startup_banner,
    _serve_from_model_dir,
    _stage_server_config,
)
from .model_download import _ensure_model_downloaded
from .preflight import (
    _check_disk_space,
    _check_memory_capacity,
    _gather_kv_cache_dtype_inputs,
)

logger = logging.getLogger(__name__)


def serve_command(args):
    """Start the OpenAI-compatible server."""
    import logging
    import os
    import sys

    # Install the M5 hardware-compat shim BEFORE any `from .server import`
    # (line ~1150), which transitively imports mlx_lm.generate -- that module
    # captures mx.new_thread_local_stream at module-import time, and on M5
    # single-stream GPUs the captured stream is unusable (#404). Idempotent,
    # no-op on hardware where the original API works. Mirrors bench_command.
    from .. import _mlx_compat as _mlx_compat

    _mlx_compat.install()

    # Released 1.0/2.0/3.0 contract: `serve --model-dir <dir>` boots the
    # multi-model engine-pool server via create_app(ServerConfig(model_dir)).
    # The Rapid-MLX migration rerouted `serve` to the single-model Scheduler
    # path (`serve <model>`); this branch restores the released --model-dir
    # contract so existing docs/scripts keep working. The Rapid-MLX
    # single-model path below is unchanged.

    # Released --model flag (docs/cli-reference.md: `serve --model X`) folds
    # into the same single-model path as the positional <model>. The parser
    # keeps both forms (dest=model_flag vs positional dest=model).
    if getattr(args, "model_flag", None):
        if getattr(args, "model", None):
            print("Error: --model and a positional <model> are mutually exclusive.")
            sys.exit(1)
        args.model = args.model_flag

    # D3.2/N2: --beginner one-click preset. Maps to lite profile + a small
    # recommended 4-bit model when none is given. No yaml double-track —
    # composes onto the existing ServerConfig via the same --profile/model
    # args the normal path reads. Mutually exclusive with --profile turbo.
    if getattr(args, "beginner", False):
        if args.profile == "turbo":
            print("Error: --beginner and --profile turbo are mutually exclusive.")
            sys.exit(1)
        args.profile = "lite"
        if not getattr(args, "model", None) and not getattr(args, "model_dir", None):
            args.model = "qwen3.5-4b-4bit"
            print(
                "[beginner] preset: profile=lite, model=qwen3.5-4b-4bit, "
                "port=11434. A small 4-bit model — safe for 16GB+ Macs."
            )
        else:
            print("[beginner] preset: profile=lite (model kept as given).")

    # FusionMLX macOS app / fusion-mlx-style launch: `serve --base-path <dir>` serves
    # <dir>/models via the multi-model engine-pool server (the app spawns this
    # with --base-path ~/.fusion-mlx). Mutually exclusive with model selection.
    base_path = getattr(args, "base_path", None)
    if base_path:
        if getattr(args, "model_dir", None) or getattr(args, "model", None):
            print(
                "Error: --base-path is mutually exclusive with <model>/--model/--model-dir."
            )
            sys.exit(1)
        args.model_dir = os.path.join(base_path, "models")
        try:
            os.makedirs(args.model_dir, exist_ok=True)
        except OSError as exc:
            print(f"Error: cannot create model dir {args.model_dir}: {exc}")
            sys.exit(1)
        return _serve_from_model_dir(args)

    if getattr(args, "model_dir", None):
        if getattr(args, "model", None):
            print("Error: --model-dir and a positional <model> are mutually exclusive.")
            print("  Use either: fusion-mlx serve --model-dir <dir>")
            print("       or:    fusion-mlx serve <model>")
            sys.exit(1)
        return _serve_from_model_dir(args)
    if not getattr(args, "model", None):
        print("Error: serve requires a model or --model-dir/--base-path <dir>.")
        print("  fusion-mlx serve --model Qwen3-4B-Q4_K_M --port 11434")
        print("  fusion-mlx serve --model-dir ~/.fusion-mlx/models --port 11435")
        print("  fusion-mlx serve --base-path ~/.fusion-mlx --port 11434")
        print()
        print('  Tip: set "default_model" in ~/.fusion-mlx/settings.json')
        print("       to make `fusion-mlx serve` work with no model arg.")
        sys.exit(1)

    _arg_max_tokens = getattr(args, "max_tokens", None)
    _max_tokens_is_explicit = _arg_max_tokens is not None
    effective_max_tokens = _arg_max_tokens if _arg_max_tokens is not None else 32768

    if _boot_guard_checks(args, effective_max_tokens):
        return

    # R10-C1: AUDIO-SERVE-MODE FORK. The boot guard above only checks
    # that the ``[audio]`` extra is installed — it doesn't route the
    # alias anywhere. Pre-R10 every short alias (``kokoro``, ``whisper``,
    # ``parakeet``...) fell through to ``_ensure_model_downloaded``
    # and 404'd at HF, while full HF ids of audio models downloaded
    # successfully but then crashed inside ``mlx_lm.load_model``
    # because audio repos don't ship safetensors. Bo r10-R1: 0/8 audio
    # aliases boot on 0.8.11 (codex r8-A r3 predicted this exact shape).
    #
    # The fix is a clean fork: if the registry resolves the model to
    # an audio entry, route to ``_serve_audio_mode`` (which skips
    # ``_ensure_model_downloaded``, the text loader, pflash, parser
    # detection, etc.) and return. Everything below this block remains
    # untouched for the text path so text-model boot does NOT regress.
    audio_entry = _resolve_audio_model_for_serve(getattr(args, "model", None))
    if audio_entry is not None:
        # Stamp the alias hop so /v1/models, telemetry, and the banner
        # all show the same name pair. ``_original_alias`` is set by
        # the main() alias resolver for text models; we mirror that
        # contract here for audio.
        if not hasattr(args, "_original_alias") or args._original_alias is None:
            args._original_alias = args.model
        # Replace the alias on args.model with the resolved HF id so
        # any downstream code that reads ``args.model`` (eg. session
        # telemetry, ps_command) sees a real repo path. The audio
        # routes still accept both forms because the registry's
        # reverse HF-id index covers full ids too.
        args.model = audio_entry.hf_id
        _serve_audio_mode(args, audio_entry)
        return

    # Interactive auto-upgrade prompt — when serve runs interactively and a
    # newer release is available, ask once before booting the model. Honors
    # FUSION_MLX_DISABLE_VERSION_CHECK, CI=1, and non-TTY stdin. Cached
    # piggy-backs on the existing staleness check's cache (24h TTL).
    from fusion_mlx._version_check import prompt_upgrade_if_available

    if prompt_upgrade_if_available():
        sys.exit(0)

    # Pre-fetch the model via the R2 mirror (with HF fallback) BEFORE the
    # heavy server boot. Without this, ``serve`` falls into
    # ``mlx_lm.load`` → ``huggingface_hub.snapshot_download`` directly and
    # skips the mirror entirely (#651). ``_ensure_model_downloaded`` is a
    # no-op on local paths and on fully-cached repos, so this is free on
    # the warm path.
    _ensure_model_downloaded(args.model)

    # Import unified server
    from .. import server
    from ..scheduler import SchedulerConfig
    from ..server import app, load_model

    logger = logging.getLogger(__name__)
    uvicorn_log_level = server.configure_logging(args.log_level)

    # Validate tool calling arguments
    if args.enable_auto_tool_choice and not args.tool_call_parser:
        print("Error: --enable-auto-tool-choice requires --tool-call-parser")
        print("Example: --enable-auto-tool-choice --tool-call-parser mistral")
        sys.exit(1)

    # Validate --tool-call-parser against the live registry (not the
    # stale argparse choices list). v0.6.63 onboarding sweep finding #1.
    if args.tool_call_parser:
        # Narrow the catch: only swallow import-time / attribute access
        # failures (broken install, missing module file). Anything else
        # — a corrupt registry that's loaded but malformed, a TypeError
        # from a buggy parser's __init_subclass__, etc. — is a real bug
        # we want to surface, not paper over with "validation skipped".
        # Codex follow-up to PR #433.
        valid: list[str] | None = None
        try:
            from ..tool_parsers import ToolParserManager

            valid = sorted(ToolParserManager.tool_parsers.keys())
        except (ImportError, AttributeError) as e:
            print(
                "warning: --tool-call-parser validation skipped — "
                f"tool_parsers registry unavailable ({type(e).__name__}: {e}). "
                "Proceeding without input check.",
                file=sys.stderr,
            )
        # Treat an empty registry (degenerate install) the same as a
        # failed import — skip validation rather than reject every input.
        # Without this guard, a successful import with zero registered
        # parsers would hard-fail every CLI invocation; DeepSeek
        # follow-up to PR #434.
        if valid and args.tool_call_parser not in valid:
            print(
                f"error: argument --tool-call-parser: invalid choice: "
                f"{args.tool_call_parser!r} "
                f"(choose from: {', '.join(valid)})",
                file=sys.stderr,
            )
            sys.exit(2)

    # Validate gpu-memory-utilization range
    if not (0.0 < args.gpu_memory_utilization <= 1.0):
        print(
            "Error: --gpu-memory-utilization must be between 0.0 (exclusive) and 1.0 (inclusive)"
        )
        sys.exit(1)

    # Validate PFlash config and reject unsupported model combinations
    # at startup. Done here (not lazily in the scheduler) so a typo in
    # --pflash-keep-ratio doesn't surface as a model-load failure
    # after a multi-minute weight download. See #287.
    #
    # ``resolve_pflash_mode_default`` runs before ``config_from_args``
    # so the per-alias default (``"always"`` for verified Qwen3.5 /
    # Qwen3.6 aliases, ``"off"`` everywhere else) is materialized into
    # ``args.pflash``. The resolved value then flows through the same
    # validation path the user-explicit case takes.
    from ..api.utils import is_mllm_model
    from ..pflash import (
        config_from_args,
        resolve_pflash_mode_default,
        validate_model_support,
    )

    args.pflash = resolve_pflash_mode_default(args, model_name=args.model)
    try:
        pflash_config = config_from_args(args)
        validate_model_support(
            pflash_config,
            model_name=args.model,
            is_mllm=is_mllm_model(args.model),
        )
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Auto-detect parser config from model name when not explicitly set.
    # --no-tool-call-parser / --no-reasoning-parser are escape hatches
    # (SOP §10): if the user opts out, do NOT let the AliasProfile auto-
    # populate args.tool_call_parser / args.reasoning_parser. Past
    # incidents: #393-class (auto-detect false positive with no opt-out).
    _autoconfig_parsers(args, logger)
    cors_origins, gc_control = _stage_server_config(args, server, logger)

    _print_startup_banner(args, cors_origins, gc_control, logger)

    # Pre-load embedding model if specified.
    #
    # H-08 install guard + D-EMBED-ALIAS alias-resolution + clean
    # ModelNotFoundError wrapping all live in the shared helper so the
    # standalone ``python -m fusion_mlx.server`` entry behaves identically.
    # See :func:`_load_embedding_model_or_exit` for the full contract;
    # F-H08-INCOMPLETE / D-CAPABILITIES already pre-flighted
    # ``require_mlx_embeddings_or_exit`` at the top of ``serve_command``
    # but the helper re-probes defensively so any caller that
    # synthesizes an ``args`` namespace and jumps into the load path
    # still gets the install-hint exit instead of a raw
    # ``ModuleNotFoundError``.
    if args.embedding_model:
        _load_embedding_model_or_exit(args, server.load_embedding_model)

    # Warn about deprecated flags
    if getattr(args, "simple_engine", False):
        print(
            "\n  ⚠ --simple-engine is deprecated and has no effect."
            "\n    BatchedEngine is now the sole engine — it handles both"
            "\n    single-user and multi-user workloads with equal performance.\n"
        )
    if getattr(args, "kv_bits", None) is not None:
        print(
            "\n  ⚠ --kv-bits is deprecated and has no effect."
            "\n    For prefix cache quantization, use --kv-cache-quantization instead.\n"
        )
    if getattr(args, "draft_model", None):
        print(
            "\n  ⚠ --draft-model is deprecated and has no effect."
            "\n    For DFlash speculative decoding, use --enable-dflash "
            "(requires a DFlash-eligible alias). "
            "For MTP, use --enable-mtp (requires a model with MTP head).\n"
        )
    if getattr(args, "specprefill", False):
        print("\n  ⚠ --specprefill is deprecated and has no effect.\n")

    # Resolve per-alias TurboQuant default before the mutual-exclusion
    # check below — operator-explicit values still win.
    from ..turboquant import resolve_turboquant_mode_default

    args.kv_cache_turboquant = resolve_turboquant_mode_default(
        args, model_name=args.model
    )

    # Mutual exclusion: turboquant (any mode) vs standard quantization.
    # The argparse layer normalizes the flag to either ``None`` (off),
    # ``"v4"``, or ``"k8v4"``. Anything truthy means TurboQuant is on.
    if args.kv_cache_turboquant and args.kv_cache_quantization:
        print(
            "\n  Error: --kv-cache-turboquant and --kv-cache-quantization are "
            "mutually exclusive. Choose one.\n"
        )
        sys.exit(1)

    # R15 #300: resolve --kv-cache-dtype + --reasoning + safelist BEFORE
    # the legacy --kv-cache-quantization flag wins. When --kv-cache-
    # turboquant is on, leave the kv-cache-dtype path alone — TurboQuant
    # owns the V cache and would conflict with QuantizedKVCache. When
    # the legacy --kv-cache-quantization flag is passed, honor it
    # verbatim for backwards compatibility; the new dtype flag only
    # takes effect on operators who haven't pinned the legacy bool.
    kv_cache_decision = None
    if not args.kv_cache_turboquant and not args.kv_cache_quantization:
        from ..kv_cache_dtype import (
            dtype_to_quantization_bits,
            log_kv_cache_decision,
            resolve_kv_cache_dtype,
        )

        hf_cfg, alias_meta = _gather_kv_cache_dtype_inputs(args.model)
        kv_cache_decision = resolve_kv_cache_dtype(
            args.kv_cache_dtype,
            reasoning=args.reasoning,
            model_name=args.model,
            hf_path=(alias_meta or {}).get("hf_path"),
            hf_config=hf_cfg,
            alias_metadata=alias_meta,
        )
        log_kv_cache_decision(kv_cache_decision, model_name=args.model)
        quant, bits = dtype_to_quantization_bits(kv_cache_decision.dtype)
        # Mutate args so the existing SchedulerConfig wiring picks up
        # the resolved values without a second code path.
        args.kv_cache_quantization = quant
        args.kv_cache_quantization_bits = bits
        # Stash on the shared ServerConfig so /metrics surfaces the
        # effective dtype during the pre-engine load window — operator
        # uptime dashboards scrape within ms of process start.
        try:
            from fusion_mlx.config import get_config as _get_config

            _get_config().kv_cache_dtype = kv_cache_decision.dtype
        except Exception:
            # ServerConfig is best-effort observability; never block
            # serve start on a metrics-only side effect.
            pass
    elif args.kv_cache_quantization:
        # Legacy flag took precedence — synthesize a decision so
        # observability still has a single source of truth.
        from ..kv_cache_dtype import (
            REASONING_KV_CACHE_DTYPE,
            KVCacheDtypeDecision,
        )

        # codex r1 BLOCKING #1: ``--reasoning`` must override the
        # legacy ``--kv-cache-quantization`` flag too — otherwise
        # ``fusion-mlx serve --reasoning --kv-cache-quantization
        # --kv-cache-quantization-bits 4`` silently resolves to int4
        # and the operator who deliberately asked for the reasoning
        # profile gets the AIME-class quality cliff. Reject the
        # conflicting combo with an explicit error: silently flipping
        # the legacy bits to 8 would hide the misconfiguration.
        # bits=8 is equivalent to --reasoning's int8 pin and is
        # harmless; only bits=4 conflicts.
        if args.reasoning and args.kv_cache_quantization_bits == 4:
            print(
                "\n  Error: --reasoning is incompatible with "
                "--kv-cache-quantization --kv-cache-quantization-bits 4. "
                "The reasoning profile pins KV cache to int8 because "
                "sub-4-bit drops -20pt on AIME-class math. Either drop "
                "--reasoning or drop --kv-cache-quantization-bits 4 "
                "(or both; use --kv-cache-dtype int8 instead).\n"
            )
            sys.exit(1)

        # codex r2 BLOCKING #1: argparse pins ``--kv-cache-quantization-bits``
        # to ``choices={4,8}``, but programmatic callers (tests, library
        # users that bypass argparse) can land an out-of-range bits value
        # here. The old ``"int4" if bits == 4 else "int8"`` silently
        # labeled every non-4 value as ``int8`` even when KV would actually
        # be quantized at the requested bit width. Fail fast instead so
        # the gauge / banner / SchedulerConfig never lie about the
        # active dtype.
        if args.kv_cache_quantization_bits not in (4, 8):
            print(
                f"\n  Error: --kv-cache-quantization-bits must be 4 or 8 "
                f"(got {args.kv_cache_quantization_bits}). Use "
                f"--kv-cache-dtype for the canonical knob.\n"
            )
            sys.exit(1)
        legacy_dtype = "int4" if args.kv_cache_quantization_bits == 4 else "int8"
        # When --reasoning is set alongside the (compatible) bits=8
        # legacy flag, the operator-facing reason should still
        # advertise the reasoning profile so the startup banner is
        # consistent across the two CLI shapes.
        if args.reasoning:
            assert legacy_dtype == REASONING_KV_CACHE_DTYPE  # by the guard above
            reason = (
                f"legacy --kv-cache-quantization flag + --reasoning — "
                f"resolved to {REASONING_KV_CACHE_DTYPE} (reasoning profile "
                f"pin matches legacy bits=8)"
            )
        else:
            reason = (
                f"legacy --kv-cache-quantization flag (bits="
                f"{args.kv_cache_quantization_bits}) — equivalent to "
                f"--kv-cache-dtype {legacy_dtype}"
            )
        kv_cache_decision = KVCacheDtypeDecision(
            dtype=legacy_dtype,
            reason=reason,
            downgraded=False,
            requested=legacy_dtype,
        )
        try:
            from fusion_mlx.config import get_config as _get_config

            _get_config().kv_cache_dtype = legacy_dtype
        except Exception:
            pass

    # --suffix-decoding + --enable-mtp may coexist: mtp takes priority for
    # MTP-eligible decode steps (verify+accept inside GenerationBatch.next)
    # and suffix runs only on steps mtp did not own (fallback / non-MTP
    # models). The scheduler's _try_spec_decode guard (last_step_was_mtp)
    # prevents double-spec. This is the mtp<->suffix per-request routing
    # path. (The DFlash-vs-{suffix,mtp} check is upstream, before the banner;
    # dflash/dspark still early-fork and stay mutually exclusive here.)
    if args.suffix_decoding and args.enable_mtp:
        print(
            "\n  --suffix-decoding + --enable-mtp: mtp takes priority for\n"
            "  MTP-eligible steps; suffix runs when mtp did not handle the\n"
            "  step (per-request routing, no double-spec).\n"
        )

    # Build scheduler config
    enable_prefix_cache = args.enable_prefix_cache and not args.disable_prefix_cache

    scheduler_config = SchedulerConfig(
        max_num_seqs=args.max_num_seqs,
        max_concurrent_requests=args.max_concurrent_requests,
        prefill_batch_size=args.prefill_batch_size,
        completion_batch_size=args.completion_batch_size,
        enable_prefix_cache=enable_prefix_cache,
        prefix_cache_size=args.prefix_cache_size,
        # R15-P1 (task #303): radix-tree prefix-cache index.
        prefix_cache_index=getattr(args, "prefix_cache_index", "radix"),
        # Memory-aware cache options
        use_memory_aware_cache=not args.no_memory_aware_cache,
        cache_memory_mb=args.cache_memory_mb,
        cache_memory_percent=args.cache_memory_percent,
        # Paged cache options
        use_paged_cache=args.use_paged_cache,
        paged_cache_block_size=args.paged_cache_block_size,
        max_cache_blocks=args.max_cache_blocks,
        # Chunked prefill
        chunked_prefill_tokens=args.chunked_prefill_tokens,
        # Prefill step size (chunk size). Must be plumbed here — BatchedEngine
        # reads it off scheduler_config only; the legacy load_model kwarg was
        # accepted but never used. See #400 and the CLI ↔ Config fidelity
        # audit at scripts/audit_cli_config_fidelity.py.
        prefill_step_size=args.prefill_step_size,
        # MTP
        enable_mtp=args.enable_mtp,
        mtp_num_draft_tokens=args.mtp_num_draft_tokens,
        mtp_optimistic=args.mtp_optimistic,
        mtp_sidecar=getattr(args, "mtp_sidecar", None),
        # R15-P1 #302/#313: --spec-decode {none,mtp,dflash}. Plumb the
        # raw choice through; the boot-time eligibility check below
        # validates that ``mtp`` was only passed for a config.json with
        # ``mtp_num_hidden_layers >= 1`` and ``dflash`` requires a
        # Qwen3.5/3.6 model + a bound DFlash drafter.
        spec_decode=getattr(args, "spec_decode", "none"),
        dflash_drafter_path=getattr(args, "dflash_drafter_path", "") or "",
        # DFlash2 (block-diffusion spec decode, z-lab dflash pkg, Qwen3.8).
        # In-place load via BatchedEngine (no forked server, unlike dflash v1).
        dflash2_drafter_path=getattr(args, "dflash2_drafter_path", "") or "",
        dflash2_block_size=getattr(args, "dflash2_block_size", 5) or 5,
        dflash2_draft_bits=getattr(args, "dflash2_draft_bits", 4),
        # SuffixDecoding
        enable_suffix_decoding=args.suffix_decoding,
        suffix_max_draft=args.suffix_max_draft,
        suffix_max_suffix_len=args.suffix_max_suffix_len,
        suffix_min_confidence=args.suffix_min_confidence,
        suffix_min_draft_len=args.suffix_min_draft_len,
        # KV cache quantization (R15 #300: dtype string is the canonical
        # observability surface; ``_quantization`` / ``_bits`` are the
        # wire-level toggles that drive ``mlx_lm.QuantizedKVCache``).
        kv_cache_dtype=(
            kv_cache_decision.dtype if kv_cache_decision is not None else "bf16"
        ),
        kv_cache_quantization=args.kv_cache_quantization,
        kv_cache_quantization_bits=args.kv_cache_quantization_bits,
        kv_cache_quantization_group_size=args.kv_cache_quantization_group_size,
        kv_cache_min_quantize_tokens=args.kv_cache_min_quantize_tokens,
        # TurboQuant compression (R15 Phase 4: mode-aware)
        # ``--kv-cache-turboquant`` now carries a mode value: ``None``
        # when off, ``"v4"`` for the legacy V-only path, ``"k8v4"`` for
        # the K-8bit + V-4bit mix. SchedulerConfig keeps the boolean
        # ``kv_cache_turboquant`` for downstream callers; the mode
        # string rides on the dedicated field below.
        kv_cache_turboquant=bool(args.kv_cache_turboquant),
        kv_cache_turboquant_bits=args.kv_cache_turboquant_bits,
        kv_cache_turboquant_group_size=args.kv_cache_turboquant_group_size,
        # R15-P1 (task #296): disk-backed KV checkpointing at 256-tok
        # boundaries. ``0`` disables; the runtime module guards every
        # hot-path call with ``should_checkpoint`` so the cost when off
        # is one int comparison.
        kv_disk_checkpoint_interval=getattr(args, "kv_disk_checkpoint_interval", 256),
        kv_cache_turboquant_mode=(args.kv_cache_turboquant or "v4"),
        # PFlash long-prompt compression (#287)
        pflash_config=pflash_config,
        # D-METAL-CAP: thread the user's --gpu-memory-utilization into
        # SchedulerConfig so the admission gate enforces the same cap
        # that ``mx.set_memory_limit`` only treats as a guideline. The
        # CLI ↔ Config fidelity audit blocks merges where this kwarg
        # exists on SchedulerConfig but is missing at the construction
        # site — without this line, ``--gpu-memory-utilization 0.45``
        # would still set the soft Metal hint but the admission-time
        # check would stay disabled (SchedulerConfig default 0.0),
        # silently recreating the D-METAL-CAP regression.
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    print("Mode: Continuous batching (for multiple concurrent users)")
    if args.chunked_prefill_tokens > 0:
        print(f"Chunked prefill: {args.chunked_prefill_tokens} tokens per step")
    if args.enable_mtp:
        print(f"MTP: enabled, draft_tokens={args.mtp_num_draft_tokens}")
    # --spec-decode auto: ask SpecAutoRouter to pick a zero-config
    # method (mtp for MTP-eligible checkpoints, n-gram suffix otherwise)
    # from the model's shape. Drafter-backed methods stay operator-
    # selected. See speculative/auto_resolve.py. Runs before the mtp
    # eligibility check so a resolved "mtp" still gets validated below.
    if getattr(args, "spec_decode", "none") == "auto":
        from fusion_mlx.speculative.auto_resolve import (
            apply_resolution,
            resolve_spec_auto,
        )

        try:
            _hf_cfg_auto, _ = _gather_kv_cache_dtype_inputs(args.model)
        except Exception:
            _hf_cfg_auto = None
        _auto_family = None
        _auto_moe = False
        _auto_qbits = None
        try:
            from ..model_auto_config import detect_model_config

            _ac = detect_model_config(args.model)
            if _ac:
                _auto_family = getattr(_ac, "model_family", None)
                _auto_moe = getattr(_ac, "is_moe", False)
                _auto_qbits = getattr(_ac, "quant_bits", None)
        except Exception:
            logger.debug(
                "spec-auto: family detection failed (non-fatal)", exc_info=True
            )
        _resolution = resolve_spec_auto(
            _hf_cfg_auto,
            model_family=_auto_family,
            is_moe=_auto_moe,
            quant_bits=_auto_qbits,
        )
        # auto is authoritative — clear operator-set spec flags so the
        # resolved method doesn't collide with a stale enable_*.
        args.suffix_decoding = False
        args.enable_mtp = False
        args.enable_dflash = False
        args.enable_dflash2 = False
        args.enable_dspark = False
        apply_resolution(args, _resolution)
        print(f"Spec-decode: auto → {_resolution.cli_target} ({_resolution.reason})")
    # R15-P1 #302: native Qwen3.5/3.6 MTP via vendored mlx-lm PR #990.
    # Banner line + boot-time eligibility check fires here so misuse
    # (--spec-decode mtp on a non-Qwen3.5/3.6 model) bounces with a
    # clear error rather than discovering the mismatch when the first
    # backbone forward pass raises ``AttributeError`` mid-generation.
    if getattr(args, "spec_decode", "none") == "mtp":
        from fusion_mlx.speculative.mtp import (
            MTPEligibility,
            detect_mtp_eligibility,
        )

        # ``_gather_kv_cache_dtype_inputs`` already reads
        # ``config.json`` for the same model the operator passed in;
        # reuse it so a side-loaded HF path or alias path both work.
        try:
            hf_cfg_eligibility, _ = _gather_kv_cache_dtype_inputs(args.model)
        except Exception:  # pragma: no cover — best-effort
            hf_cfg_eligibility = None
        eligibility = detect_mtp_eligibility(hf_cfg_eligibility)
        if eligibility is MTPEligibility.NONE:
            print(
                "error: --spec-decode mtp requires a Qwen3.5 / Qwen3.6 "
                "checkpoint with mtp_num_hidden_layers >= 1 in "
                "config.json. The loaded model does not qualify "
                "(re-convert from HF with mlx-lm PR #990's sanitize() "
                "path to preserve mtp.* weights).",
                file=sys.stderr,
            )
            sys.exit(2)
        print(f"Spec-decode: mtp ({eligibility.value})")

    # Reconcile scheduler_config.mtp_model_type against MTP eligibility so the
    # BatchedEngine dispatch gate sees the vetted target type. Runs for both
    # --spec-decode mtp (native Qwen3.5/3.6) and --mtp-sidecar (gemma4 unified
    # assistant drafter). The --enable-mtp (Qwen3-Next BatchGenerator inject)
    # path is intentionally excluded: its architecture is outside the MTP
    # dispatch set and it does not route through dispatch_mtp_inject.
    if getattr(args, "spec_decode", "none") == "mtp" or getattr(
        args, "mtp_sidecar", None
    ):
        try:
            _hf_cfg_reconcile, _ = _gather_kv_cache_dtype_inputs(args.model)
        except Exception:
            _hf_cfg_reconcile = None
        _apply_mtp_cli_model_type_reconciliation(
            scheduler_config,
            _hf_cfg_reconcile,
            has_external_sidecar=bool(getattr(args, "mtp_sidecar", None)),
        )

    # ``--spec-decode dflash`` is normalized to ``--enable-dflash`` near
    # the top of serve_command (#318 redirect); by the time we reach
    # here, args.spec_decode is "none" for dflash callers. The
    # speculative.dflash gate at the start of serve_command runs the
    # actual eligibility + drafter-binding checks via the prod bridge.
    if args.suffix_decoding:
        print(
            f"SuffixDecoding: enabled, max_draft={args.suffix_max_draft}, "
            f"max_suffix={args.suffix_max_suffix_len}, "
            f"min_conf={args.suffix_min_confidence}"
        )
    print(f"Stream interval: {args.stream_interval} tokens")
    if args.use_paged_cache:
        print(
            f"Paged cache: block_size={args.paged_cache_block_size}, max_blocks={args.max_cache_blocks}"
        )
    elif enable_prefix_cache and not args.no_memory_aware_cache:
        cache_info = (
            f"{args.cache_memory_mb}MB"
            if args.cache_memory_mb
            else f"{args.cache_memory_percent * 100:.0f}% of RAM"
        )
        index_choice = getattr(args, "prefix_cache_index", "radix")
        print(f"Memory-aware cache: {cache_info} (index={index_choice})")
        if args.kv_cache_turboquant:
            mode = args.kv_cache_turboquant
            if mode == "k8v4":
                print(
                    f"TurboQuant K8V4: K=8-bit Walsh-Hadamard, V=4-bit Lloyd-Max, "
                    f"group_size={args.kv_cache_turboquant_group_size}"
                )
            else:
                bits_str = (
                    str(args.kv_cache_turboquant_bits)
                    if args.kv_cache_turboquant_bits
                    else "auto"
                )
                print(
                    f"TurboQuant V-cache ({mode}): {bits_str}-bit, "
                    f"group_size={args.kv_cache_turboquant_group_size} (K stays FP16)"
                )
        elif args.kv_cache_quantization:
            print(
                f"KV cache quantization: {args.kv_cache_quantization_bits}-bit, "
                f"group_size={args.kv_cache_quantization_group_size}"
            )
    elif enable_prefix_cache:
        print(f"Prefix cache: max_entries={args.prefix_cache_size}")

    # Check port availability before loading model (avoid wasting RAM on conflict).
    # Set SO_REUSEADDR to match uvicorn's bind behavior — without it, this
    # preflight fails on a port still in TCP TIME_WAIT (e.g. just after a
    # previous fusion-mlx process exited), even though uvicorn would happily
    # bind it. Caused spurious "port in use" errors for back-to-back server
    # starts in the validation pipeline.
    #
    # Skip in --listen-fd mode: the supervisor has already bound the socket
    # and handed us the fd. There is no host/port for us to check, and any
    # bind we attempt here would race or collide with the inherited socket.
    # --host unix: also skips (#351: UDS mode has no TCP port to probe).
    if (
        getattr(args, "listen_fd", None) is None
        and _uds_path_from_host(args.host) is None
    ):
        # Shared helper so the legacy ``python -m fusion_mlx.server``
        # entrypoint (fusion_mlx/server.py) can call the same probe
        # without duplicating the wildcard-alias / loopback-shadow
        # logic. See ``_port_preflight_or_die`` for why we probe both
        # the requested host AND 127.0.0.1 when the requested host is
        # a wildcard alias.
        _port_preflight_or_die(args.host, args.port, model=args.model)

    # Check disk space before downloading model
    _check_disk_space(args.model, force=getattr(args, "force_disk_check", False))

    # Pre-flight memory check — warn (don't abort) if model + working set
    # would push unified memory past the kernel-panic threshold (issue #324).
    _check_memory_capacity(args.model)

    # DFlash fork: when --enable-dflash is set, skip BatchedEngine entirely
    # and run the dedicated DFlash server. The eligibility check above has
    # already validated the alias, so by here we have a known-good profile.
    if args.enable_dflash:
        # DFlash IS a speculative-decode path. The --no-spec-decode escape
        # hatch (SOP §10) must reject it here — otherwise the user thinks
        # they've disabled spec-decode but DFlash silently proceeds via
        # its dedicated server, never touching EngineCore / ModelConfig.
        if getattr(args, "no_spec_decode", False):
            print(
                "error: --enable-dflash and --no-spec-decode are mutually "
                "exclusive — DFlash is a speculative-decode mode.",
                file=sys.stderr,
            )
            sys.exit(2)
        from ..model_aliases import resolve_profile
        from ..speculative.dflash.server import run_dflash_server

        _alias_name = getattr(args, "_original_alias", None) or args.model
        _profile = resolve_profile(_alias_name)
        # The eligibility check at top of serve_command guarantees this
        # passes — assert to be defensive against future refactors.
        assert (
            _profile is not None and _profile.supports_dflash
        ), f"DFlash profile invariant violated for {_alias_name!r}"
        # ``--dflash-drafter-path`` override stays valid through both
        # ``--enable-dflash`` and the ``--spec-decode dflash`` redirect
        # path (#318): an operator-supplied path wins over the profile
        # default. Empty string / missing attr falls back to the alias
        # registry entry (validated non-None by _coerce_alias_dflash).
        _dflash_drafter_override = (
            getattr(args, "dflash_drafter_path", "") or ""
        ).strip()
        run_dflash_server(
            main_model_repo=_profile.hf_path,
            drafter_repo=_dflash_drafter_override or _profile.dflash_draft_model,
            host=args.host,
            port=args.port,
            served_model_name=args.served_model_name or _alias_name,
            default_max_tokens=effective_max_tokens,
            cors_origins=cors_origins,
            uvicorn_log_level=uvicorn_log_level,
            no_thinking=args.no_thinking,
        )
        return

    # Load model with unified server
    if getattr(args, "force_hybrid", False) and getattr(args, "no_hybrid", False):
        print(
            "error: --force-hybrid and --no-hybrid are mutually exclusive — "
            "pick one to override auto-detection.",
            file=sys.stderr,
        )
        sys.exit(2)
    if getattr(args, "force_spec_decode", False) and getattr(
        args, "no_spec_decode", False
    ):
        print(
            "error: --force-spec-decode and --no-spec-decode are mutually "
            "exclusive — pick one to override auto-detection.",
            file=sys.stderr,
        )
        sys.exit(2)
    if getattr(args, "force_openai_harmony_streaming", False) and getattr(
        args, "no_openai_harmony_streaming", False
    ):
        print(
            "error: --force-openai-harmony-streaming and "
            "--no-openai-harmony-streaming are mutually exclusive — pick one "
            "to override the HarmonyStreamingRouter auto-upgrade gate (#516).",
            file=sys.stderr,
        )
        sys.exit(2)
    try:
        load_model(
            args.model,
            scheduler_config=scheduler_config,
            stream_interval=args.stream_interval,
            max_tokens=effective_max_tokens,
            max_tokens_is_explicit=_max_tokens_is_explicit,
            force_text=args.no_mllm,
            gpu_memory_utilization=args.gpu_memory_utilization,
            cloud_model=args.cloud_model,
            cloud_threshold=args.cloud_threshold,
            cloud_api_base=args.cloud_api_base,
            cloud_api_key=args.cloud_api_key,
            served_model_name=args.served_model_name,
            mtp=args.enable_mtp,
            force_hybrid=getattr(args, "force_hybrid", False),
            no_hybrid=getattr(args, "no_hybrid", False),
            force_spec_decode=getattr(args, "force_spec_decode", False),
            no_spec_decode=getattr(args, "no_spec_decode", False),
            force_openai_harmony_streaming=getattr(
                args, "force_openai_harmony_streaming", False
            ),
            no_openai_harmony_streaming=getattr(
                args, "no_openai_harmony_streaming", False
            ),
            lora_path=getattr(args, "lora_path", None),
        )
    except Exception as e:
        # Show clean error instead of raw traceback. Catch the typed
        # HF exception class for the 404 case; fall back to substring
        # match for legacy callers (older huggingface_hub) and for
        # non-HF errors that still spell out "not found".
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

    # load_model() above called get_app(), which instantiated the singleton
    # Server and set the module-level ``server.app``. The local ``app``
    # imported at the top of serve_command is stale (None at import time) —
    # rebind to the real FastAPI app before handing it to uvicorn.
    app = server.app

    # Task #292 / codex r1 BLOCKING defense-in-depth: ``load_model``
    # already invokes ``register_audio_routes_if_enabled`` at its tail.
    # Calling it AGAIN here makes the wire-up explicit at the CLI
    # surface — a future refactor that moves the hook out of
    # ``load_model`` (e.g. into a lifespan event) won't silently drop
    # ``--enable-audio`` for the ``fusion-mlx serve`` path. The helper
    # is idempotent (app-local sentinel) so the second call is a
    # cheap attribute read.
    server.register_audio_routes_if_enabled()

    # Start server
    # Note: Metal shader warmup runs in the FastAPI lifespan hook (server.py).
    # The "Ready:" banner is printed FROM that hook once warmup completes and
    # the port is actually bound — printing it here would lie to users who
    # curl immediately and get connection-refused while shaders compile.
    print()
    _print_profile_banner(server)
    host_display, uds_path = _display_host(args.host)
    listen_fd = getattr(args, "listen_fd", None)
    if uds_path is not None:
        print(
            f"  Starting server on unix socket: {uds_path} "
            "(warming up - this can take a few seconds)"
        )
    elif listen_fd is not None:
        # Socket activation path — supervisor pre-bound the listening
        # socket. We don't know the actual address from the fd without a
        # ``getsockname`` lookup; surfacing fd=<N> in the banner is the
        # honest thing to print here.
        print(
            f"  Starting server on inherited fd {listen_fd} "
            "(warming up — this can take a few seconds)"
        )
    else:
        print(
            f"  Starting server on http://{host_display}:{args.port} (warming up — this can take a few seconds)"
        )
    from fusion_mlx._version_check import print_staleness_warning_if_any

    print_staleness_warning_if_any()
    print()

    # Stash the source of truth for the lifespan "Ready:" banner —
    # which shape depends on the bind mode:
    #
    #   * Default (host+port): stamp ``bind_host``/``bind_port`` so the
    #     banner prints ``Ready: http://host:port/v1``.
    #   * ``--listen-fd``: stamp ``bind_listen_fd`` instead. The
    #     supervisor's ``getsockname`` is the only honest source for the
    #     address — stamping ``args.host``/``args.port`` here would lie
    #     to log readers (the supervisor might have bound to a different
    #     address). Codex rounds 1+3 PR #696 review.
    from fusion_mlx.config import get_config

    # Always reset BOTH source-of-truth fields before stamping the
    # active branch — the singleton config persists across in-process
    # ``serve_command`` invocations (test harnesses, embedded usage), so
    # a prior host/port stash would otherwise take precedence over a
    # subsequent fd stash (and vice-versa) and the Ready banner would
    # lie about which listener is live. Codex round-4 PR #696 review.
    _cfg = get_config()
    _cfg.bind_host = None
    _cfg.bind_port = None
    _cfg.bind_listen_fd = None
    _cfg.bind_uds = None
    if uds_path is not None:
        _cfg.bind_uds = uds_path
    elif listen_fd is None:
        _cfg.bind_host = host_display
        _cfg.bind_port = args.port
    else:
        _cfg.bind_listen_fd = listen_fd

    _run_uvicorn(app, args, uvicorn_log_level)
