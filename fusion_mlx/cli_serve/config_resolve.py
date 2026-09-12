# SPDX-License-Identifier: Apache-2.0
"""Server config staging: pflash args, profile banner, serve-from-model-dir,
boot guards, autoconfig parsers, server config staging, startup banner,
MTP CLI model-type reconciliation."""

import logging
import sys

from fusion_mlx._cli_base import (
    _apply_body_receive_timeout_env,
    _auth_feature_str,
    _run_uvicorn,
)

from .audio_mode import _display_host

logger = logging.getLogger(__name__)


def _add_pflash_args(parser) -> None:
    """Attach PFlash long-prompt-compression CLI flags to an argparse parser.

    Used by both ``serve`` and ``bench`` so the flag surface stays in
    sync. The default for ``--pflash`` is intentionally ``None``
    (sentinel for "user passed nothing") so the per-alias resolver in
    ``pflash.resolve_pflash_mode_default`` can switch the engine to
    ``always`` for ``pflash_tier="verified"`` aliases (Qwen3.5 /
    Qwen3.6 family per #287) without breaking the explicit-override
    contract: passing ``--pflash off`` still wins.
    """
    parser.add_argument(
        "--pflash",
        choices=["off", "auto", "always"],
        default=None,
        help="Enable PFlash long-prompt prefill compression "
        "(off, auto, always). Default: 'always' for verified aliases "
        "(Qwen3.5 / Qwen3.6 family per #287), 'off' for everything else.",
    )
    parser.add_argument(
        "--pflash-threshold",
        type=int,
        default=32_768,
        help="Minimum prompt tokens before --pflash auto compresses (default: 32768).",
    )
    parser.add_argument(
        "--pflash-keep-ratio",
        type=float,
        default=0.20,
        help="Fraction of prompt tokens to keep when compressing "
        "(default: 0.20 — matches the bench-validated profile in PR #649: "
        "TTFT 3.87x-8.5x, needle recall 5/5 across tested cells).",
    )
    parser.add_argument(
        "--pflash-min-keep-tokens",
        type=int,
        default=2_048,
        help="Minimum tokens to keep when compressing (default: 2048).",
    )
    parser.add_argument(
        "--pflash-sink-tokens",
        type=int,
        default=256,
        help="Leading prompt tokens always kept by PFlash (default: 256).",
    )
    parser.add_argument(
        "--pflash-tail-tokens",
        type=int,
        default=2_048,
        help="Trailing prompt tokens always kept by PFlash (default: 2048).",
    )
    parser.add_argument(
        "--pflash-block-size",
        type=int,
        default=128,
        help="Middle-token scoring block size (default: 128).",
    )
    parser.add_argument(
        "--pflash-query-window",
        type=int,
        default=512,
        help="Trailing query window used to score middle blocks (default: 512).",
    )
    parser.add_argument(
        "--pflash-stride-blocks",
        type=int,
        default=8,
        help="Keep every Nth middle block as an anchor during scoring "
        "(0 disables anchors, default: 8).",
    )
    parser.add_argument(
        "--pflash-include-tools",
        action="store_true",
        help="Allow PFlash compression on prompts with tool definitions. "
        "By default tool prompts are skipped for tool-call reliability.",
    )


def _print_profile_banner(server_module) -> None:
    """§4.4: print profile capability list at startup.

    Shows the operator what's actually enabled: profile name, mounted
    routes, skipped routes, spec-decode status, memory watermarks, and
    wired limit. The commercial ops first need is "what is this machine
    actually running?"
    """
    try:
        from ..config import get_config as _get_config

        _cfg = _get_config()
        _profile_name = getattr(_cfg, "profile", None) or "standard"
        _srv = getattr(server_module, "_server", None) or getattr(
            server_module, "server", None
        )
        _mounted = getattr(_srv, "_profile_mounted_routes", []) or []
        _skipped = getattr(_srv, "_profile_skipped_routes", []) or []

        from ..profile import _PRESET_SPEC_DEFAULT

        _spec_default = _PRESET_SPEC_DEFAULT.get(_profile_name, True)
        _spec_flag = getattr(_cfg, "spec_decode_enabled", None)
        if _spec_flag is not None:
            _spec_status = "ON" if _spec_flag else "OFF"
        else:
            _spec_status = "ON(default)" if _spec_default else "OFF(default)"

        _mem_cfg = getattr(_cfg, "memory", None)
        _mem_tier = getattr(_mem_cfg, "tier", "balanced") if _mem_cfg else "balanced"
        _hard_limit = getattr(_mem_cfg, "hard_limit_gb", None) if _mem_cfg else None

        lines = [
            f"  Profile: {_profile_name}  (modalities: {len(_mounted)} routes mounted, {len(_skipped)} skipped)",
        ]
        if _mounted:
            lines.append(f"    Mounted: {', '.join(sorted(_mounted))}")
        if _skipped:
            lines.append(f"    Skipped: {', '.join(sorted(_skipped))}")
        lines.append(f"    Spec decode: {_spec_status}")
        lines.append(
            f"    Memory tier: {_mem_tier}"
            + (f"  (hard limit: {_hard_limit}G)" if _hard_limit else "")
        )

        import os

        _wired = os.environ.get("MTL_WIRED_LIMIT_MB", "")
        if _wired:
            lines.append(
                f"    Wired limit: {int(_wired) // 1024}G (MTL_WIRED_LIMIT_MB={_wired})"
            )

        print()
        print("\n".join(lines))
        print()
    except Exception as e:
        logging.getLogger(__name__).debug("profile banner skipped: %s", e)


def _build_benchmark_context(target_tokens: int) -> str:
    """Build a deterministic long-context filler for the bench command.

    Used by ``--long-prompt-tokens`` to construct repeatable long
    prompts for TTFT replication runs without depending on a real
    long-context corpus. The block is intentionally generic so the
    measurement targets prefill cost, not semantic difficulty.
    """
    if target_tokens <= 0:
        return ""
    block = (
        "Reference context for long prompt benchmarking. "
        "Fusion MLX evaluates prompt prefill latency, prefix cache behavior, "
        "tool instructions, JSON schema preservation, and model output quality. "
        "The assistant must preserve system instructions and answer only the "
        "final user request after reviewing all reference material. "
    )
    approx_block_tokens = max(1, len(block.split()))
    repeats = max(1, target_tokens // approx_block_tokens)
    return (block * repeats).strip()


def _serve_from_model_dir(args):
    # Released --model-dir multi-model server path (1.0/2.0/3.0 contract).
    # Boots the engine-pool server that auto-discovers every model in the
    # directory via create_app(ServerConfig(model_dir)). Mirrors the
    # pre-Rapid-MLX-migration serve_command; kept as the compat path so
    # existing docs/scripts (`serve --model-dir <dir>`) keep working while
    # the Rapid-MLX single-model path (`serve <model>`) remains the default
    # for explicit model selection.
    import logging

    logger = logging.getLogger(__name__)

    from .. import server
    from ..config import ServerConfig

    # Issue #636: stage the CLI --api-key into the server module global
    # BEFORE create_app constructs Server(), whose __init__ auth-sync reads
    # the bare ``_api_key`` global (server.py:696). The text/audio paths
    # stage it via ``server._api_key = server._resolve_api_key(args.api_key)``
    # (cli_serve.py:102/1154); this model-dir path previously imported only
    # ``create_app`` and skipped the staging, so ``--api-key X`` was dropped
    # and auth fell through to settings.json — /v1/* rejected X with 401.
    server._api_key = server._resolve_api_key(args.api_key)
    create_app = server.create_app

    # Issue #692: --rate-limit 0 must disable the limiter on the model-dir
    # path too. The module-level RateLimiter defaults to enabled=True @ 60rpm
    # (middleware/auth.py). #637 patched only _serve_audio_mode and
    # _stage_server_config; this path built the app directly without calling
    # configure_rate_limiter, so the module default leaked and throttled
    # bursty workloads despite the documented-disable flag. Configure
    # unconditionally and gate on the flag (0 = disabled), matching the
    # other two paths — before create_app, since the limiter is read during
    # app construction the same way server._api_key is (#636 ordering).
    from ..middleware.auth import configure_rate_limiter

    configure_rate_limiter(args.rate_limit, enabled=args.rate_limit > 0)

    host = getattr(args, "host", "0.0.0.0") or "0.0.0.0"
    # Honor an explicit --port 0 (OS-assigned ephemeral port, valid for
    # uvicorn). `or 11434` would collapse 0 -> 11434 since 0 is falsy, so only
    # fall back to the default when the flag was not provided at all.
    # (code-review #75)
    port_raw = getattr(args, "port", None)
    port = 11434 if port_raw is None else int(port_raw)
    config = ServerConfig(host=host, port=port, model_dir=args.model_dir)
    # R-7: pass --profile into ServerConfig
    _profile = getattr(args, "profile", None)
    if _profile:
        config.profile = _profile

    # Pass spec-decode / dflash2 / dspark CLI flags through to the engine
    # pool's scheduler_config. Without this, --enable-dflash2 +
    # --dflash2-drafter-path are silently dropped in --model-dir mode:
    # _convert_scheduler_config reads ServerConfig.scheduler, which defaults
    # to an empty SchedulerConfig, so VLMBatchedEngine._apply_dflash2 /
    # BatchedEngine._apply_dflash2 see an empty drafter path and skip
    # loading. The single-model serve_command path builds scheduler_config
    # at line ~1856; this mirrors the dflash2/dspark/suffix fields for the
    # multi-model path.
    from ..scheduler.config import SchedulerConfig as _SchedCfg

    _sched = _SchedCfg(
        model_name="",
        spec_decode=getattr(args, "spec_decode", "none"),
        dflash_drafter_path=getattr(args, "dflash_drafter_path", "") or "",
        dflash2_drafter_path=getattr(args, "dflash2_drafter_path", "") or "",
        dflash2_block_size=getattr(args, "dflash2_block_size", 5) or 5,
        dflash2_draft_bits=getattr(args, "dflash2_draft_bits", 4),
        dspark_drafter_path=getattr(args, "dspark_drafter_path", "") or "",
        dspark_draft_quant_bits=getattr(args, "dspark_draft_quant_bits", 8),
        enable_suffix_decoding=getattr(args, "suffix_decoding", False),
        suffix_max_draft=getattr(args, "suffix_max_draft", 0),
        suffix_max_suffix_len=getattr(args, "suffix_max_suffix_len", 0),
        suffix_min_confidence=getattr(args, "suffix_min_confidence", 0.0),
        suffix_min_draft_len=getattr(args, "suffix_min_draft_len", 0),
        chunked_prefill=(getattr(args, "chunked_prefill_tokens", 0) or 0) > 0,
        prefill_step_size=getattr(args, "chunked_prefill_tokens", 4096) or 4096,
    )
    config.scheduler = _sched

    logger.info(
        "serve --model-dir=%s host=%s port=%d (multi-model engine-pool server)",
        args.model_dir,
        host,
        port,
    )
    logger.info("serving models from %s on %s:%s", args.model_dir, host, port)

    log_level = getattr(args, "log_level", "INFO")
    if not isinstance(log_level, str):
        log_level = "INFO"
    uvicorn_log_level = log_level.lower()

    from ..server import configure_logging as _configure_logging

    _configure_logging(log_level)

    app = create_app(config)

    _print_profile_banner(server)

    # #569: route --model-dir through the same UDS-aware dispatch the
    # single-model serve path uses (_run_uvicorn → Server.run()), instead
    # of a bare uvicorn.run that treats ``unix:/path`` as a TCP host and
    # fails with Errno 8. Mirror the bind-config stamp so the lifespan
    # "Ready:" banner reports the real listener (uds/host/fd).
    listen_fd = getattr(args, "listen_fd", None)
    host_display, uds_path = _display_host(args.host)

    if uds_path is not None:
        print(
            f"  Starting server on unix socket: {uds_path} "
            "(warming up - this can take a few seconds)"
        )
    elif listen_fd is not None:
        print(
            f"  Starting server on inherited fd {listen_fd} "
            "(warming up — this can take a few seconds)"
        )
    else:
        print(
            f"  Starting server on http://{host_display}:{port} "
            "(warming up — this can take a few seconds)"
        )

    from ..config import get_config

    _cfg = get_config()
    _cfg.bind_host = None
    _cfg.bind_port = None
    _cfg.bind_listen_fd = None
    _cfg.bind_uds = None
    if uds_path is not None:
        _cfg.bind_uds = uds_path
    elif listen_fd is None:
        _cfg.bind_host = host_display
        _cfg.bind_port = port
    else:
        _cfg.bind_listen_fd = listen_fd

    sys.stdout.flush()
    _run_uvicorn(app, args, uvicorn_log_level)


def _boot_guard_checks(args, effective_max_tokens):
    """Run early boot-guard checks (watchdog, extras, dspark fork).

    Returns True if an early fork was taken and the caller should return.
    Calls sys.exit on validation failures.
    """
    import sys

    from .._parent_watchdog import install_parent_watchdog, resolve_expected_ppid

    install_parent_watchdog(resolve_expected_ppid(getattr(args, "watchdog_ppid", None)))

    # [embeddings] extra guard
    if getattr(args, "embedding_model", None):
        from ..embedding import require_mlx_embeddings_or_exit

        require_mlx_embeddings_or_exit()

    # [audio] extra guard
    from ..audio.probe import is_audio_model_alias, require_audio_or_exit

    if is_audio_model_alias(getattr(args, "model", None)):
        require_audio_or_exit(args.model)

    # [dflash] extra guard
    _wants_dflash = getattr(args, "enable_dflash", False) or (
        getattr(args, "spec_decode", "none") == "dflash"
    )
    if _wants_dflash:
        from ..speculative.dflash.eligibility import have_runtime

        if not have_runtime():
            print(
                "\n  Error: --enable-dflash (and --spec-decode dflash) "
                "requires mlx-vlm 0.5.0+ for the DFlash drafter hooks. "
                "Install with: ``pip install 'fusion-mlx[dflash]'``.\n"
            )
            sys.exit(1)

    # [dflash2] extra guard — official z-lab dflash pip pkg (Qwen3.8).
    _wants_dflash2 = getattr(args, "enable_dflash2", False) or (
        getattr(args, "spec_decode", "none") == "dflash2"
    )
    if _wants_dflash2:
        from ..speculative.dflash2 import have_runtime as _dflash2_have_runtime

        if not _dflash2_have_runtime():
            print(
                "\n  Error: --enable-dflash2 (and --spec-decode dflash2) "
                "requires the official dflash pkg. Install with: "
                "``pip install 'fusion-mlx[dflash2]'``.\n"
            )
            sys.exit(1)

        _dflash2_drafter = getattr(args, "dflash2_drafter_path", "")
        if not _dflash2_drafter:
            print(
                "\n  Error: --enable-dflash2 requires --dflash2-drafter-path "
                "(HF id or local path of the DFlash2 draft repo, e.g. "
                "z-lab/Qwen3.8-27B-DFlash2).\n"
            )
            sys.exit(1)

    # [dspark] early fork
    _wants_dspark = getattr(args, "enable_dspark", False) or (
        getattr(args, "spec_decode", "none") == "dspark"
    )
    if _wants_dspark:
        from ..speculative.dspark.eligibility import (
            have_runtime as _dspark_have_runtime,
        )

        if not _dspark_have_runtime():
            print(
                "\n  Error: --enable-dspark (and --spec-decode dspark) requires "
                "dspark-metal (DeepSeek DeepSpec MLX port). Install with "
                "`pip install -e /path/to/dspark-metal` or `uv add dspark-metal`.\n"
            )
            sys.exit(1)

        _dspark_drafter = getattr(args, "dspark_drafter_path", "")
        if not _dspark_drafter:
            print(
                "\n  Error: --enable-dspark requires --dspark-drafter-path "
                "<path-to-converted-mlx-draft>. Convert one with:\n"
                "    dspark-metal-convert deepseek-ai/dspark_qwen3_8b_block7 "
                "--target mlx-community/Qwen3-8B-bf16\n"
            )
            sys.exit(1)

        from ..speculative.dspark.server import run_dspark_server

        if not hasattr(args, "_original_alias") or args._original_alias is None:
            args._original_alias = args.model
        run_dspark_server(
            target_model_repo=args.model,
            drafter_path=_dspark_drafter,
            draft_quant_bits=getattr(args, "dspark_draft_quant_bits", 8),
            host=args.host,
            port=args.port,
            served_model_name=args._original_alias or args.model,
            default_max_tokens=effective_max_tokens,
            uvicorn_log_level=getattr(args, "log_level", "info"),
            enable_thinking=False,
            vlm_dev=getattr(args, "vlm_dev", False),
        )
        return True

    return False


def _autoconfig_parsers(args, logger):
    """Auto-detect tool/reasoning parsers from model name.

    Modifies args.tool_call_parser, args.reasoning_parser, and
    args.enable_auto_tool_choice in place. Calls sys.exit on conflicts.
    """
    import sys

    _opt_out_tool = getattr(args, "no_tool_call_parser", False)
    _opt_out_reasoning = getattr(args, "no_reasoning_parser", False)
    if args.tool_call_parser and _opt_out_tool:
        print(
            "error: --tool-call-parser and --no-tool-call-parser are "
            "mutually exclusive — pick one to override auto-detection.",
            file=sys.stderr,
        )
        sys.exit(2)
    if args.reasoning_parser and _opt_out_reasoning:
        print(
            "error: --reasoning-parser and --no-reasoning-parser are "
            "mutually exclusive — pick one to override auto-detection.",
            file=sys.stderr,
        )
        sys.exit(2)

    _user_explicit_tool_call_parser = bool(args.tool_call_parser)
    if not args.tool_call_parser or not args.reasoning_parser:
        try:
            from ..model_auto_config import detect_model_config

            auto_config = detect_model_config(args.model)
            if auto_config is None:
                # R-27 (#811): detect_model_config returns None for any
                # model whose path matches no alias profile and no regex
                # family pattern. The branches below treat None as "skip",
                # so tool_call_parser / reasoning_parser stay unset — the
                # server boots and serves, but tool calls come back empty
                # and reasoning tags leak into content, which users
                # misread as the model getting dumber. Surface this loudly
                # at startup so the operator knows the model was not
                # classified and tool/reasoning parsing is OFF unless they
                # pass --tool-call-parser / --reasoning-parser explicitly.
                logger.warning(
                    "Auto-config could not classify model '%s' — "
                    "tool-call-parser and reasoning-parser are UNSET. "
                    "Tool calling will silently return empty tool_calls "
                    "and reasoning tags will not be separated. If this "
                    "model supports tool calls or thinking, pass "
                    "--tool-call-parser <name> / --reasoning-parser "
                    "<name> explicitly.",
                    args.model,
                )
            if auto_config:
                if (
                    not args.tool_call_parser
                    and not _opt_out_tool
                    and auto_config.tool_call_parser
                ):
                    args.tool_call_parser = auto_config.tool_call_parser
                    args.enable_auto_tool_choice = True
                    logger.info(
                        f"Auto-configured --tool-call-parser {auto_config.tool_call_parser}"
                    )
                if (
                    not args.reasoning_parser
                    and not _opt_out_reasoning
                    and not args.no_thinking
                    and auto_config.reasoning_parser
                ):
                    args.reasoning_parser = auto_config.reasoning_parser
                    logger.info(
                        f"Auto-configured --reasoning-parser {auto_config.reasoning_parser}"
                    )
        except Exception as e:
            logger.debug(f"Auto-detection failed (non-fatal): {e}")
    if _opt_out_tool:
        logger.info(
            "Tool-call parser auto-detection disabled via --no-tool-call-parser"
        )
    if _opt_out_reasoning:
        logger.info(
            "Reasoning parser auto-detection disabled via --no-reasoning-parser"
        )

    # Misbind check for deepseek_v3 parsers
    try:
        from ..model_auto_config import warn_misbound_deepseek_v3_parser

        misbind_warning = warn_misbound_deepseek_v3_parser(
            args.model, args.tool_call_parser
        )
        if misbind_warning:
            logger.warning(misbind_warning, stacklevel=2)
            if not _user_explicit_tool_call_parser:
                logger.warning(
                    "  Auto-detect note: this binding came from "
                    "AUTO-DETECT, not an explicit --tool-call-parser "
                    "flag. The detect_model_config() regex was fooled "
                    "by the path. Override with --tool-call-parser "
                    "hermes (or whatever your checkpoint actually "
                    "emits) to recover tool-call capability."
                )
    except Exception as e:
        logger.debug(f"deepseek_v3 misbind check failed (non-fatal): {e}")


def _stage_server_config(args, server, logger):
    """Stage all server configuration from CLI args into singletons.

    Returns (cors_origins, gc_control).
    """
    import os
    import sys

    from fusion_mlx.config import get_config as _get_config

    # Alias info for /v1/models
    _get_config().model_alias = getattr(args, "_original_alias", None)

    # R-7: profile gate — --profile flag > settings.json profile field
    _profile_arg = getattr(args, "profile", None)
    if _profile_arg:
        _get_config().profile = _profile_arg
        logger.info("profile from --profile flag: %s", _profile_arg)
    else:
        from .._cli_base import _settings_profile as _sp

        _sp_val = _sp()
        if _sp_val:
            _get_config().profile = _sp_val
            logger.info("profile from settings.json: %s", _sp_val)

    # §6.3: disabled_modules from settings.json
    from .._cli_base import _settings_disabled_modules as _sdm

    _sdm_val = _sdm()
    if _sdm_val:
        _get_config().disabled_modules = _sdm_val
        logger.info("disabled_modules from settings.json: %s", _sdm_val)

    # API key
    server._api_key = server._resolve_api_key(args.api_key)
    _get_config().default_timeout = args.timeout

    # Body-size cap
    _max_body_arg = getattr(args, "max_request_bytes", None)
    if _max_body_arg is not None:
        _get_config().max_request_bytes = max(0, int(_max_body_arg))
    else:
        _env_name = "FUSION_MLX_MAX_REQUEST_BYTES"
        _env = os.environ.get(_env_name, "").strip()
        if _env:
            try:
                _get_config().max_request_bytes = max(0, int(_env))
            except ValueError:
                _get_config().max_request_bytes = 8 * 1024 * 1024
                logger.warning(
                    "%s=%r is not an integer; falling back to the 8 MiB default",
                    _env_name,
                    _env,
                )

    # SSE keepalive
    _sse_env_name = "FUSION_MLX_SSE_KEEPALIVE_SECONDS"
    _sse_env = os.environ.get(_sse_env_name, "").strip()
    if _sse_env:
        try:
            _get_config().sse_keepalive_seconds = max(0.0, float(_sse_env))
        except ValueError:
            logger.warning(
                "%s=%r is not a number; falling back to the 20 s default",
                _sse_env_name,
                _sse_env,
            )
            _get_config().sse_keepalive_seconds = 20.0

    # Body-receive idle timeout
    _apply_body_receive_timeout_env(server, logger=logger)

    # CORS
    cors_origins = server.configure_cors_from_env(args.cors_origins)

    # Rate limit
    from ..middleware.auth import configure_rate_limiter

    # Issue #635: configure unconditionally so 0 disables the limiter.
    configure_rate_limiter(args.rate_limit, enabled=args.rate_limit > 0)

    # GC control
    gc_control = args.gc_control and not args.no_gc_control
    _get_config().gc_control = gc_control

    # mDNS/Bonjour cluster advertising
    _cluster_adv = getattr(args, "cluster_advertise", False)
    _cluster_env = os.environ.get("FUSION_MLX_CLUSTER_ADVERTISE", "").strip()
    if _cluster_env.lower() in ("1", "true", "yes"):
        _cluster_adv = True
    _get_config().cluster_advertise = _cluster_adv

    # No-thinking flag
    _get_config().no_thinking = args.no_thinking

    # System prompt pinning
    _get_config().pin_system_prompt = args.pin_system_prompt

    # Tool calling
    if args.enable_auto_tool_choice and args.tool_call_parser:
        _get_config().enable_auto_tool_choice = True
        _get_config().tool_call_parser = args.tool_call_parser
    else:
        _get_config().enable_auto_tool_choice = False
        _get_config().tool_call_parser = None

    # Generation defaults
    if args.default_temperature is not None:
        _get_config().default_temperature = args.default_temperature
    if args.default_top_p is not None:
        _get_config().default_top_p = args.default_top_p
    if args.default_top_k is not None:
        _get_config().default_top_k = args.default_top_k
    if args.default_min_p is not None:
        _get_config().default_min_p = args.default_min_p
    if args.default_repetition_penalty is not None:
        _get_config().default_repetition_penalty = args.default_repetition_penalty
    if args.default_presence_penalty is not None:
        _get_config().default_presence_penalty = args.default_presence_penalty
    if args.default_frequency_penalty is not None:
        _get_config().default_frequency_penalty = args.default_frequency_penalty

    # Reasoning parser
    if args.reasoning_parser:
        try:
            from ..reasoning import get_parser

            parser_cls = get_parser(args.reasoning_parser)
            _get_config().reasoning_parser = parser_cls()
            _get_config().reasoning_parser_name = args.reasoning_parser
            logger.info(f"Reasoning parser enabled: {args.reasoning_parser}")
        except KeyError as e:
            print(f"Error: {e}")
            sys.exit(1)
        except ImportError as e:
            print(f"Error: Failed to import reasoning module: {e}")
            sys.exit(1)
        except Exception as e:
            print(
                f"Error: Failed to initialize reasoning parser "
                f"'{args.reasoning_parser}': {e}"
            )
            sys.exit(1)
    else:
        _get_config().reasoning_parser = None

    return cors_origins, gc_control


def _print_startup_banner(args, cors_origins, gc_control, logger):
    """Print the startup banner with feature summary."""
    features = []
    if args.enable_auto_tool_choice:
        features.append(f"tools: {args.tool_call_parser}")
    if args.reasoning_parser:
        features.append(f"reasoning: {args.reasoning_parser}")
    auth_feature = _auth_feature_str(args.api_key)
    if auth_feature:
        features.append(auth_feature)
    if args.rate_limit > 0:
        features.append(f"rate-limit: {args.rate_limit}/min")
    if args.cloud_model:
        features.append(f"cloud: {args.cloud_model}")
    if gc_control:
        features.append("gc-control")
    if args.pin_system_prompt:
        features.append("pin-system-prompt")
    if cors_origins:
        features.append(f"cors: {', '.join(cors_origins)}")
    if args.enable_dflash:
        features.append("dflash: single-user")
    if getattr(args, "enable_dflash2", False):
        features.append("dflash2: single-user")
    _profile = getattr(args, "profile", None)
    if not _profile:
        from .._cli_base import _settings_profile as _sp

        _profile = _sp() or "standard"
    features.append(f"profile: {_profile}")
    print()
    print("  🐆 Fusion-MLX")
    print("  ─────────")
    if features:
        print(f"  Features: {', '.join(features)}")
    print(f"  Model: {args.model}")
    if args.mcp_config:
        print(f"MCP config: {args.mcp_config}")
        import os

        os.environ["FUSION_MLX_MCP_CONFIG"] = args.mcp_config


def _apply_mtp_cli_model_type_reconciliation(
    scheduler_config,
    hf_config,
    *,
    has_external_sidecar: bool = False,
):
    # Promote the MTP-eligibility read into scheduler_config.mtp_model_type
    # so the BatchedEngine dispatch gate (dispatch_mtp_inject) sees the vetted
    # target type. Native Qwen3.5/3.6 -> CHAIN promotes model_type; the gemma4
    # sidecar path keys dispatch on the gemma4_unified backbone. Hard-fail when
    # MTP is active but no supported model_type can be resolved - silent
    # fallback would dispatch on an unknown type and crash mid-generation.
    import logging

    logger = logging.getLogger(__name__)
    from fusion_mlx.speculative.mtp.detect import (
        MTPEligibility,
        _detect_mtp_eligibility_verbose,
    )

    # Probe the backbone with has_external_sidecar=False so the real model_type
    # is recovered even when a sidecar overrides eligibility to TREE.
    backbone = _detect_mtp_eligibility_verbose(hf_config, has_external_sidecar=False)
    _CHAIN_SIDECAR_TYPES = {"qwen3_5", "qwen3_5_moe"}
    if has_external_sidecar:
        if backbone.model_type == "gemma4_unified":
            scheduler_config.mtp_model_type = backbone.model_type
            logger.info(
                "MTP sidecar reconciled -> backbone %s (%s)",
                backbone.model_type,
                backbone.reason,
            )
            return
        if backbone.model_type in _CHAIN_SIDECAR_TYPES:
            scheduler_config.mtp_model_type = backbone.model_type
            logger.info(
                "MTP sidecar reconciled -> backbone %s (%s, chain+sidecar)",
                backbone.model_type,
                backbone.reason,
            )
            return
        print(
            "error: --mtp-sidecar requires a gemma4_unified or "
            f"qwen3_5/qwen3_5_moe backbone, "
            f"got model_type={backbone.model_type!r} ({backbone.reason}).",
            file=sys.stderr,
        )
        sys.exit(2)
    if backbone.eligibility is MTPEligibility.CHAIN:
        scheduler_config.mtp_model_type = backbone.model_type
        logger.info(
            "MTP model_type reconciled -> %s (%s)",
            backbone.model_type,
            backbone.reason,
        )
        return
    print(
        "error: MTP enabled but the model is not MTP-eligible "
        "(need a Qwen3.5/3.6 checkpoint with mtp_num_hidden_layers >= 1, "
        f"or --mtp-sidecar <dir>). {backbone.reason}",
        file=sys.stderr,
    )
    sys.exit(2)
