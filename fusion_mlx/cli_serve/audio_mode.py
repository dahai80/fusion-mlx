# SPDX-License-Identifier: Apache-2.0
"""Audio-only serve path + host display + embedding model loader."""

import logging
import sys

from fusion_mlx._cli_base import (
    _apply_body_receive_timeout_env,
    _embedding_not_found_exception_classes,
    _port_preflight_or_die,
    _resolve_embedding_alias,
    _run_uvicorn,
    _uds_path_from_host,
)

logger = logging.getLogger(__name__)


def _display_host(host: str) -> tuple[str, str | None]:
    # R-P1-12 (#0908 audit): consolidated host_display + UDS computation
    # previously duplicated 3× in cli_serve.py.
    host_display = "localhost" if host == "0.0.0.0" else host
    uds_path = _uds_path_from_host(host)
    return host_display, uds_path


def _serve_audio_mode(args, entry) -> None:
    """Bind the audio-only serve path for a resolved registry entry.

    R10-C1 audio-serve-mode. Pre-fix every ``fusion-mlx serve kokoro``
    crash-looped because the text-model boot path was the ONLY path:

    1. ``_ensure_model_downloaded(args.model)`` queried HF for the
       short alias and 404'd — there's no ``hf.co/kokoro`` repo.
    2. Even when the user supplied a full HF id, ``load_model``
       (text path) called ``mlx_lm.load_model`` which expects
       safetensors. Audio repos ship npz/mlx weights, so the loader
       crashed with "no safetensors found".
    3. ``pflash.validate_model_support`` and the parser auto-detection
       both consult ``args.model`` assuming it's a text-LM alias —
       a wrong tool for audio.

    The audio-serve-mode bypasses all of the above:

    * Print the resolved alias -> HF id banner so the operator sees
      the same alias-resolution UX they get for text models.
    * Stamp the resolved HF id on ``args.model`` so the audio routes
      treat it as a known engine (``STT_MODEL_ALIASES`` /
      ``TTS_MODEL_ALIASES`` map both the short and full forms).
    * Capture the alias on ``cfg.model_alias`` so ``/v1/models``
      advertises it.
    * Configure server security knobs (api-key, body-size cap, CORS)
      the SAME way the text path does — audio endpoints share the
      same middleware stack.
    * Skip the text-LM loader. The audio engines are loaded LAZILY
      on the first request by the route handlers (``STTEngine.load``
      / ``TTSEngine.load``), so there's nothing to boot at startup —
      and a Kokoro/Whisper weight download mid-boot would only add
      cold-start latency without buying anything.
    * Run uvicorn with the same FastAPI ``app`` text models use; the
      ``/v1/audio/*`` routes are already mounted on it.
    """
    import os
    import sys

    # Late imports — audio mode runs on the lighter base install +
    # ``[audio]`` extra; we don't want the text-LM engine machinery to
    # boot until / unless it's actually needed.
    from .. import server
    from ..config import get_config
    from ..middleware.auth import configure_rate_limiter
    from ..server import app

    uvicorn_log_level = server.configure_logging(args.log_level)

    # Stamp the resolved model id so the audio routes find the same
    # alias mapping the registry has. Written directly to ServerConfig
    # (#50): ``cfg.model_alias`` is read by ``/v1/models`` to surface
    # the operator-facing alias name; ``cfg.model_name`` / ``cfg.model_path``
    # let /v1/models list the served audio model (codex r1 HIGH #1 follow-up).
    _cfg = get_config()
    if hasattr(args, "_original_alias") and args._original_alias is not None:
        _cfg.model_alias = args._original_alias
    else:
        # No prior alias hop (e.g. user passed a full HF id). Use the
        # short alias from the registry so /v1/models still shows the
        # friendly name, not the bare HF path.
        _cfg.model_alias = entry.alias
    # R11-K / task #258: honor ``--served-model-name`` on the audio
    # path, mirroring the text-mode contract at ``server.load_model``
    # (``cfg.model_name = served_model_name or model_name``). Pre-fix the
    # audio dispatcher ignored the flag, so operators wrapping
    # ``fusion-mlx serve kokoro`` behind a gateway with a stable
    # ``model_name`` saw the raw HF id on ``/v1/models`` and the
    # gateway's model-id allowlist 404'd. The underlying HF id stays
    # on ``cfg.model_path`` (cache dir / engine input), and the friendly
    # short alias stays on ``cfg.model_alias`` so ``/v1/models`` lists
    # both the custom name AND the alias — same wire shape as text.
    _served_name = getattr(args, "served_model_name", None)
    _cfg.model_name = _served_name or entry.hf_id
    _cfg.model_path = entry.hf_id

    # Mirror the text path's security configuration. Audio routes use
    # the SAME middleware stack as chat/embeddings — the same env vars
    # and CLI flags govern auth + body-size caps + CORS. Diverging
    # here would silently weaken the deployment posture for anyone who
    # added ``--api-key`` to their ``fusion-mlx serve kokoro`` command.
    server._api_key = server._resolve_api_key(args.api_key)
    get_config().default_timeout = args.timeout

    _max_body_arg = getattr(args, "max_request_bytes", None)
    if _max_body_arg is not None:
        get_config().max_request_bytes = max(0, int(_max_body_arg))
    else:
        _env = os.environ.get("FUSION_MLX_MAX_REQUEST_BYTES", "").strip()
        if _env:
            try:
                get_config().max_request_bytes = max(0, int(_env))
            except ValueError:
                get_config().max_request_bytes = 8 * 1024 * 1024

    # Body-receive timeout — same env-driven hook the text path uses.
    _apply_body_receive_timeout_env(server)

    # CORS — same friendly default the text path uses.
    server.configure_cors_from_env(args.cors_origins)
    # Issue #635: --rate-limit 0 must disable the limiter. The module-level
    # RateLimiter defaults to enabled=True @ 60rpm, so configure unconditionally
    # and gate on the flag value (0 = disabled, matching the documented semantics).
    configure_rate_limiter(args.rate_limit, enabled=args.rate_limit > 0)

    # CRITICAL: copy the just-set server globals into the
    # ServerConfig singleton the middleware actually reads.
    # ``server.load_model`` does this on the text path (calls
    # ``_sync_config`` after wiring globals); the audio path skips
    # ``load_model`` so we must call it explicitly here. Without this
    # sync the auth middleware reads ``cfg.api_key`` (still ``None``
    # because nothing populated it) instead of ``server._api_key``,
    # so ``fusion-mlx serve kokoro --api-key SECRET`` would silently
    # accept unauthenticated /v1/audio/* requests. Codex r1 HIGH #1.
    server._sync_config()

    # Task #292: register ``/v1/audio/*`` routes. ``cfg.model_alias``
    # / ``cfg.model_name`` were just stamped with the registry-known
    # audio alias above, so the registry-driven branch of
    # :func:`register_audio_routes_if_enabled` is what fires here — the
    # ``--enable-audio`` flag is for the text-mode-with-audio escape
    # hatch, not the audio-mode boot path. Skipping the call would leave
    # text-only behaviour on an audio server, with /v1/audio/* returning
    # 404 (the exact symmetric mistake the unconditional pre-fix made on
    # text-only servers). Idempotent — safe even if a future refactor
    # adds a second call site.
    server.register_audio_routes_if_enabled()

    # Print the resolution banner so the operator sees what loaded.
    family_tag = f"[audio:{entry.type}]"
    shown_alias = getattr(args, "_original_alias", args.model)
    print()
    print(f"  Audio mode: {shown_alias} → {entry.hf_id} {family_tag}")
    if entry.type == "tts" and entry.default_voice:
        print(f"  Default voice: {entry.default_voice}")
    if entry.type == "stt" and entry.languages:
        print(f"  Languages: {entry.languages}")
    print(
        "  Audio engines load lazily on the first /v1/audio/* request "
        "(no boot-time weight download)."
    )

    # R11-K / task #258: honor ``--embedding-model`` on the audio
    # path. The shared helper (``_load_embedding_model_or_exit``) is
    # intentionally orthogonal to the text-LM engine — it only goes
    # through ``server.load_embedding_model`` — so audio + embedding
    # compose cleanly: the audio engines stay lazy on /v1/audio/*
    # while the embeddings sidecar serves /v1/embeddings from the
    # same FastAPI app. Mirrors the text-mode call site at
    # ``serve_command`` (post-``load_model``); see the helper's
    # docstring "Audio-mode integration" note (R11-K coordination)
    # — single source of truth for the install + alias + error wrap.
    # Ordered after the banner so the operator sees the audio model
    # banner FIRST (matches the text-mode visual ordering where the
    # ``Model:`` line prints before ``Pre-loading embedding model:``).
    if getattr(args, "embedding_model", None):
        _load_embedding_model_or_exit(args, server.load_embedding_model)

    # Stamp the bind source-of-truth so the lifespan "Ready:" banner
    # prints the right URL. Mirrors the text-path block.
    host_display, uds_path = _display_host(args.host)
    listen_fd = getattr(args, "listen_fd", None)

    # Port preflight — same friendly "port already in use" probe the
    # text path runs. Skip in --listen-fd mode (the supervisor owns
    # the socket; binding here would race). Mirrors the rationale on
    # the text-path call site. --host unix: also skips (#351: no
    # TCP port to probe).
    if listen_fd is None and uds_path is None:
        _port_preflight_or_die(args.host, args.port, model=args.model)

    if uds_path is not None:
        print(
            f"  Starting server on unix socket: {uds_path} "
            "(audio routes ready immediately)"
        )
    elif listen_fd is not None:
        print(
            f"  Starting server on inherited fd {listen_fd} "
            "(audio routes ready immediately)"
        )
    else:
        print(
            f"  Starting server on http://{host_display}:{args.port} "
            "(audio routes ready immediately)"
        )

    from fusion_mlx._version_check import print_staleness_warning_if_any

    print_staleness_warning_if_any()
    print()

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

    # Use sys.stdout.flush so the banner lands before uvicorn's own
    # startup logs interleave — operators expect to see the audio
    # banner FIRST.
    sys.stdout.flush()

    _run_uvicorn(app, args, uvicorn_log_level)


def _load_embedding_model_or_exit(args, load_fn) -> None:
    """Pre-load ``--embedding-model`` with the H-08 install guard and
    the D-EMBED-ALIAS alias-resolution + clean error-wrapping path.

    Lifted out of ``serve_command`` so the dispatch sequence can be
    unit-tested without booting the full engine — the pr_validate
    codex r0 BLOCKING #1 noted that the in-test exercising the
    behaviour at module scope didn't actually invoke the CLI path,
    so a regression that removed the alias resolution would pass.
    Calling this helper directly gives the test surgical coverage.

    ``args`` mirrors the ``argparse.Namespace`` shape — only
    ``embedding_model`` is read and (on alias hit) mutated.
    ``load_fn`` is the embedding-loader callable
    (``fusion_mlx.server.load_embedding_model``) — passed in so tests
    can mock it without monkeypatching the server module.

    Failure modes that exit cleanly:

    * Missing ``[embeddings]`` extra → ``sys.exit(2)`` with install
      hint (H-08, ``require_mlx_embeddings_or_exit``).
    * Loader raises ``ModelNotFoundError`` / ``RepositoryNotFoundError``
      / ``FileNotFoundError`` → ``sys.exit(1)`` with an actionable
      hint pointing at the alias registry and the canonical HF id
      format. Any OTHER ``Exception`` re-raises so unrelated bugs
      surface with their real trace.

    Audio-mode integration (deferred #258 / r11-K coordination): if
    ``_serve_audio_mode`` ever needs to honour ``--embedding-model``
    (e.g. an STT lane that exposes embeddings of the transcript), the
    audio path MUST route through this helper rather than duplicate
    the guard logic. The probe + alias resolve + error-wrap are a
    single source of truth — a second copy in the audio dispatcher
    would drift on the next H-08/H-09/H-13 follow-up. The helper is
    intentionally independent of the text-LM serve path so the audio
    boot path can call it without dragging in the chat-engine
    machinery.
    """
    from ..embedding import require_mlx_embeddings_or_exit

    require_mlx_embeddings_or_exit()

    original_embed = args.embedding_model
    resolved_embed, did_resolve = _resolve_embedding_alias(original_embed)
    if did_resolve:
        print(f"  Embedding alias: {original_embed} → {resolved_embed}")
        args.embedding_model = resolved_embed
    print(f"Pre-loading embedding model: {args.embedding_model}")
    # Bind to the concrete not-found classes the loader can raise
    # (mlx_embeddings.utils.ModelNotFoundError +
    # huggingface_hub.errors.RepositoryNotFoundError/EntryNotFoundError
    # + stdlib FileNotFoundError for the local-path branch). Any OTHER
    # exception class falls through unchanged so unrelated bugs (corrupt
    # safetensors mid-load, Metal OOM, schema mismatch) surface with
    # their real trace — pr_validate codex r1 NIT closure (the prior
    # ``"not found"`` substring match was too loose).
    not_found_exc_classes = _embedding_not_found_exception_classes()
    try:
        load_fn(args.embedding_model, lock=True)
    except not_found_exc_classes as exc:
        print(
            f"\n  Error: --embedding-model '{original_embed}' could not "
            f"be loaded ({type(exc).__name__}: {exc})."
        )
        print(
            "  Tip: use a registered embedding alias (see "
            "``fusion-mlx ls`` for the list — e.g. "
            "``embeddinggemma-300m-6bit``) or pass the full "
            "HuggingFace id (e.g. "
            "``mlx-community/embeddinggemma-300m-6bit``).\n"
        )
        sys.exit(1)
    except NotImplementedError as exc:
        # FC-1 / EF-10 (#0907 audit): server.load_embedding_model raises
        # NotImplementedError to redirect to POST /v1/embeddings (CLI
        # pre-load of embedding models is no longer the supported path).
        # Without this handler the exception escaped _load_embedding_model_or_exit
        # and crashed boot with a raw traceback. Exit cleanly with the guidance.
        print(
            f"\n  Error: --embedding-model pre-load is not supported "
            f"({type(exc).__name__}: {exc})."
        )
        print(
            "  Tip: load embedding models via POST /v1/embeddings (the "
            "pool lazy-loads on first request) or the CLI "
            "'fusion load <model>'. See GET /v1/models for available models.\n"
        )
        sys.exit(2)
    print(f"Embedding model loaded: {args.embedding_model}")
