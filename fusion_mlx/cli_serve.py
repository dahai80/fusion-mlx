#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CLI serve and bench commands for fusion-mlx."""

import os
import sys

from fusion_mlx._cli_base import (
    _apply_body_receive_timeout_env,
    _auth_feature_str,
    _embedding_not_found_exception_classes,
    _port_preflight_or_die,
    _print_unknown_model_help,
    _resolve_audio_model_for_serve,
    _resolve_embedding_alias,
    _run_uvicorn,
)


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
    * Capture the alias on ``server._model_alias`` so ``/v1/models``
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
    from . import server
    from .middleware.auth import configure_rate_limiter
    from .server import app

    uvicorn_log_level = server.configure_logging(args.log_level)

    # Stamp the resolved model id so the audio routes find the same
    # alias mapping the registry has. ``server._model_alias`` is read
    # by ``/v1/models`` to surface the operator-facing alias name;
    # ``server._model_name`` / ``server._model_path`` populate
    # ``ServerConfig.model_name`` / ``model_path`` so /v1/models lists
    # the served audio model (codex r1 HIGH #1 follow-up).
    if hasattr(args, "_original_alias") and args._original_alias is not None:
        server._model_alias = args._original_alias
    else:
        # No prior alias hop (e.g. user passed a full HF id). Use the
        # short alias from the registry so /v1/models still shows the
        # friendly name, not the bare HF path.
        server._model_alias = entry.alias
    # R11-K / task #258: honor ``--served-model-name`` on the audio
    # path, mirroring the text-mode contract at ``server.load_model``
    # (``_model_name = served_model_name or model_name``). Pre-fix the
    # audio dispatcher ignored the flag, so operators wrapping
    # ``fusion-mlx serve kokoro`` behind a gateway with a stable
    # ``model_name`` saw the raw HF id on ``/v1/models`` and the
    # gateway's model-id allowlist 404'd. The underlying HF id stays
    # on ``_model_path`` (cache dir / engine input), and the friendly
    # short alias stays on ``_model_alias`` so ``/v1/models`` lists
    # both the custom name AND the alias — same wire shape as text.
    _served_name = getattr(args, "served_model_name", None)
    server._model_name = _served_name or entry.hf_id
    server._model_path = entry.hf_id

    # Mirror the text path's security configuration. Audio routes use
    # the SAME middleware stack as chat/embeddings — the same env vars
    # and CLI flags govern auth + body-size caps + CORS. Diverging
    # here would silently weaken the deployment posture for anyone who
    # added ``--api-key`` to their ``fusion-mlx serve kokoro`` command.
    server._api_key = server._resolve_api_key(args.api_key)
    server._default_timeout = args.timeout

    _max_body_arg = getattr(args, "max_request_bytes", None)
    if _max_body_arg is not None:
        server._max_request_bytes = max(0, int(_max_body_arg))
    else:
        _env = os.environ.get("FUSION_MLX_MAX_REQUEST_BYTES", "").strip()
        if _env:
            try:
                server._max_request_bytes = max(0, int(_env))
            except ValueError:
                server._max_request_bytes = 8 * 1024 * 1024

    # Body-receive timeout — same env-driven hook the text path uses.
    _apply_body_receive_timeout_env(server)

    # CORS — same friendly default the text path uses.
    server.configure_cors_from_env(args.cors_origins)
    if args.rate_limit > 0:
        server._rate_limiter = configure_rate_limiter(args.rate_limit, enabled=True)

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

    # Task #292: register ``/v1/audio/*`` routes. ``server._model_alias``
    # / ``server._model_name`` were just stamped with the registry-known
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
    host_display = "localhost" if args.host == "0.0.0.0" else args.host
    listen_fd = getattr(args, "listen_fd", None)

    # Port preflight — same friendly "port already in use" probe the
    # text path runs. Skip in --listen-fd mode (the supervisor owns
    # the socket; binding here would race). Mirrors the rationale on
    # the text-path call site.
    if listen_fd is None:
        _port_preflight_or_die(args.host, args.port, model=args.model)

    if listen_fd is not None:
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
    from fusion_mlx.config import get_config

    print_staleness_warning_if_any()
    print()

    _cfg = get_config()
    _cfg.bind_host = None
    _cfg.bind_port = None
    _cfg.bind_listen_fd = None
    if listen_fd is None:
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
    from .embedding import require_mlx_embeddings_or_exit

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
    print(f"Embedding model loaded: {args.embedding_model}")


def _check_disk_space(model_name: str, force: bool = False) -> None:
    """Verify there's enough disk space to download the model.

    Queries HuggingFace for the repo size and compares with available space
    on the resolved HF cache filesystem (respects ``HF_HOME`` /
    ``HF_HUB_CACHE`` rather than the hard-coded ``~/.cache/huggingface``).

    Behaviour:

    - Model is already a local path → return.
    - ``config.json`` is in the cache → assume already downloaded → return.
    - HF API call fails (offline, gated repo, etc.) → return silently. The
      loader's 404/auth handlers will surface the real error if there is one.
    - Determined size and disk is insufficient → print actionable error
      and ``sys.exit(1)``. ``force=True`` warns instead of aborting.

    The previous behaviour was to print a soft warning then continue. Users
    burned 30+ minutes downloading a 141 GB model on an 8.8 GB disk before
    HF Hub crashed with ``OSError: No space left on device``.
    """
    # Skip if model is a local path that already exists.
    if os.path.exists(model_name):
        return

    # Skip if model is already in the HF cache.
    try:
        from huggingface_hub import try_to_load_from_cache

        cached = try_to_load_from_cache(model_name, "config.json")
        if isinstance(cached, str) and os.path.exists(cached):
            return
    except Exception:
        pass

    # Query HF for repo size + free space on the actual HF cache filesystem.
    try:
        from huggingface_hub import model_info
        from huggingface_hub.constants import HF_HUB_CACHE

        info = model_info(model_name, files_metadata=True)
        model_size_bytes = sum(
            (s.size or 0)
            for s in (getattr(info, "siblings", None) or [])
            if hasattr(s, "size")
        )
        if model_size_bytes == 0:
            return  # Can't determine size — skip rather than guess.

        # statvfs needs an existing path; HF_HUB_CACHE may not exist yet on
        # a fresh install. Walk up to the first ancestor that does.
        # Resolve to absolute up front so a relative HF_HUB_CACHE doesn't
        # short-circuit to CWD when an ancestor walk hits ".".
        probe = os.path.abspath(HF_HUB_CACHE) if HF_HUB_CACHE else ""
        while probe and not os.path.exists(probe):
            parent = os.path.dirname(probe)
            if parent == probe:
                break
            probe = parent
        if not probe or not os.path.exists(probe):
            probe = os.path.expanduser("~")

        stat = os.statvfs(probe)
        available_bytes = stat.f_bavail * stat.f_frsize

        # ~10% headroom for temp files during xet_get / move-into-place.
        required_bytes = int(model_size_bytes * 1.1)
        if available_bytes >= required_bytes:
            return

        model_size_gb = model_size_bytes / (1024**3)
        available_gb = available_bytes / (1024**3)
        need_to_free_gb = (required_bytes - available_bytes) / (1024**3)

        print()
        print("  Error: Insufficient disk space for download.")
        print(f"    Model size:    {model_size_gb:>7.1f} GB")
        print(f"    Free space:    {available_gb:>7.1f} GB  ({probe})")
        print(f"    Need to free:  {need_to_free_gb:>7.1f} GB")
        print()
        print("  Suggestions:")
        print("    - Free disk space, or set HF_HOME to a drive with more room")
        print("    - Pick a smaller variant: fusion-mlx models")
        if not force:
            print(
                "    - Bypass this check (download will likely fail mid-way): "
                "--force-disk-check"
            )
            print()
            sys.exit(1)
        # ``force=True``: warn loudly, let the user proceed at their own risk.
        print("  --force-disk-check set — proceeding anyway.")
        print()
    except SystemExit:
        raise
    except Exception:
        # Network / auth / etc. failures are non-critical — fall through to
        # the loader's own error handling rather than blocking startup on a
        # flaky HF metadata query.
        pass


def _gather_kv_cache_dtype_inputs(model_name: str) -> tuple[dict | None, dict | None]:
    """Best-effort collect the inputs ``resolve_kv_cache_dtype`` consumes.

    R15 task #300: the safelist that downgrades int4 → bf16 for sliding-
    window and MLA models needs the HF ``config.json`` (``sliding_window``,
    ``q_lora_rank`` / ``kv_lora_rank``) plus any alias-level
    ``sliding_window`` / ``is_mla`` hints. We intentionally avoid
    network fetches here — both signals come from data that's already
    on disk (aliases.json) or that will be downloaded for the model
    load anyway (HF config). If neither is available (offline, gated
    repo, brand-new release), the substring fallback in
    :func:`fusion_mlx.kv_cache_dtype.resolve_kv_cache_dtype` still catches
    the documented families by name.

    Returns:
        A ``(hf_config, alias_metadata)`` pair. Either or both may be
        ``None`` when the inputs aren't reachable.
    """
    hf_cfg: dict | None = None
    alias_meta: dict | None = None

    # Alias metadata — pull straight from the loaded profile so a
    # contributor-curated override (``"sliding_window": true``) wins
    # over the substring heuristic.
    try:
        from .model_aliases import resolve_profile

        profile = resolve_profile(model_name)
        if profile is not None:
            alias_meta = {
                "hf_path": getattr(profile, "hf_path", None),
                # AliasProfile doesn't have ``sliding_window`` / ``is_mla``
                # fields today (R15 #300 intentionally avoids a frozen-
                # dataclass schema bump). The substring fallback covers
                # the in-tree aliases; we leave the hook here so a
                # future closed-key extension picks them up automatically.
                "sliding_window": getattr(profile, "sliding_window", False),
                "is_mla": getattr(profile, "is_mla", False),
            }
    except Exception:
        # Alias resolution must never block server start. The substring
        # fallback covers the documented families even with no profile.
        alias_meta = None

    # HF config — read from the local HF cache only. We're inside the
    # serve preflight path so a network round-trip would be cheap (the
    # model load follows immediately anyway), but staying file-local
    # keeps this helper safe to call in tests and air-gapped installs.
    try:
        import json as _json
        import os as _os

        from huggingface_hub import try_to_load_from_cache as _cache_lookup

        hf_path = (alias_meta or {}).get("hf_path") or model_name
        if hf_path:
            cached = _cache_lookup(repo_id=hf_path, filename="config.json")
            if cached and _os.path.exists(cached):
                with open(cached) as fh:
                    hf_cfg = _json.load(fh)
    except Exception:
        hf_cfg = None

    return hf_cfg, alias_meta


def _check_memory_capacity(model_name: str) -> None:
    """Pre-flight memory check — warn loudly if loading this model is
    likely to push unified memory past the danger threshold.

    On low-memory Apple Silicon (especially Mac mini M4 24 GB), loading
    a model that forces unified memory past ~85% of total can trip the
    iBoot AMCC async-abort firmware path and **kernel-panic the entire
    machine** rather than raise a userspace OOM. See issue #324.

    This check is best-effort: it warns the user, never aborts. If we
    can't read the model size (offline / gated repo), or psutil isn't
    importable, fall through silently — the existing loader paths still
    surface real failures.

    Working-set estimate is ``model_size * 1.5`` for a typical short
    chat workload — covers KV cache, activations, and OS reserve.
    Long-context (32k+) or high-concurrency serving pushes the
    multiplier higher; the warning under-predicts in those modes
    rather than over-predicts, so a user who configures aggressively
    may still crash. We err on the side of warning earlier than later.

    **Pressure formula uses already-used memory** rather than just
    ``working / total``. The kernel panic fires on absolute unified-
    memory pressure, so a 10 GB model on a 24 GB Mac that already has
    8 GB used by macOS + Chrome lands at projected ``(8 + 15) / 24``
    = 95.8% — kernel-panic territory. The naive formula would have
    reported only 62.5% and stayed silent.
    """
    try:
        import psutil
    except Exception:
        return

    # Resolve model size in bytes — local path, then HF cache, then HF API.
    model_size_bytes = 0
    try:
        if os.path.isdir(model_name):
            for root, _dirs, files in os.walk(model_name):
                for f in files:
                    try:
                        model_size_bytes += os.path.getsize(os.path.join(root, f))
                    except OSError:
                        continue
        else:
            from huggingface_hub import model_info, try_to_load_from_cache

            cached = try_to_load_from_cache(model_name, "config.json")
            if isinstance(cached, str) and os.path.exists(cached):
                # Already-downloaded model: walk the snapshot directory.
                snapshot_dir = os.path.dirname(cached)
                for root, _dirs, files in os.walk(snapshot_dir):
                    for f in files:
                        try:
                            model_size_bytes += os.path.getsize(os.path.join(root, f))
                        except OSError:
                            continue
            else:
                info = model_info(model_name, files_metadata=True)
                model_size_bytes = sum(
                    (s.size or 0)
                    for s in (getattr(info, "siblings", None) or [])
                    if hasattr(s, "size")
                )
    except Exception:
        return  # Network / auth failure — fall through.

    if model_size_bytes <= 0:
        return

    try:
        vm = psutil.virtual_memory()
        total_ram_bytes = vm.total
        available_ram_bytes = vm.available
    except Exception:
        return

    if total_ram_bytes <= 0:
        return

    # Projected post-load pressure: already-used + estimated working set.
    # ``available`` is psutil's best estimate of "memory we can grab without
    # swapping," which on macOS includes inactive + cached pages that the
    # kernel will reclaim under pressure. ``total - available`` is therefore
    # a tighter "currently-pinned" floor than ``total - free``.
    estimated_working = int(model_size_bytes * 1.5)
    used_ram_bytes = max(0, total_ram_bytes - available_ram_bytes)
    projected_use = used_ram_bytes + estimated_working
    ratio = projected_use / total_ram_bytes
    if ratio < 0.65:
        return  # Comfortable headroom — no warning.

    model_gb = model_size_bytes / (1024**3)
    working_gb = estimated_working / (1024**3)
    used_gb = used_ram_bytes / (1024**3)
    total_gb = total_ram_bytes / (1024**3)

    is_tty = sys.stdout.isatty() and "NO_COLOR" not in os.environ
    YELLOW = "\x1b[33m" if is_tty else ""
    RED = "\x1b[31m" if is_tty else ""
    BOLD = "\x1b[1m" if is_tty else ""
    DIM = "\x1b[2m" if is_tty else ""
    RESET = "\x1b[0m" if is_tty else ""

    print()
    if ratio >= 0.85:
        print(
            f"  {RED}{BOLD}!! Memory pressure warning:{RESET} "
            f"this model is likely too large for your hardware."
        )
        print(
            f"  {DIM}Continuing may trigger a macOS kernel panic "
            f"(see issue #324).{RESET}"
        )
    else:
        print(
            f"  {YELLOW}{BOLD}Memory pressure note:{RESET} "
            f"this model uses a large fraction of system RAM."
        )
    print()
    print(f"    Model on disk:           {model_gb:>6.1f} GB")
    print(
        f"    Est. working set:        {working_gb:>6.1f} GB  "
        f"{DIM}(model x 1.5 — short-chat workload; long-context serving will use more){RESET}"
    )
    print(f"    Currently used by OS:    {used_gb:>6.1f} GB")
    print(
        f"    Total system RAM:        {total_gb:>6.1f} GB  "
        f"({ratio * 100:.0f}% projected utilization)"
    )
    print()
    if ratio >= 0.85:
        print("  Apple Silicon firmware can panic the whole system rather than")
        print("  raise an OOM error when unified-memory pressure exceeds the")
        print("  iBoot AMCC threshold. Recommended actions:")
        print()
        print("    - Close other apps to free RAM, or")
        print("    - Pick a smaller model:    fusion-mlx models")
        print(
            "    - Or lower memory headroom: "
            "fusion-mlx serve <model> --gpu-memory-utilization 0.75"
        )
    else:
        print(
            "  If you see crashes or kernel panics, try: --gpu-memory-utilization 0.85"
        )
    print()


def _try_mirror_prefetch(model_name: str) -> bool:
    """Pre-fetch a HuggingFace repo via R2-first / HF-fallback (per file).

    Delegates to :func:`fusion_mlx._mirror.download_with_mirror_fallback`.
    Returns ``True`` if the snapshot is fully populated (any mix of R2
    and HF). Returns ``False`` if the caller should fall through to the
    plain ``snapshot_download(repo_id)`` path (catalog unavailable for
    catalog-only paths, or one or more files failed both R2 and HF).

    Set ``FUSION_MLX_MODEL_MIRROR=""`` to disable R2 entirely and force
    HuggingFace.

    Codex round-6 BLOCKING #2: the mirror module already returns
    ``False`` on every recoverable network/cache error, so the only
    catch worth doing here is ``ImportError`` (mirror module disabled
    or missing in a minimal install). Programmer errors propagate so
    bugs in the mirror module surface as real stack traces instead of
    silently routing to ``snapshot_download``.
    """
    try:
        from fusion_mlx._mirror import download_with_mirror_fallback
    except ImportError:
        # Mirror module not available (minimal-deps install or
        # deliberately removed). Use the legacy HF path.
        return False
    return download_with_mirror_fallback(model_name)


def _ensure_model_downloaded(model_name: str) -> None:
    """Pre-fetch a model in the foreground so HF's tqdm progress is visible.

    Used by ``fusion-mlx chat``: the chat REPL spawns ``serve`` as a
    subprocess with stdout/stderr redirected to a log file. If the model
    isn't cached, the user sees a silent multi-minute hang while several
    GB downloads behind the log. Calling ``snapshot_download`` here first
    surfaces the standard HF progress bars on the user's terminal, then
    the spawned server starts as a cache hit.

    No-op when the model is already cached, when ``model_name`` is a local
    path, or when the HF lookup fails (let the loader's own error paths
    handle it).
    """
    if os.path.exists(model_name):
        return
    # Bare names (e.g. ``Qwen3.5-9B-4bit``) already cached in the omlx /
    # fusion-mlx / HF-snapshot model dirs need no HuggingFace fetch. Reuse
    # the same resolver the server's load_model uses so ``serve --model
    # <bare-name>`` does not attempt a doomed HF lookup for a model that
    # is already on disk locally. Slash-names and genuine HF repos fall
    # through unchanged (resolver returns them as-is when no local path
    # exists).
    try:
        from . import server as _server_mod

        resolved = _server_mod._resolve_single_model_path(model_name)
        if os.path.exists(resolved):
            return
    except Exception:
        pass
    # Reuse the same weight-file-presence probe as ``is_repo_cached``:
    # the older ``try_to_load_from_cache('config.json')`` check
    # short-circuits on a partial cache (metadata downloaded, weight
    # shards still in flight), letting the spawned ``serve`` quietly
    # finish the download inside its logfile. Codex round-3 BLOCKING #2.
    try:
        from fusion_mlx._download_gate import is_repo_cached

        if is_repo_cached(model_name):
            return
    except Exception:
        # Probe failed (filesystem permission error, unexpected layout) —
        # fall through to the heavy snapshot_download path; HF will
        # short-circuit on its own cache check if the repo really is
        # fully present.
        pass

    # Disk-space gate: a 20 GB partial download that fails on the last
    # shard wastes the user's time. ``_check_disk_space`` queries HF for
    # the repo size and aborts with a clear message + exit(1) if there
    # isn't enough room on the resolved HF cache filesystem.
    _check_disk_space(model_name)

    # User-configured mirror path (R2/S3/any HTTP host). When the mirror
    # serves every file the repo declares, populate the HF cache layout
    # ourselves and skip snapshot_download. On any miss we fall through
    # to the normal HuggingFace download below.
    if _try_mirror_prefetch(model_name):
        return

    try:
        from huggingface_hub import model_info, snapshot_download

        size_gb = 0.0
        try:
            info = model_info(model_name, files_metadata=True)
            size_bytes = sum(
                (s.size or 0)
                for s in (getattr(info, "siblings", None) or [])
                if hasattr(s, "size")
            )
            size_gb = size_bytes / (1024**3)
        except Exception:
            pass

        is_tty = sys.stdout.isatty() and "NO_COLOR" not in os.environ
        BOLD = "\x1b[1m" if is_tty else ""
        DIM = "\x1b[2m" if is_tty else ""
        RESET = "\x1b[0m" if is_tty else ""
        if size_gb > 0:
            print(
                f"\n  {BOLD}First-time download{RESET} — "
                f"fetching {model_name} {DIM}(~{size_gb:.1f} GB){RESET} "
                "from HuggingFace ..."
            )
        else:
            print(
                f"\n  {BOLD}First-time download{RESET} — "
                f"fetching {model_name} from HuggingFace ..."
            )

        snapshot_download(model_name)
        print()
    except SystemExit:
        # _check_disk_space aborts via sys.exit(1) — let it through.
        raise
    except Exception as e:
        # Definitive 404s are surfaced so callers (e.g. ``/model bogus``)
        # can refuse fast instead of spawning a doomed serve subprocess
        # that fails after ``--ready-timeout``. Other transient errors
        # (network, auth) fall through silently — the spawned server's
        # own loader will retry and surface a real error if needed.
        from huggingface_hub.utils import RepositoryNotFoundError

        if isinstance(e, RepositoryNotFoundError) or "404" in str(e):
            raise RuntimeError(f"Model {model_name!r} not found on HuggingFace") from e
        print(f"\n  Pre-download skipped ({type(e).__name__}); server will retry.")


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

    from .config import ServerConfig
    from .server import create_app

    host = getattr(args, "host", "0.0.0.0") or "0.0.0.0"
    # Honor an explicit --port 0 (OS-assigned ephemeral port, valid for
    # uvicorn). `or 8000` would collapse 0 -> 8000 since 0 is falsy, so only
    # fall back to the default when the flag was not provided at all.
    # (code-review #75)
    port_raw = getattr(args, "port", None)
    port = 8000 if port_raw is None else int(port_raw)
    config = ServerConfig(host=host, port=port, model_dir=args.model_dir)

    logger.info(
        "serve --model-dir=%s host=%s port=%d (multi-model engine-pool server)",
        args.model_dir,
        host,
        port,
    )
    print(f"fusion-mlx: serving models from {args.model_dir} on {host}:{port}")

    app = create_app(config)

    import uvicorn

    log_level = getattr(args, "log_level", "INFO")
    if not isinstance(log_level, str):
        log_level = "INFO"
    uvicorn.run(app, host=host, port=port, log_level=log_level.lower())


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
    from . import _mlx_compat as _mlx_compat

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

    # FusionMLX macOS app / omlx-style launch: `serve --base-path <dir>` serves
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
        print("  fusion-mlx serve --model Qwen3-4B-Q4_K_M --port 8000")
        print("  fusion-mlx serve --model-dir ~/.omlx/models --port 11435")
        print("  fusion-mlx serve --base-path ~/.fusion-mlx --port 8000")
        sys.exit(1)

    # Parent-PID watchdog (rapid-desktop issue #449): if the supervisor
    # passed its own PID via ``--watchdog-ppid`` or
    # ``$FUSION_MLX_WATCHDOG_PPID``, spawn a daemon thread that polls
    # ``os.getppid()`` every 2 s. When the parent dies (re-parented to
    # launchd / init), the watchdog sends SIGTERM to ourselves so the
    # FastAPI lifespan can flush + release the model — falling back to
    # SIGKILL after a 5 s grace. Installed at the top of serve_command
    # so it covers BOTH the text-LM path and the audio-mode fork below,
    # AND so it arms before the (potentially multi-minute) model
    # download — an operator who kills the desktop mid-download still
    # gets a clean reap on the sidecar. No-op when no supervisor PID
    # was passed (default).
    from ._parent_watchdog import install_parent_watchdog, resolve_expected_ppid

    install_parent_watchdog(resolve_expected_ppid(getattr(args, "watchdog_ppid", None)))

    _arg_max_tokens = getattr(args, "max_tokens", None)
    _max_tokens_is_explicit = _arg_max_tokens is not None
    effective_max_tokens = _arg_max_tokens if _arg_max_tokens is not None else 32768

    # F-H08-INCOMPLETE: the ``[embeddings]`` extra-required guard MUST
    # fire first thing in ``serve_command`` — before
    # ``prompt_upgrade_if_available`` (which may exit 0 on user
    # decline), before ``_ensure_model_downloaded`` (which can take
    # minutes on a cold cache), and well before the startup banner
    # gets printed. Pre-fix the check lived deeper in the function so
    # the operator saw the alias-resolved log line, the startup banner,
    # the feature list, AND the model id BEFORE the
    # "requires the [embeddings] extra" error and ``sys.exit(2)``,
    # which read as a successful boot followed by a mysterious failure
    # — Diego logged this as a warning-and-fall-through bug because
    # the banner masked the actual exit. Hoisting the probe to the
    # very top of ``serve_command`` puts the error first, with no
    # banner output before it. ``mlx_embeddings`` import stays lazy so
    # the base install (no ``[embeddings]`` extra) keeps booting.
    if getattr(args, "embedding_model", None):
        from .embedding import require_mlx_embeddings_or_exit

        require_mlx_embeddings_or_exit()

    # R6-H4 (Eva 0.8.7 dogfood): same boot-guard shape for audio aliases.
    # ``mlx-audio`` lives behind the ``[audio]`` extra; pre-fix
    # ``fusion-mlx serve kokoro`` (or whisper/parakeet/chatterbox/...) on
    # a base install printed the startup banner, opened the port, and
    # only crashed on the first audio request (the in-route lane probe).
    # That looked like "successful boot, broken inference" instead of
    # the obvious "you need the [audio] extra". Probe at flag-parse
    # time so the operator sees an actionable hint with rc=2 before
    # any download / banner output, mirroring r5-C's UI-TARS guard.
    #
    # Recognition is alias-substring based (``whisper``, ``parakeet``,
    # ``kokoro``, ``chatterbox``, ``vibevoice``, ``voxcpm``) so the
    # quantised variants (``kokoro-4bit``) and HF-style ids
    # (``mlx-community/Kokoro-82M-bf16``) trip it the same way bare
    # aliases do. A model name that doesn't match an audio token falls
    # through unchanged — text/vision/embedding models never see this
    # probe.
    from .audio.probe import is_audio_model_alias, require_audio_or_exit

    if is_audio_model_alias(getattr(args, "model", None)):
        require_audio_or_exit(args.model)

    # 0.9.2 dogfood (parallels the [embeddings]/[vision]/[audio] guards
    # immediately above): ``--enable-dflash`` and the equivalent
    # ``--spec-decode dflash`` both depend on the optional ``mlx-vlm``
    # bridge that ships in the ``[dflash]`` extra. Pre-0.9.3 the missing-
    # runtime error only surfaced ~50 lines into serve_command, AFTER:
    #   - alias profile resolved (logged twice via pflash)
    #   - tool/reasoning parsers auto-configured
    #   - CORS allow-origin warning printed
    # so the operator saw five INFO lines and a banner before the
    # actionable ``Install with: pip install 'fusion-mlx[dflash]'`` line,
    # matching Diego's earlier ``[embeddings]`` regression shape exactly.
    # Hoist the cheap ``have_runtime()`` probe to the same boot-guard tier
    # as the other extras so the error lands FIRST. ``importlib.util.
    # find_spec("mlx_vlm")`` doesn't trigger a load — safe to run on the
    # hot CLI path.
    _wants_dflash = getattr(args, "enable_dflash", False) or (
        getattr(args, "spec_decode", "none") == "dflash"
    )
    if _wants_dflash:
        from .speculative.dflash.eligibility import have_runtime

        if not have_runtime():
            print(
                "\n  Error: --enable-dflash (and --spec-decode dflash) "
                "requires mlx-vlm 0.5.0+ for the DFlash drafter hooks. "
                "Install with: ``pip install 'fusion-mlx[dflash]'``.\n"
            )
            sys.exit(1)

    # DSpark — DeepSeek DeepSpec lossless block spec-decode. The generator is
    # self-contained (loads its own target + converted MLX draft, taps the
    # target's own hidden states — no mlx-vlm hook, no BatchedEngine), so this
    # is an EARLY fork that returns before _ensure_model_downloaded / pflash /
    # parser detection, exactly like the audio-mode fork below. Single-user
    # serial (1-worker pool) — same concurrency contract as DFlash.
    _wants_dspark = getattr(args, "enable_dspark", False) or (
        getattr(args, "spec_decode", "none") == "dspark"
    )
    if _wants_dspark:
        from .speculative.dspark.eligibility import have_runtime as _dspark_have_runtime

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

        from .speculative.dspark.server import run_dspark_server

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
    from . import server
    from .middleware.auth import configure_rate_limiter
    from .scheduler import SchedulerConfig
    from .server import app, load_model

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
            from .tool_parsers import ToolParserManager

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
    from .api.utils import is_mllm_model
    from .pflash import (
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
    # R12-S1: snapshot whether the user explicitly passed
    # ``--tool-call-parser`` BEFORE auto-detect mutates ``args``. The
    # misbind warning below only consults this snapshot — auto-detected
    # bindings are guaranteed to match the model family by construction
    # (the auto path picks the same parser the helper would suggest), so
    # warning on them would be a contradictory "drop the flag" nudge
    # against a flag the user never passed. (Codex r4 NIT — keeps the
    # warning grounded in user intent even if a helper-side regression
    # ever started flagging in-spec cases.)
    _user_explicit_tool_call_parser = bool(args.tool_call_parser)
    if not args.tool_call_parser or not args.reasoning_parser:
        try:
            from .model_auto_config import detect_model_config

            auto_config = detect_model_config(args.model)
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

    # R12-S1: surface a startup warning when ``args.tool_call_parser``
    # is a ``deepseek_v3`` / ``deepseek_v31`` / ``deepseek_r1_0528``
    # binding but the model can't emit the matching V3 fenced-JSON wire
    # shape. See Sven r12 dogfood HIGH-1: forcing ``--tool-call-parser
    # deepseek_v3`` on ``DeepSeek-R1-Distill-Qwen-1.5B-4bit`` lands tool
    # calls with ``arguments="{}"`` because the parser correctly refuses
    # to parse the non-V3 prose the model emits.
    #
    # Runs on BOTH explicit overrides AND auto-detected bindings,
    # because ``detect_model_config`` still scans the full path — a
    # parent dir like ``/models/DeepSeek-V3/qwen-model`` can fool
    # auto-detect into ``deepseek_v3`` even though the tail model name
    # is out-of-lineage (pr-validate codex r7 BLOCKING). The helper's
    # canonical model-name classification catches that mis-pick that
    # auto-detect missed. Whether the user explicitly bound the flag is
    # tracked in ``_user_explicit_tool_call_parser`` and threaded into
    # the warning so the operator can tell who to blame: a misbound
    # flag (user error) vs. a fooled auto-detect (regex needs
    # tightening — tracked as follow-up so this PR stays scoped).
    try:
        from .model_auto_config import warn_misbound_deepseek_v3_parser

        misbind_warning = warn_misbound_deepseek_v3_parser(
            args.model, args.tool_call_parser
        )
        if misbind_warning:
            # ``logger.warning`` so the message lands in any structured
            # log sink AND surfaces in the terminal at the default
            # ``WARNING`` level (no stderr-print needed).
            # ``stacklevel=2`` so log frameworks attribute the call
            # site to the CLI entry rather than the helper module.
            logger.warning(misbind_warning, stacklevel=2)
            if not _user_explicit_tool_call_parser:
                # Auto-detect mis-pick. Emit a second WARNING line so
                # the operator knows the user didn't bind anything —
                # the ``detect_model_config`` regex was fooled by the
                # path. Forces the user to add an explicit
                # ``--tool-call-parser hermes`` (or similar) to recover
                # tool-call capability, which is the actually-correct
                # action for an out-of-lineage checkpoint.
                logger.warning(
                    "  Auto-detect note: this binding came from "
                    "AUTO-DETECT, not an explicit --tool-call-parser "
                    "flag. The detect_model_config() regex was fooled "
                    "by the path. Override with --tool-call-parser "
                    "hermes (or whatever your checkpoint actually "
                    "emits) to recover tool-call capability."
                )
    except Exception as e:  # noqa: BLE001
        logger.debug(f"deepseek_v3 misbind check failed (non-fatal): {e}")

    # Pass alias info to server (for /v1/models)
    server._model_alias = getattr(args, "_original_alias", None)

    # Task #292: forward the ``--enable-audio`` opt-in to the server
    # module BEFORE ``load_model`` runs — the post-load hook in
    # ``load_model`` calls ``register_audio_routes_if_enabled``, which
    # reads ``server._enable_audio_lane`` to decide whether to mount
    # the audio router on a text-only server. Setting it after
    # ``load_model`` would leave the router unmounted on the very boot
    # that asked for it.
    #
    # Codex r2 NIT #2: assign from the parsed value directly so a second
    # in-process ``serve_command`` call (test harness, embedded usage)
    # without ``--enable-audio`` clears any stale ``True`` from a prior
    # run — the singleton ``server`` module persists across calls in
    # the same process.
    server._enable_audio_lane = bool(getattr(args, "enable_audio", False))

    # Configure server security settings. ``FUSION_MLX_API_KEY`` env var
    # is the secret-friendly form ``fusion-mlx share`` uses to avoid
    # exposing the key in argv; inline ``--api-key`` overrides it for
    # backwards-compat with existing scripts. ``_resolve_api_key`` is
    # the single SSOT — both this entrypoint and the ``fusion_mlx.server``
    # ``python -m`` entry call into it, so a future policy tweak (e.g.
    # a deprecation warning when argv is used) lands in one place.
    server._api_key = server._resolve_api_key(args.api_key)
    server._default_timeout = args.timeout

    # Per-request body-size cap. Resolution order:
    #   1. ``--max-request-bytes`` (explicit CLI flag, including 0 to disable)
    #   2. ``FUSION_MLX_MAX_REQUEST_BYTES`` env var
    #   3. ``ServerConfig`` dataclass default (8 MiB)
    # See fusion_mlx/middleware/body_size.py for the DoS rationale.
    _max_body_arg = getattr(args, "max_request_bytes", None)
    if _max_body_arg is not None:
        server._max_request_bytes = max(0, int(_max_body_arg))
    else:
        _env_name = "FUSION_MLX_MAX_REQUEST_BYTES"
        _env = os.environ.get(_env_name, "").strip()
        if _env:
            try:
                server._max_request_bytes = max(0, int(_env))
            except ValueError:
                # Explicit reset (codex round-2 NIT): without this,
                # an in-process callsite that mutated ``_max_request_bytes``
                # before serve_command runs would silently leak a stale
                # value past a malformed env var, which is the worst
                # possible failure shape — bigger cap than the operator
                # intended. Fall back to the documented 8 MiB default
                # explicitly.
                server._max_request_bytes = 8 * 1024 * 1024
                logger.warning(
                    "%s=%r is not an integer; falling back to the 8 MiB default",
                    _env_name,
                    _env,
                )

    # SSE keepalive interval (F-070). Env-only (no CLI flag yet — keep
    # the surface small until operators ask for it). 0 disables. The
    # value lands on ``server._sse_keepalive_seconds`` so ``_sync_config``
    # propagates it into the live ``ServerConfig`` after ``load_model``;
    # writing the config singleton directly here would be clobbered by
    # the subsequent ``_sync_config`` (mirrors the ``_max_request_bytes``
    # pattern just above).
    _sse_env_name = "FUSION_MLX_SSE_KEEPALIVE_SECONDS"
    _sse_env = os.environ.get(_sse_env_name, "").strip()
    if _sse_env:
        try:
            server._sse_keepalive_seconds = max(0.0, float(_sse_env))
        except ValueError:
            # NOTE: the env-var name is interpolated via ``%s`` (not baked
            # into the format string) so the
            # ``tests/test_no_out_of_band_routing.py`` constant scan
            # doesn't see the literal ``FUSION_MLX_…=%r is not a number``
            # as a stand-alone string and false-positive on a routing
            # match. Same pattern the body-receive timeout block below
            # uses.
            logger.warning(
                "%s=%r is not a number; falling back to the 20 s default",
                _sse_env_name,
                _sse_env,
            )
            server._sse_keepalive_seconds = 20.0

    # Body-receive idle timeout (F-072 / H-14 slow-DoS gate). Env-only.
    # 0 disables. Same ``_sync_config``-then-route-handler ordering
    # rationale as the SSE keepalive above. Extracted into
    # :func:`_apply_body_receive_timeout_env` so tests can call the
    # SAME resolver the production binary uses — codex round-2 BLOCKING
    # on PR #786 flagged that an inline-only resolver couldn't be
    # exercised by a unit test without duplicating its logic, which
    # would mask a regression that deleted the wire-up entirely.
    _apply_body_receive_timeout_env(server, logger=logger)

    # Configure CORS (F-090 + F-091). Default: wildcard ``*`` for friendly
    # single-machine UX — fusion-mlx is primarily run locally and a
    # browser frontend at ``http://localhost:3000`` hitting the API at
    # ``http://localhost:8000`` "just works". Operators on multi-tenant /
    # production deployments lock down via
    # ``FUSION_MLX_CORS_ALLOW_ORIGINS=https://your.app,https://other.app``.
    # The full env-var family (METHODS / HEADERS / MAX_AGE /
    # ALLOW_CREDENTIALS) still applies; see
    # ``fusion_mlx/server.py::configure_cors_from_env``.
    #
    # Wildcard + credentials is spec-invalid (Fetch spec rejects the
    # combination), so the resolver forces ``allow_credentials=False``
    # when ``*`` is in the origin list. Operators who need cookie /
    # ``Authorization`` auto-forwarding must pin to specific origins.
    cors_origins = server.configure_cors_from_env(args.cors_origins)
    if args.rate_limit > 0:
        server._rate_limiter = configure_rate_limiter(args.rate_limit, enabled=True)

    # Configure GC control
    gc_control = args.gc_control and not args.no_gc_control
    server._gc_control = gc_control

    # Configure --no-thinking: suppress chain-of-thought in chat template
    server._no_thinking = args.no_thinking

    # Configure system prompt pinning
    server._pin_system_prompt = args.pin_system_prompt

    # Configure tool calling
    if args.enable_auto_tool_choice and args.tool_call_parser:
        server._enable_auto_tool_choice = True
        server._tool_call_parser = args.tool_call_parser
        server._enable_tool_logits_bias = getattr(
            args, "enable_tool_logits_bias", False
        )
    else:
        server._enable_auto_tool_choice = False
        server._tool_call_parser = None
        server._enable_tool_logits_bias = False

    # Configure generation defaults
    if args.default_temperature is not None:
        server._default_temperature = args.default_temperature
    if args.default_top_p is not None:
        server._default_top_p = args.default_top_p
    if args.default_top_k is not None:
        server._default_top_k = args.default_top_k
    if args.default_min_p is not None:
        server._default_min_p = args.default_min_p
    if args.default_repetition_penalty is not None:
        server._default_repetition_penalty = args.default_repetition_penalty
    if args.default_presence_penalty is not None:
        server._default_presence_penalty = args.default_presence_penalty
    if args.default_frequency_penalty is not None:
        server._default_frequency_penalty = args.default_frequency_penalty

    # Configure reasoning parser
    if args.reasoning_parser:
        try:
            from .reasoning import get_parser

            parser_cls = get_parser(args.reasoning_parser)
            server._reasoning_parser = parser_cls()
            server._reasoning_parser_name = args.reasoning_parser
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
        server._reasoning_parser = None

    # R15-P1 #313 follow-up (#318): ``--spec-decode dflash`` routes to
    # the prod path. The originally vendored BatchedEngine adapter at
    # ``fusion_mlx/speculative/dflash/drafter.py:275`` called
    # ``drafter.draft_block(prefix_tokens, current_position)`` with 2 args,
    # but mlx-vlm 0.5.0's ``DFlashDraftModel.draft_block`` requires 6 args:
    # ``(last_bonus, hidden, cache, block_size, sampler, token_dtype)``.
    # The BatchedEngine adapter never wired the verifier→drafter hidden-
    # state + cache + sampler thread, so the new --spec-decode dflash
    # flag was 100% broken at first request — never validated end-to-end.
    # The OLD ``--enable-dflash`` flag (``fusion_mlx/speculative/dflash/`` +
    # mlx-vlm's ``_dflash_rounds``) IS the prod-tested path. We unify the
    # CLI surface by routing ``--spec-decode dflash`` to
    # ``--enable-dflash`` so users hit the working bridge. The
    # ``speculative/dflash/{generator,drafter,verifier}.py`` modules
    # remain importable but inert; the ``accept_counter`` /
    # ``drafter_registry`` siblings stay active for metric scaffolding.
    if getattr(args, "spec_decode", "none") == "dflash":
        args.enable_dflash = True
        args.spec_decode = "none"
        print(
            "Spec-decode: --spec-decode dflash routed to --enable-dflash "
            "(mlx-vlm bridge; BatchedEngine integration deferred to 0.10).",
            file=sys.stderr,
        )

    # DFlash mutual-exclusion gate fires BEFORE the startup banner so
    # the user sees a clean error instead of an optimistic "Features:
    # dflash" line immediately followed by an exit. The deeper SchedulerConfig
    # mutex (suffix vs. mtp) stays below since it doesn't involve DFlash.
    if args.enable_dflash and (args.suffix_decoding or args.enable_mtp):
        print(
            "\n  Error: --enable-dflash cannot combine with --suffix-decoding "
            "or --enable-mtp. DFlash runs a dedicated single-user server "
            "that bypasses BatchedEngine; other spec-decode methods only "
            "apply to the BatchedEngine path.\n"
        )
        sys.exit(1)

    # DFlash eligibility gate fires here, BEFORE the startup banner —
    # so the user sees a clean error rather than an optimistic "DFlash
    # enabled" feature line followed by an exit. Cheap (just reads
    # aliases.json + checks the module spec); no model load yet.
    if args.enable_dflash:
        from .model_aliases import resolve_profile
        from .speculative.dflash import DFlashUnavailable, check

        # ``have_runtime()`` validated at the top-of-function boot-guard
        # tier — see the 0.9.2 dogfood comment near the audio probe.
        _alias_name = getattr(args, "_original_alias", None) or args.model
        _profile = resolve_profile(_alias_name)
        if _profile is None:
            print(
                f"\n  Error: --enable-dflash requires a known alias, got "
                f"{_alias_name!r}. DFlash eligibility is recorded per-alias "
                f"in aliases.json; ad-hoc HuggingFace paths can't be "
                f"validated. Try ``fusion-mlx info qwen3.5-27b-8bit``.\n"
            )
            sys.exit(1)
        try:
            check(_profile, alias=_alias_name)
        except DFlashUnavailable as e:
            print(f"\n  Error: {e}\n")
            sys.exit(1)
        # ``have_runtime()`` is already validated by the boot-guard tier
        # at the top of ``serve_command`` — see the 0.9.2 dogfood comment
        # there. We keep the import + the deeper DFlashUnavailable / alias
        # check here because they need the resolved profile, but the
        # extras-not-installed branch is unreachable by the time control
        # reaches this point.

        # Warn about flags that BatchedEngine honours but the DFlash
        # server doesn't — better to surface this once at startup than
        # to let users wonder why their tuning has no effect. Inspected
        # against the actual argparse Namespace so we only mention flags
        # the user explicitly set away from their default.
        _GPU_MEM_DEFAULT = 0.90  # keep in sync with the serve_parser default
        _dflash_ignored: list[str] = []
        if getattr(args, "enable_prefix_cache", False):
            _dflash_ignored.append("--enable-prefix-cache")
        if getattr(args, "kv_cache_quantization", None):
            _dflash_ignored.append("--kv-cache-quantization")
        # gpu-memory-utilization defaults to 0.90 (not None) in the serve
        # parser, so an ``is not None`` check would fire on every invocation.
        # Compare to the real default — only warn when the user explicitly
        # tuned it. Tolerate a tiny float-equality slack for safety.
        _gpu_mem = getattr(args, "gpu_memory_utilization", _GPU_MEM_DEFAULT)
        if _gpu_mem is not None and abs(_gpu_mem - _GPU_MEM_DEFAULT) > 1e-6:
            _dflash_ignored.append("--gpu-memory-utilization")
        if getattr(args, "enable_auto_tool_choice", False):
            _dflash_ignored.append("--enable-auto-tool-choice")
        if getattr(args, "tool_call_parser", None):
            _dflash_ignored.append("--tool-call-parser")
        if getattr(args, "reasoning_parser", None):
            _dflash_ignored.append("--reasoning-parser")
        if getattr(args, "embedding_model", None):
            _dflash_ignored.append("--embedding-model")
        if getattr(args, "mcp_config", None):
            _dflash_ignored.append("--mcp-config")
        if _dflash_ignored:
            print(
                "\n  ⚠ The following flags are ignored under --enable-dflash"
                "\n    (DFlash uses a dedicated single-user server that bypasses"
                "\n    BatchedEngine):"
                f"\n      {', '.join(_dflash_ignored)}"
                "\n    Drop them from your serve command, or run without"
                "\n    --enable-dflash if you need them.\n"
            )

    # Startup summary
    print()
    print("  🐆 Fusion-MLX")
    print("  ─────────")
    features = []
    if args.enable_auto_tool_choice:
        bias_info = (
            " + logits bias" if getattr(args, "enable_tool_logits_bias", False) else ""
        )
        features.append(f"tools: {args.tool_call_parser}{bias_info}")
    if args.reasoning_parser:
        features.append(f"reasoning: {args.reasoning_parser}")
    # Banner mirrors the effective auth state via ``_auth_feature_str``
    # so the test can call the same function. Pre-fix the gate said
    # ``if args.api_key`` directly — a sidecar that set env-only saw
    # ``auth: off`` printed even though ``verify_api_key`` was
    # enforcing. ``_auth_feature_str`` keeps the banner and the actual
    # enforcement aligned and is directly unit-testable.
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
    # Show CORS in the startup banner when CLI flag or env-var-driven
    # config produced an origin list (``configure_cors_from_env`` is what
    # actually resolved it — see the call site earlier in this function).
    if cors_origins:
        features.append(f"cors: {', '.join(cors_origins)}")
    if args.enable_dflash:
        features.append("dflash: single-user")
    if features:
        print(f"  Features: {', '.join(features)}")
    print(f"  Model: {args.model}")
    # Store MCP config path for FastAPI startup
    if args.mcp_config:
        print(f"MCP config: {args.mcp_config}")
        os.environ["FUSION_MLX_MCP_CONFIG"] = args.mcp_config

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
    from .turboquant import resolve_turboquant_mode_default

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
        from .kv_cache_dtype import (
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
        from .kv_cache_dtype import (
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
        # R15-P1 #302/#313: --spec-decode {none,mtp,dflash}. Plumb the
        # raw choice through; the boot-time eligibility check below
        # validates that ``mtp`` was only passed for a config.json with
        # ``mtp_num_hidden_layers >= 1`` and ``dflash`` requires a
        # Qwen3.5/3.6 model + a bound DFlash drafter.
        spec_decode=getattr(args, "spec_decode", "none"),
        dflash_drafter_path=getattr(args, "dflash_drafter_path", "") or "",
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
        _resolution = resolve_spec_auto(_hf_cfg_auto)
        # auto is authoritative — clear operator-set spec flags so the
        # resolved method doesn't collide with a stale enable_*.
        args.suffix_decoding = False
        args.enable_mtp = False
        args.enable_dflash = False
        args.enable_dspark = False
        apply_resolution(args, _resolution)
        print(
            f"Spec-decode: auto → {_resolution.cli_target} " f"({_resolution.reason})"
        )
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
    if getattr(args, "listen_fd", None) is None:
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
        from .model_aliases import resolve_profile
        from .speculative.dflash.server import run_dflash_server

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
    host_display = "localhost" if args.host == "0.0.0.0" else args.host
    listen_fd = getattr(args, "listen_fd", None)
    if listen_fd is not None:
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
    if listen_fd is None:
        _cfg.bind_host = host_display
        _cfg.bind_port = args.port
    else:
        _cfg.bind_listen_fd = listen_fd

    _run_uvicorn(app, args, uvicorn_log_level)


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
    from .bench.tier_runner import run_tier

    rc, tier_results = run_tier(
        model=args.model,
        tier=tier,
        base_url=getattr(args, "base_url", None),
        sampled=getattr(args, "sampled", False),
        return_results=True,
        skip_speed=True,
    )
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

    from .community_bench.hardware import collect as collect_hw
    from .community_bench.hardware import is_apple_silicon
    from .community_bench.runner import run_standardized_bench
    from .community_bench.submission import (
        build_submission_payload,
        submit_interactive,
    )
    from .engine_core import AsyncEngineCore, EngineConfig
    from .model_aliases import resolve_profile
    from .scheduler import SchedulerConfig

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

        from .engine_core import _init_mlx_step_thread

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
    from . import _mlx_compat as _mlx_compat

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
        from .bench.tier_runner import run_tier

        sys.exit(
            run_tier(
                model=args.model,
                tier=args.tier,
                base_url=getattr(args, "base_url", None),
                sampled=getattr(args, "sampled", False),
            )
        )

    # --submit routes through the standardized community-bench runner,
    # which locks the comparability knobs the freeform path exposes.
    # Keep the branch high in this function so the rest of bench_command
    # doesn't accidentally read --submit-only args.
    if getattr(args, "submit", False):
        sys.exit(_run_submit_flow(args))

    from mlx_lm import load

    from .api.utils import is_mllm_model as _bench_is_mllm_model
    from .engine_core import AsyncEngineCore, EngineConfig
    from .pflash import config_from_args as _pflash_config_from_args
    from .pflash import resolve_pflash_mode_default as _pflash_resolve_default
    from .pflash import validate_model_support as _bench_pflash_validate
    from .request import SamplingParams
    from .scheduler import SchedulerConfig

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
