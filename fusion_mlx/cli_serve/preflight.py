# SPDX-License-Identifier: Apache-2.0
"""Pre-flight checks: disk space, KV cache dtype, memory capacity."""

import logging
import os
import sys

logger = logging.getLogger(__name__)


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

        logger.warning("Insufficient disk space for download:")
        logger.warning("  Model size:    %7.1f GB", model_size_gb)
        logger.warning("  Free space:    %7.1f GB  (%s)", available_gb, probe)
        logger.warning("  Need to free:  %7.1f GB", need_to_free_gb)
        logger.warning("  Suggestions:")
        logger.warning(
            "    - Free disk space, or set HF_HOME to a drive with more room"
        )
        logger.warning("    - Pick a smaller variant: fusion-mlx models")
        if not force:
            logger.warning(
                "    - Bypass this check (download will likely fail mid-way): --force-disk-check"
            )
            sys.exit(1)
        # ``force=True``: warn loudly, let the user proceed at their own risk.
        logger.warning("--force-disk-check set — proceeding anyway.")
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
        from ..model_aliases import resolve_profile

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

    # OPS-P3-5 (#0907 audit): route through logger so the warning lands in
    # server.log (stderr-captured) and is level-filterable, not just stdout.
    if ratio >= 0.85:
        logger.warning(
            "Memory pressure warning: this model is likely too large for your hardware. "
            "Continuing may trigger a macOS kernel panic (see issue #324)."
        )
    else:
        logger.warning(
            "Memory pressure note: this model uses a large fraction of system RAM."
        )
    logger.warning(
        "  Model on disk: %6.1f GB; Est. working set: %6.1f GB (model x 1.5); "
        "OS used: %6.1f GB; Total RAM: %6.1f GB (%.0f%% projected utilization).",
        model_gb,
        working_gb,
        used_gb,
        total_gb,
        ratio * 100,
    )
    if ratio >= 0.85:
        logger.warning(
            "Apple Silicon firmware can panic the whole system rather than raise an "
            "OOM error when unified-memory pressure exceeds the iBoot AMCC threshold. "
            "Recommended: close other apps to free RAM, pick a smaller model "
            "(fusion-mlx models), or lower memory headroom "
            "(--gpu-memory-utilization 0.75)."
        )
    else:
        logger.warning(
            "If you see crashes or kernel panics, try: --gpu-memory-utilization 0.85"
        )
