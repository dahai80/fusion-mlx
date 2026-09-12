# SPDX-License-Identifier: Apache-2.0
"""Profile formatting + suffix-decoding tier classification.

format_profile_summary / format_profile_table / get_profile +
classify_suffix_decoding_tier / suffix_decoding_hint. Split from the
original model_auto_config.py monolith.
"""

import logging

from .core import (
    ModelConfig,
    detect_model_config,
    enrich_model_config,
)

logger = logging.getLogger(__name__)


def classify_suffix_decoding_tier(speedup: dict[str, float]) -> str:
    """Map a per-workload speedup dict to a tier string.

    Empty dict → "unknown". Single-workload dicts use the special-case
    rule that an empty ``min(others)`` is treated as +∞ (the AGENT gate
    is satisfied vacuously). See ``tests/test_suffix_decoding_tier.py``
    for boundary cases including the real Qwen3-0.6B / Qwen3-14B numbers.
    """
    if not speedup:
        return "unknown"

    lo = min(speedup.values())
    hi = max(speedup.values())

    # AVOID first: any individual workload regressing past 0.85x means
    # we don't know the user's traffic mix well enough to recommend.
    if lo < 0.85:
        return "avoid"

    # AGENT — tool_loop must be the workload winning big, AND no other
    # workload regresses past 0.95x. Tool_loop missing from the dict
    # means the bench didn't measure it; we can't claim agent then.
    tool_loop = speedup.get("tool_loop")
    if tool_loop is not None and tool_loop >= 1.8:
        others = [v for k, v in speedup.items() if k != "tool_loop"]
        if not others or min(others) >= 0.95:
            return "agent"

    # STRUCTURED — some workload wins meaningfully (≥1.5x) AND the
    # weakest workload still clears 0.90x (small regression tolerated
    # because the user is opting in for the structured win).
    if hi >= 1.5 and lo >= 0.90:
        return "structured"

    # NEUTRAL — flat across the board. Tighter than STRUCTURED's 0.90
    # floor: we want true noise here, not a near-miss STRUCTURED.
    if lo >= 0.95 and hi >= 1.0 and hi < 1.5:
        return "neutral"

    # Mixed signal that didn't fit any positive bucket — recommend AVOID
    # rather than silently shipping ambiguous data.
    return "avoid"


def suffix_decoding_hint(cfg: "ModelConfig | None") -> str | None:
    """Startup hint for the SuffixDecoding flag, or ``None`` for silent tiers.

    The hint surfaces only AGENT / STRUCTURED / AVOID tiers. UNKNOWN and
    NEUTRAL stay silent — no user-visible nudge until bench data exists
    or there's a real regression to warn about.

    Hybrid arches (``supports_spec_decode=False``) always return ``None``
    even if the tier was somehow set: spec decoding is gated off at the
    engine level, and a "recommended" hint there would just confuse.
    """
    if cfg is None:
        return None
    if not cfg.supports_spec_decode:
        return None
    tier = cfg.suffix_decoding_tier
    speedup = cfg.suffix_bench_speedup or {}
    if tier == "agent":
        peak = speedup.get("tool_loop") or (max(speedup.values()) if speedup else 0)
        return (
            f"SuffixDecoding: recommended for tool/agent traffic "
            f"(tool_loop {peak:.1f}x). Pass --suffix-decoding to enable."
        )
    if tier == "structured":
        peak_key = max(speedup, key=speedup.get) if speedup else "structured"
        peak_val = speedup.get(peak_key, 0)
        return (
            f"SuffixDecoding: may help on {peak_key} ({peak_val:.2f}x). "
            "Pass --suffix-decoding if your traffic matches."
        )
    if tier == "avoid":
        worst_key = min(speedup, key=speedup.get) if speedup else "some workloads"
        worst_val = speedup.get(worst_key, 0)
        return (
            f"SuffixDecoding: NOT recommended for this model — {worst_key} "
            f"regresses to {worst_val:.2f}x. Leave --suffix-decoding off."
        )
    return None


def _arch_label(cfg: "ModelConfig") -> str:
    """One-word architecture label for human display."""
    if cfg.is_hybrid:
        return "hybrid (linear-attention/Mamba)"
    return "pure attention"


def _suffix_tier_cell(cfg: "ModelConfig", max_width: int | None = None) -> str:
    """Format the ``Suffix tier`` row for ``rapid-mlx info``.

    AGENT/STRUCTURED — surface the peak workload speedup (the reason the
    tier was assigned). AVOID — surface the worst-regressing workload so
    the user understands the warning. UNKNOWN — point them at the bench
    script. Hybrid arches always render ``n/a`` regardless of tier
    because ``supports_spec_decode=False`` gates the flag off anyway.

    When ``max_width`` is set and the produced string would exceed it,
    the parenthetical note after the tier word (``avoid``/``prefer``/
    ``neutral``/…) is truncated so the value fits inside the caller's
    box column without breaking alignment. The tier word itself is kept
    intact because it's the load-bearing signal. Truncated notes end
    with ``…)`` instead of ``)``.
    """
    if not cfg.supports_spec_decode:
        # ``supports_spec_decode=False`` covers two cases: hybrid arches
        # (Mamba / linear-attention — the runtime gates spec decode off)
        # and dense models where no MTP/drafter checkpoint is registered.
        # Surfacing the right reason is load-bearing for ``rapid-mlx info``
        # — 0.9.0 dogfood found we were reporting ``hybrid arch`` for
        # pure-attention Qwen3.5/3.6 dense aliases, which contradicts the
        # ``Architecture: pure attention`` row two lines above.
        if cfg.is_hybrid:
            text = "n/a (hybrid arch — spec decode off)"
        else:
            # Tight enough to fit the 41-char ``info`` value column
            # (``inner=60 − 17-char key − 2-char ": "``) so the row
            # renders without ``_truncate_tier_note`` clipping.
            text = "n/a (no MTP/drafter — spec decode off)"
    else:
        tier = cfg.suffix_decoding_tier
        speedup = cfg.suffix_bench_speedup or {}
        if tier == "unknown":
            text = "unknown — run scripts/bench_suffix_decoding_integrated"
        elif tier == "agent" and speedup:
            peak_key = (
                "tool_loop" if "tool_loop" in speedup else max(speedup, key=speedup.get)
            )
            text = (
                f"agent ({peak_key} {speedup[peak_key]:.2f}x"
                " — recommend --suffix-decoding)"
            )
        elif tier == "structured" and speedup:
            peak_key = max(speedup, key=speedup.get)
            text = (
                f"structured ({peak_key} {speedup[peak_key]:.2f}x"
                " — try if traffic matches)"
            )
        elif tier == "neutral":
            text = "neutral (within noise — leave off)"
        elif tier == "avoid" and speedup:
            worst_key = min(speedup, key=speedup.get)
            text = (
                f"avoid ({worst_key} {speedup[worst_key]:.2f}x regression — leave off)"
            )
        else:
            text = tier
    return _truncate_tier_note(text, max_width)


def _truncate_tier_note(text: str, max_width: int | None) -> str:
    """Shorten a ``tier (note)`` string to fit within ``max_width`` chars.

    Only the parenthetical note is trimmed; the leading tier word stays
    whole. If the tier word alone already overflows (shouldn't happen
    with current tiers but kept defensive), the full text is returned
    unchanged — the caller's column will visibly break, surfacing the
    bug instead of silently dropping load-bearing data.

    The ``tier — note`` (em-dash) form used by the ``unknown`` tier is
    handled as a fallback so that variant also fits inside the box.
    """
    if max_width is None or len(text) <= max_width:
        return text
    open_paren = text.find("(")
    if open_paren != -1 and text.endswith(")"):
        # ``prefix`` = ``tier (`` — keep verbatim. Available room for
        # note body = max_width − len(prefix) − len("…)").
        prefix = text[: open_paren + 1]
        available = max_width - len(prefix) - len("…)")
        if available < 1:
            return text
        note_body = text[open_paren + 1 : -1]
        return prefix + note_body[:available].rstrip() + "…)"
    em_dash = text.find(" — ")
    if em_dash != -1:
        prefix = text[: em_dash + 3]  # include the `` — `` separator
        available = max_width - len(prefix) - len("…")
        if available < 1:
            return text
        note_body = text[em_dash + 3 :]
        return prefix + note_body[:available].rstrip() + "…"
    return text


def format_profile_summary(model_path: str, cfg: "ModelConfig | None") -> str:
    """Single-line profile summary for startup logs (Level 1).

    Empty/no-match models return a generic line so the log is consistent
    across known and unknown models.
    """
    if cfg is None:
        return f"Model profile: {model_path} (unknown family — using defaults)"
    parts = [_arch_label(cfg)]
    parts.append(f"throttle {'ON' if cfg.is_hybrid else 'OFF'}")
    parts.append(f"spec decode {'OFF' if not cfg.supports_spec_decode else 'OK'}")
    if cfg.tool_call_parser:
        parts.append(f"tool={cfg.tool_call_parser}")
    if cfg.reasoning_parser:
        parts.append(f"reasoning={cfg.reasoning_parser}")
    return f"Model profile: {model_path} → " + ", ".join(parts)


def format_profile_table(model_path: str, cfg: "ModelConfig | None") -> str:
    """Multi-line ASCII capability table for verbose startup output and
    the ``rapid-mlx info`` CLI command (Level 2 + Level 3).

    Width is fixed at 64 cols so it renders cleanly in terminal logs.
    Note: Unicode check/cross marks count as 1 char each (no double-width).
    """
    inner = 60  # printable width between ``│ `` and `` │`` markers
    # Value column = ``inner`` minus the 17-char key field and the
    # 2-char ``": "`` separator. Used by ``_suffix_tier_cell`` to keep
    # long parenthetical notes inside the box.
    value_width = inner - 17 - 2
    sep = "─" * inner

    def _row(text: str) -> str:
        return f"│ {text:<{inner}} │"

    rows: list[tuple[str, str]]
    header = f"Model: {model_path}"
    if len(header) > inner:
        header = header[: inner - 1] + "…"

    if cfg is None:
        rows = [
            ("Profile", "(no pattern matched — using defaults)"),
            ("Tool format", "(none)"),
            ("Reasoning parser", "(none)"),
            ("Architecture", "unknown"),
            ("Spec decode", "✓ default-on"),
            ("Throttle", "✗ default-off"),
            (
                "Suffix tier",
                _truncate_tier_note(
                    "unknown — run scripts/bench_suffix_decoding_integrated",
                    value_width,
                ),
            ),
        ]
    else:
        if cfg.supports_spec_decode:
            spec = "✓ supported"
        elif cfg.is_hybrid:
            spec = "✗ disabled (hybrid arch)"
        elif cfg.supports_dflash:
            # 0.9.1 dogfood follow-up: ``qwen3.5-27b-8bit`` is THE
            # flagship DFlash alias (code median 1.85× per 0.9.0 release
            # notes), but its alias has ``supports_spec_decode=False``
            # because no MTP head is trained. Pre-0.9.2 the row claimed
            # ``(no MTP/drafter trained)`` — half-true but actively
            # misleading because the DFlash drafter IS registered.
            # Surface the actionable opt-in instead.
            spec = "✗ MTP off — try --enable-dflash"
        else:
            # 0.9.0 dogfood: non-hybrid + spec-off was rendering
            # ``hybrid arch`` next to ``Architecture: pure attention``.
            spec = "✗ disabled (no MTP/drafter trained)"
        throttle = "✓ 200ms gap" if cfg.is_hybrid else "✗ not needed"
        rows = [
            ("Tool format", cfg.tool_call_parser or "(none)"),
            ("Reasoning parser", cfg.reasoning_parser or "(none)"),
            ("Architecture", _arch_label(cfg)),
            ("Spec decode", spec),
            ("Throttle", throttle),
            ("Suffix tier", _suffix_tier_cell(cfg, max_width=value_width)),
        ]

    body = [_row(header), _row(sep)]
    for k, v in rows:
        body.append(_row(f"{k:<17}: {v}"))

    top = "┌" + "─" * (inner + 2) + "┐"
    bot = "└" + "─" * (inner + 2) + "┘"
    return "\n".join([top, *body, bot])


def get_profile(model_path: str, model: object | None = None) -> "ModelConfig":
    """One-shot profile lookup combining both stages.

    This is the public API for code that wants the final ModelConfig in
    one call: regex pattern match → optional runtime ArraysCache probe.
    Always returns a ``ModelConfig`` (never None) — falls back to defaults
    when nothing matches so downstream code doesn't need null checks.

    Args:
        model_path: Model name or HF repo path.
        model: Optional loaded mlx-lm model object. When provided, runtime
            probe runs as a safety net for unknown hybrid arches.

    Returns:
        Final merged ``ModelConfig``.
    """
    cfg = detect_model_config(model_path) or ModelConfig()
    if model is not None:
        cfg = enrich_model_config(cfg, model)
    return cfg
