# SPDX-License-Identifier: Apache-2.0
"""Language capability metadata (#1416).

Resolves the ``languages`` array surfaced on each /v1/models entry so
downstream consumers (fusion-gateway language-aware routing, fusion-bench
``run arabic``) can discover per-model language capability without
hardcoding.

Resolution order:
  1. Alias-profile ``languages`` (aliases.json) — explicit, curated, wins.
  2. Conservative family hints below — only families whose multilingual
     support is documented by the model card / vendor. Never guess.
  3. [] — unknown. Empty is honest; the acceptance criteria for #1416
     explicitly allow empty/unknown.
"""

import logging
import re

logger = logging.getLogger(__name__)

# (case-insensitive substring pattern, languages). Order matters: first
# match wins. Deliberately minimal — add a family only when its Arabic /
# multilingual claim is verifiable on the model card.
_FAMILY_HINTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    # Jais family (Inception/G42): Arabic-English bilingual by design.
    ("jais", ("ar", "en")),
    # SDAIA ALLaM: Arabic-dominant, English-capable.
    ("allam", ("ar", "en")),
    # AceGPT (FreedomIntelligence): Llama tuned for Arabic.
    ("acegpt", ("ar", "en")),
    # Qwen family: officially supports English + Chinese (and more, but we
    # only advertise the two core languages).
    ("qwen", ("en", "zh")),
    # Gemma family: officially multilingual incl. English + Mandarin.
    ("gemma", ("en", "zh")),
)


def _family_hint(model_id: str) -> tuple[str, ...]:
    lowered = model_id.lower()
    for pattern, langs in _FAMILY_HINTS:
        if re.search(pattern, lowered):
            return langs
    return ()


def languages_for(model_id: str) -> list[str]:
    """Return advertised languages for a model id (alias or HF path).

    Alias-declared languages win over family hints; unknown models get [].
    """
    if not model_id:
        return []
    try:
        from .model_aliases import resolve_profile

        profile = resolve_profile(model_id)
        if profile is not None and profile.languages:
            return list(profile.languages)
    except Exception:
        logger.debug("languages_for: profile lookup failed for %s", model_id)
    hint = _family_hint(model_id)
    if hint:
        return list(hint)
    return []
