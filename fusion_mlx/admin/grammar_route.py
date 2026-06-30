# SPDX-License-Identifier: Apache-2.0
"""Admin panel routes for oMLX server configuration.

This module provides HTTP routes for the admin panel including:
- Login/logout with API key authentication
- Dashboard for server monitoring
- Model settings management (per-model sampling parameters, pinning, default)
- Global settings management
"""

import inspect
import logging
import re

from fastapi import APIRouter, Depends

from .auth import (
    require_admin,
)

logger = logging.getLogger(__name__)

PRESET_REMOTE_URL = "http://bench.dpdns.org/assets/omlx_preset.json"




_router = APIRouter()

# =============================================================================
# Grammar API Routes
# =============================================================================


_SUPPORTED_MODELS_DOC_RE = re.compile(
    r"Supported models:\s*\n((?:\s*-\s*\S.*\n?)+)",
)


def _models_from_docstring(fn) -> list[str]:
    """Extract the ``Supported models:`` bullet list from an xgrammar 0.1.34+
    structural-tag function's docstring. Returns ``[]`` if the section is
    absent or unparseable."""
    doc = inspect.getdoc(fn) or ""
    match = _SUPPORTED_MODELS_DOC_RE.search(doc)
    if not match:
        return []
    return [
        line.strip().lstrip("-").strip()
        for line in match.group(1).splitlines()
        if line.strip().startswith("-")
    ]


@_router.get("/api/grammar/parsers")
async def list_grammar_parsers(is_admin: bool = Depends(require_admin)):
    """Return available reasoning parser names from xgrammar.

    Supports both API generations:

    - **xgrammar 0.1.34+** exposes a per-model registry at
        ``xgrammar.builtin_structural_tag._structural_tag_registry``; supported
        model names are pulled from each function's docstring.
    - **xgrammar 0.1.32–0.1.33** exposes the now-removed helper
        ``get_builtin_structural_tag_supported_models()``.

    Returns ``[]`` if xgrammar is missing, fails to load (e.g. broken native
    binding on macOS arm64), or has neither API available.
    """
    # Install the torch stub BEFORE any xgrammar import. If this lives
    # inside the first try-block, a failure on the 0.1.34+ path can leave
    # the fallback try-block importing xgrammar without the stub, which
    # is guaranteed ImportError on stub-only (DMG) deployments.
    try:
        from fusion_mlx._torch_stub import install as _install_torch_stub

        _install_torch_stub()
    except Exception as e:  # pragma: no cover — defensive
        logger.debug("torch stub install failed: %s", e)

    # Prefer the 0.1.34+ registry so newer parsers (qwen3_6, gemma4,
    # deepseek_v4, ...) are exposed.
    try:
        from xgrammar.builtin_structural_tag import _structural_tag_registry

        return [
            {"value": style, "label": style, "models": _models_from_docstring(fn)}
            for style, fn in _structural_tag_registry.items()
        ]
    except Exception as e:
        logger.debug("xgrammar 0.1.34+ registry unavailable: %s", e)

    # Fall back to the pre-0.1.34 helper.
    try:
        from xgrammar import get_builtin_structural_tag_supported_models

        supported = get_builtin_structural_tag_supported_models()
        return [
            {"value": style, "label": style, "models": models}
            for style, models in supported.items()
        ]
    except Exception as e:
        logger.warning("xgrammar parser discovery unavailable: %s", e)
        return []



router = _router
