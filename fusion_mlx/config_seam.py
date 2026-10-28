# #1439: single-point API-key/env resolution lives in fusion_core.config
# (config-conventions.md §1). fusion-core is an in-tree monorepo package,
# NOT on PyPI, so it is an optional extra here (`pip install
# "fusion-mlx[core]"`, same pattern as fusion-agent-studio's `identity`
# extra). When absent — a standalone PyPI install — a minimal stdlib
# fallback with IDENTICAL precedence keeps slim installs booting (#805).
# Nothing else in fusion_mlx may read FUSION_MLX_API_KEY or settings.json
# auth.api_key directly; route through this seam.

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

ENV_API_KEY = "FUSION_MLX_API_KEY"

try:
    from fusion_core import config as _fusion_core_config
except ImportError:
    _fusion_core_config = None
    logger.debug(
        "fusion_core not installed — standalone fallback key resolution "
        "active (install fusion-mlx[core] for the shared resolver, #1439)"
    )


def env_api_key() -> str:
    return os.environ.get(ENV_API_KEY, "")


def api_key(explicit: str | None = None) -> str:
    # Precedence (config-conventions §1):
    #   explicit  >  FUSION_MLX_API_KEY env  >  settings.json auth.api_key
    # Never raises: settings read failures warn and degrade to "" so boot
    # and CLI paths stay alive.
    if explicit:
        return explicit
    if _fusion_core_config is not None:
        try:
            return _fusion_core_config.resolve_api_key(explicit=None)
        except Exception as exc:
            logger.warning(
                "fusion_core resolve_api_key failed (%s); using local fallback", exc
            )
    key = env_api_key()
    if key:
        return key
    try:
        data = json.loads(
            (Path.home() / ".fusion-mlx" / "settings.json").read_text(encoding="utf-8")
        )
    except FileNotFoundError:
        return ""
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("settings.json read failed during api_key lookup: %s", exc)
        return ""
    auth = data.get("auth")
    if not isinstance(auth, dict):
        logger.warning("settings.json auth section is not an object; no api_key read")
        return ""
    raw = auth.get("api_key") or ""
    if not isinstance(raw, str):
        logger.warning("settings.json auth.api_key is not a string; ignoring")
        return ""
    return raw
