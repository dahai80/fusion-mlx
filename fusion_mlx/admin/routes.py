# SPDX-License-Identifier: Apache-2.0
"""Admin panel routes for Fusion-MLX server configuration.

This module provides HTTP routes for the admin panel including:
- Login/logout with API key authentication
- Dashboard for server monitoring
- Model settings management (per-model sampling parameters, pinning, default)
- Global settings management
"""

import json
import logging
from pathlib import Path

from fastapi import APIRouter
from fastapi.templating import Jinja2Templates

logger = logging.getLogger(__name__)

PRESET_REMOTE_URL = "http://bench.dpdns.org/assets/fusionmlx_preset.json"


# =============================================================================
# Router and Templates
# =============================================================================

router = APIRouter(prefix="/admin", tags=["admin"])
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")
static_dir = Path(__file__).parent / "static"


def _static_version(path: str) -> str:
    """Append file mtime as query string for cache busting."""
    file_path = static_dir / path
    if file_path.is_file():
        mtime = int(file_path.stat().st_mtime)
        return f"/admin/static/{path}?v={mtime}"
    return f"/admin/static/{path}"


templates.env.globals["static"] = _static_version

from fusion_mlx._version import __version__ as _fusionmlx_version

templates.env.globals["version"] = _fusionmlx_version

# i18n — English fallback, overridden by _refresh_i18n_globals()
_i18n_dir = Path(__file__).parent / "i18n"
_en_locale: dict = {}
try:
    _en_locale = json.loads((_i18n_dir / "en.json").read_text(encoding="utf-8"))
except Exception:
    logger.warning("Failed to load en.json i18n locale")
templates.env.globals["t"] = lambda key: _en_locale.get(key, key)
templates.env.globals["locale_json"] = json.dumps(_en_locale, ensure_ascii=False)
templates.env.globals["current_lang"] = "en"


def _load_locale(language: str) -> dict:
    path = _i18n_dir / f"{language}.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        try:
            return json.loads((_i18n_dir / "en.json").read_text(encoding="utf-8"))
        except Exception:
            return {}


def _make_t(locale: dict):
    def t(key: str) -> str:
        return locale.get(key, key)
    return t


def _refresh_i18n_globals() -> None:
    lang = "en"
    try:
        from .helpers import _get_global_settings
        settings = _get_global_settings()
        if settings:
            lang = getattr(getattr(settings, "ui", None), "language", "en")
    except Exception:
        pass
    locale = _load_locale(lang)
    templates.env.globals["t"] = _make_t(locale)
    templates.env.globals["locale_json"] = json.dumps(locale, ensure_ascii=False)
    templates.env.globals["current_lang"] = lang

# Sub-routers (split by function for maintainability)
from .accuracy_bench import router as accuracy_bench_router
from .auth_routes import router as auth_routes_router
from .bench import router as bench_router
from .grammar_route import router as grammar_router
from .hf_download import router as hf_download_router
from .hf_upload import router as hf_upload_router
from .html_routes import router as html_router
from .logs import router as logs_router
from .models_route import router as models_router
from .ms_download import router as ms_download_router
from .oq import router as oq_router
from .preset import router as preset_router
from .profile import router as profile_router
from .settings import router as settings_router
from .stats import router as stats_router
from .subkey import router as subkey_router
from .update_check import router as update_check_router

# Register all sub-routers
router.include_router(html_router)
router.include_router(auth_routes_router)
router.include_router(subkey_router)
router.include_router(grammar_router)
router.include_router(models_router)
router.include_router(profile_router)
router.include_router(preset_router)
router.include_router(settings_router)
router.include_router(logs_router)
router.include_router(stats_router)
router.include_router(hf_download_router)
router.include_router(ms_download_router)
router.include_router(accuracy_bench_router)
router.include_router(bench_router)
router.include_router(update_check_router)
router.include_router(oq_router)
router.include_router(hf_upload_router)
