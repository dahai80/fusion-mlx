"""Compatibility shim: re-exports the /v1/completions router.

The original ``routes_internal/completions.py`` was never ported (the
legacy completions route lives on ``api.openai_routes.router`` alongside
/v1/chat/completions). Tests import ``get_engine`` + ``router`` from this
path, so this shim re-exports both — mirroring ``routes_internal/chat.py``.
"""

from ..api.openai_routes import router  # noqa: F401
from ..service.helpers import get_engine  # noqa: F401
