import importlib

import pytest

# P0-1 (#0912 audit): ollama_routes deferred-imported SystemMessage/UserMessage
# from api.models, but the Message refactor had dropped those names -> ImportError
# at request time -> 500 on every /api/chat + /api/generate. This smoke test
# imports every route module + the symbols they deferred-import so a regression
# fails at collection time, not in production.

ROUTE_MODULES = [
    "fusion_mlx.api.ollama_routes",
    "fusion_mlx.api.openai",
    "fusion_mlx.api.anthropic_routes",
    "fusion_mlx.api.audio_routes",
    "fusion_mlx.api.images",
    "fusion_mlx.api.embeddings_routes",
    "fusion_mlx.api.rerank_routes",
    "fusion_mlx.api.videos_routes",
    "fusion_mlx.api.mcp_routes",
    "fusion_mlx.api.session_routes",
]

# Symbols that route modules deferred-import from api.models (the P0-1 surface).
# Must remain importable for back-compat with the Message refactor.
MODELS_SYMBOLS = [
    "Message",
    "AssistantMessage",
    "SystemMessage",
    "UserMessage",
    "ChatCompletionRequest",
    "ChatCompletionResponse",
]


@pytest.mark.parametrize("mod_name", ROUTE_MODULES)
def test_route_module_imports_clean(mod_name):
    mod = importlib.import_module(mod_name)
    assert mod is not None, f"{mod_name} failed to import"


@pytest.mark.parametrize("sym", MODELS_SYMBOLS)
def test_models_backcompat_symbols_present(sym):
    from fusion_mlx.api import models

    assert hasattr(models, sym), (
        f"api.models.{sym} missing — route modules deferred-import this name; "
        f"its absence caused P0-1 (#0912 audit) ImportError -> 500 on /api/chat"
    )


def test_system_user_message_alias_identity():
    from fusion_mlx.api.models import (
        AssistantMessage,
        Message,
        SystemMessage,
        UserMessage,
    )

    assert SystemMessage is Message
    assert UserMessage is Message
    assert AssistantMessage is not Message
