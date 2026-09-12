# SPDX-License-Identifier: Apache-2.0
"""
OpenAI-compatible API routes package for fusion-mlx.

Split from the original monolithic ``openai_routes.py`` into:
- ``_common``    — module-level state + shared helpers + the FastAPI router
- ``grammar``    — grammar compilation + schema extraction helpers
- ``streaming``  — streaming chat generator and streaming chat entry point
- ``markitdown`` — MarkItDown chat completion helpers
- ``chat``       — non-streaming chat completion + the /chat/completions route
- ``completions`` — legacy text completion + model listing routes
- ``resume``     — resumable streaming + resume-completion routes

All route decorators register on the shared ``router`` from ``_common``.
"""

from __future__ import annotations

# Import submodules to trigger route registration on the shared router.
from . import (  # noqa: F401
    _common,
    chat,
    completions,
    grammar,
    markitdown,
    resume,
    streaming,
)

# Public surface — must match what server.py / ollama_routes.py / routes_internal import.
from ._common import (
    _adapter,
    _build_sampling_params,
    _detect_prefix_cache_boundary,
    _extract_text,
    _get_settings,
    _inject_web_search,
    _messages_for_engine,
    _release_engine,
    _resolve_capabilities,
    _resolve_engine,
    _resolve_modality,
    logger,
    router,
    set_openai_context,
)
from .chat import (
    _run_chat,
    chat_completions,
)
from .completions import completions, list_models
from .grammar import (
    _compile_grammar_for_request,
    _extract_strict_json_schema,
    _gen_to_internal,
)
from .markitdown import _create_markitdown_chat_completion
from .resume import (
    ResumeRequest,
    lookup_resumable_stream,
    resume_completion,
    start_resumable_stream,
)
from .streaming import (
    _CHANNEL_REASONING_PARSERS,
    _resolve_streaming_reasoning_parser,
    _resolve_streaming_tool_parser,
    _stream_chat,
    _stream_chat_generator,
)

__all__ = [
    "router",
    "set_openai_context",
    "_resolve_engine",
    "_release_engine",
    "_adapter",
    "logger",
    "_pool",
    "_run_chat",
    "_stream_chat",
    "_stream_chat_generator",
    "_build_sampling_params",
    "_resolve_modality",
    "_resolve_capabilities",
    "_detect_prefix_cache_boundary",
    "_compile_grammar_for_request",
    "_extract_text",
    "_inject_web_search",
    "_messages_for_engine",
    "_get_settings",
    "_extract_strict_json_schema",
    "_gen_to_internal",
    "_resolve_streaming_tool_parser",
    "_resolve_streaming_reasoning_parser",
    "_CHANNEL_REASONING_PARSERS",
    "_create_markitdown_chat_completion",
    "chat_completions",
    "completions",
    "list_models",
    "resume_completion",
    "ResumeRequest",
    "start_resumable_stream",
    "lookup_resumable_stream",
]


def __getattr__(name: str):
    if name == "_pool":
        return _common._pool
    if name == "_request_router":
        return _common._request_router
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
