# SPDX-License-Identifier: Apache-2.0
"""
OpenAI-compatible API routes for fusion-mlx — shared module.

Module-level state + shared helpers used by chat, completions, and resume
submodules. The FastAPI ``router`` and ``set_openai_context`` live here so
all route decorators across the package register on the same router.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter

from ...dispatch import RequestRouter
from ...pool import EnginePool
from ...request import SamplingParams
from .._concurrency import (
    init_request_semaphore,
)
from .._engine_helpers import release_engine as _shared_release
from .._engine_helpers import resolve_engine as _shared_resolve
from ..adapters.openai import OpenAIAdapter
from ..openai_models import (
    ChatCompletionRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["openai"])

# Set by server.py during startup
_pool: Any = None
_request_router: Any = None
_adapter = OpenAIAdapter()
log = logging.getLogger(__name__)

_MODEL_TYPE_TO_MODALITY: dict[str, str] = {
    "llm": "text",
    "vlm": "text",
    "embedding": "text",
    "reranker": "text",
    "ner": "text",
    "audio_stt": "audio",
    "audio_tts": "audio",
    "audio_sts": "audio",
    "image": "image",
    "video": "video",
}


def _resolve_modality(model_id: str) -> str:
    from ...model_aliases import resolve_profile

    profile = resolve_profile(model_id)
    if profile is not None and profile.modality:
        return profile.modality
    if _pool is not None:
        entry = _pool.get_entry(model_id)
        if entry is not None:
            mt = getattr(entry, "model_type", None)
            if mt and mt in _MODEL_TYPE_TO_MODALITY:
                return _MODEL_TYPE_TO_MODALITY[mt]
    return "text"


def _resolve_capabilities(model_id: str) -> dict:
    caps = {
        "text_generation": False,
        "tool_calling": False,
        "structured_output": False,
        "vision": False,
        "embedding": False,
    }
    if _pool is not None:
        entry = _pool.get_entry(model_id)
        if entry is not None:
            mt = getattr(entry, "model_type", None)
            if mt == "llm":
                caps["text_generation"] = True
                caps["tool_calling"] = True
                caps["structured_output"] = True
            elif mt == "vlm":
                caps["text_generation"] = True
                caps["tool_calling"] = True
                caps["structured_output"] = True
                caps["vision"] = True
            elif mt == "embedding":
                caps["embedding"] = True
    else:
        caps["text_generation"] = True
    return caps


def set_openai_context(pool: EnginePool, req_router: RequestRouter) -> None:
    """Inject engine pool and request router into this module."""
    global _pool, _request_router
    _pool = pool
    _request_router = req_router
    try:
        from ...config import get_config

        cfg = get_config()
        init_request_semaphore(getattr(cfg.scheduler, "max_num_seqs", 8))
    except Exception:
        logger.debug("request semaphore init deferred", exc_info=True)


async def _resolve_engine(model_name: str, adapter_path=None):
    return await _shared_resolve(model_name, _pool, adapter_path=adapter_path)


async def _release_engine(model_name: str, adapter_path=None):
    await _shared_release(model_name, _pool, adapter_path=adapter_path)


def _extract_text(msg: Any) -> str:
    """Extract plain text from a message's content field."""
    if isinstance(msg, dict):
        content = msg.get("content", "")
    else:
        content = getattr(msg, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text":
                    parts.append(part.get("text", ""))
                elif part.get("type") == "image_url":
                    parts.append("[image]")
                elif part.get("type") == "audio_url":
                    parts.append("[audio]")
                elif part.get("type") in ("video", "video_url"):
                    parts.append("[video]")
            elif hasattr(part, "text") and part.text:
                parts.append(part.text)
        return "\n".join(parts)
    return str(content) if content else ""


def _detect_prefix_cache_boundary(messages: Any) -> int | None:
    """Auto-detect prefix cache boundary from cache_control hints in messages.

    Scans system messages for cache_control markers (Anthropic-compatible).
    Returns the estimated token boundary for KV prefix cache reuse, or None.

    The boundary is set at the end of the last system message that carries
    a cache_control block, enabling the engine to reuse cached KV states
    for shared system prompt prefixes across requests.
    """
    char_boundary = 0
    found = False
    for m in messages:
        role = getattr(m, "role", "")
        if role != "system":
            break
        content = getattr(m, "content", "")
        has_cache_control = False
        if isinstance(content, str):
            has_cache_control = bool(getattr(m, "cache_control", None))
            if has_cache_control:
                char_boundary += len(content)
                found = True
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    has_cc = bool(part.get("cache_control"))
                else:
                    has_cc = bool(getattr(part, "cache_control", None))
                if has_cc:
                    found = True
                text = ""
                if isinstance(part, dict):
                    text = part.get("text", "")
                elif hasattr(part, "text"):
                    text = part.text or ""
                char_boundary += len(text)
        else:
            break
        if not has_cache_control and found:
            break
    if not found:
        return None
    return max(1, char_boundary // 4)


async def _inject_web_search(request: ChatCompletionRequest) -> None:
    """When request.web_search is True, search DuckDuckGo for the user's last
    message and prepend the results as a system message into the context."""
    if not getattr(request, "web_search", False):
        return

    query = None
    for msg in reversed(request.messages):
        role = getattr(msg, "role", "")
        content = getattr(msg, "content", "")
        if role == "user" and content:
            query = _extract_text(msg).strip()
            break

    if not query:
        return

    logger.info("web_search: querying '%s'", query[:80])
    try:
        import httpx as _httpx

        from fusion_mlx._http_limits import bounded_limits

        snippets: list[str] = []
        async with _httpx.AsyncClient(
            timeout=8.0, follow_redirects=True, limits=bounded_limits()
        ) as client:
            resp = await client.get(
                "https://html.duckduckgo.com/html/",
                params={"q": query},
                headers={
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36",
                },
            )
            if resp.status_code == 200:
                import re

                text = resp.text
                results = re.findall(
                    r'<a[^>]+class="result__a"[^>]*>(.*?)</a>', text, re.DOTALL
                )
                for i, raw_title in enumerate(results[:6]):
                    title = re.sub(r"<[^>]+>", "", raw_title).strip()
                    if title:
                        snippets.append(f"{i + 1}. {title}")

        if snippets:
            search_ctx = (
                "[Web Search Results]\n"
                + "\n".join(snippets)
                + "\n\nUse the above results to inform your answer when relevant."
            )
            from ..models import Message

            search_msg = Message(role="system", content=search_ctx)
            request.messages.insert(0, search_msg)
            logger.info("web_search: injected %d results", len(snippets))
        else:
            logger.info("web_search: no results found for '%s'", query[:60])
    except Exception as exc:
        logger.warning("web_search failed: %s(%s)", type(exc).__name__, exc)


def _messages_for_engine(request_msgs: Any, is_mllm: bool) -> list[dict]:
    """Convert request messages to the dict list engines expect.

    Text-only models (and plain-string content) get the flattened text form
    from _extract_text. Multimodal models keep the structured content parts
    (image_url / video_url blocks) as dicts so the VLM engine can extract the
    media - flattening here would discard the URLs and the model would see
    only "[video]" / "[image]" placeholders.
    """
    out: list[dict] = []
    for m in request_msgs:
        content = getattr(m, "content", "")
        if is_mllm and isinstance(content, list):
            parts: list[dict] = []
            for part in content:
                if isinstance(part, dict):
                    parts.append(part)
                elif hasattr(part, "model_dump"):
                    parts.append(part.model_dump(exclude_none=True))
                else:
                    parts.append(dict(part))
            out.append({"role": m.role, "content": parts})
        else:
            out.append({"role": m.role, "content": _extract_text(m)})
    return out


def _build_sampling_params(
    req: ChatCompletionRequest,
    profile_overrides: dict | None = None,
) -> SamplingParams:
    """Convert ChatCompletionRequest to SamplingParams.

    profile_overrides: dict from model:profile resolution.
    Request-level params take precedence; profile fills in unset defaults.
    """
    po = profile_overrides or {}
    # Fallback when neither the request nor a profile sets max_tokens
    # (e.g. OpenAI-compatible clients that omit it, or AI SDK v6 which
    # silently drops the renamed maxTokens param). Use the operator-
    # configured ServerConfig.default_max_tokens, NOT a hard-coded 2048 —
    # 2048 truncates long structured completions (~3900 chars) before the
    # JSON closes, surfacing as finish=length + client-side parse failure.
    from ...config import get_config

    return SamplingParams(
        max_tokens=req.max_tokens
        or po.get("max_tokens")
        or get_config().default_max_tokens,
        temperature=(
            req.temperature
            if req.temperature is not None
            else po.get("temperature", 0.7)
        ),
        top_p=(req.top_p if req.top_p is not None else po.get("top_p", 0.9)),
        top_k=getattr(req, "top_k", 0) or po.get("top_k") or 0,
        min_p=getattr(req, "min_p", 0.0) or po.get("min_p") or 0.0,
        presence_penalty=(
            req.presence_penalty
            if req.presence_penalty is not None
            else po.get("presence_penalty", 0.0)
        ),
        frequency_penalty=(
            req.frequency_penalty if req.frequency_penalty is not None else 0.0
        ),
        stop=(
            req.stop
            if isinstance(req.stop, list)
            else ([req.stop] if req.stop else None)
        ),
        stop_token_ids=getattr(req, "stop_token_ids", None),
        logprobs=bool(req.logprobs),
        top_logprobs=req.top_logprobs,
    )


def _get_settings() -> Any:
    from ...server import get_settings

    return get_settings()
