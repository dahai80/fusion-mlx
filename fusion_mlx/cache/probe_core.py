# SPDX-License-Identifier: Apache-2.0
"""Shared cache-probe core (#1559).

Extracted from admin/stats.py probe_cache so both the admin endpoint and
the harness consumer routes answer from one implementation. Adds the RAM
radix-trie walk that the admin probe did not cover: the admin probe only
classified SSD hot/disk blocks, while the primary reuse path for live
sessions is the in-RAM RadixPrefixCache trie.

The walk is read-only: no block tables are created, no refcounts are
touched, no stats counters move. A block counts as reusable when it is
reachable from the radix root (RAM) or retrievable from the SSD tier
(hot RAM copy or on-disk index). The walk stops at the first block that
is in neither tier — the chain hash makes every later block cold.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _render_prompt(engine: Any, tokenizer: Any, request: Any) -> str:
    messages = request.messages
    if hasattr(engine, "_preprocess_messages"):
        messages = engine._preprocess_messages(messages)
    try:
        from ..api.tool_calling import convert_tools_for_template

        template_tools = (
            convert_tools_for_template(request.tools) if request.tools else None
        )
    except Exception:
        template_tools = request.tools or None
    if hasattr(engine, "_apply_chat_template"):
        return engine._apply_chat_template(
            messages,
            template_tools,
            chat_template_kwargs=getattr(request, "chat_template_kwargs", None),
        )
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def probe_prompt_cache(
    engine_pool: Any,
    model_id: str,
    messages: list[dict],
    tools: list[dict] | None = None,
    chat_template_kwargs: dict | None = None,
) -> dict:
    """Classify each chain-hashed block of a rendered prompt by tier.

    Returns a dict with per-tier block/token counts plus the unified
    cached_prefix_tokens (root-chained blocks reusable at prefill time).
    Raises HTTPException-compatible ValueError when prerequisites fail.
    """
    from fastapi import HTTPException

    if engine_pool is None:
        raise HTTPException(status_code=503, detail="Engine pool not initialized")

    entry = engine_pool._entries.get(model_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")
    if entry.engine is None:
        return {
            "model_id": model_id,
            "model_loaded": False,
            "reason": "Model is not loaded — load it to enable cache probing.",
        }

    engine = entry.engine
    tokenizer = getattr(engine, "_tokenizer", None)
    if tokenizer is None or not hasattr(tokenizer, "encode"):
        raise HTTPException(
            status_code=400,
            detail="Model tokenizer does not support encoding.",
        )

    async_core = getattr(engine, "_engine", None)
    core = getattr(async_core, "engine", None) if async_core is not None else None
    scheduler = getattr(core, "scheduler", None) if core is not None else None
    if scheduler is None:
        raise HTTPException(
            status_code=500, detail="Scheduler unavailable for loaded model."
        )

    prefix_cache = getattr(scheduler, "block_aware_cache", None)
    ssd_manager = getattr(scheduler, "paged_ssd_cache_manager", None)
    paged_cache = getattr(scheduler, "paged_cache_manager", None)
    block_size = getattr(
        getattr(scheduler, "config", None), "paged_cache_block_size", 0
    )
    if not block_size and prefix_cache is not None:
        block_size = getattr(prefix_cache, "block_size", 0)
    if not block_size:
        raise HTTPException(
            status_code=500,
            detail="Cache block size unavailable — cache may not be enabled.",
        )

    class _ProbeRequest:
        pass

    req = _ProbeRequest()
    req.messages = messages
    req.tools = tools
    req.chat_template_kwargs = chat_template_kwargs
    try:
        prompt = _render_prompt(engine, tokenizer, req)
        token_ids = list(tokenizer.encode(prompt))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"Failed to tokenize messages: {exc}"
        )

    total_tokens = len(token_ids)
    base = {
        "model_id": model_id,
        "model_loaded": True,
        "total_tokens": total_tokens,
        "block_size": block_size,
    }
    if total_tokens == 0:
        base.update(
            {
                "total_blocks": 0,
                "blocks_ram": 0,
                "blocks_ssd_hot": 0,
                "blocks_ssd_disk": 0,
                "blocks_cold": 0,
                "ram_hit_tokens": 0,
                "ssd_hit_tokens": 0,
                "cached_prefix_tokens": 0,
            }
        )
        return base

    from .paged_cache import compute_block_hash

    model_name = getattr(paged_cache, "model_name", None) if paged_cache else None
    ssd_index = getattr(ssd_manager, "_index", None) if ssd_manager else None
    ssd_hot = getattr(ssd_manager, "_hot_cache", None) if ssd_manager else None

    is_radix = hasattr(prefix_cache, "_root") and hasattr(prefix_cache, "_block_slices")

    total_blocks = (total_tokens + block_size - 1) // block_size
    blocks_ram = 0
    blocks_ssd_hot = 0
    blocks_ssd_disk = 0
    cached_prefix_tokens = 0
    cached_block_count = 0
    ssd_hit_tokens = 0
    parent_hash = b""

    # Single-pass walk: the radix descent keeps a live node pointer (each
    # block descends one trie level), the SSD side keeps the chain hash.
    # Both are root-chained, so the first miss ends the reusable prefix.
    ram_node = getattr(prefix_cache, "_root", None) if is_radix else None
    ram_dead = not is_radix
    ram_allocated = (
        getattr(getattr(prefix_cache, "paged_cache", None), "allocated_blocks", None)
        if is_radix
        else None
    )

    for start in range(0, total_tokens, block_size):
        end = min(start + block_size, total_tokens)
        block_tokens = token_ids[start:end]
        if not block_tokens:
            break

        in_ram = False
        if not ram_dead:
            key = tuple(token_ids[start : start + block_size])
            child = ram_node.children.get(key[0])
            if child is None or child.key != key or child.block_id is None:
                ram_dead = True
            else:
                block = ram_allocated.get(child.block_id)
                if block is None or getattr(block, "block_hash", None) is None:
                    ram_dead = True
                else:
                    ram_node = child
                    in_ram = True

        block_hash = compute_block_hash(
            parent_hash,
            block_tokens,
            extra_keys=None,
            model_name=model_name,
        )
        parent_hash = block_hash

        in_ssd_hot = ssd_hot is not None and block_hash in ssd_hot
        in_ssd_disk = False
        if ssd_index is not None:
            try:
                in_ssd_disk = ssd_index.contains(block_hash)
            except Exception:
                in_ssd_disk = False

        if not (in_ram or in_ssd_hot or in_ssd_disk):
            break

        if in_ram:
            blocks_ram += 1
        if in_ssd_hot:
            blocks_ssd_hot += 1
        elif in_ssd_disk:
            blocks_ssd_disk += 1
        cached_prefix_tokens += len(block_tokens)
        cached_block_count += 1
        if in_ssd_hot or in_ssd_disk:
            ssd_hit_tokens += len(block_tokens)

    base.update(
        {
            "total_blocks": total_blocks,
            "blocks_ram": blocks_ram,
            "blocks_ssd_hot": blocks_ssd_hot,
            "blocks_ssd_disk": blocks_ssd_disk,
            "blocks_cold": max(total_blocks - cached_block_count, 0),
            "ram_hit_tokens": blocks_ram * block_size if is_radix else 0,
            "ssd_hit_tokens": ssd_hit_tokens,
            "cached_prefix_tokens": cached_prefix_tokens,
            "cache_index": "radix" if is_radix else "block_aware",
        }
    )
    return base
