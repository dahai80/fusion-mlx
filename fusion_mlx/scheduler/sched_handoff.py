# SPDX-License-Identifier: Apache-2.0
"""
Phase split KV handoff — export KV state from prefill engine, import into decode engine.

Enables split-phase requests: prefill on omlx (fast matmul), decode on Rapid-MLX (lightweight KV).
The KV cache (prompt_cache + block_table) is transferred between engines as MLX array references
(Metal buffers), so this is a zero-copy handoff at the GPU level.
"""

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def export_kv_state(self, request_id: str) -> Optional[dict[str, Any]]:
    """Export KV state for a completed prefill request.

    Called after all prompt tokens are processed but before decode begins.
    Captures the prompt_cache (per-layer KV), block_table (paged cache),
    and computed token count. Then removes the request from the running set.

    Args:
        request_id: The request to export

    Returns:
        Dict with prompt_cache, block_table, num_computed_tokens, prompt_token_ids,
        or None if request not found.
    """
    request = self.requests.get(request_id)
    if request is None:
        logger.warning("export_kv_state: request %s not found", request_id)
        return None

    # Only export if prefill is complete (no remaining prompt tokens)
    remaining = request.remaining_tokens if request.remaining_tokens is not None else []
    if len(remaining) > 0:
        logger.debug(
            "export_kv_state: request %s still has %d remaining prefill tokens, "
            "deferring export", request_id, len(remaining)
        )
        return None

    kv_state = {
        "prompt_cache": request.prompt_cache,
        "block_table": request.block_table,
        "num_computed_tokens": request.num_computed_tokens,
        "prompt_token_ids": request.prompt_token_ids,
        "cached_tokens": request.cached_tokens,
        "shared_prefix_blocks": request.shared_prefix_blocks,
    }

    # Remove from running set so prefill engine doesn't continue decode
    self._remove_request_from_running(request_id)

    logger.info(
        "[KVHandoff] Exported %s: %d computed tokens, %d prefix blocks",
        request_id, kv_state["num_computed_tokens"],
        kv_state["shared_prefix_blocks"],
    )
    return kv_state


def import_kv_state(self, request_id: str, kv_state: dict[str, Any]) -> None:
    """Import KV state into a request, skipping prefill and starting decode directly.

    Restores the prompt_cache, block_table, and computed token count from
    a prefill engine's export. Increments paged cache block ref counts so
    the blocks aren't evicted while the decode engine holds them.

    Args:
        request_id: The request to import state into
        kv_state: KV state dict from export_kv_state()
    """
    request = self.requests.get(request_id)
    if request is None:
        logger.warning("import_kv_state: request %s not found", request_id)
        return

    request.prompt_cache = kv_state.get("prompt_cache")
    request.num_computed_tokens = kv_state.get("num_computed_tokens", 0)
    request.cached_tokens = kv_state.get("cached_tokens", 0)
    request.shared_prefix_blocks = kv_state.get("shared_prefix_blocks", 0)

    # Restore block table and increment ref counts
    block_table = kv_state.get("block_table")
    if block_table is not None:
        request.block_table = block_table
        if self.paged_cache_manager is not None:
            block_ids = getattr(block_table, "block_ids", [])
            for block_id in block_ids:
                self.paged_cache_manager.increment_ref_count(block_id)

    # Set remaining_tokens to empty since prefill is done
    request.remaining_tokens = []

    logger.info(
        "[KVHandoff] Imported into %s: %d computed tokens, block_table=%s",
        request_id, request.num_computed_tokens,
        "yes" if block_table is not None else "none",
    )


def _remove_request_from_running(self, request_id: str) -> None:
    """Remove a request from the running set without marking it finished."""
    if request_id in self.running:
        del self.running[request_id]
        # Also remove from batch generator if present
        if self.batch_generator is not None and request_id in self._uid_to_request:
            uid = self._uid_to_request.pop(request_id, None)
            if uid is not None:
                try:
                    self.batch_generator.remove(uid)
                except Exception:
                    logger.debug("Failed to remove uid %s from batch_generator", uid)
